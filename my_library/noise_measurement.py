"""Project 1, measurements 3 and 4: the photon transfer fit, and dynamic range from it.

One fit feeds every number, so the conversion gain, read noise, saturation capacity and
dynamic range are always a consistent set (Prof. McNames, HW3 feedback: "Pick one
consistent pair and say which"). The fit is the pair-difference photon transfer curve
of Prof. McNames' 05_photon_transfer.ipynb, sections 1 and 3, as implemented for HW3 in
my_library/photon_transfer_curve.py.

The fit uses only pairs below LOW_SIGNAL_CUTOFF of the saturation span. HW3 showed the
pair-difference curve bending upward at higher signal (local slope ~0.25 below 150 DN,
~0.49 at 1100-2100 DN), which drove a fit up to 70% to a negative intercept and a NaN
read noise; the professor attributes the bend to 0.5-0.7% of the mean surviving the pair
difference (the phone or the scene changing between frames). The straight low-signal
part gives K = 0.2567 DN/e- and 6.99 e-, which match a cutoff near 10%.

Dynamic range. With variance(S) = K S + c in DN^2 (S = mean above black, in DN), the read
noise is sqrt(c) DN = sqrt(c) / K electrons and the saturation capacity is span / K
electrons, so

    DR = log2(capacity / read noise) = log2(span / sqrt(c))     (K cancels)

and its uncertainty comes from the intercept alone: sd(DR) = sd(c) / (2 c ln 2).

Units. Photographers count stops (factors of 2): stops = log2(ratio). Sensor datasheets
use decibels on the same signal ratio: dB = 20 log10(ratio) = 20 log10(2) x stops, about
6.02 dB per stop. Both describe one ratio, so they convert exactly.

Where the floor sits. With signal S electrons and read noise sigma_r, SNR = S / sqrt(S +
sigma_r^2) (05_photon_transfer.ipynb section 4). The knee, S = sigma_r^2, is where shot
and read noise are equal; SNR there is sigma_r / sqrt(2). A floor at SNR = 1 lies below
the knee (read-noise limited); a floor at SNR = 10 lies above it once sigma_r is under
about 14 e- (shot-noise limited).
"""
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import my_library.image_metadata as meta
import my_library.photon_transfer_curve as ptc

LOW_SIGNAL_CUTOFF = 0.10          # fraction of the saturation span; see module docstring

CRITERION = ("DR = log2(S_max / S_min) with S_max = white level - black level and S_min = "
             "the read noise (the signal equal to the noise floor, i.e. SNR = 1 counting read "
             "noise only); read noise from the intercept of a pair-difference photon transfer "
             "fit below {cutoff:.0%} of the span. Equivalently {db:.1f} dB as 20 log10(S_max / "
             "S_min). The exact SNR = 1 point, which includes the signal's own shot noise, "
             "gives {exact:.2f} stops.")


# ----------------------------------------------------------------- pure functions

def dynamic_range_from_fit(span_dn, c_dn2, sd_c_dn2):
    """Dynamic range from a photon transfer fit: log2(span / sqrt(c)).

    The conversion gain cancels (see the module docstring), so only the fit's intercept
    and the span are needed.

    Parameters
    ----------
    span_dn : float
        Saturation span, white level minus black level, in DN.
    c_dn2 : float
        Fit intercept: the read-noise variance in DN^2.
    sd_c_dn2 : float
        Standard deviation of c_dn2 from the fit, in DN^2.

    Returns
    -------
    (dr, sd) : tuple of float
        Dynamic range in stops and its standard deviation, sd_c / (2 c ln 2).
        Both are nan if c_dn2 <= 0 (no read-noise floor to divide by).
    """
    if c_dn2 <= 0:
        return float("nan"), float("nan")
    dr = np.log2(span_dn / np.sqrt(c_dn2))
    return float(dr), float(sd_c_dn2 / (2.0 * c_dn2 * np.log(2.0)))


def signal_at_snr(K, c_dn2, target_snr):
    """The signal level at which the camera's SNR reaches target_snr.

    target_snr is a chosen value (e.g. 1 or 10), not a measured SNR. The answer is the
    bottom of the dynamic range for that criterion, counting both shot and read noise.

    Derivation. In electrons (05_photon_transfer.ipynb section 4), with mean signal mu
    and read noise sigma_r:

        SNR = mu / sqrt(mu + sigma_r^2)          (shot-noise variance = mu)

    The camera multiplies electrons by the conversion gain K (DN per electron), so in DN
    the signal is S = K mu, and every variance scales by K^2:

        noise variance = K^2 (mu + sigma_r^2) = K S + c,   with c = (K sigma_r)^2

    K S is the shot noise in DN^2 and c is the read noise in DN^2: the slope and
    intercept of the photon transfer fit, variance = K S + c. The K's cancel in the
    ratio, so this is the same SNR:

        SNR = S / sqrt(K S + c) = K mu / sqrt(K^2 mu + K^2 sigma_r^2)

    Setting SNR = target_snr (t) and squaring both sides gives a quadratic in S,

        S^2 - t^2 K S - t^2 c = 0,

    whose positive root (the only physical signal) is returned:

        S = [ t^2 K + sqrt(t^4 K^2 + 4 t^2 c) ] / 2

    The electron form, mu = [ t^2 + sqrt(t^4 + 4 t^2 sigma_r^2) ] / 2, is S / K.
    Check: with t = 1 and no shot noise (K -> 0), S = sqrt(c), the read noise in DN.

    Parameters
    ----------
    K : float
        Conversion gain in DN per electron (the slope of the photon transfer fit).
    c_dn2 : float
        Read-noise variance in DN^2 (the fit's intercept), c = (K sigma_r)^2.
    target_snr : float
        The SNR that defines the floor (dimensionless), e.g. 1 or 10.

    Returns
    -------
    float or ndarray
        Signal S in DN above black at which SNR = target_snr. Divide by K for electrons.
    """
    t2 = target_snr ** 2
    return 0.5 * (t2 * K + np.sqrt((t2 * K) ** 2 + 4.0 * t2 * c_dn2))


def dynamic_range_at_snr(span_dn, K, c_dn2, target_snr=1.0):
    """Dynamic range with the floor at an exact SNR, shot noise included.

    Parameters
    ----------
    span_dn : float
        Saturation span, white level minus black level, in DN.
    K : float
        Conversion gain in DN per electron.
    c_dn2 : float
        Read-noise variance in DN^2.
    target_snr : float, default 1.0
        The SNR that defines the bottom of the range (see signal_at_snr).

    Returns
    -------
    float
        log2(span / S) in stops, with S the signal at which SNR = target_snr.
    """
    return float(np.log2(span_dn / signal_at_snr(K, c_dn2, target_snr)))


DB_PER_STOP = 20.0 * np.log10(2.0)          # 6.0206 dB per factor of 2


def stops_to_db(stops):
    """Convert stops to decibels: the same ratio on two scales.

    20 log10(ratio) = 20 log10(2) x log2(ratio), about 6.02 dB per stop.

    Parameters
    ----------
    stops : float or array_like
        A ratio expressed as log2(ratio), in stops (also works for a standard deviation).

    Returns
    -------
    numpy.float64 or ndarray
        The same ratio in dB (20 log10): a numpy float for a scalar input, an array
        for an array input.
    """
    return DB_PER_STOP * np.asarray(stops, float)


def dynamic_range_table(span_dn, K, c_dn2, sd_c_dn2=0.0):
    """Dynamic range for three choices of floor, all from the same fit.

    Rows: the read-noise floor (S = sqrt(c), the convention reported in results.json),
    the exact SNR = 1 point (shot noise included), and SNR = 10 (the stricter
    photographic floor). The knee is returned for reference.

    Parameters
    ----------
    span_dn : float
        Saturation span, white level minus black level, in DN.
    K : float
        Conversion gain in DN per electron.
    c_dn2 : float
        Read-noise variance in DN^2 (must be > 0).
    sd_c_dn2 : float, default 0.0
        Standard deviation of c_dn2; gives the uncertainty of the first row only.

    Returns
    -------
    (rows, knee) : tuple
        rows : list of dict, one per floor, with keys
            floor (str, description), floor_dn (DN above black), floor_e (electrons),
            snr_at_floor, ratio (span / floor), stops, sd_stops (None if undefined),
            db (20 log10), sd_db (None if undefined), and region ("below the knee" or
            "above the knee").
        knee : dict with knee_e (sigma_r^2, electrons), knee_dn (DN) and snr_at_knee
            (sigma_r / sqrt(2)): where shot and read noise are equal.
    """
    knee_dn = c_dn2 / K                       # sigma_r^2 electrons, in DN
    dr, sd_dr = dynamic_range_from_fit(span_dn, c_dn2, sd_c_dn2)
    floors = [
        ("read-noise floor, S = sigma_r (reported)", np.sqrt(c_dn2), sd_dr),
        ("exact SNR = 1 (shot noise included)", signal_at_snr(K, c_dn2, 1.0), None),
        ("SNR = 10", signal_at_snr(K, c_dn2, 10.0), None),
    ]
    rows = []
    for name, floor_dn, sd_stops in floors:
        snr = floor_dn / np.sqrt(K * floor_dn + c_dn2)
        stops = float(np.log2(span_dn / floor_dn))
        rows.append(dict(floor=name, floor_dn=float(floor_dn), floor_e=float(floor_dn / K),
                         snr_at_floor=float(snr), ratio=float(span_dn / floor_dn),
                         stops=stops, sd_stops=sd_stops, db=float(stops_to_db(stops)),
                         sd_db=None if sd_stops is None else float(stops_to_db(sd_stops)),
                         region="below the knee" if floor_dn < knee_dn else "above the knee"))
    knee = dict(knee_e=float(knee_dn / K), knee_dn=float(knee_dn),
                snr_at_knee=float(np.sqrt(knee_dn / K) / np.sqrt(2.0)))
    return rows, knee


def saturation_capacity_e(span_dn, K, sd_K):
    """Saturation capacity in electrons: span / K.

    Called saturation capacity, not full well: at a given ISO the ADC can clip before
    the photodiode fills.

    Parameters
    ----------
    span_dn : float
        Saturation span, white level minus black level, in DN.
    K : float
        Conversion gain in DN per electron.
    sd_K : float
        Standard deviation of K, in DN per electron.

    Returns
    -------
    (capacity, sd) : tuple of float
        Capacity in electrons and its standard deviation, span sd_K / K^2.
    """
    return float(span_dn / K), float(span_dn * sd_K / K ** 2)


# ----------------------------------------------------------------- the measurement

def flat_field_files(folder):
    """Raw files in a flat-field folder, searched recursively.

    Parameters
    ----------
    folder : str or Path
        Folder holding the flat-field pairs.

    Returns
    -------
    list of Path
        Raw files sorted by name; consecutive files form the pairs.
    """
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in ptc.RAW_SUFFIXES)


def gain_matched_pair_variance(a, b, black):
    """Temporal noise variance of one frame from a pair, with the pair's means matched.

    The math. Two frames of the same flat scene, above black, are
        a = s(x) + n_a(x),   b = s(x) (1 - eps) + n_b(x)
    where s(x) is the signal pattern (the mean S times pixel-gain variation and lens
    shading), n_a and n_b are independent temporal noise, and eps is the fraction by
    which the light changed between the frames. The plain pair difference is
        a - b = eps s(x) + n_a - n_b,
    and the variance (which removes the mean) is 2 sigma^2 + eps^2 var_x[s(x)], with
    var_x[s(x)] = (p S)^2, p the relative pattern spread (PRNU plus shading). The mean
    offset eps S itself never enters the variance. So a uniform change matters only
    through eps p S: with eps = 0.6% and p = 2% that is 0.012% of S, a variance about
    a thousandth of the shot noise even near saturation. Gain matching is therefore
    nearly a no-op on a flat field, and computing it shows that numerically.

    It does NOT explain or fix the HW3 bend. Prof. McNames measured 0.5-0.7% *of the
    mean* surviving the difference; a uniform change leaves only eps p S. A residual
    that large must vary across the patch (light changing differently across it, or
    the camera moving across a non-flat scene): see plane_removed_pair_variance and
    difference_structure.

    Scaling b by g = mean(a) / mean(b) (means above black) removes eps s(x) for a
    uniform change: a - g b = n_a - g n_b, var = sigma^2 (1 + g^2), so the per-frame
    variance is var(a - g b) / (1 + g^2) (dividing by 1 + g^2, not 2, accounts for
    scaling b's noise by g).

    Parameters
    ----------
    a, b : ndarray
        The two patches, raw DN (black not yet subtracted), same shape.
    black : float
        Black level in DN.

    Returns
    -------
    (var, eps) : tuple of float
        var : per-frame temporal noise variance in DN^2, matched means.
        eps : mean(a) / mean(b) - 1, the relative change in light between the frames.
    """
    sa = np.asarray(a, float) - black
    sb = np.asarray(b, float) - black
    g = sa.mean() / sb.mean()
    return float(np.var(sa - g * sb) / (1.0 + g ** 2)), float(g - 1.0)


def _plane(h, w):
    """Design matrix [1, x, y] for a least-squares plane over an h x w patch."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.column_stack([np.ones(h * w), xx.ravel(), yy.ravel()])


def plane_removed_pair_variance(a, b, black=0.0):
    """Temporal noise variance of one frame from a pair, after removing a fitted plane
    from the difference image.

    The math. If the light changed by a different fraction across the patch between the
    two frames (a sky gradient shifting, the camera moving slightly across a scene that
    is not perfectly flat), the difference a - b holds that change as a smooth surface
    on top of the noise, and its variance grows with the square of the signal: the
    upward bend in HW3's photon transfer curve. To first order such a change is a
    gradient, a plane d0 + gx x + gy y. Fitting that plane to d = a - b by least
    squares and subtracting it leaves the noise, n_a - n_b, whose variance is 2 sigma^2.
    Removing 3 fitted parameters from N pixels lowers the residual variance by
    (N - 3) / N, which is corrected, so

        sigma^2 = var(d - plane) * N / (N - 3) / 2.

    Curvature or blotches are not removed; difference_structure's block ratio says
    whether they remain.

    Parameters
    ----------
    a, b : ndarray
        The two patches of one pair, raw DN, same shape.
    black : float, default 0.0
        Black level in DN (cancels in the difference; kept for a uniform signature).

    Returns
    -------
    float
        Per-frame temporal noise variance in DN^2.
    """
    d = (np.asarray(a, float) - black) - (np.asarray(b, float) - black)
    h, w = d.shape
    A = _plane(h, w)
    coef, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
    resid = d.ravel() - A @ coef
    n = resid.size
    return float(np.var(resid) * n / (n - 3) / 2.0)


def fit_ptc_weighted(mean, var, span, cutoff, n_pixels, n_params_removed=0, iterations=2):
    """Weighted straight-line fit var = K S + c over pairs below cutoff x span.

    The math. Each pair's variance is estimated from N pixels. For Gaussian noise, a
    sample variance has a relative standard deviation of sqrt(2 / (N - 1)), so its
    absolute error is proportional to the variance itself:
        sd(var_i) = (K S_i + c) sqrt(2 / (N_i - 1 - n_params_removed)).
    Bright pairs therefore carry much larger absolute errors than dim ones. An
    unweighted fit (photon_transfer_curve.fit_ptc) treats all points as equally
    uncertain, which misplaces the uncertainty between slope and intercept: on a
    simulated sensor its reported sd(K) was 1.6x too small and sd(c) 1.6x too large.
    Weighted least squares minimizes sum((var_i - K S_i - c)^2 / sd_i^2), and the
    covariance of (K, c) then follows from the known sd_i (np.polyfit with
    cov="unscaled").

    The weights use the model's predicted variance, not the measured one (weighting by
    the measured value would favor points that happened to come out low): start from
    the unweighted fit, then refit `iterations` times with sd_i from the latest line.

    Verified on 30 simulated sensors: sd(K) is calibrated (actual scatter / reported =
    0.98); sd(c) is conservative (0.66), partly because rounding to whole DN gives the
    noise lighter tails than a Gaussian, so variance estimates scatter about 13% less
    than sqrt(2 / (N - 1)) assumes. Both are 1.3x (K) to 4x (c) more precise than the
    unweighted fit.

    Check of the assumption: if the points scatter about the line more than the sd_i
    predict (reduced chi-square chi2_red = chi^2 / (n - 2) > 1, e.g. from drift between
    frames), the covariance is scaled by chi2_red, so the uncertainties are never
    smaller than the scatter supports.

    Parameters
    ----------
    mean, var : array_like
        Mean signal above black (DN) and variance (DN^2) of each pair.
    span : float
        Saturation span in DN.
    cutoff : float
        Fit only pairs with mean < cutoff x span.
    n_pixels : array_like or int
        Pixels behind each variance (N_i).
    n_params_removed : int, default 0
        Parameters fitted out of each difference image before its variance (3 for the
        plane-removed estimate), reducing the degrees of freedom.
    iterations : int, default 2
        Reweighting passes after the initial unweighted fit.

    Returns
    -------
    dict
        K, sK (DN/e-); c, sc (DN^2); read, s_read (e-, nan if c <= 0, with the K-c
        covariance included); used (boolean mask of pairs in the fit); chi2_red; cov
        (2 x 2 covariance of K, c, after any chi2_red scaling).
    """
    mean = np.asarray(mean, float)
    var = np.asarray(var, float)
    n = np.broadcast_to(np.asarray(n_pixels, float), mean.shape)
    used = mean < cutoff * span
    x, y, dof_pix = mean[used], var[used], n[used] - 1 - n_params_removed
    K, c = np.polyfit(x, y, 1)
    for _ in range(iterations):
        sd = np.maximum(K * x + c, 1e-12) * np.sqrt(2.0 / dof_pix)
        (K, c), cov = np.polyfit(x, y, 1, w=1.0 / sd, cov="unscaled")
    chi2_red = float(np.sum(((y - K * x - c) / sd) ** 2) / max(len(x) - 2, 1))
    cov = cov * max(chi2_red, 1.0)
    sK, sc = np.sqrt(np.diag(cov))
    if c > 0:
        read = np.sqrt(c) / K
        d_dc, d_dK = 0.5 / (np.sqrt(c) * K), -np.sqrt(c) / K ** 2
        s_read = np.sqrt(d_dc ** 2 * sc ** 2 + d_dK ** 2 * sK ** 2 + 2 * d_dc * d_dK * cov[0, 1])
    else:
        read = s_read = float("nan")
    return dict(K=float(K), sK=float(sK), c=float(c), sc=float(sc), read=float(read),
                s_read=float(s_read), used=used, chi2_red=chi2_red, cov=cov)


PAIR_METHODS = ("pair difference", "gain-matched", "plane-removed")
_VAR_KEY = {"pair difference": "var", "gain-matched": "var_matched", "plane-removed": "var_plane"}


def measure_ptc(folder, white_level, black_level, cutoff=LOW_SIGNAL_CUTOFF, method="pair difference"):
    """Photon transfer fit on the consecutive flat-field pairs in folder.

    The math (05_photon_transfer.ipynb sections 1 and 3). For a mean signal S (DN above
    black) the temporal noise variance of one frame is
        var(S) = K S + c,   K = conversion gain (DN/e-), c = (K sigma_r)^2 (DN^2)
    because shot noise in electrons has variance equal to the mean (Poisson) and the
    camera scales electrons by K (variances by K^2). Each pair gives one (S, var) point;
    a straight-line fit over the pairs below cutoff x span gives K (slope) and c
    (intercept), weighted by each point's known sampling error (fit_ptc_weighted). The
    read noise in electrons is sqrt(c) / K.

    Three variances are computed for every pair: the plain pair difference (HW3), the
    gain-matched one (gain_matched_pair_variance) and the plane-removed one
    (plane_removed_pair_variance). All three are fitted; `method` picks which fit is
    "the" fit used for results.

    Runs the lens_id assertion and series identity on the files first, and uses the
    central 128 x 128 patch of one green plane, as in HW3. Stops (SystemExit) if there
    are fewer than two pairs, if the pairs mix ISOs, or if fewer than three pairs fall
    below the cutoff.

    Parameters
    ----------
    folder : str or Path
        Flat-field pairs, consecutive by file name.
    white_level : float
        Saturation level in DN (e.g. 4095).
    black_level : float
        Black level in DN (e.g. 528). The span is white_level - black_level.
    cutoff : float, default LOW_SIGNAL_CUTOFF (0.10)
        Fit only pairs whose mean is below this fraction of the span.
    method : str, default "pair difference"
        One of PAIR_METHODS: which fit the results use.

    Returns
    -------
    dict
        fit : the selected fit, a dict from fit_ptc_weighted (K, sK in DN/e-; c, sc in
            DN^2; read, s_read in e-; used, a boolean mask of pairs; chi2_red; cov).
        fits : {method name: fit} for all three methods.
        method : the selected method's name.
        rows : list of dict, one per pair (mean, var, var_matched, var_plane, eps, var_single,
            iso, black, white, peak, flagged, files, ...).
        pairs : list of dict, the patches (a, b) and settings of each pair.
        span : float, the span in DN.  iso : the ISO of every pair.
        cutoff : float, as given.  n_used : int, pairs in the selected fit.
        ident : dict, the camera identity from series_identity.
        files : list of Path, the flat-field files.
        white_level, black_level : as given.
    """
    paths = flat_field_files(folder)
    if len(paths) < 4:
        raise SystemExit(f"need at least two flat-field pairs in {folder}; found {len(paths)} files")
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(paths)} flat-field files:")
    meta.assert_one_camera(paths)
    ident, n = meta.series_identity(paths)

    with contextlib.redirect_stdout(io.StringIO()):     # load_flat_pairs prints a HW3 heading
        pairs = ptc.load_flat_pairs(folder)
    rows = ptc.analyze_pairs(pairs)
    for row, pair in zip(rows, pairs):
        row["var_matched"], row["eps"] = gain_matched_pair_variance(pair["a"], pair["b"], pair["black"])
        row["var_plane"] = plane_removed_pair_variance(pair["a"], pair["b"], pair["black"])
    isos = sorted({r["iso"] for r in rows})
    blacks = sorted({r["black"] for r in rows})
    if len(isos) != 1:
        raise SystemExit(f"flat fields mix ISOs {isos}; one ISO per measurement")
    if blacks != [black_level]:
        print(f"  note: files declare black level {blacks}; the span uses {black_level} DN")
    span = float(white_level - black_level)

    if method not in PAIR_METHODS:
        raise ValueError(f"method must be one of {PAIR_METHODS}")
    mean = np.array([r["mean"] for r in rows])
    n_pix = np.array([r["n_pixels"] for r in rows])
    fits = {m: fit_ptc_weighted(mean, np.array([r[_VAR_KEY[m]] for r in rows]), span, cutoff,
                                n_pix, n_params_removed=3 if m == "plane-removed" else 0)
            for m in PAIR_METHODS}
    fit = fits[method]
    n_used = int(fit["used"].sum())
    flagged = sum(r["flagged"] for r in rows)
    print(f"{len(rows)} pairs at ISO {isos[0]}; {n_used} below {cutoff:.0%} of the "
          f"{span:.0f} DN span used in the fit ({method}); {flagged} pair(s) flagged for mean mismatch")
    if n_used < 3:
        raise SystemExit(f"only {n_used} pairs below {cutoff:.0%} of the span; too few to fit")
    return dict(fit=fit, fits=fits, method=method, rows=rows, pairs=pairs, span=span,
                iso=isos[0], cutoff=cutoff, n_used=n_used, ident=ident, files=paths,
                white_level=white_level, black_level=black_level)


def report_dynamic_range(ptc_result):
    """Print the dynamic range, the floor comparison table and the knee.

    Parameters
    ----------
    ptc_result : dict
        The return value of measure_ptc.

    Returns
    -------
    dict
        dr, sd_dr : dynamic range and sd in stops (nan if the intercept is not positive).
        dr_db, sd_dr_db : the same in dB (20 log10).
        exact_snr1, snr10 : stops with the floor at exact SNR = 1 and at SNR = 10.
        table, knee : as returned by dynamic_range_table ([] and None if c <= 0).
        capacity_e, sd_capacity_e : saturation capacity in electrons and its sd.
        K, sK, c, sc, read_e, s_read_e : the fit values (DN/e-, DN^2, e-).
    """
    fit, span = ptc_result["fit"], ptc_result["span"]
    K, sK, c, sc = fit["K"], fit["sK"], fit["c"], fit["sc"]
    dr, sd_dr = dynamic_range_from_fit(span, c, sc)
    cap, sd_cap = saturation_capacity_e(span, K, sK)
    exact = dynamic_range_at_snr(span, K, c, 1.0) if c > 0 else float("nan")
    snr10 = dynamic_range_at_snr(span, K, c, 10.0) if c > 0 else float("nan")
    table, knee = dynamic_range_table(span, K, c, sc) if c > 0 else ([], None)
    print(f"\nPhoton transfer fit below {ptc_result['cutoff']:.0%} of span "
          f"({ptc_result['n_used']} pairs, ISO {ptc_result['iso']}):")
    print(f"  K = {K:.4f} +/- {sK:.4f} DN/e-,  c = {c:.3f} +/- {sc:.3f} DN^2,  "
          f"read noise = {np.sqrt(max(c, 0)):.3f} DN = {fit['read']:.2f} +/- {fit['s_read']:.2f} e-")
    print(f"  saturation capacity = span / K = {span:.0f} / {K:.4f} = {cap:.0f} +/- {sd_cap:.0f} e-")
    print(f"  dynamic range = log2(span / sqrt(c)) = {dr:.2f} +/- {sd_dr:.2f} stops "
          f"= {stops_to_db(dr):.1f} +/- {stops_to_db(sd_dr):.1f} dB (K cancels; uncertainty from the intercept)")
    if table:
        print(f"\n  Same fit, three floors. Ceiling = span = {span:.0f} DN = {cap:.0f} e-. "
              f"Knee (shot = read noise) at {knee['knee_e']:.1f} e- = {knee['knee_dn']:.2f} DN, "
              f"SNR {knee['snr_at_knee']:.2f} there.")
        print(f"  {'floor':42s} {'S_min DN':>9s} {'S_min e-':>9s} {'SNR':>5s} {'ratio':>7s} "
              f"{'stops':>12s} {'dB (20 log10)':>14s}  where")
        for r in table:
            pm = lambda v, sd, f: f"{v:{f}}" + (f" +/- {sd:.2f}" if sd is not None else "")
            print(f"  {r['floor']:42s} {r['floor_dn']:9.3f} {r['floor_e']:9.2f} {r['snr_at_floor']:5.2f} "
                  f"{r['ratio']:7.0f} {pm(r['stops'], r['sd_stops'], '.2f'):>12s} "
                  f"{pm(r['db'], r['sd_db'], '.1f'):>14s}  {r['region']}")
        print(f"  ({DB_PER_STOP:.4f} dB per stop: both columns describe the same ratio.)")
    return dict(dr=dr, sd_dr=sd_dr, exact_snr1=exact, snr10=snr10,
                dr_db=float(stops_to_db(dr)) if np.isfinite(dr) else float("nan"),
                sd_dr_db=float(stops_to_db(sd_dr)) if np.isfinite(sd_dr) else float("nan"),
                table=table, knee=knee,
                capacity_e=cap, sd_capacity_e=sd_cap, K=K, sK=sK, c=c, sc=sc,
                read_e=fit["read"], s_read_e=fit["s_read"])


def write_dynamic_range_table(summary, path):
    """Save the floor comparison table as a CSV for the write-up.

    Parameters
    ----------
    summary : dict
        The return value of report_dynamic_range (uses its "table").
    path : str or Path
        Where to write the CSV; parent folders are created.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["table"]).to_csv(path, index=False)
    return path


def results_entries(summary, cutoff=LOW_SIGNAL_CUTOFF, decimals=2):
    """The dynamic-range keys of results.json.

    Parameters
    ----------
    summary : dict
        The return value of report_dynamic_range.
    cutoff : float, default LOW_SIGNAL_CUTOFF
        The fit cutoff used, stated in the criterion string.
    decimals : int, default 2
        Rounding for the stops value and its uncertainty.

    Returns
    -------
    dict
        dynamic_range_stops : [value, sd] in stops, or None if the dynamic range is
            undefined (then no criterion is written).
        dynamic_range_criterion : str stating the convention, the cutoff and the dB
            and exact-SNR equivalents.
    """
    if not np.isfinite(summary["dr"]):
        return {"dynamic_range_stops": None}
    return {"dynamic_range_stops": [round(summary["dr"], decimals), round(summary["sd_dr"], decimals)],
            "dynamic_range_criterion": CRITERION.format(cutoff=cutoff, exact=summary["exact_snr1"],
                                                        db=summary["dr_db"])}


# ================================================================= measurement 3: noise
# The functions below are the noise step (project1.py --noise). They share
# measure_ptc with the dynamic-range step, so the gain, read noise and dynamic range
# always come from one fit.

def neighbor_correlation(d):
    """Correlation of each pixel's value with its right and lower neighbor.

    The math. For independent pixel noise the correlation between horizontally (or
    vertically) adjacent pixels of a difference image is 0, within about
    1 / sqrt(number of pairs). A negative value means a pixel's noise tends to go
    opposite to its neighbors': sharpening (a high-pass boost) does exactly that. A
    positive value means smoothing (blurring, denoising). The difference image is
    used because the scene and the fixed pattern cancel in it, leaving only noise.

    Note the spacing: in one green plane of the Bayer mosaic, adjacent samples are two
    full-resolution pixels apart.

    Parameters
    ----------
    d : ndarray
        A 2-D noise image (e.g. a difference image), any offset.

    Returns
    -------
    (rho_x, rho_y) : tuple of float
        Lag-1 correlation along rows (x) and down columns (y).
    """
    d = np.asarray(d, float) - np.mean(d)
    rho_x = np.sum(d[:, :-1] * d[:, 1:]) / np.sqrt(np.sum(d[:, :-1] ** 2) * np.sum(d[:, 1:] ** 2))
    rho_y = np.sum(d[:-1, :] * d[1:, :]) / np.sqrt(np.sum(d[:-1, :] ** 2) * np.sum(d[1:, :] ** 2))
    return float(rho_x), float(rho_y)


def autocorrelation(d, max_lag):
    """Correlation of a noise image with itself shifted by (u, v), for |u|, |v| <= max_lag.

    Parameters
    ----------
    d : ndarray
        A 2-D noise image (e.g. a difference image).
    max_lag : int
        Largest shift in pixels, in each direction.

    Returns
    -------
    ndarray, shape (2 max_lag + 1, 2 max_lag + 1)
        rho[max_lag + v, max_lag + u] is the correlation at shift u along rows (x) and v
        down columns (y); the center is 1.
    """
    d = np.asarray(d, float) - np.mean(d)
    h, w = d.shape
    rho = np.empty((2 * max_lag + 1, 2 * max_lag + 1))
    for v in range(-max_lag, max_lag + 1):
        for u in range(-max_lag, max_lag + 1):
            a = d[max(v, 0):h + min(v, 0), max(u, 0):w + min(u, 0)]
            b = d[max(-v, 0):h + min(-v, 0), max(-u, 0):w + min(-u, 0)]
            rho[max_lag + v, max_lag + u] = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return rho


def predicted_block_ratio(d, block=8):
    """Block-structure ratio R expected from the noise's own spatial correlation.

    The math. For noise with variance sigma^2 and correlation rho(u, v) between pixels
    offset by (u, v), the mean of a b x b block has variance
        (sigma^2 / b^4) sum over all pixel pairs in the block of rho(offset)
      = (sigma^2 / b^2) sum_{|u|,|v| < b} (b - |u|)(b - |v|) rho(u, v) / b^2,
    because a b-wide block holds (b - |u|) pairs of pixels offset by u along a row (and
    likewise in v). Dividing by the independent-noise value sigma^2 / b^2 gives
        R^2 = sum_{|u|,|v| < b} (b - |u|)(b - |v|) rho(u, v) / b^2.
    With no correlation only rho(0, 0) = 1 survives and R = 1. Keeping only adjacent
    neighbors gives the shorthand R ~ sqrt(1 + 2 (rho_x + rho_y)(b - 1) / b), but a
    sharpening kernel also correlates diagonal and two-apart pixels, so the full sum is
    used. If the measured R matches this prediction, the correlation alone accounts for
    it; if the measured R is larger, there is also structure (a gradient, blotches).

    Parameters
    ----------
    d : ndarray
        A 2-D noise image (e.g. the plane-removed difference image).
    block : int, default 8
        Block side in pixels.

    Returns
    -------
    float
        The predicted R (nan if the sum is not positive).
    """
    lag = block - 1
    rho = autocorrelation(d, lag)
    u = np.arange(-lag, lag + 1)
    weight = np.outer(block - np.abs(u), block - np.abs(u)) / block ** 2
    r2 = float(np.sum(weight * rho))
    return float(np.sqrt(r2)) if r2 > 0 else float("nan")


def difference_structure(a, b, black, block=8):
    """Does the pair's difference image hold structure beyond noise? A number, not a look.

    The math. If a and b differ only by temporal noise, d = a - b is white noise with
    some standard deviation sigma_d, and the mean of a block of block x block pixels has
    standard deviation sigma_d / block. So the ratio
        R = std(block means of d) / (sigma_d / block)
    is about 1 for pure noise. A uniform change in light (eps) leaves eps s(x) in d,
    whose pixel-gain and shading pattern survives block averaging, so R > 1. A
    gradient or blotches (light that changed differently across the patch) also give
    R > 1 and, unlike a uniform change, survive gain matching. A plane fitted to d,
    d ~ d0 + gx x + gy y, measures any gradient directly.

    R is computed for the plain difference, the gain-matched one (b scaled by
    g = mean(a) / mean(b)) and the plane-removed one. A uniform change in light barely
    shows (it leaves only eps x the small pattern; see gain_matched_pair_variance), so
    R_plain well above 1 means the light changed differently across the patch. If
    removing a plane brings R back to about 1, the change was a gradient and
    plane_removed_pair_variance handles it; if not, blotches or curvature remain.

    Parameters
    ----------
    a, b : ndarray
        The two patches of one pair, raw DN, same shape (cropped to a multiple of block).
    black : float
        Black level in DN.
    block : int, default 8
        Side of the averaging blocks, in pixels.

    Returns
    -------
    dict
        eps : mean(a) / mean(b) - 1 (means above black).
        R_plain, R_matched, R_plane : the block-structure ratios (about 1 for pure noise).
        rho : (rho_x, rho_y), neighbor correlations of the plane-removed difference
            (neighbor_correlation); R_predicted : the R its full spatial correlation
            implies (predicted_block_ratio).
        gradient_dn : (gx, gy) of the plane fitted to the plain difference, in DN per
            pixel; gradient_matched_dn likewise for the matched difference.
        d_plain, d_matched, d_plane : the difference images (DN), for plotting.
        blocks_plain, blocks_matched, blocks_plane : their block-mean maps (DN).
    """
    sa = np.asarray(a, float) - black
    sb = np.asarray(b, float) - black
    h, w = (sa.shape[0] // block) * block, (sa.shape[1] // block) * block
    sa, sb = sa[:h, :w], sb[:h, :w]
    g = sa.mean() / sb.mean()

    A = _plane(h, w)

    def analyze(d):
        blocks = d.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
        ratio = blocks.std() / (d.std() / block)
        coef, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
        return float(ratio), (float(coef[1]), float(coef[2])), blocks, coef

    d_plain, d_matched = sa - sb, sa - g * sb
    R_plain, grad_plain, blocks_plain, coef = analyze(d_plain)
    R_matched, grad_matched, blocks_matched, _ = analyze(d_matched)
    d_plane = d_plain - (A @ coef).reshape(h, w)
    R_plane, _, blocks_plane, _ = analyze(d_plane)
    rho = neighbor_correlation(d_plane)
    return dict(eps=float(g - 1.0), R_plain=R_plain, R_matched=R_matched, R_plane=R_plane,
                rho=rho, R_predicted=predicted_block_ratio(d_plane, block=block),
                gradient_dn=grad_plain, gradient_matched_dn=grad_matched,
                d_plain=d_plain, d_matched=d_matched, d_plane=d_plane,
                blocks_plain=blocks_plain, blocks_matched=blocks_matched, blocks_plane=blocks_plane)


def noise_power_spectrum(diffs, band_low=(0.02, 0.15), band_high=(0.35, 0.5)):
    """Normalized noise power spectrum of difference images, averaged over pairs.

    The math. For each noise image d (zero mean, N pixels) the periodogram
        P(fx, fy) = |FFT2(d)|^2 / (N var(d))
    is normalized so that its average over all frequencies is 1 (Parseval's theorem:
    the sum of |FFT|^2 equals N times the sum of d^2). For independent pixels the
    expected periodogram is flat at 1 everywhere. By the Wiener-Khinchin theorem P is
    the Fourier transform of the autocorrelation rho(u, v), so the anti-correlated
    neighbors measured by neighbor_correlation must appear here as a spectrum that
    rises toward high frequency: for lag-1 correlations alone,
        P(fx, fy) = 1 + 2 rho_x cos(2 pi fx) + 2 rho_y cos(2 pi fy),
    below 1 near zero frequency and above 1 near Nyquist (0.5) when rho < 0. If the noise
    is white noise passed through a filter H, P is |H(f)|^2 normalized to mean 1, so the
    spectrum shows the filter's frequency response directly.

    One periodogram scatters by about 100% per frequency bin. Averaging the normalized
    periodograms of n pairs reduces that by sqrt(n); the 1-D profiles average many
    bins further. Frequencies are in cycles per sample of the image (cycles per
    green-plane pixel for a green plane; Nyquist 0.5). The zero-frequency bin carries
    no information after mean and plane removal and is left out of the profiles.

    Parameters
    ----------
    diffs : sequence of ndarray
        Noise images of equal shape, e.g. plane-removed pair differences.
    band_low, band_high : (float, float), default (0.02, 0.15) and (0.35, 0.5)
        Radial frequency bands (cycles per sample) for the summary ratio.

    Returns
    -------
    dict
        nps : 2-D averaged spectrum, zero frequency at the center (fftshift).
        fx, fy : frequency axes for nps (cycles per sample).
        horizontal, vertical : (f, P) 1-D spectra of rows and of columns (P averaged
            over the other frequency axis), f > 0.
        radial : (f, P) averaged in rings of width 1/min(h, w).
        low, high, ratio : mean P in band_low and band_high, and high / low
            (about 1 for independent noise; > 1 for sharpening, < 1 for smoothing).
        n_pairs : how many images were averaged.
    """
    diffs = [np.asarray(d, float) for d in diffs]
    h, w = diffs[0].shape
    acc = np.zeros((h, w))
    for d in diffs:
        d0 = d - d.mean()
        acc += np.abs(np.fft.fft2(d0)) ** 2 / (d0.size * d0.var())
    nps = np.fft.fftshift(acc / len(diffs))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    horizontal = nps.mean(axis=0)            # spectrum of rows: average over fy
    vertical = nps.mean(axis=1)              # spectrum of columns: average over fx
    keep_x, keep_y = fx > 0, fy > 0
    fr = np.hypot(*np.meshgrid(fx, fy))
    edges = np.arange(0.0, 0.5 + 1e-9, 1.0 / min(h, w))
    centers, ring = [], []
    for lo, hi in zip(edges[1:-1], edges[2:]):
        sel = (fr >= lo) & (fr < hi)
        centers.append(0.5 * (lo + hi)); ring.append(nps[sel].mean())
    band = lambda b: float(nps[(fr >= b[0]) & (fr < b[1])].mean())
    low, high = band(band_low), band(band_high)
    return dict(nps=nps, fx=fx, fy=fy,
                horizontal=(fx[keep_x], horizontal[keep_x]), vertical=(fy[keep_y], vertical[keep_y]),
                radial=(np.array(centers), np.array(ring)),
                low=low, high=high, ratio=high / low, n_pairs=len(diffs),
                band_low=band_low, band_high=band_high)


def remove_plane(d):
    """Subtract the least-squares plane d0 + gx x + gy y from a 2-D image.

    Parameters
    ----------
    d : ndarray
        A 2-D image, e.g. a pair difference.

    Returns
    -------
    ndarray
        The residual, same shape as d.
    """
    d = np.asarray(d, float)
    h, w = d.shape
    A = _plane(h, w)
    coef, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
    return d - (A @ coef).reshape(h, w)


def dark_frame_pairs(folder, size=128):
    """Pairs of dark frames taken at the same exposure time and ISO.

    Frames are grouped by (exposure, ISO) from their metadata and paired in file-name
    order within each group (1st with 2nd, 3rd with 4th, ...); an odd frame left over
    in a group is not used.

    Parameters
    ----------
    folder : str or Path
        Dark frames (lens covered).
    size : int, default 128
        Side of the central green-plane patch, as for the flat fields.

    Returns
    -------
    (pairs, isos) : tuple
        pairs : list of (a, b, exposure_s) with the two patches in DN.
        isos : sorted list of the ISO values present.
    """
    groups = {}
    for path in flat_field_files(folder):
        md = meta.raw_metadata(path)
        groups.setdefault((md["exposure_s"], md["iso"]), []).append(path)
    pairs = []
    for (exposure, _), paths in sorted(groups.items()):
        for p1, p2 in zip(paths[0::2], paths[1::2]):
            a, _ = meta.green_patch(p1, size)
            b, _ = meta.green_patch(p2, size)
            pairs.append((a, b, exposure))
    return pairs, sorted({iso for _, iso in groups})


def dark_difference_spectrum(folder):
    """Noise power spectrum of dark-frame differences: read noise only.

    The math. A dark frame has no photons, so the difference of two dark frames taken
    at the same settings holds only read noise (plus a negligible amount of dark-
    current shot noise), with the fixed pattern cancelled. Read noise arises in the
    readout electronics, after the photodiode. If its spectrum has the same shape as
    the flat-field spectrum (shot plus read noise), whatever shapes the spectrum acts
    after readout, on all noise alike: a digital step in the path to the DNG. If the
    dark spectrum were flat while the flat-field one rises, the shaping would act only
    on the photo-signal (for example optical or electrical crosstalk between photodiodes).

    Each difference has its best-fit plane removed (remove_plane), then the normalized
    periodograms are averaged (noise_power_spectrum).

    Parameters
    ----------
    folder : str or Path
        Dark frames, several per exposure time.

    Returns
    -------
    dict or None
        None if fewer than one pair. Otherwise the noise_power_spectrum dict, plus
        iso (list of ISOs present), rho (mean lag-1 neighbor correlation over pairs,
        (rho_x, rho_y)), and n_pairs.
    """
    pairs, isos = dark_frame_pairs(folder)
    if not pairs:
        return None
    diffs = [remove_plane(a - b) for a, b, _ in pairs]
    spec = noise_power_spectrum(diffs)
    rhos = np.array([neighbor_correlation(d) for d in diffs])
    spec.update(iso=isos, rho=tuple(rhos.mean(axis=0)))
    return spec


def plot_spectra_compared(spectra, title, out_png):
    """Radial noise power spectra of several sources on one plot.

    Parameters
    ----------
    spectra : list of (label, spectrum dict from noise_power_spectrum)
    title : str
    out_png : str or Path

    Returns
    -------
    Path
        The path written.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, spec in spectra:
        f, p = spec["radial"]
        ax.plot(f, p, lw=1.4, label=f"{label} ({spec['n_pairs']} pairs, high/low {spec['ratio']:.2f})")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_xlim(0, 0.5); ax.set_xlabel("frequency (cycles / sample; Nyquist 0.5)")
    ax.set_ylabel("normalized power (1 = independent noise)")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def plot_noise_power_spectrum(spec, title, out_png):
    """The averaged 2-D noise power spectrum (log scale) and its 1-D profiles.

    Parameters
    ----------
    spec : dict
        The return value of noise_power_spectrum.
    title : str
        Figure title.
    out_png : str or Path
        Where to save the figure.

    Returns
    -------
    Path
        The path written.
    """
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fx, fy = spec["fx"], spec["fy"]
    im = a1.imshow(np.log10(np.maximum(spec["nps"], 1e-3)), cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   extent=(fx[0], fx[-1], fy[-1], fy[0]))
    fig.colorbar(im, ax=a1, label="log10 P  (0 = flat, independent noise)")
    a1.set_xlabel("fx (cycles / sample)"); a1.set_ylabel("fy (cycles / sample)")
    a1.set_title(f"2-D spectrum, {spec['n_pairs']} pairs averaged")
    for (f, p), name, style in ((spec["horizontal"], "rows (along x)", "-"),
                                (spec["vertical"], "columns (along y)", "-"),
                                (spec["radial"], "radial average", "k--")):
        a2.plot(f, p, style, lw=1.5 if style == "k--" else 1.2, label=name)
    a2.axhline(1.0, color="0.5", ls=":", lw=1)
    for b, c in ((spec["band_low"], "#cfe3cf"), (spec["band_high"], "#f3d6c4")):
        a2.axvspan(*b, color=c, alpha=0.5, lw=0)
    a2.set_xlim(0, 0.5); a2.set_xlabel("frequency (cycles / sample; Nyquist 0.5)")
    a2.set_ylabel("normalized power (1 = independent noise)")
    a2.set_title(f"profiles: high / low band = {spec['ratio']:.2f}")
    a2.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def brightest_unclipped_pair(ptc_result, max_peak_frac=0.98):
    """Index of the pair with the highest mean whose peak stays below max_peak_frac of
    the white level (the most sensitive pair for spotting drift; HW3 feedback).

    Parameters
    ----------
    ptc_result : dict
        The return value of measure_ptc.
    max_peak_frac : float, default 0.98
        Pairs whose brightest pixel reaches this fraction of the white level are skipped.

    Returns
    -------
    int or None
        Index into ptc_result["rows"] and ["pairs"], or None if every pair is clipped.
    """
    rows = ptc_result["rows"]
    ok = [i for i, r in enumerate(rows) if r["peak"] < max_peak_frac * ptc_result["white_level"]]
    return max(ok, key=lambda i: rows[i]["mean"]) if ok else None


def single_frame_bias(ptc_result):
    """Why a single frame gives the wrong answer, in numbers.

    The math. The variance of one frame across the patch is spatial, not temporal: it
    adds every pixel-to-pixel difference that is the same in every frame,
        var_single(S) = K S + c + DSNU^2 + (p S)^2,
    where DSNU is the dark-signal (offset) pattern and p the relative gain pattern:
    photo-response non-uniformity (PRNU) plus any lens shading inside the patch. The
    pair difference cancels both, leaving K S + c. Two consequences:
      1. A straight line through single-frame variances has a slope inflated by the
         (p S)^2 term: a bias toward higher K, growing with the fit range, not just a
         noisier estimate (Prof. McNames, HW3 feedback, item 7).
      2. The excess, var_single - var_pair = DSNU^2 + (p S)^2, is a parabola in S;
         fitting it gives p, the pattern's relative size.

    Parameters
    ----------
    ptc_result : dict
        The return value of measure_ptc (its rows carry var_single and the pair
        variance of the selected method).

    Returns
    -------
    dict
        fit_single : straight-line fit (fit_ptc) of the single-frame variances over the
            same pairs as the selected fit.
        K_ratio : K_single / K_pair (> 1 means the single-frame slope is too steep).
        p : relative pattern spread (PRNU plus in-patch shading), from
            excess = DSNU^2 + p^2 S^2 fitted over all unclipped pairs.
        dsnu_dn : sqrt of the fitted offset term in DN (nan if negative).
    """
    rows, span = ptc_result["rows"], ptc_result["span"]
    key = _VAR_KEY[ptc_result["method"]]
    mean = np.array([r["mean"] for r in rows])
    single = np.array([r["var_single"] for r in rows])
    pair = np.array([r[key] for r in rows])
    fit_single = ptc.fit_ptc(mean, single, span, ptc_result["cutoff"])
    ok = np.array([r["peak"] < 0.98 * ptc_result["white_level"] for r in rows])
    A = np.column_stack([np.ones(ok.sum()), mean[ok] ** 2])
    (offset, p2), *_ = np.linalg.lstsq(A, (single - pair)[ok], rcond=None)
    return dict(fit_single=fit_single, K_ratio=float(fit_single["K"] / ptc_result["fit"]["K"]),
                p=float(np.sqrt(p2)) if p2 > 0 else float("nan"),
                dsnu_dn=float(np.sqrt(offset)) if offset > 0 else float("nan"))


def vendor_noise_profile(path, K, read, sK=None, s_read=None):
    """The camera's own noise model (DNG NoiseProfile tag), converted to K and read noise.

    The math. The DNG NoiseProfile gives, per color plane, a scale S and offset O of the
    noise variance in normalized units x = (D - black) / span:
        var(x) = S x + O.
    In DN, D - black = span x and variances scale by span^2, so
        var_DN = S span (D - black) + O span^2,
    which is the photon transfer line with K = S span and c = O span^2. The read noise
    in electrons is sqrt(c) / K = sqrt(O) / S. (image_metadata.compare_noise_profile
    does this conversion and prints it.)

    Parameters
    ----------
    path : str or Path
        A raw file from the same series (same ISO) as the measurement.
    K, read : float
        Your conversion gain (DN/e-) and read noise (e-), for the ratios.
    sK, s_read : float, optional
        Their standard deviations, printed alongside.

    Returns
    -------
    dict or None
        None if the file has no NoiseProfile. Otherwise: planes (list of dicts with S,
        O, K, read per plane), green (the plane matching the green measurement: the
        middle of three, or the only one), K_ratio and read_ratio (vendor / yours).
    """
    planes = meta.compare_noise_profile(path, K=K, read=read, s_K=sK, s_read=s_read)
    if not planes:
        return None
    green = planes[1] if len(planes) == 3 else planes[0]
    return dict(planes=planes, green=green, K_ratio=green["K"] / K,
                read_ratio=green["read"] / read if read else float("nan"))


def measure_dark_current(folder, K, sK, plot_path=None):
    """Dark current from lens-capped frames at several exposure times.

    The math. With no light, the mean level rises with exposure time t only through
    thermally generated electrons: mean(t) = black + K D t, D in e-/s. A straight-line
    fit of each color channel's mean against t gives the slope K D in DN/s; dividing by
    K gives D, with the fit's standard error and the uncertainty of K combined
    (photon_transfer_curve.dark_current_fit). A slope consistent with zero is a
    result: it says dark current is below what these exposures can detect.

    Runs the lens_id assertion on the dark frames first (LensModel is recorded even
    with the lens covered).

    Parameters
    ----------
    folder : str or Path
        Dark frames, any number per exposure time, at one ISO.
    K, sK : float
        Conversion gain and its sd (DN/e-), from the photon transfer fit.
    plot_path : str or Path, optional
        Where to save the level-against-exposure figure.

    Returns
    -------
    dict
        rows : per-exposure statistics (photon_transfer_curve.dark_frame_stats).
        fit : per-channel slopes and rates, plus "summary" (photon_transfer_curve.
            dark_current_fit).
        rate_e_s, sd_rate_e_s : the green-average dark current and its sd (e-/s),
            the value reported (the noise measurement is on a green plane).
        upper_95_e_s : the largest channel rate + 2 sd, a conservative bound.
        black_measured_dn : the green-average fit's level at zero exposure, a measured
            black level to compare with the declared one. (The shortest exposure's mean
            is not used: it includes dark signal accumulated during that exposure.)
    """
    files = flat_field_files(folder)
    if not files:
        raise SystemExit(f"no dark frames in {folder}")
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(files)} dark frames:")
    meta.assert_one_camera(files)
    rows = ptc.dark_frame_stats(folder, K=K)
    fit = ptc.dark_current_fit(rows, K, sK, plot_path=plot_path, title="Dark frames")
    plt.close("all")
    g = fit["G avg"]
    return dict(rows=rows, fit=fit, rate_e_s=float(g["rate"]), sd_rate_e_s=float(g["s_rate"]),
                upper_95_e_s=float(fit["summary"]["upper_95"]),
                black_measured_dn=float(g["level0"] + rows[0]["black_levels"][0]))


def plot_ptc_methods(ptc_result, single, out_png):
    """Photon transfer curve with the three variance estimates and their fits, log-log.

    Parameters
    ----------
    ptc_result : dict
        The return value of measure_ptc.
    single : dict
        The return value of single_frame_bias.
    out_png : str or Path
        Where to save the figure.

    Returns
    -------
    Path
        The path written.
    """
    rows, span, cutoff = ptc_result["rows"], ptc_result["span"], ptc_result["cutoff"]
    mean = np.array([r["mean"] for r in rows])
    series = [(m, _VAR_KEY[m], ptc_result["fits"][m], mk) for m, mk in zip(PAIR_METHODS, "osd")]
    series.append(("single frame", "var_single", single["fit_single"], "^"))
    grid = np.logspace(np.log10(max(mean.min(), 1e-3)), np.log10(span), 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, key, fit, marker in series:
        y = np.array([r[key] for r in rows])
        line, = ax.loglog(mean, y, marker, mfc="none", ms=5,
                          label=f"{name}: K = {fit['K']:.4f} DN/e-")
        ax.loglog(grid, fit["K"] * grid + max(fit["c"], 1e-9), "-", lw=1, color=line.get_color())
    ax.axvline(cutoff * span, color="0.4", ls=":", lw=1)
    ax.text(cutoff * span * 1.05, 0.05, f"fit cutoff {cutoff:.0%}", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("mean signal above black (DN)"); ax.set_ylabel("variance (DN$^2$)")
    ax.set_title(f"Photon transfer, ISO {ptc_result['iso']}: variance estimates compared")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def plot_difference_image(structure, title, out_png):
    """Plain and plane-removed difference images of one pair, with their block means.

    Parameters
    ----------
    structure : dict
        The return value of difference_structure.
    title : str
        Figure title (e.g. the file names).
    out_png : str or Path
        Where to save the figure.

    Returns
    -------
    Path
        The path written.
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    lim = 3 * structure["d_plain"].std()
    blim = 4 * structure["d_plain"].std() / (structure["d_plain"].shape[0] / structure["blocks_plain"].shape[0])
    for ax, img, name, v in ((axes[0], structure["d_plain"], "difference a - b", lim),
                             (axes[1], structure["blocks_plain"], f"block means, R = {structure['R_plain']:.2f}", blim),
                             (axes[2], structure["d_plane"], "plane removed", lim),
                             (axes[3], structure["blocks_plane"], f"block means, R = {structure['R_plane']:.2f}", blim)):
        im = ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(name, fontsize=10); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="DN")
    fig.suptitle(f"{title}   (eps = {100 * structure['eps']:+.2f}%; R about 1 means noise only)")
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def report_noise(ptc_result, output_dir, dark_folder=None, noise_profile_file=None):
    """Run every part of the noise measurement on one photon transfer result and print it.

    Parts: the gain and read noise from the selected fit, both fits side by side; the
    knee where read noise and shot noise are equal (sigma_r^2 electrons, where
    SNR = sigma_r / sqrt(2)) with the SNR curve; the difference image of the brightest
    unclipped pair; the single-frame bias; the vendor NoiseProfile; and dark current.

    Parameters
    ----------
    ptc_result : dict
        The return value of measure_ptc.
    output_dir : str or Path
        Folder for the figures and the per-pair CSV.
    dark_folder : str or Path, optional
        Dark frames for the dark current; skipped if None or missing.
    noise_profile_file : str or Path, optional
        Raw file whose NoiseProfile to compare; defaults to the first flat field.

    Returns
    -------
    dict
        K, sK, read_e, s_read_e, c, sc : the selected fit (DN/e-, e-, DN^2).
        knee_e, sd_knee_e, knee_dn : where shot and read noise are equal.
        method : the selected fit's name.  iso : the ISO.
        structure : difference_structure of the brightest unclipped pair (or None).
        single : single_frame_bias.  vendor : vendor_noise_profile (or None).
        spectrum : noise_power_spectrum over the unclipped pairs (or None).
        dark_spectrum : dark_difference_spectrum (or None).
        dark is None when the dark frames' ISO differs from the flat fields'.
        dark : measure_dark_current (or None).
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fit, fits, iso = ptc_result["fit"], ptc_result["fits"], ptc_result["iso"]
    K, sK, read, s_read = fit["K"], fit["sK"], fit["read"], fit["s_read"]

    print(f"\nPhoton transfer, ISO {iso}, fit below {ptc_result['cutoff']:.0%} of the span "
          f"({ptc_result['n_used']} pairs); selected: {ptc_result['method']}")
    for name, f in fits.items():
        mark = "  <- selected" if name == ptc_result["method"] else ""
        print(f"  {name:16s} K = {f['K']:.4f} +/- {f['sK']:.4f} DN/e-   read noise = "
              f"{f['read']:.2f} +/- {f['s_read']:.2f} e-   intercept c = {f['c']:.3f} +/- {f['sc']:.3f} DN^2{mark}")
    eps = np.array([r["eps"] for r in ptc_result["rows"]])
    print(f"  light change between frames (eps): median |eps| {100 * np.median(np.abs(eps)):.2f}%, "
          f"largest {100 * eps[np.argmax(np.abs(eps))]:+.2f}%")

    knee = read ** 2
    sd_knee = 2 * read * s_read
    print(f"\nRegimes: read noise and shot noise are equal at the knee, S = sigma_r^2 = "
          f"{knee:.1f} +/- {sd_knee:.1f} e- = {knee * K:.2f} DN above black (SNR there "
          f"{read / np.sqrt(2):.2f}). Below it the camera is read-noise limited, above it "
          "photon (shot-noise) limited.")
    if np.isfinite(read):
        with contextlib.redirect_stdout(io.StringIO()):
            _, fig = ptc.plot_snr(ptc_result["rows"], fit, saturation_dn=ptc_result["white_level"],
                                  fit_frac=ptc_result["cutoff"], plot_path=output_dir / "noise_snr.png",
                                  title=f"SNR vs signal, ISO {iso}")
        plt.close(fig)

    i = brightest_unclipped_pair(ptc_result)
    structure = None
    if i is not None:
        pair = ptc_result["pairs"][i]
        structure = difference_structure(pair["a"], pair["b"], pair["black"])
        label = f"{pair['files'][0]} / {pair['files'][1]}"
        plot_difference_image(structure, label, output_dir / "noise_difference_image.png")
        print(f"\nDifference image of the brightest unclipped pair ({label}, "
              f"{ptc_result['rows'][i]['mean']:.0f} DN): eps = {100 * structure['eps']:+.2f}%")
        print(f"  block structure ratio R (about 1 = noise only): plain {structure['R_plain']:.2f}, "
              f"gain-matched {structure['R_matched']:.2f}, plane removed {structure['R_plane']:.2f}")
        rx, ry = structure["rho"]
        n_pairs = structure["d_plain"].size
        print(f"  neighbor correlation of the noise: {rx:+.3f} along rows, {ry:+.3f} down columns "
              f"(independent pixels: 0 +/- {1 / np.sqrt(n_pairs):.3f}); the noise's spatial "
              f"correlation alone predicts R = {structure['R_predicted']:.2f}")
        if min(rx, ry) < -3 / np.sqrt(n_pairs):
            print("  -> negative: neighbors' noise goes opposite ways, the signature of sharpening "
                  "before the DNG is written (it would also raise the measured MTF)")
        elif max(rx, ry) > 3 / np.sqrt(n_pairs):
            print("  -> positive: neighbors' noise moves together, the signature of smoothing or "
                  "noise reduction before the DNG is written")
        gx, gy = structure["gradient_dn"]
        n = structure["d_plain"].shape[0]
        print(f"  fitted gradient: {gx * n:+.2f} DN across the patch in x, {gy * n:+.2f} DN in y")
        verdict = ("no structure beyond noise" if structure["R_plain"] < 1.5
                   else "a gradient: removing a plane fixes it (see the plane-removed fit)"
                   if structure["R_plane"] < 1.5
                   else "blotches or curvature survive plane removal: drop or reshoot such pairs")
        print(f"  reading: {verdict}")

    noise_power_spec = None
    unclipped = [p for p, r in zip(ptc_result["pairs"], ptc_result["rows"])
                 if r["peak"] < 0.98 * ptc_result["white_level"]]
    if unclipped:
        diffs = [difference_structure(p["a"], p["b"], p["black"])["d_plane"] for p in unclipped]
        noise_power_spec = noise_power_spectrum(diffs)
        plot_noise_power_spectrum(noise_power_spec, f"Noise power spectrum of plane-removed pair differences, ISO {iso}",
                                  output_dir / "noise_power_spectrum.png")
        (fh, ph), (fv, pv) = noise_power_spec["horizontal"], noise_power_spec["vertical"]
        print(f"\nNoise power spectrum ({noise_power_spec['n_pairs']} unclipped pairs averaged; 1 = independent noise):")
        print(f"  low band {noise_power_spec['band_low'][0]}-{noise_power_spec['band_low'][1]} cy/sample: {noise_power_spec['low']:.2f}; "
              f"high band {noise_power_spec['band_high'][0]}-{noise_power_spec['band_high'][1]}: {noise_power_spec['high']:.2f}; "
              f"high / low = {noise_power_spec['ratio']:.2f}")
        print(f"  near Nyquist: rows {ph[-3:].mean():.2f}, columns {pv[-3:].mean():.2f}"
              + ("  -> rising toward Nyquist: sharpening-like filtering" if noise_power_spec["ratio"] > 1.1
                 else "  -> falling toward Nyquist: smoothing-like filtering" if noise_power_spec["ratio"] < 0.9
                 else "  -> about flat: independent noise"))

    single = single_frame_bias(ptc_result)
    print(f"\nSingle frame vs pair difference: single-frame K = {single['fit_single']['K']:.4f} DN/e- "
          f"({100 * (single['K_ratio'] - 1):+.1f}% vs the pair fit, same pairs)")
    print(f"  the excess variance fits DSNU^2 + (p S)^2 with p = {100 * single['p']:.2f}% "
          "(PRNU plus shading inside the patch): a bias that grows as S^2, which the pair "
          "difference cancels")
    plot_ptc_methods(ptc_result, single, output_dir / "noise_ptc_methods.png")

    print()
    vendor = vendor_noise_profile(noise_profile_file or ptc_result["files"][0], K, read, sK, s_read)
    if vendor:
        print(f"  green plane: vendor/yours K {vendor['K_ratio']:.3f}, read noise {vendor['read_ratio']:.3f}")

    dark = None
    dark_spec = None
    if dark_folder and Path(dark_folder).exists():
        dark_spec = dark_difference_spectrum(dark_folder)
        if dark_spec is not None:
            rx, ry = dark_spec["rho"]
            print(f"\nDark-frame difference spectrum (read noise only; {dark_spec['n_pairs']} pairs, "
                  f"ISO {', '.join(str(i) for i in dark_spec['iso'])}): low {dark_spec['low']:.2f}, "
                  f"high {dark_spec['high']:.2f}, high / low = {dark_spec['ratio']:.2f}; "
                  f"neighbor correlation {rx:+.3f} rows, {ry:+.3f} columns")
            if noise_power_spec is not None:
                same = abs(dark_spec["ratio"] - noise_power_spec["ratio"]) < 0.15 * noise_power_spec["ratio"]
                print("  -> " + ("same shape as the flat fields: the filtering acts after readout, on all noise"
                                 if same else
                                 "different shape from the flat fields: the filtering does not act on read noise alike"))
                plot_spectra_compared([("flat-field differences (shot + read noise)", noise_power_spec),
                                       ("dark-frame differences (read noise only)", dark_spec)],
                                      "Noise power spectra, radial average",
                                      output_dir / "noise_power_spectrum_dark_vs_flat.png")

    dark_iso = None
    if dark_folder and Path(dark_folder).exists():
        files = flat_field_files(dark_folder)
        dark_iso = meta.raw_metadata(files[0])["iso"] if files else None
    if dark_iso is not None and dark_iso != iso:
        print(f"\nDark current skipped: the dark frames are ISO {dark_iso} but the flat fields are "
              f"ISO {iso}, so this fit's gain would convert them wrongly. Use dark frames at ISO {iso}.")
    elif dark_iso is not None:
        print()
        dark = measure_dark_current(dark_folder, K, sK, plot_path=output_dir / "noise_dark_current.png")
        print(f"  reported (green average): {dark['rate_e_s']:+.3f} +/- {dark['sd_rate_e_s']:.3f} e-/s; "
              f"conservative bound {dark['upper_95_e_s']:.3f} e-/s; zero-exposure black "
              f"{dark['black_measured_dn']:.2f} DN vs declared {ptc_result['black_level']}")
    elif dark_folder:
        print(f"\nDark frames: {dark_folder} not found; dark current skipped")

    pd.DataFrame([{k: v for k, v in r.items() if k not in ("files",)} | {"file_a": r["files"][0], "file_b": r["files"][1]}
                  for r in ptc_result["rows"]]).to_csv(output_dir / "noise_pairs.csv", index=False)
    return dict(K=K, sK=sK, read_e=read, s_read_e=s_read, c=fit["c"], sc=fit["sc"],
                knee_e=knee, sd_knee_e=sd_knee, knee_dn=knee * K, method=ptc_result["method"],
                iso=iso, structure=structure, single=single, vendor=vendor, dark=dark,
                spectrum=noise_power_spec, dark_spectrum=dark_spec)


def noise_results_entries(summary):
    """The noise keys of results.json.

    Parameters
    ----------
    summary : dict
        The return value of report_noise.

    Returns
    -------
    dict
        conversion_gain_dn_per_e [K, sd], read_noise_e [read, sd],
        dark_current_e_per_s [rate, sd] (green average; omitted if no dark frames),
        iso_of_noise_measurement (int), vendor_noiseprofile_present (bool).
    """
    out = {"conversion_gain_dn_per_e": [round(summary["K"], 4), round(summary["sK"], 4)],
           "read_noise_e": [round(summary["read_e"], 2), round(summary["s_read_e"], 2)],
           "iso_of_noise_measurement": int(summary["iso"]),
           "vendor_noiseprofile_present": summary["vendor"] is not None}
    if summary["dark"] is not None:
        out["dark_current_e_per_s"] = [round(summary["dark"]["rate_e_s"], 3),
                                       round(summary["dark"]["sd_rate_e_s"], 3)]
    return out
