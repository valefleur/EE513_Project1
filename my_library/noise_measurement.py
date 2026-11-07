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
from pathlib import Path

import numpy as np

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


def measure_ptc(folder, white_level, black_level, cutoff=LOW_SIGNAL_CUTOFF):
    """Pair-difference photon transfer fit on the consecutive flat-field pairs in folder.

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

    Returns
    -------
    dict
        fit : dict from photon_transfer_curve.fit_ptc (K, sK in DN/e-; c, sc in DN^2;
            read, s_read in e-; used, a boolean mask of the pairs in the fit).
        rows : list of dict, one per pair (mean, var, iso, black, flagged, ...).
        span : float, the span in DN.  iso : the ISO of every pair.
        cutoff : float, as given.  n_used : int, pairs in the fit.
        ident : dict, the camera identity from series_identity.
    """
    paths = flat_field_files(folder)
    if len(paths) < 4:
        raise SystemExit(f"need at least two flat-field pairs in {folder}; found {len(paths)} files")
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(paths)} flat-field files:")
    meta.assert_one_camera(paths)
    ident, n = meta.series_identity(paths)

    rows = ptc.analyze_pairs(ptc.load_flat_pairs(folder))
    isos = sorted({r["iso"] for r in rows})
    blacks = sorted({r["black"] for r in rows})
    if len(isos) != 1:
        raise SystemExit(f"flat fields mix ISOs {isos}; one ISO per measurement")
    if blacks != [black_level]:
        print(f"  note: files declare black level {blacks}; the span uses {black_level} DN")
    span = float(white_level - black_level)

    mean = np.array([r["mean"] for r in rows])
    var = np.array([r["var"] for r in rows])
    fit = ptc.fit_ptc(mean, var, span, cutoff)
    n_used = int(fit["used"].sum())
    flagged = sum(r["flagged"] for r in rows)
    print(f"{len(rows)} pairs at ISO {isos[0]}; {n_used} below {cutoff:.0%} of the "
          f"{span:.0f} DN span used in the fit; {flagged} pair(s) flagged for mean mismatch")
    if n_used < 3:
        raise SystemExit(f"only {n_used} pairs below {cutoff:.0%} of the span; too few to fit")
    return dict(fit=fit, rows=rows, span=span, iso=isos[0], cutoff=cutoff,
                n_used=n_used, ident=ident)


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
    import pandas as pd
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
