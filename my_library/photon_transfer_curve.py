"""Photon transfer curve (PTC) helpers for the Noise and Linear Light assignment.
Written by Claude Sonnet 5.5

Model (lecture 5):  Var[D] = K (E[D] - B) + K^2 sigma_r^2
    D        raw pixel value in digital numbers (DN)
    B        black level (DN)
    K        conversion gain (DN per electron)
    sigma_r  read noise (electrons RMS)

The pair-difference estimator, the 70 percent saturation cut, and the
read-noise formula sqrt(intercept)/K follow the Lecture 5 demo notebook
(05_photon_transfer.ipynb) by Professor James McNames. The fit uncertainties,
error propagation, and the packaging into functions are new here.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import warnings
from my_library.image_metadata import RAW_SUFFIXES, green_patch, raw_metadata, channel_means
import my_library.latex_formatter as tex
import my_library.image_metadata as img


def pair_mean_check(a, b, black, tol=0.003, nsigma=4.0):
    """Compare the two frame means of a pair.

    Returns (relative_difference, flagged). The difference is relative to the
    signal (mean above black). A pair is flagged when the means differ by more
    than tol * signal plus nsigma times the difference that noise alone would
    produce, sqrt(Var[a - b] / N). The noise allowance matters at the dark end,
    where a 0.3 percent tolerance is smaller than the statistical scatter of
    the means.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a.mean() - b.mean()
    signal = (a.mean() + b.mean()) / 2.0 - black
    sigma = np.sqrt(np.var(a - b) / a.size)
    rel = diff / signal if signal > 0 else np.nan
    flagged = abs(diff) > tol * max(signal, 0.0) + nsigma * sigma
    return rel, bool(flagged)


def pair_statistics(a, b, black, tol=0.003, warn=True):
    """Mean above black and noise variance from two frames of one scene.

    mean     = average of the two frame means, minus black
    variance = half the variance of the difference (fixed pattern cancels)
    Also returns the pixel count of the patch. Warns if the two frame means
    disagree by more than pair_mean_check allows.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mean = (a.mean() + b.mean()) / 2.0 - black
    var = np.var(a - b) / 2.0
    if warn:
        rel, flagged = pair_mean_check(a, b, black, tol=tol)
        if flagged:
            warnings.warn(
                f"Pair means differ by {rel * 100:+.2f}% of the signal "
                f"(mean above black = {mean:.1f} DN). Check for illumination "
                f"drift or motion before using this pair.", stacklevel=2)
    return mean, var, a.size


def single_frame_variance(a):
    """Variance of one frame. Includes fixed-pattern noise, which the model omits."""
    return np.var(np.asarray(a, dtype=float))


def fit_ptc(mean, var, saturation_span, frac=0.7):
    """Fit var = K*mean + c on points with mean < frac * saturation_span.

    saturation_span is (white - black) in DN.
    c is (K**2 * sigma_r**2), the intercept of the PTC.

    Returns a dict with K, c, their 1-sigma uncertainties from the fit
    covariance, the read noise sigma_r = sqrt(c)/K with propagated
    uncertainty, and the boolean mask of points used.
    """
    mean = np.asarray(mean, dtype=float)
    var = np.asarray(var, dtype=float)
    used = mean < saturation_span * frac
    (K, c), cov = np.polyfit(mean[used], var[used], 1, cov=True)
    sK, sc = np.sqrt(np.diag(cov))
    sKc = cov[0, 1]

    if c > 0:
        read = np.sqrt(c) / K
        # sigma_r = c^(1/2) K^(-1). Partial derivatives:
        d_dc = 0.5 / (np.sqrt(c) * K)
        d_dK = -np.sqrt(c) / K ** 2
        s_read = np.sqrt(d_dc ** 2 * sc ** 2 + d_dK ** 2 * sK ** 2
                         + 2 * d_dc * d_dK * sKc)
    else:
        # Negative intercept: K^2 sigma_r^2 < 0 is unphysical, so the read
        # noise is undefined for this fit rather than zero.
        read = np.nan
        s_read = np.nan
    return dict(K=K, sK=sK, c=c, sc=sc, read=read, s_read=s_read, used=used)


def full_well_and_dr(K, read, span):
    """Full well in electrons and dynamic range in stops.

    Convention here: DR = log2(full_well / read_noise), i.e. the signal at
    which SNR is about 1 when read noise dominates (noise floor = sigma_r).
    """
    full_well = span / K
    return full_well, np.log2(full_well / read)


def pct_error(est, true):
    return (est - true) / true * 100.0


def load_flat_pairs(directory, size=128, which=0, suffixes=RAW_SUFFIXES):
    """Read every raw file in a directory as consecutive pairs (sorted by file name).

    Raises if the two files of a pair differ in exposure time, ISO or black level.
    Returns a list of dicts with the two patches and their settings.
    """
    paths = sorted(p for p in Path(directory).rglob("*")
                   if p.is_file() and p.suffix.lower() in suffixes)
    if len(paths) % 2:
        raise ValueError(f"{len(paths)} files in {directory}: expected an even number")
    pairs = []
    for p1, p2 in zip(paths[0::2], paths[1::2]):
        m1, m2 = raw_metadata(p1), raw_metadata(p2)
        if (m1["exposure_s"], m1["iso"]) != (m2["exposure_s"], m2["iso"]):
            raise ValueError(f"{p1.name} and {p2.name} differ in exposure or ISO; "
                             f"they are not a matched pair")
        a, black_a = green_patch(p1, size, which)
        b, black_b = green_patch(p2, size, which)
        if black_a != black_b:
            raise ValueError(f"{p1.name} and {p2.name} have different black levels")
        pairs.append(dict(files=(p1.name, p2.name), exposure_s=m1["exposure_s"],
                          iso=m1["iso"], a=a, b=b, black=black_a,
                          white=m1["white_level"]))
    # Print size of one patch, from the first pair, to confirm the patch size is as expected
    print(f"**** Problem 5: Size of one patch: {a.shape}")
    return pairs


def analyze_pairs(pairs, tol=0.003, output_path=None):
    """Per-pair mean above black, pair-difference variance, single-frame variance, and flags."""
    rows = []
    for p in pairs:
        mean, var, n = pair_statistics(p["a"], p["b"], p["black"], warn=False)
        rel, flagged = pair_mean_check(p["a"], p["b"], p["black"], tol=tol)
        rows.append(dict(files=p["files"], exposure_s=p["exposure_s"], iso=p["iso"],
                         mean=mean, var=var, var_single=single_frame_variance(p["a"]),
                         rel_diff=rel, flagged=flagged, n_pixels=n,
                         peak=max(p["a"].max(), p["b"].max()),
                         black=p["black"], white=p["white"]))

    if output_path:
        # generate latex table of means and variances for each pair using
        tex.generate_table([[r['files'][0], r['files'][1], img._format_exposure(r['exposure_s']), r['mean'], r['var']] for r in rows], labels=['File 1', 'File 2', 'Exposure Time', 'Mean', 'Variance'], caption='Mean and Variance for each pair', output_path=f'{output_path}05_ptc_pair_table.txt')

    return rows


def report_ptc(rows, frac=0.7, saturation_dn=None, exclude_flagged=False,
               plot_path=None, title="", output=None):
    """Print the table and fit results for a set of pairs, and plot the PTC.

    saturation_dn is the raw value at which the sensor really clips. It defaults
    to the white level in the file, which may be higher than where the sensor
    actually saturates; check the `peak` column of the brightest pairs.
    """

    black = rows[0]["black"]
    span = (saturation_dn if saturation_dn is not None else rows[0]["white"]) - black
    keep = [r for r in rows if not (exclude_flagged and r["flagged"])]
    mean = np.array([r["mean"] for r in keep])
    var = np.array([r["var"] for r in keep])
    var1 = np.array([r["var_single"] for r in keep])

    print(f"{title}  black {black:.0f} DN, saturation span {span:.0f} DN, "
          f"{rows[0]['n_pixels']} px per patch")
    print(f"{'pair':22s} {'exp':>8s} {'mean':>8s} {'%span':>6s} {'var pair':>9s} "
          f"{'var 1fr':>9s} {'dmean%':>7s} {'peak':>6s}  flag")
    for r in rows:
        exp = f"1/{round(1 / r['exposure_s'])}" if r["exposure_s"] < 1 else f"{r['exposure_s']:g}"
        print(f"{r['files'][0][-8:-4] + '/' + r['files'][1][-8:-4]:22s} {exp:>8s} "
              f"{r['mean']:8.1f} {100 * r['mean'] / span:6.1f} {r['var']:9.2f} "
              f"{r['var_single']:9.2f} {100 * r['rel_diff']:+7.2f} {r['peak']:6.0f}  "
              f"{'CHECK' if r['flagged'] else ''}")

    fit = fit_ptc(mean, var, span, frac)
    fit1 = fit_ptc(mean, var1, span, frac)
    if fit["c"] <= 0:
        print("note: the fitted intercept is not positive, so read noise is undefined "
              "for this fit range; try a smaller frac")
    full_well, dr = full_well_and_dr(fit["K"], fit["read"], span)
    print(f"\npoints fitted: {fit['used'].sum()} of {len(mean)} (below {100 * frac:.0f}% of span)")
    print(f"pair fit   K = {fit['K']:.4f} +/- {fit['sK']:.4f} DN/e-,  "
          f"read noise = {fit['read']:.2f} +/- {fit['s_read']:.2f} e-")
    print(f"**** Problem 7 ****")
    print(f"single fit K = {fit1['K']:.4f} +/- {fit1['sK']:.4f} DN/e-   "
          f"({pct_error(fit1['K'], fit['K']):+.1f}% vs pair fit)")
    print(f"**** Problem 8 ****")
    print(f"full well = {full_well:.0f} e-, dynamic range = {dr:.2f} stops "
          f"(log2(full well / read noise))")

    xs = np.linspace(0, mean.max() * 1.02, 100)
    used = fit["used"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(mean[used], var[used], "o", label="fitted")
    ax.plot(mean[~used], var[~used], "o", mfc="none", label=f"excluded (> {100 * frac:.0f} % sat.)")
    ax.plot(xs, fit["K"] * xs + fit["c"], "-",
            label=f"K = {fit['K']:.4f} DN/e-, read noise = {fit['read']:.2f} e-")
    ax.set_xlabel("mean above black (DN)")
    ax.set_ylabel(r"variance (DN$^2$)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if plot_path:
        fig.savefig(plot_path, dpi=150)
    return fit, fit1, fig



def run_real_ptc(directory, tol=0.003, iso=None, output_path=None, **kwargs):
    """Load a directory of consecutive pairs and run the whole HW 3 Part 2 analysis."""
    pairs = load_flat_pairs(directory)
    rows = analyze_pairs(pairs, tol=tol)
    plot_residuals(rows, fracs=(0.7, 0.1), saturation_dn=4077,
                title=f"ISO {iso}", plot_path=f"{output_path}06_resid_iso{iso}.png")
    fit_range_table(rows, saturation_dn=4077)
    # ptc.report_ptc(rows, title=f"ISO {iso}, low signal", plot_path=f"{output_path}06_ptc_iso{iso}_low.png", saturation_dn=4077)

    return report_ptc(rows, **kwargs)


def fit_range_table(rows, saturation_dn=None, fracs=(0.7, 0.3, 0.2, 0.1, 0.05)):
    """Print K, intercept and read noise for several upper limits on the fit range."""
    black = rows[0]["black"]
    span = (saturation_dn if saturation_dn is not None else rows[0]["white"]) - black
    mean = np.array([r["mean"] for r in rows])
    var = np.array([r["var"] for r in rows])
    print(f"{'fit range':>18s} {'n':>3s} {'K (DN/e-)':>18s} {'intercept (DN^2)':>20s} {'read (e-)':>10s}")
    for frac in fracs:
        f = fit_ptc(mean, var, span, frac)
        read = f"{f['read']:.2f}" if f["c"] > 0 else "undefined"
        print(f"{'< ' + format(100 * frac, 'g') + '% of span':>18s} {f['used'].sum():3d} "
              f"{f['K']:8.4f} +/- {f['sK']:.4f} {f['c']:9.2f} +/- {f['sc']:5.2f} {read:>10s}")


def plot_residuals(rows, fracs=(0.7, 0.1), saturation_dn=None, max_frac=0.95,
                   plot_path=None, title=""):
    """Plot the percent residual of pair variance from straight-line fits.

    One panel per entry in `fracs`. Each panel fits a line to the points below that
    fraction of saturation, then shows how every point (up to max_frac of saturation)
    departs from it. The grey band is +/- 2 standard errors of a variance estimate
    from n_pixels independent pixels, so points outside it are not just sampling
    scatter. Pairs at or near clipping are left out because their variance collapses.
    Residuals are relative to the measured variance; positive means the measured
    variance is above the line.
    """

    black = rows[0]["black"]
    span = (saturation_dn if saturation_dn is not None else rows[0]["white"]) - black
    mean = np.array([r["mean"] for r in rows])
    var = np.array([r["var"] for r in rows])
    shown = mean < max_frac * span
    band = 200 * np.sqrt(2.0 / (rows[0]["n_pixels"] - 1))

    fig, axes = plt.subplots(1, len(fracs), figsize=(5.5 * len(fracs), 3.8),
                             squeeze=False)
    for ax, frac in zip(axes[0], fracs):
        f = fit_ptc(mean, var, span, frac)
        pred = f["K"] * mean + f["c"]
        resid = np.full_like(var, np.nan)
        ok = var > 0
        resid[ok] = 100 * (var[ok] - pred[ok]) / var[ok]
        used = f["used"]
        ax.axhspan(-band, band, color="0.85", label=r"$\pm$2 s.e. of a variance estimate")
        ax.axhline(0, color="0.4", lw=1)
        ax.semilogx(mean[used & shown], resid[used & shown], "o", label="fitted")
        ax.semilogx(mean[~used & shown], resid[~used & shown], "o", mfc="none",
                    label="not fitted")
        ax.set_xlabel("mean above black (DN)")
        ax.set_ylabel("residual (% of measured variance)")
        ax.set_title(f"line fitted below {100 * frac:g}% of span\n"
                     f"K = {f['K']:.4f}, intercept = {f['c']:.2f} DN$^2$", fontsize=10)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if plot_path:
        fig.savefig(plot_path, dpi=130)
    return fig


def plot_snr(rows, fit, saturation_dn=None, fit_frac=0.1, max_frac=0.95,
             plot_path=None, title=""):
    """SNR versus signal in electrons: measured pairs, the model curve, and the knee.

    Measured SNR is the mean above black over the square root of the pair variance,
    both in DN, so it does not depend on K. Signal in electrons is mean / K. The model
    is SNR = mu / sqrt(mu + sigma_r^2) with the K and read noise in `fit` (a dict from
    fit_ptc, which must have a positive intercept so the read noise is defined).
    Pairs with zero variance or above `max_frac` of saturation (clipped) are skipped.
    Points above `fit_frac` of saturation are drawn hollow; pass the same fraction the
    fit used. Returns (knee in electrons, figure).
    """
    import matplotlib.pyplot as plt

    K, read = fit["K"], fit["read"]
    if not np.isfinite(read):
        raise ValueError("fit has no read noise (intercept <= 0); pass a fit over a "
                         "narrower range, e.g. fit_ptc(mean, var, span, 0.1)")
    black = rows[0]["black"]
    span = (saturation_dn if saturation_dn is not None else rows[0]["white"]) - black
    mean = np.array([r["mean"] for r in rows])
    var = np.array([r["var"] for r in rows])
    ok = (var > 0) & (mean < max_frac * span)
    mean, var = mean[ok], var[ok]

    e = mean / K
    snr = mean / np.sqrt(var)
    model = e / np.sqrt(e + read ** 2)
    fitted = mean < fit_frac * span
    knee = read ** 2
    knee_err = 2 * read * fit["s_read"]

    grid = np.logspace(np.log10(e.min() / 20), np.log10(e.max() * 1.5), 400)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.loglog(grid, grid / np.sqrt(grid + read ** 2), "-", lw=2,
              label=rf"model, $\mu/\sqrt{{\mu+\sigma_r^2}}$, $\sigma_r$ = {read:.2f} e$^-$")
    ax.loglog(grid, np.sqrt(grid), "--", lw=1.2, color="0.45", label=r"photon limit, $\sqrt{\mu}$")
    ax.loglog(grid, grid / read, ":", lw=1.2, color="0.45", label=r"read-noise limit, $\mu/\sigma_r$")
    ax.loglog(e[fitted], snr[fitted], "o", label="measured (in fit range)")
    ax.loglog(e[~fitted], snr[~fitted], "o", mfc="none", label="measured (above fit range)")
    ax.axvline(knee, color="0.2", lw=1)
    ax.text(knee * 1.1, 0.04, rf"knee: $\sigma_r^2$ = {knee:.0f} e$^-$",
            transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set_xlabel("signal (electrons)")
    ax.set_ylabel("SNR")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    if plot_path:
        fig.savefig(plot_path, dpi=150)

    dev = 100 * (snr / model - 1)
    print(f"knee = sigma_r^2 = {knee:.1f} +/- {knee_err:.1f} e- "
          f"({knee * K:.1f} DN above black)")
    print(f"signal range measured: {e.min():.0f} to {e.max():.0f} e-; "
          f"{(e < knee).sum()} of {len(e)} pairs below the knee")
    print(f"measured / model - 1: median {np.median(dev):+.1f}%, "
          f"{dev[np.argmin(e)]:+.1f}% at the dimmest, {dev[np.argmax(e)]:+.1f}% at the brightest")
    return knee, fig


def dark_frame_stats(directory, K=None, size=128, suffixes=RAW_SUFFIXES):
    """Summarize dark frames (lens covered), grouped by exposure time and ISO.

    For each group prints the number of frames, the mean of every color channel over
    the whole frame, the mean of the centre green patch, and the temporal variance of
    that patch (per-pixel variance across the frames of the group, averaged over the
    patch). A dark frame has no photon signal, so that variance is K^2 sigma_r^2 plus
    any dark-current shot noise; with K given, the read noise it implies is printed in
    electrons. Finishes by comparing the shortest exposure's channel means with the
    black level stored in the file. Returns a list of dicts, one per group.
    """
    paths = sorted(p for p in Path(directory).rglob("*")
                   if p.is_file() and p.suffix.lower() in suffixes)
    if not paths:
        print(f"No raw files found in {directory}")
        return []
    groups = {}
    for p in paths:
        m = raw_metadata(p)
        groups.setdefault((m["exposure_s"], m["iso"]), []).append((p, m))

    rows = []
    for (exposure, iso), items in sorted(groups.items(), key=lambda kv: kv[0][0]):
        results = [channel_means(p) for p, _ in items]
        stack = np.stack([green_patch(p, size)[0] for p, _ in items])
        var_t = stack.var(axis=0, ddof=1).mean() if len(items) > 1 else float("nan")
        read = np.sqrt(var_t) / K if (K and var_t > 0) else float("nan")
        rows.append(dict(exposure_s=exposure, iso=iso, n=len(items), labels=results[0][1],
                         channel_means=np.mean([r[0] for r in results], axis=0),
                         black_levels=items[0][1]["black_levels"],
                         patch_mean=float(stack.mean()), var_t=float(var_t),
                         read_e=float(read)))

    labels = rows[0]["labels"]
    print("black level stored in the file:", rows[0]["black_levels"])
    head = (f"{'exposure':>9s} {'n':>2s} {'ISO':>4s} "
            + " ".join(f"{label:>8s}" for label in labels)
            + f" {'g patch':>8s} {'var (DN^2)':>11s}")
    print(head + (f" {'read (e-)':>10s}" if K else ""))
    for r in rows:
        e = r["exposure_s"]
        exp = f"1/{round(1 / e)}" if e < 1 else f"{e:g}"
        line = (f"{exp:>9s} {r['n']:2d} {r['iso']:4d} "
                + " ".join(f"{v:8.2f}" for v in r["channel_means"])
                + f" {r['patch_mean']:8.2f} {r['var_t']:11.2f}")
        print(line + (f" {r['read_e']:10.2f}" if K else ""))

    first = rows[0]
    diffs = [m - b for m, b in zip(first["channel_means"], first["black_levels"])]
    print("\nshortest exposure, channel mean minus file black level (DN): "
          + ", ".join(f"{label} {d:+.2f}" for label, d in zip(labels, diffs)))
    return rows


def dark_current_fit(rows, K, s_K=None, plot_path=None, title=""):
    """Fit dark current from the group means returned by dark_frame_stats.

    For each color channel (and the average of the two greens) fit mean level against
    exposure time with a straight line. The slope in DN/s divided by the gain K is the
    dark current in e-/s; its standard error comes from the scatter about the line
    (np.polyfit with cov=True, 6 degrees of freedom for 8 exposures) plus the gain
    uncertainty. The spread between channels is an extra systematic. The reported
    upper bound is the largest channel slope + 2 s.e., converted to e-/s.
    Returns a dict and, if plot_path is given, saves a figure.
    """

    t = np.array([r["exposure_s"] for r in rows])
    labels = list(rows[0]["labels"])
    M = np.array([r["channel_means"] for r in rows])          # groups x channels
    black = float(rows[0]["black_levels"][0])
    series = {lab: M[:, j] for j, lab in enumerate(labels)}
    greens = [lab for lab in labels if lab.startswith("G")]
    series["G avg"] = np.mean([series[g] for g in greens], axis=0)
    for g in greens:
        series.pop(g)

    out = {}
    print(f"longest exposure {t.max():g} s over {len(t)} exposure groups; K = {K:.4f} DN/e-")
    print(f"{'channel':>8s} {'slope (DN/s)':>16s} {'e-/s':>14s} {'level at t=0 - black':>22s}")
    for lab, y in series.items():
        (s, c), cov = np.polyfit(t, y, 1, cov=True)
        ss = np.sqrt(cov[0, 0])
        rate = s / K
        err = np.hypot(ss / K, (s * s_K / K ** 2) if s_K else 0.0)
        out[lab] = dict(slope=s, s_slope=ss, rate=rate, s_rate=err, level0=c - black)
        print(f"{lab:>8s} {s:+9.4f} +/-{ss:6.4f} {rate:+8.2f} +/-{err:5.2f} {c - black:+14.2f} DN")
    rates = np.array([v["rate"] for v in out.values()])
    upper = max(v["rate"] + 2 * v["s_rate"] for v in out.values())
    out["summary"] = dict(mean_rate=rates.mean(), spread=rates.max() - rates.min(),
                          upper_95=upper, accumulated_e_at_tmax=upper * t.max())
    print(f"channels range {rates.min():+.2f} to {rates.max():+.2f} e-/s; "
          f"upper bound (max channel + 2 s.e.) = {upper:.2f} e-/s -> "
          f"{upper * t.max():.2f} e- accumulated at {t.max():g} s")

    if plot_path:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        grid = np.linspace(0, t.max() * 1.03, 50)
        for lab, y in series.items():
            s, c = np.polyfit(t, y, 1)
            line, = ax.plot(t, y - black, "o", label=f"{lab}: {s / K:+.2f} e$^-$/s")
            ax.plot(grid, s * grid + c - black, "-", color=line.get_color(), lw=1)
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.set_xlabel("exposure time (s)")
        ax.set_ylabel("mean $-$ file black level (DN)")
        ax.secondary_yaxis("right", functions=(lambda y: y / K, lambda y: y * K)
                           ).set_ylabel("electrons")
        ax.set_title(title)
        ax.legend(fontsize=8, title="fitted slope")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
    return out