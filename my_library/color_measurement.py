"""Project 1, measurement 5: color accuracy from ColorChecker captures.

Per chart: locate and sample the 24 patches; white balance on an unclipped mid gray
(von Kries gains); set exposure with one number, by default a least-squares fit over
all six neutrals (HW4 feedback: the white patch reads low); fit a 3 x 3 least-squares
matrix from camera RGB to reference XYZ; and compute CIEDE2000 per patch before
(balanced camera RGB treated as linear sRGB) and after the matrix. Per lighting round:
the pooled matrix (one fit over every chart, the CCM an ISP would store) and the
per-chart matrices, the mean and worst patch as the mean over charts with the sample
standard deviation across the nominally identical charts, the per-patch table, and the
neutral-patch diagnostics (ratios to white and flare).

Started from hw4.py Problems 5, 6, 9 and 10, packaged as functions so that project1.py
can run it, then extended with the HW4 feedback. It reuses my_library.color_chart
(patch sampling rule from Prof. James McNames, 07_build_an_isp.ipynb section 4) and
my_library.white_balance (least-squares fit and CIEDE2000, following Prof. McNames'
06_color_matrix.ipynb). Run with exposure="white" to reproduce HW4 exactly.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import psu_imaging as psu
import my_library.catalog as datalog
import my_library.color_chart as chart
import my_library.image_metadata as meta
import my_library.white_balance as wb

METRIC = "CIEDE2000"

# Lighting rounds, relative to the images root. Names match hw4.py's `rounds`, so the
# output can be compared with HW4's results file for the same round.
COLOR_ROUNDS = {
    "florescent": "hw4_color_charts/florescent",
    "led": "hw4_color_charts/led",
    "incandesent": "hw4_color_charts/incandesent",
    "cloudy1": "hw4_color_charts/outdoors_cloudy/set1",
    "cloudy2": "hw4_color_charts/outdoors_cloudy/set2",
    "shade": "hw4_color_charts/outdoors_shade",
}


def chart_catalog(rounds, saturation_dn, black_level=None):
    """Catalog of every chart capture (white sheets dropped), numbered within each round.

    Checks, before anything is fitted: the lens_id assertion and the series identity
    over every file (one camera), and ISO, exposure and declared black level constant
    within each round (HW4 Problem 1 requires identical settings within a round; this
    check is for ColorChecker rounds only). Files whose name contains "_white" are the
    white sheets shot for shading and are left out.

    Parameters
    ----------
    rounds : dict
        {round name: folder of raw files}.
    saturation_dn : float
        Saturation level in DN, used for the per-file clipping statistics.
    black_level : float, optional
        The black level that will be subtracted; if it differs from what a file
        declares, a note is printed.

    Returns
    -------
    (catalog, ident, n) : tuple
        catalog : pandas.DataFrame, one row per chart capture (round, file, path, chart
            number within the round, metadata and clipping statistics).
        ident : dict, the camera identity of the first file (series_identity).
        n : int, the number of files checked.
    """
    catalog = datalog.add_clip_stats(datalog.build_catalog(rounds), saturation_dn)
    catalog = catalog[~catalog["file"].str.contains("_white")].reset_index(drop=True)
    catalog["chart"] = catalog.groupby("round").cumcount() + 1
    datalog.check_constant_within(catalog, "round", ["iso", "exposure_s", "black_levels"])
    paths = catalog["path"].tolist()
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(paths)} chart files:")
    meta.assert_one_camera(paths)
    ident, n = meta.series_identity(paths)
    if black_level is not None:
        declared = sorted({b for blacks in catalog["black_levels"] for b in blacks})
        if declared != [black_level]:
            print(f"  note: subtracting black level {black_level} DN; files declare {declared}")
    return catalog, ident, n


EXPOSURE_METHODS = ("grays", "white")
NEUTRALS = list(range(18, 24))          # 0-based: patches 19 (white) .. 24 (black)


def gray_ratios(balanced, ref_XYZ, usable):
    """How each neutral compares with the white patch, against the published ratios.

    The math. On a white-balanced chart a neutral patch's green is proportional to its
    luminance Y, so green_i / green_19 should equal Y_i / Y_19 from the reference.
    Green is only a proxy for Y (the reference grays are not perfectly neutral and a
    camera's green is not exactly Y): on an exact simulated camera the ratios come out
    within about 1%, so only deviations well beyond that mean anything. If
    every neutral reads 12-17% high relative to white (Prof. McNames, HW4 feedback), the
    white patch reads low. A deviation that grows or shrinks along the bottom row
    (patches 19 to 24 run left to right) points instead to uneven light across the chart.

    Parameters
    ----------
    balanced : ndarray, 24 x 3
        White-balanced linear camera RGB.
    ref_XYZ : ndarray, 24 x 3
        Reference XYZ of the patches.
    usable : ndarray of bool, 24
        Patches that are not clipped.

    Returns
    -------
    dict
        {patch number (20..24): dict(measured, published, deviation)} for usable neutrals,
        where measured = green_i / green_19, published = Y_i / Y_19 and deviation =
        measured / published - 1. Empty if patch 19 is not usable.
    """
    if not usable[18]:
        return {}
    out = {}
    for i in NEUTRALS[1:]:
        if usable[i]:
            measured = balanced[i, 1] / balanced[18, 1]
            published = ref_XYZ[i, 1] / ref_XYZ[18, 1]
            out[i + 1] = dict(measured=float(measured), published=float(published),
                              deviation=float(measured / published - 1))
    return out


def exposure_scale(balanced, ref_XYZ, usable, method="grays"):
    """One number that puts white-balanced camera RGB on the reference's luminance scale.

    The math. method "white" (HW4): s = Y_19 / green_19, from the white patch alone, so
    any error in patch 19 becomes an error in every color. method "grays" (HW4
    feedback): least squares over all usable neutrals 19-24 in relative terms,
        minimize sum_i ((s green_i - Y_i) / Y_i)^2   ->   s = sum(r_i) / sum(r_i^2),
    with r_i = green_i / Y_i, so each neutral counts equally whatever its brightness and
    one mis-reading patch moves s only a little.

    Parameters
    ----------
    balanced, ref_XYZ, usable : as in gray_ratios.
    method : {"grays", "white"}

    Returns
    -------
    float
        The scale s (camera units to reference Y).
    """
    if method == "white":
        k = wb.pick_white(usable)
        return float(ref_XYZ[k, 1] / balanced[k, 1])
    idx = [i for i in NEUTRALS if usable[i]]
    r = balanced[idx, 1] / ref_XYZ[idx, 1]
    return float(np.sum(r) / np.sum(r ** 2))


def delta_e(scaled, ref_XYZ, usable, M, method="grays"):
    """CIEDE2000 per patch between camera colors (through M) and the reference.

    method "white": wb.delta_e_vs_reference, the HW4 calculation, which takes the
    measured patch 19 as the CIELAB white for the measured colors. method "grays": the
    measured colors are already on the reference's scale (exposure_scale), so both sides
    use the same white, the reference patch 19, and an error in the measured white patch
    no longer shifts every other color.

    Parameters
    ----------
    scaled : ndarray, 24 x 3
        Balanced camera RGB times the exposure scale.
    ref_XYZ : ndarray, 24 x 3
    usable : ndarray of bool, 24
    M : ndarray, 3 x 3
        Camera RGB to XYZ (wb.SRGB_TO_XYZ for "before", the fitted matrix for "after").
    method : {"grays", "white"}

    Returns
    -------
    ndarray, 24
        Delta E 2000 per patch.
    """
    if method == "white":
        return wb.delta_e_vs_reference(scaled, ref_XYZ, usable, M=M)[0]
    XYZ_meas = np.asarray(scaled, float) @ np.asarray(M, float).T
    white = ref_XYZ[18]
    lab_meas = wb.lab_with_white(XYZ_meas, white)
    lab_ref = wb.lab_with_white(ref_XYZ, white)
    return np.asarray(wb.colour.delta_E(lab_meas, lab_ref, method="CIE 2000"))


def correct_chart(means, usable, ref_XYZ, exposure="grays"):
    """White balance, set exposure, fit this chart's matrix, and score it. No I/O.

    Steps: gains from a mid gray make it R = G = B (von Kries); one exposure number puts
    the balanced RGB on the reference's luminance scale (exposure_scale); a 3 x 3 matrix
    M is fitted by least squares from balanced camera RGB to reference XYZ
    (wb.fit_matrix); Delta E 2000 is computed before (balanced RGB treated as linear
    sRGB) and after (through M).

    Parameters
    ----------
    means : ndarray, 24 x 3
        Linear camera RGB of each patch.
    usable : ndarray of bool, 24
        False for clipped or near-ceiling patches (left out of every fit).
    ref_XYZ : ndarray, 24 x 3
        Reference XYZ.
    exposure : {"grays", "white"}, default "grays"
        How exposure and the CIELAB white are set (see exposure_scale, delta_e).

    Returns
    -------
    dict or None
        None if no mid gray is usable. Otherwise gray_idx (0-based patch used for white
        balance), gains, scale, scaled (the 24 x 3 RGB the matrix was fitted to), M,
        de_before, de_after, gray_ratios (see gray_ratios) and white_reads (patch 19's
        green relative to the six-gray fit: s_grays / s_white - 1; negative means the
        white reads low), flare (flare_fit, or None).
    """
    gray_idx = wb.pick_gray(usable)
    if gray_idx is None:
        return None
    gains = wb.gains_from_gray(means[gray_idx])
    balanced = means * gains
    flare = flare_fit(balanced, ref_XYZ, usable)
    scale = exposure_scale(balanced, ref_XYZ, usable, exposure)
    scaled = balanced * scale
    M = wb.fit_matrix(scaled, ref_XYZ, usable)
    white_reads = float("nan")
    if usable[18]:
        white_reads = exposure_scale(balanced, ref_XYZ, usable, "grays") / \
            exposure_scale(balanced, ref_XYZ, usable, "white") - 1
    return dict(gray_idx=gray_idx, gains=gains, scale=scale, scaled=scaled, M=M,
                de_before=delta_e(scaled, ref_XYZ, usable, wb.SRGB_TO_XYZ, exposure),
                de_after=delta_e(scaled, ref_XYZ, usable, M, exposure),
                gray_ratios=gray_ratios(balanced, ref_XYZ, usable), white_reads=white_reads,
                flare=flare)


def flare_fit(balanced, ref_XYZ, usable):
    """Separate a gain and an additive offset (flare) in the neutral patches.

    The math. Veiling glare (flare) is stray light scattered inside the lens and camera
    that lands roughly evenly across the image, adding about the same amount to every
    patch. For the neutrals, whose balanced green tracks luminance Y,
        green_i = g Y_i + o,
    with g the exposure gain and o the flare offset. A straight-line fit of green
    against the reference Y over the usable grays 20-24 gives both. Patch 19 is left out
    of the fit because it is the suspect: included, a white that reads low drags the
    line toward itself, hiding most of its own error and inflating the offset. The
    offset as a
    fraction of the white's ideal level, f = o / (g Y_19), is the flare level; it matters
    most for dark patches (it adds f Y_19 / Y_i of their own level), so it shows as a
    neutrals/white deviation that grows toward black. Patch 19's residual from the line,
        w = green_19 / (g Y_19 + o) - 1,
    then says whether the white patch reads low beyond what flare explains, judged
    against a line it did not influence.

    Parameters
    ----------
    balanced : ndarray, 24 x 3
        White-balanced linear camera RGB.
    ref_XYZ : ndarray, 24 x 3
        Reference XYZ.
    usable : ndarray of bool, 24
        Patches that are not clipped.

    Returns
    -------
    dict or None
        None if fewer than three of patches 20-24 are usable. Otherwise gain g, offset o
        (camera units), flare (o / (g Y_19), a fraction), white_residual (w, nan if
        patch 19 is not usable) and n_neutrals (grays used in the fit).
    """
    idx = [i for i in NEUTRALS[1:] if usable[i]]
    if len(idx) < 3:
        return None
    g, o = np.polyfit(ref_XYZ[idx, 1], balanced[idx, 1], 1)
    white_residual = (balanced[18, 1] / (g * ref_XYZ[18, 1] + o) - 1) if usable[18] else float("nan")
    return dict(gain=float(g), offset=float(o), flare=float(o / (g * ref_XYZ[18, 1])),
                white_residual=float(white_residual), n_neutrals=len(idx))


def per_patch_table(charts, ref_XYZ, pooled, exposure="grays"):
    """Delta E 2000 for each of the 24 patches: mean and sd across charts.

    Columns: before the matrix, after each chart's own matrix, and after the pooled
    matrix (the CCM). A chart contributes to a patch only where that patch is usable.

    Parameters
    ----------
    charts : list of dict
        Results of correct_chart (with "scaled", "usable", "de_before", "de_after").
    ref_XYZ : ndarray, 24 x 3
    pooled : ndarray, 3 x 3
        The pooled matrix (fit_pooled_matrix).
    exposure : {"grays", "white"}

    Returns
    -------
    pandas.DataFrame
        One row per patch: patch, name, n (charts used), and mean and sd of
        dE_before, dE_after_per_chart and dE_after_pooled.
    """
    pooled_de = [delta_e(c["scaled"], ref_XYZ, c["usable"], pooled, exposure) for c in charts]
    rows = []
    for i in range(24):
        use = [k for k, c in enumerate(charts) if c["usable"][i]]
        row = dict(patch=i + 1, name=psu.COLORCHECKER_NAMES[i], n=len(use))
        for key, values in (("dE_before", [charts[k]["de_before"][i] for k in use]),
                            ("dE_after_per_chart", [charts[k]["de_after"][i] for k in use]),
                            ("dE_after_pooled", [pooled_de[k][i] for k in use])):
            v = np.asarray(values, float)
            row[f"{key}_mean"] = float(v.mean()) if len(v) else float("nan")
            row[f"{key}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def neutral_diagnostics(charts):
    """The gray-ratio and white-patch diagnostics of a round, averaged over its charts.

    Only charts whose patch 19 is usable contribute to the ratios to white (on a chart
    where the white is clipped, those ratios are undefined).

    Parameters
    ----------
    charts : list of dict
        Results of correct_chart.

    Returns
    -------
    dict
        n_with_white (charts with a usable patch 19), deviation ({patch: mean
        neutral/white deviation}), white_reads (mean, or None if no chart has a usable
        patch 19), flare (mean flare fraction, nan if none) and white_residual (mean
        residual of patch 19 from the flare line, nan if unavailable).
    """
    with_white = [c for c in charts if c["usable"][18]]
    dev = {p: float(np.mean([c["gray_ratios"][p]["deviation"] for c in with_white if p in c["gray_ratios"]]))
           for p in range(20, 25) if any(p in c["gray_ratios"] for c in with_white)}
    flares = [c["flare"] for c in charts if c["flare"] is not None]
    residuals = [f["white_residual"] for f in flares if np.isfinite(f["white_residual"])]
    return dict(n_with_white=len(with_white), deviation=dev,
                white_reads=float(np.mean([c["white_reads"] for c in with_white])) if with_white else None,
                flare=float(np.mean([f["flare"] for f in flares])) if flares else float("nan"),
                white_residual=float(np.mean(residuals)) if residuals else float("nan"))


def fit_pooled_matrix(charts, ref_XYZ):
    """One least-squares matrix fitted to every usable patch of every chart in a round.

    The math. Stack each chart's scaled RGB (n_charts x 24 rows, usable only) as C and
    the matching reference XYZ as X; the matrix minimizing |C M^T - X|^2 is M^T =
    lstsq(C, X). With the charts nominally identical, this is the single matrix a camera
    would store for that illuminant (HW4 Problem 7), and the best one to hand to an ISP.

    Parameters
    ----------
    charts : list of dict
        Results of correct_chart, each with "scaled" and "usable".
    ref_XYZ : ndarray, 24 x 3

    Returns
    -------
    ndarray, 3 x 3
        The pooled matrix M (XYZ = M @ rgb).
    """
    C = np.vstack([c["scaled"][c["usable"]] for c in charts])
    X = np.vstack([ref_XYZ[c["usable"]] for c in charts])
    Mt, *_ = np.linalg.lstsq(C, X, rcond=None)
    return Mt.T


def summarize_round(charts):
    """Round summary from the per-chart results: mean, worst patch and their spread.

    The math. For each chart, over its usable patches: mean Delta E before, mean after
    (with that chart's own matrix), and the worst patch after. Each of those three is
    then summarized across the n charts as
        mean  = (1/n) sum x_k,
        sd    = sqrt( sum (x_k - mean)^2 / (n - 1) )      (sample sd, ddof = 1),
        sem   = sd / sqrt(n)                               (standard error of the mean).
    The sd describes how much one more capture would vary and settles toward the true
    chart-to-chart spread as n grows; the sem describes how well the mean is known and
    shrinks as 1 / sqrt(n). Both cover only what changes between captures, not shared
    effects such as flare from the same setup (Prof. McNames, HW4 feedback). The
    datasheet reports the sd (plan Q24); with n = 3 both are rough (an sd from three
    values is itself uncertain by about 50%).

    "Worst" is the mean over charts of each chart's worst patch, which may be different
    patches on different charts; worst_patch names the one that is worst most often.

    Parameters
    ----------
    charts : list of dict
        Results of correct_chart, each with de_before, de_after and usable.

    Returns
    -------
    dict
        n_charts; mean_before, mean_after, worst_after (each a dict with mean, std, sem,
        lo, hi; std and sem are None for one chart); worst_patch ("number name") and
        worst_patch_count (how many charts it was worst on).
    """
    def over_charts(values):
        v = np.asarray(values, float)
        sd = float(v.std(ddof=1)) if len(v) > 1 else None
        return dict(mean=float(v.mean()), std=sd, sem=None if sd is None else sd / np.sqrt(len(v)),
                    lo=float(v.min()), hi=float(v.max()))

    worst_idx = [int(np.argmax(np.where(c["usable"], c["de_after"], -np.inf))) for c in charts]
    most_common = max(set(worst_idx), key=worst_idx.count)
    return dict(
        n_charts=len(charts),
        mean_before=over_charts([c["de_before"][c["usable"]].mean() for c in charts]),
        mean_after=over_charts([c["de_after"][c["usable"]].mean() for c in charts]),
        worst_after=over_charts([c["de_after"][c["usable"]].max() for c in charts]),
        worst_patch=f"{most_common + 1} {psu.COLORCHECKER_NAMES[most_common]}",
        worst_patch_count=worst_idx.count(most_common),
    )


def results_entries(summary, decimals=2, matrix="pooled"):
    """The color keys of results.json for one round's summary.

    Parameters
    ----------
    summary : dict
        One round's summary from measure_color.
    decimals : int, default 2
    matrix : {"pooled", "per chart"}, default "pooled"
        Which matrix the "after" values describe. The pooled matrix is the CCM (the one
        matrix an ISP would store for this illuminant), so it is the default; "before"
        does not depend on the matrix.

    Returns
    -------
    dict
        delta_e_metric, colorchecker_reference_set, and [mean, sd] for
        delta_e_mean_before_ccm, delta_e_mean_after_ccm and delta_e_worst_after_ccm.
    """
    def pair(s):
        std = None if s["std"] is None else round(s["std"], decimals)
        return [round(s["mean"], decimals), std]
    after, worst = ((summary["pooled_after"], summary["pooled_worst"]) if matrix == "pooled"
                    else (summary["mean_after"], summary["worst_after"]))
    return {"delta_e_metric": METRIC,
            "colorchecker_reference_set": wb.REFERENCE,
            "delta_e_mean_before_ccm": pair(summary["mean_before"]),
            "delta_e_mean_after_ccm": pair(after),
            "delta_e_worst_after_ccm": pair(worst)}


def measure_color(rounds, saturation_dn, output_dir, black_level=None, manual_clicks=None,
                  exposure="grays"):
    """Run the color measurement over every round; print, and write tables and figures.

    For each round: correct_chart on every capture (charts the detector misses or with
    no usable mid gray are skipped with a message), summarize_round, the pooled matrix
    (fit_pooled_matrix) and its Delta E, the per-patch table, and the neutral
    diagnostics (ratios to white and flare). Written to output_dir:
    color_dE_per_patch.csv (every patch of every chart), color_summary.csv (one row per
    round), color_<round>_per_patch_table.csv, color_<round>_dE_before_after.png, and
    color_matrices.json (per-chart and pooled matrices with row sums; the pooled one is
    marked as the CCM).

    Parameters
    ----------
    rounds : dict
        {round name: folder of raw ColorChecker captures}.
    saturation_dn : float
        Saturation level in DN.
    output_dir : str or Path
        Folder for the tables and figures (created if needed).
    black_level : float, optional
        DN to subtract; None uses each file's declared value.
    manual_clicks : dict, optional
        {file name: four (x, y) corner-patch centers} for charts the detector misses
        (see color_chart.homography_from_clicks).
    exposure : {"grays", "white"}, default "grays"
        See correct_chart; "white" reproduces HW4.

    Returns
    -------
    dict
        {round name: summary} for rounds with at least one usable chart. Each summary is
        summarize_round's dict plus pooled_M (3 x 3), pooled_after and pooled_worst
        (mean, std; pooled_after also sem), per_patch (pandas.DataFrame from
        per_patch_table), neutrals (neutral_diagnostics) and flare (mean, white_residual).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_clicks = manual_clicks or {}
    ref_names, ref_XYZ, _ = wb.load_reference()
    print(f"Reference: {wb.REFERENCE}. {wb.check_numbering(ref_names, psu.COLORCHECKER_NAMES)}")

    catalog, ident, n = chart_catalog(rounds, saturation_dn, black_level)
    meta.print_identity(ident, n)

    per_patch, summaries, matrices = [], {}, {}
    for rnd, group in catalog.groupby("round", sort=False):
        charts = []
        for _, row in group.iterrows():
            found = chart.measure_chart(row.path, saturation_dn, clicks=manual_clicks.get(row.file),
                                        black_level=black_level)
            if found is None:
                print(f"  {rnd} {row.file}: chart not found; add it to manual_clicks. Skipped.")
                continue
            c = correct_chart(found["means"], found["usable"], ref_XYZ, exposure)
            if c is None:
                print(f"  {rnd} {row.file}: no usable mid gray. Skipped.")
                continue
            c.update(file=row.file, chart=int(row.chart), usable=found["usable"])
            charts.append(c)
            per_patch.append(pd.DataFrame({
                "round": rnd, "chart": c["chart"], "file": row.file, "patch": range(1, 25),
                "name": psu.COLORCHECKER_NAMES, "usable": c["usable"],
                "dE00_before": c["de_before"], "dE00_after": c["de_after"]}))
        if not charts:
            print(f"  {rnd}: no usable charts")
            continue
        summaries[rnd] = summarize_round(charts)
        pooled = fit_pooled_matrix(charts, ref_XYZ)
        pooled_de = [delta_e(c["scaled"], ref_XYZ, c["usable"], pooled, exposure) for c in charts]
        pooled_after = np.array([d[c["usable"]].mean() for d, c in zip(pooled_de, charts)])
        pooled_worst = np.array([d[c["usable"]].max() for d, c in zip(pooled_de, charts)])
        sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else None
        table = per_patch_table(charts, ref_XYZ, pooled, exposure)
        table.to_csv(output_dir / f"color_{rnd}_per_patch_table.csv", index=False)
        diag = neutral_diagnostics(charts)
        summaries[rnd].update(
            pooled_M=pooled, per_patch=table, neutrals=diag,
            pooled_after=dict(mean=float(pooled_after.mean()), std=sd(pooled_after),
                              sem=None if sd(pooled_after) is None else sd(pooled_after) / np.sqrt(len(charts))),
            pooled_worst=dict(mean=float(pooled_worst.mean()), std=sd(pooled_worst)),
            flare=dict(mean=diag["flare"], white_residual=diag["white_residual"]))
        matrices[rnd] = dict(exposure=exposure, ccm="pooled", pooled=pooled.tolist(),
                             pooled_row_sums=pooled.sum(axis=1).tolist(),
                             per_chart={c["file"]: c["M"].tolist() for c in charts})
        if diag["white_reads"] is None:
            print(f"  {rnd}: patch 19 (white) is clipped on every chart, so there are no "
                  "ratios to white; exposure comes from the usable grays")
        else:
            print(f"  {rnd}: neutrals / white vs published (mean over {diag['n_with_white']} charts): "
                  + ", ".join(f"{p} {100 * d:+.1f}%" for p, d in diag["deviation"].items())
                  + f"; white patch reads {100 * diag['white_reads']:+.1f}% vs the six-gray fit")
        if np.isfinite(diag["flare"]):
            wr = diag["white_residual"]
            print(f"  {rnd}: flare (gain + offset fit to the neutrals) {100 * diag['flare']:+.1f}% of white"
                  + (f"; white patch {100 * wr:+.1f}% from that line (beyond flare)" if np.isfinite(wr)
                     else "; white patch clipped"))
        wb.plot_de_before_after(
            [(f"{rnd} chart {c['chart']}", c["de_before"], c["de_after"], c["usable"]) for c in charts],
            psu.COLORCHECKER_NAMES, f"Per-patch error before and after each chart's own matrix, {rnd}",
            output_dir / f"color_{rnd}_dE_before_after.png")

    if per_patch:
        pd.concat(per_patch).to_csv(output_dir / "color_dE_per_patch.csv", index=False)
    rows = [dict(round=r, charts=s["n_charts"],
                 **{f"{k}_{stat}": s[k][stat] for k in ("mean_before", "mean_after", "worst_after")
                    for stat in ("mean", "std")},
                 worst_patch=s["worst_patch"]) for r, s in summaries.items()]
    for row, (r, s_) in zip(rows, summaries.items()):
        row["mean_after_sem"] = s_["mean_after"]["sem"]
        row["pooled_after_mean"] = s_["pooled_after"]["mean"]
        row["pooled_after_std"] = s_["pooled_after"]["std"]
        row["pooled_worst_mean"] = s_["pooled_worst"]["mean"]
        row["flare_pct"] = 100 * s_["flare"]["mean"]
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "color_summary.csv", index=False)
    (output_dir / "color_matrices.json").write_text(json.dumps(matrices, indent=2) + "\n")
    print(f"\n{METRIC} (exposure from {'all six grays' if exposure == 'grays' else 'patch 19'}), "
          "mean over charts, sample std and standard error of the mean:")
    print(table.to_string(index=False, float_format="{:.2f}".format))
    return summaries


def compare_with_hw4(rnd, entries, hw4_output="output/hw4"):
    """Check that the datasheet entries reproduce HW4's results file for the same round.

    A regression check: with exposure="white" the color step uses HW4's method, so its
    entries should be identical to the ones hw4.py wrote. With the default six-gray
    exposure they are expected to differ, and project1.py does not call this.

    Parameters
    ----------
    rnd : str
        Round name (e.g. "led").
    entries : dict
        The results.json color entries from results_entries.
    hw4_output : str or Path, default "output/hw4"
        Where hw4.py wrote its per-round results files.

    Returns
    -------
    dict or None
        {key: True if identical} for each entry, or None if the HW4 file does not exist.
        Prints "identical" or which keys differ.
    """
    import json
    path = Path(hw4_output) / rnd / f"10b_{rnd}_results_color.json"
    if not path.exists():
        print(f"  (no HW4 file at {path} to compare against)")
        return None
    old = json.loads(path.read_text())
    same = {k: old.get(k) == v for k, v in entries.items()}
    print(f"  compared with {path}: " + ("identical" if all(same.values())
          else "DIFFERS in " + ", ".join(k for k, ok in same.items() if not ok)))
    return same
