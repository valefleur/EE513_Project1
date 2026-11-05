"""Project 1, measurement 5: color accuracy from ColorChecker captures.

Per chart: locate and sample the 24 patches, white balance on an unclipped mid gray,
fit a 3x3 least-squares color correction matrix (camera RGB -> reference XYZ) to that
chart alone, and compute CIEDE2000 per patch before and after the matrix. Per lighting
round: mean and worst patch, each as the mean over charts with the sample standard
deviation across the nominally identical charts as its uncertainty.

This is the method of hw4.py Problems 5, 6, 9 and 10, packaged as functions so that
project1.py can run it. It reuses my_library.color_chart (patch sampling rule from
Prof. James McNames, 07_build_an_isp.ipynb section 4) and my_library.white_balance
(least-squares fit and CIEDE2000, following Prof. McNames' 06_color_matrix.ipynb).
"""
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

    Checks, before anything is fitted: one camera across every file, and ISO, exposure
    and declared black level constant within each round (HW4 Problem 1 requires
    identical settings within a round; this check is for ColorChecker rounds only).
    If black_level is given and differs from what a file declares, that is printed.
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


def correct_chart(means, usable, ref_XYZ):
    """White balance, fit this chart's matrix, and score it. Pure arithmetic, no I/O.

    means: 24 x 3 linear camera RGB; usable: 24 bools (False = clipped or near ceiling).
    Returns dict with gray_idx (0-based index of the neutral patch used for white
    balance), gains, matrix M, and per-patch dE before (balanced camera RGB treated as
    linear sRGB) and after (fitted M). Returns None if no mid gray is usable.
    """
    gray_idx = wb.pick_gray(usable)
    if gray_idx is None:
        return None
    gains = wb.gains_from_gray(means[gray_idx])
    scaled, _ = wb.scale_to_reference(means * gains, ref_XYZ, usable)
    M = wb.fit_matrix(scaled, ref_XYZ, usable)
    before, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable)
    after, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable, M=M)
    return dict(gray_idx=gray_idx, gains=gains, M=M, de_before=before, de_after=after)


def summarize_round(charts):
    """Round summary from per-chart results (each with de_before, de_after, usable).

    Each statistic is computed per chart over its usable patches, then reported as the
    mean over charts and the sample standard deviation across charts (ddof=1).
    """
    def over_charts(values):
        v = np.asarray(values, float)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else None,
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


def results_entries(summary, decimals=2):
    """The color keys of results.json for one round's summary."""
    def pair(s):
        std = None if s["std"] is None else round(s["std"], decimals)
        return [round(s["mean"], decimals), std]
    return {"delta_e_metric": METRIC,
            "colorchecker_reference_set": wb.REFERENCE,
            "delta_e_mean_before_ccm": pair(summary["mean_before"]),
            "delta_e_mean_after_ccm": pair(summary["mean_after"]),
            "delta_e_worst_after_ccm": pair(summary["worst_after"])}


def measure_color(rounds, saturation_dn, output_dir, black_level=None, manual_clicks=None):
    """Run the color measurement over every round. Writes CSVs and figures to output_dir.

    rounds: {round name: folder}. black_level: DN to subtract (None = each file's
    declared value). manual_clicks: {file name: four (x, y) corner-patch centers} for
    charts the detector misses (see color_chart.homography_from_clicks).
    Returns {round name: summary} for the rounds that had at least one usable chart.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_clicks = manual_clicks or {}
    ref_names, ref_XYZ, _ = wb.load_reference()
    print(f"Reference: {wb.REFERENCE}. {wb.check_numbering(ref_names, psu.COLORCHECKER_NAMES)}")

    catalog, ident, n = chart_catalog(rounds, saturation_dn, black_level)
    meta.print_identity(ident, n)

    per_patch, summaries = [], {}
    for rnd, group in catalog.groupby("round", sort=False):
        charts = []
        for _, row in group.iterrows():
            found = chart.measure_chart(row.path, saturation_dn, clicks=manual_clicks.get(row.file),
                                        black_level=black_level)
            if found is None:
                print(f"  {rnd} {row.file}: chart not found; add it to manual_clicks. Skipped.")
                continue
            c = correct_chart(found["means"], found["usable"], ref_XYZ)
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
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "color_summary.csv", index=False)
    print(f"\n{METRIC}, mean over charts and sample std across charts:")
    print(table.to_string(index=False, float_format="{:.2f}".format))
    return summaries


def compare_with_hw4(rnd, entries, hw4_output="output/hw4"):
    """Print how these entries compare with HW4's results file for the same round, if any."""
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
