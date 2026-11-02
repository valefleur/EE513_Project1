# catalog.py
from pathlib import Path
import pandas as pd
import my_library.image_metadata as meta
import my_library.analyze as analyze

# A top-of-histogram bin this many times taller than the occupied bins just below it
# is a saturation pile-up (see analyze.top_spike). Unclipped frames sit near 1.
SPIKE_THRESHOLD = 10.0

def build_catalog(rounds: dict[str, str | Path]) -> pd.DataFrame:
    """One row per raw file. Metadata only, so any assignment can use it."""
    rows = []
    for round_name, folder in rounds.items():
        for path in meta.get_image_paths(folder):
            m = meta.raw_metadata(path)
            rows.append(dict(
                round=round_name, file=path.name, path=path,
                iso=m["iso"], exposure_s=m["exposure_s"],
                black_levels=tuple(m["black_levels"]),    # tuple is hashable, so nunique() works
                white_level_dn=m["white_level"], bayer=m["bayer"], shape=m["shape"]))
    df = pd.DataFrame(rows)
    df["exposure"] = df["exposure_s"].map(meta._format_exposure)
    return df


def add_clip_stats(df: pd.DataFrame, sat_dn: int) -> pd.DataFrame:
    rows = []
    for p in df["path"]:
        c = analyze.clip_report(p, sat=sat_dn)
        rows.append(dict(max_dn=c["max_dn"], n_at_max=c["n_at_max"],
                         clip_frame_pct=100 * c["frame"], spike_ratio=c["spike_ratio"],
                         **{f"clip_{k}_pct": 100 * v for k, v in c["channels"].items()}))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_saturation_stats(df: pd.DataFrame, window_frac=0.02, spike_threshold=SPIKE_THRESHOLD):
    """HW4 Problem 1: measure where the files actually saturate, per CFA channel.

    For every frame and channel, analyze.top_spike() looks for a pile-up at the top
    of the histogram. A channel's saturation is the LOWEST spike value among frames
    whose spike ratio reaches `spike_threshold`; frames that did not clip say nothing
    about the ceiling. A channel that never clipped in any frame cannot be measured;
    it is held to the lowest saturation measured on the other channels, which is the
    conservative choice for deciding what counts as clipped.

    Returns (df with new columns, dict of measured saturation per channel).
    New columns: peak_<ch> and spike_<ch> per channel, sat_frame_pct (percent of all
    sites at or above their channel's measured saturation) and sat_<ch>_pct.
    """
    hists, rows = [], []
    for p in df["path"]:
        h, white = analyze.channel_histograms(p)
        hists.append(h)
        row = {}
        for ch, counts in h.items():
            t = analyze.top_spike(counts, white, window_frac)
            row[f"peak_{ch}"], row[f"spike_{ch}"] = t["peak_dn"], t["spike_ratio"]
        rows.append(row)
    stats = pd.DataFrame(rows)
    channels = list(hists[0].keys())

    measured = {}
    for ch in channels:
        clipped = stats[f"spike_{ch}"] >= spike_threshold
        measured[ch] = int(stats.loc[clipped, f"peak_{ch}"].min()) if clipped.any() else None
    found = [v for v in measured.values() if v is not None]
    floor = min(found) if found else None
    applied = {ch: (v if v is not None else floor) for ch, v in measured.items()}

    frac_rows = []
    for h in hists:
        r, n_at, n_all = {}, 0, 0
        for ch, counts in h.items():
            at = int(counts[applied[ch]:].sum()) if applied[ch] is not None else 0
            r[f"sat_{ch}_pct"] = 100 * at / counts.sum()
            n_at, n_all = n_at + at, n_all + int(counts.sum())
        r["sat_frame_pct"] = 100 * n_at / n_all
        frac_rows.append(r)
    out = pd.concat([df.reset_index(drop=True), stats, pd.DataFrame(frac_rows)], axis=1)
    return out, measured


def print_inventory(df):
    cols = ["round", "kind", "chart", "file", "iso", "exposure",
            "max_dn", "n_at_max", "clip_frame_pct"]
    print(df[cols].to_string(index=False, float_format="{:.3f}".format))


def check_constant_within(df: pd.DataFrame, by: str, cols: list[str]) -> None:
    """Assert each of `cols` has exactly one value within every group in `by`."""
    n = df.groupby(by)[cols].nunique()
    bad = n[(n > 1).any(axis=1)]
    assert bad.empty, f"values vary within a {by}:\n{bad}"

