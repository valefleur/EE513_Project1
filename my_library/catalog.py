# catalog.py
from pathlib import Path
import pandas as pd
import my_library.image_metadata as meta
import my_library.analyze as analyze

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


def print_inventory(df):
    cols = ["round", "kind", "chart", "file", "iso", "exposure",
            "max_dn", "n_at_max", "clip_frame_pct"]
    print(df[cols].to_string(index=False, float_format="{:.3f}".format))


def check_constant_within(df: pd.DataFrame, by: str, cols: list[str]) -> None:
    """Assert each of `cols` has exactly one value within every group in `by`."""
    n = df.groupby(by)[cols].nunique()
    bad = n[(n > 1).any(axis=1)]
    assert bad.empty, f"values vary within a {by}:\n{bad}"

