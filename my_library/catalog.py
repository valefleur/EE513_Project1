# catalog.py

import pandas as pd
import my_library.image_metadata as meta
import my_library.analyze as analyze

def build_catalog(rounds, saturation_dn, white_tag="white"):
    """One row per raw file. Columns carry units in their names (_dn, _s, _pct)."""
    rows = []
    for round_name, folder in rounds.items():
        chart = 0
        for path in meta.get_image_paths(folder):
            m = meta.raw_metadata(path)
            is_white = white_tag in path.stem.lower()
            chart += not is_white
            clip = analyze.clip_report(path, sat=saturation_dn)
            rows.append(dict(
                round=round_name, kind="white" if is_white else "chart",
                chart=None if is_white else chart, file=path.name,
                iso=m["iso"], exposure_s=m["exposure_s"],
                black_dn=max(m["black_levels"]),
                black_uniform=len(set(m["black_levels"])) == 1,
                white_level_dn=m["white_level"], bayer=m["bayer"],
                max_dn=clip["max_dn"], n_at_max=clip["n_at_max"],
                clip_frame_pct=100 * clip["frame"],
                spike_ratio=clip["spike_ratio"],
                **{f"clip_{k}_pct": 100 * v for k, v in clip["channels"].items()},
                path=path))
    df = pd.DataFrame(rows)
    df["chart"] = df["chart"].astype("Int64")
    df["exposure"] = df["exposure_s"].map(meta._format_exposure)
    return df

def print_inventory(df):
    cols = ["round", "kind", "chart", "file", "iso", "exposure",
            "max_dn", "n_at_max", "clip_frame_pct"]
    print(df[cols].to_string(index=False, float_format="{:.3f}".format))

def check_rounds(df, min_clipped_px=100):
    """The Problem 1 assertions, in one place."""
    g = df.groupby("round").agg(n=("file", "size"), n_iso=("iso", "nunique"),
                                n_exp=("exposure_s", "nunique"))
    assert (g.n_iso == 1).all() and (g.n_exp == 1).all(), g
    assert df.black_uniform.all()
    df["camera_id"] = df.path.map(meta.lens_id)
    assert df.camera_id.nunique() == 1, df.camera_id.unique()
    clipped = df[df.n_at_max >= min_clipped_px]
    print(f"{len(clipped)}/{len(df)} frames have a real spike at their max.")
    print(f"They stop at {sorted(clipped.max_dn.unique())} "
          f"(declared white level: {sorted(df.white_level_dn.unique())})")
    return g