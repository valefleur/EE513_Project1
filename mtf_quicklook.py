"""Quick look at slanted-edge captures: find the center and four corner squares, measure
one near-vertical edge of each, and plot ESF / LSF / MTF.

    uv run python mtf_quicklook.py images/iPhone16/Lightroom/<folder>   [--out output/quicklook]

Bare minimum, for checking captures before moving on; not the Project 1 MTF step.
Works on one green phase of the raw mosaic, black subtracted (linear light, nothing
interpolated). Frequencies are cycles per green-plane pixel; green-plane Nyquist is 0.5.
The edge analysis is my_library.mtf_analysis.slanted_edge_mtf (HW2, adapted from
Prof. James McNames' 04_measure_your_camera.ipynb).
"""
import argparse
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import rawpy

import my_library.image_metadata as meta
from my_library.geometry_measurement import green_phase
import my_library.mtf_analysis as mtf

REGIONS = ("center", "top left", "top right", "bottom left", "bottom right")


def green_plane(path):
    """One green phase of the mosaic, black subtracted, plus the white level above black."""
    with rawpy.imread(str(path)) as raw:
        bayer = np.asarray(raw.raw_pattern)
        r, c = green_phase(bayer, raw.color_desc.decode())
        plane = raw.raw_image_visible[r::2, c::2].astype(float)
        black = float(raw.black_level_per_channel[bayer[r, c]])
        white = float(raw.white_level)
    return plane - black, white - black


def find_squares(plane):
    """Dark, square blobs: the largest is the center square; in each quadrant around it,
    the farthest blob of roughly 0.1-0.4 of its area is that corner's square."""
    img = cv.GaussianBlur(np.clip(plane / np.percentile(plane, 99.5) * 255, 0, 255).astype(np.uint8), (5, 5), 0)
    _, dark = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # RETR_CCOMP marks holes (e.g. the white sheet's outline inside a dark table) as
    # children; only outer boundaries of dark blobs (no parent) can be squares.
    contours, hierarchy = cv.findContours(dark, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    h, w = plane.shape
    squares = []
    for cnt, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent != -1:
            continue
        x, y, bw, bh = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        (_, _), (rw, rh), _ = cv.minAreaRect(cnt)
        if (area < 0.0004 * h * w or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
                or min(rw, rh) == 0 or not 0.85 < rw / rh < 1.18 or area / (rw * rh) < 0.9):
            continue
        squares.append(dict(contour=cnt.reshape(-1, 2), area=area, box=(x, y, bw, bh),
                            center=(x + bw / 2, y + bh / 2)))
    if not squares:
        return {}
    middle = max(squares, key=lambda s: s["area"])
    found = {"center": middle}
    mx, my = middle["center"]
    for name, (sx, sy) in zip(REGIONS[1:], ((-1, -1), (1, -1), (-1, 1), (1, 1))):
        cands = [s for s in squares if s is not middle and 0.1 < s["area"] / middle["area"] < 0.4
                 and np.sign(s["center"][0] - mx) == sx and np.sign(s["center"][1] - my) == sy]
        if cands:
            found[name] = max(cands, key=lambda s: np.hypot(s["center"][0] - mx, s["center"][1] - my))
    return found


def edge_roi(square, side, margin=16):
    """Rows: the middle half of the square's left or right side. Columns: that side's
    edge points, widened by the edge's slant across those rows plus margin."""
    x, y, bw, bh = square["box"]
    pts = square["contour"]
    r0, r1 = int(y + 0.25 * bh), int(y + 0.75 * bh)
    on_side = pts[(pts[:, 1] >= r0) & (pts[:, 1] <= r1)]
    on_side = on_side[np.abs(on_side[:, 0] - (x if side == "left" else x + bw)) < 0.15 * bw]
    xc = np.median(on_side[:, 0])
    half = margin + 0.5 * (on_side[:, 0].max() - on_side[:, 0].min())
    return r0, r1, int(xc - half), int(xc + half) + 1


def analyze(path):
    plane, span = green_plane(path)
    squares = find_squares(plane)
    results = {}
    w = plane.shape[1]
    for name in REGIONS:
        if name not in squares:
            continue
        sq = squares[name]
        side = "right" if name == "center" else ("left" if sq["center"][0] < w / 2 else "right")
        r0, r1, c0, c1 = edge_roi(sq, side)
        roi = plane[r0:r1, c0:c1]
        freq, m, esf_x, esf, lsf, *_ = mtf.slanted_edge_mtf(roi)
        results[name] = dict(freq=freq, mtf=m, esf_x=esf_x, esf=esf, lsf=lsf,
                             mtf50=mtf.mtf50(freq, m), box=(r0, r1, c0, c1), side=side,
                             peak_frac=float(roi.max() / span))
    return plane, results


def plot(path, plane, results, out_png):
    fig = plt.figure(figsize=(15, 3.2 * (len(results) + 1)))
    ax = fig.add_subplot(len(results) + 1, 1, 1)
    ax.imshow(plane, cmap="gray", vmin=0, vmax=np.percentile(plane, 99.5))
    for name, r in results.items():
        r0, r1, c0, c1 = r["box"]
        ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec="red", lw=1.5))
        ax.text(c0, r0 - 8, name, color="red", fontsize=8)
    ax.set_title(f"{Path(path).name}: green plane (linear) and edge regions"); ax.axis("off")
    # Right edges first, then left edges: a left edge runs white to black, so its ESF and
    # LSF are flipped, and grouping them avoids switching direction from row to row.
    ordered = sorted(results.items(), key=lambda kv: (kv[1]["side"] == "left", REGIONS.index(kv[0])))
    for i, (name, r) in enumerate(ordered):
        a1, a2, a3 = (fig.add_subplot(len(results) + 1, 3, 3 * (i + 1) + k) for k in (1, 2, 3))
        a1.plot(r["esf_x"], r["esf"], ".", ms=3); a1.set_title(f"{name} ({r['side']} edge): ESF")
        a2.plot(r["esf_x"], r["lsf"]); a2.set_title("LSF")
        a3.plot(r["freq"], r["mtf"]); a3.axvline(0.5, color="gray", ls=":", lw=1)
        a3.axhline(0.5, color="gray", ls=":", lw=1); a3.set_xlim(0, 1); a3.set_ylim(0, 1.05)
        a3.set_title(f"MTF, MTF50 = {r['mtf50']:.3f} cy/green-px")
        a1.set_xlabel("px from edge"); a2.set_xlabel("px from edge"); a3.set_xlabel("cycles / green-plane px")
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", type=Path, help="folders or raw files")
    p.add_argument("--out", type=Path, default=Path("output/quicklook"))
    args = p.parse_args()
    files = []
    for item in args.inputs:
        files += meta.get_image_paths(item) if item.is_dir() else [item]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(files)} files:")
    meta.assert_one_camera(files)
    table = {name: [] for name in REGIONS}
    for f in files:
        md = meta.raw_metadata(f)
        plane, results = analyze(f)
        plot(f, plane, results, args.out / f"{f.stem}_mtf.png")
        cells = []
        for name in REGIONS:
            r = results.get(name)
            if r is None:
                cells.append(f"{name}: not found")
                continue
            table[name].append(r["mtf50"])
            clip = " CLIPPED?" if r["peak_frac"] > 0.98 else ""
            cells.append(f"{name} {r['mtf50']:.3f} (peak {100 * r['peak_frac']:.0f}%){clip}")
        print(f"{f.name}  ISO {md['iso']:.0f}  1/{1 / md['exposure_s']:.0f} s:  " + " | ".join(cells))
    print(f"\nMTF50 in cycles per green-plane pixel (green Nyquist 0.5), over {len(files)} files:")
    for name, vals in table.items():
        v = np.array(vals, float)
        v = v[np.isfinite(v)]
        if len(v):
            sd = f" +/- {v.std(ddof=1):.3f}" if len(v) > 1 else ""
            print(f"  {name:13s} {v.mean():.3f}{sd}   (n = {len(v)})")
    print(f"\nFigures: {args.out}/<file>_mtf.png")


if __name__ == "__main__":
    main()
