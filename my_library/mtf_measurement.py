"""Project 1, measurement 2: spatial resolution (MTF) by the slanted-edge method.

The math (Prof. McNames, 04_measure_your_camera.ipynb sections 2-4; HW2's
my_library/mtf_analysis.slanted_edge_mtf). A straight edge tilted a few degrees from
the pixel grid crosses each row at a slightly different sub-pixel position. Locating
the edge in every row, fitting a line through those positions, and binning every
pixel by its distance from that line gives the edge spread function (ESF) sampled
four times finer than the pixels. Its derivative is the line spread function (LSF,
the camera's blur across the edge), and the magnitude of the LSF's Fourier transform,
normalized to 1 at zero frequency, is the MTF. MTF50 is the frequency where it falls
to 0.5. The ESF is always cropped to +/-12 px of the edge before differentiating, for
test and real data alike (see slanted_edge_mtf). In DSP terms: ESF = step response, LSF = impulse response, MTF = |frequency
response|; the slant is equivalent-time sampling.

Data. One green phase of the raw mosaic, black level subtracted (linear light, nothing
interpolated), so frequencies are in cycles per green-plane sample: green-plane
Nyquist is 0.5. One green-plane sample spans two raw pixels, so cycles per raw pixel
are half these values, and cycles per micrometer divide by 2 x the pixel pitch.

Target. A printed sheet with a large tilted center square and four smaller corner
squares. For each square the step measures one near-vertical and one near-horizontal
edge, on the side facing the frame's outer edge (the center square: right and bottom).

Series. Frames are grouped into series by subfolder (one target position per series).
Within a series nothing should move, so the spread over its frames is the measurement
noise (notebook 04 section 4: mean +/- sample sd). The datasheet takes the center from
the center series and each corner from the series that put that corner's square
closest to the frame corner.

Note (noise step, Fri Nov 6): the raw data's noise shows a fixed sharpening-like filter
within each color plane, applied after readout; the MTF measured here is of lens +
sensor + that processing.
"""
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy

import my_library.geometry_measurement as gm
import my_library.image_metadata as meta
import my_library.mtf_analysis as mtfa

REGIONS = ("center", "top left", "top right", "bottom left", "bottom right")
CORNERS = REGIONS[1:]
DIRECTIONS = ("vertical", "horizontal")
CLIPPED_PEAK = 0.98          # brightest pixel above this fraction of the span: clipped
MIN_EDGE_DRIFT_PX = 4.0      # edge must drift this far across its rows (see edge_drift)
RADIUS_MATCH = 0.05          # series this close to a corner's farthest radius count as repeats
UNDEREXPOSED_PAPER_LEVEL = 0.20   # paper (bright) side below this fraction of the span: low SNR


# ----------------------------------------------------------------- reading and finding

def green_plane(path):
    """One green phase of the raw mosaic, black level subtracted, and the span above it.

    Parameters
    ----------
    path : str or Path
        A raw file.

    Returns
    -------
    (plane, span) : tuple
        plane : ndarray, float, linear DN above the black level (mosaic[r::2, c::2] for the
            first green site (r, c) of the Bayer tile).
        span : float, white level minus black level, in DN.
    """
    with rawpy.imread(str(path)) as raw:
        bayer = np.asarray(raw.raw_pattern)
        r, c = gm.green_phase(bayer, raw.color_desc.decode())
        plane = raw.raw_image_visible[r::2, c::2].astype(float)
        black_level = float(raw.black_level_per_channel[bayer[r, c]])
        white_level = float(raw.white_level)
    return plane - black_level, white_level - black_level


def _squares_at_threshold(img, threshold):
    """Target squares in an 8-bit (log-scaled) image, with dark = below threshold.

    Keeps outer boundaries only (RETR_CCOMP, no parent, so a dark surround cannot
    contribute its inner outline) of blobs that are square (aspect 0.85-1.18, fill > 0.9
    of their rotated bounding box), large enough, and not touching the frame edge. The
    largest is the center square; in each quadrant around it, the farthest blob of
    0.1-0.4 of its area is that corner's square (45 mm / 100 mm squares give about 0.2).

    Parameters
    ----------
    img : ndarray, uint8
    threshold : int
        Pixels below this are dark.

    Returns
    -------
    dict
        {region name: dict(contour, area, box (x, y, w, h), center)} for regions found.
    """
    dark = ((img < threshold) * 255).astype(np.uint8)
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, hierarchy = cv.findContours(dark, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    h, w = img.shape
    squares = []
    for cnt, (_, _, _, parent) in zip(contours, hierarchy[0] if hierarchy is not None else []):
        if parent != -1:
            continue
        x, y, bw, bh = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)
        (_, _), (rw, rh), _ = cv.minAreaRect(cnt)
        if (area < 0.0004 * h * w or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
                or min(rw, rh) == 0 or not 0.85 < rw / rh < 1.18 or area / (rw * rh) < 0.9):
            continue
        squares.append(dict(contour=cnt.reshape(-1, 2), area=area, box=(x, y, bw, bh),
                            center=(x + (bw - 1) / 2, y + (bh - 1) / 2)))
    if not squares:
        return {}
    middle = max(squares, key=lambda s: s["area"])
    found = {"center": middle}
    mx, my = middle["center"]
    for name, (sx, sy) in zip(CORNERS, ((-1, -1), (1, -1), (-1, 1), (1, 1))):
        cands = [s for s in squares if s is not middle and 0.1 < s["area"] / middle["area"] < 0.4
                 and np.sign(s["center"][0] - mx) == sx and np.sign(s["center"][1] - my) == sy]
        if cands:
            found[name] = max(cands, key=lambda s: np.hypot(s["center"][0] - mx, s["center"][1] - my))
    return found


def find_squares(plane, thresholds=range(20, 240, 10)):
    """The target's dark squares: the center square and the farthest square per quadrant.

    Two problems shape the method, both met on real captures.

    1. Lighting is multiplicative: the paper can be 2-3x dimmer on one side of the sheet
       (light falloff, a clip's shadow), and a linear threshold then classes the dim paper
       as dark, merging the squares with the background (APC_0500, Sat Nov 7). Ink
       contrast is multiplicative too (black about 14x below paper). So the threshold
       works on the log of the image, where the lighting gradient becomes a small
       additive shift (log 2.5 ~ 0.9) and the paper-to-ink gap stays large (log 14 ~ 2.6).

    2. The scene around the target varies. Outdoors with dark foreground and foliage
       (APC_0658 and APC_0677, Sun Nov 8), a single automatic (Otsu) threshold lands
       between the dark scene and the gray board, so the squares, dark only relative to
       the paper, end up on the bright side and nothing is found. Measured there: paper
       1741 DN, squares 123, board 206, foreground 9. Any threshold inside the
       paper-to-ink gap isolates every square, because the paper surrounds each one,
       whatever the rest of the scene does.

    So the log image is scaled to 8 bits and a range of thresholds is tried
    (_squares_at_threshold). Each result is scored by whether it found a center square
    and then by how many corner squares it found (a real target gives a center and up
    to four corners of the right relative size, arranged one per quadrant, which a
    stray square-ish shadow does not). Among the thresholds with the best score, the
    middle one is used: the most robust choice inside the working range.

    Parameters
    ----------
    plane : ndarray
        Linear green plane.
    thresholds : iterable of int, default range(20, 240, 10)
        Thresholds to try on the 8-bit log image.

    Returns
    -------
    dict
        {region name: dict(contour (N x 2 points), area, box (x, y, w, h), center)},
        only for the regions found; empty if no threshold found a center square.
    """
    L = np.log(np.maximum(plane, 1.0))                      # lighting becomes additive
    lo, hi = np.percentile(L, 1), np.percentile(L, 99.5)
    if hi - lo < 1e-6:                                      # a uniform frame has no target
        return {}
    img = cv.GaussianBlur(np.clip((L - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), (5, 5), 0)
    results = [(t, _squares_at_threshold(img, t)) for t in thresholds]
    score = lambda found: (("center" in found), len(found))
    best = max(score(f) for _, f in results)
    if not best[0]:
        return {}
    tied = [f for _, f in results if score(f) == best]
    return tied[len(tied) // 2]


def edge_side(region, square, shape, direction):
    """Which side of a square to measure: the one facing the frame's outer edge.

    Parameters
    ----------
    region : str
        Region name.
    square : dict
        From find_squares.
    shape : (h, w)
        Plane shape.
    direction : {"vertical", "horizontal"}
        A vertical edge is a left or right side; a horizontal edge is a top or bottom side.

    Returns
    -------
    str
        "left", "right", "top" or "bottom". The center square uses right and bottom.
    """
    h, w = shape
    cx, cy = square["center"]
    if direction == "vertical":
        return "right" if region == "center" else ("left" if cx < (w - 1) / 2 else "right")
    return "bottom" if region == "center" else ("top" if cy < (h - 1) / 2 else "bottom")


def edge_roi(square, side, margin=16):
    """Region of interest across one side of a square.

    Along the edge: the middle half of the side (away from the corners). Across it:
    the side's contour points, widened by the edge's slant over that span plus margin.

    Parameters
    ----------
    square : dict
        From find_squares.
    side : {"left", "right", "top", "bottom"}
    margin : int, default 16
        Pixels kept on each side of the edge beyond its slant.

    Returns
    -------
    (r0, r1, c0, c1) : tuple of int
        Rows r0:r1 and columns c0:c1 of the plane.
    """
    x, y, bw, bh = square["box"]
    pts = square["contour"]
    if side in ("left", "right"):
        r0, r1 = int(y + 0.25 * bh), int(y + 0.75 * bh)
        on = pts[(pts[:, 1] >= r0) & (pts[:, 1] <= r1)]
        on = on[np.abs(on[:, 0] - (x if side == "left" else x + bw - 1)) < 0.15 * bw]
        mid, spread = np.median(on[:, 0]), on[:, 0].max() - on[:, 0].min()
        half = margin + 0.5 * spread
        return r0, r1, int(mid - half), int(mid + half) + 1
    c0, c1 = int(x + 0.25 * bw), int(x + 0.75 * bw)
    on = pts[(pts[:, 0] >= c0) & (pts[:, 0] <= c1)]
    on = on[np.abs(on[:, 1] - (y if side == "top" else y + bh - 1)) < 0.15 * bh]
    mid, spread = np.median(on[:, 1]), on[:, 1].max() - on[:, 1].min()
    half = margin + 0.5 * spread
    return int(mid - half), int(mid + half) + 1, c0, c1


def measure_edge(roi, direction):
    """MTF of one edge region with HW2's slanted_edge_mtf.

    slanted_edge_mtf expects a near-vertical edge (rows along the edge), so a
    horizontal edge's region is transposed first; the result is the MTF across that
    edge either way.

    Parameters
    ----------
    roi : ndarray
        The region (plane[r0:r1, c0:c1]).
    direction : {"vertical", "horizontal"}

    Returns
    -------
    dict
        freq (cycles per sample), mtf, esf_x, esf, lsf, mtf50 (cycles per sample, nan
        if the MTF never falls below 0.5), slant_deg (edge angle from the pixel grid).
    """
    data = roi.T if direction == "horizontal" else roi
    freq, m, esf_x, esf, lsf, _, slope, _ = mtfa.slanted_edge_mtf(data)
    return dict(freq=freq, mtf=m, esf_x=esf_x, esf=esf, lsf=lsf, mtf50=float(mtfa.mtf50(freq, m)),
                slant_deg=float(np.degrees(np.arctan(slope))))


def edge_drift(n_rows, slant_deg):
    """How far an edge moves across the pixel grid over the rows of its region.

    The math. The slanted-edge method samples the edge at sub-pixel positions because
    each row crosses it at a slightly different phase (equivalent-time sampling). Over
    n rows an edge tilted theta from the grid drifts n |tan theta| pixels. With 4 bins
    per pixel, each bin needs rows from many different phases: if the drift is only a
    pixel or two, the bins are filled by a few rows at the same few phases, and the ESF
    and LSF show a comb (values alternating from bin to bin) instead of a smooth curve.
    On APC_0500 (Sat Nov 7) an edge tilted 0.57 deg drifted 1.4 px over 143 rows and its
    LSF was such a comb; edges tilted 2-5.7 deg drifted 5-14 px and were smooth. Edges
    drifting less than MIN_EDGE_DRIFT_PX are excluded from the statistics.

    Parameters
    ----------
    n_rows : int
        Rows along the edge in its region.
    slant_deg : float
        The edge's angle from the pixel grid, degrees.

    Returns
    -------
    float
        Drift in pixels.
    """
    return float(n_rows * abs(np.tan(np.radians(slant_deg))))


def analyze_frame(plane, span):
    """Every region and edge direction of one frame.

    Parameters
    ----------
    plane : ndarray
        Linear green plane.
    span : float
        White level minus black level, DN.

    Returns
    -------
    list of dict
        One row per measured edge: region, direction, side, mtf50, slant_deg, radius
        (region center's distance from the frame center as a fraction of the
        half-diagonal: 0 at center, 1 at a frame corner), peak (brightest pixel / span),
        paper_level (the paper side's level, its 90th percentile / span; not the sensor's
        white level), drift_px (edge_drift), reliable (drift at
        least MIN_EDGE_DRIFT_PX), roi, and the curves (freq, mtf, esf_x, esf, lsf).
    """
    shape = plane.shape
    fc = np.array(gm.frame_center(shape[::-1]))
    half_diag = np.hypot(shape[1] / 2, shape[0] / 2)
    rows = []
    for region, sq in find_squares(plane).items():
        for direction in DIRECTIONS:
            side = edge_side(region, sq, shape, direction)
            try:
                r0, r1, c0, c1 = edge_roi(sq, side)
                roi = plane[max(r0, 0):r1, max(c0, 0):c1]
                if min(roi.shape) < 8:
                    continue
                e = measure_edge(roi, direction)
            except (ValueError, np.linalg.LinAlgError, TypeError):
                continue
            hi = np.percentile(roi, 90)
            n_along = roi.shape[1] if direction == "horizontal" else roi.shape[0]
            drift = edge_drift(n_along, e["slant_deg"])
            center = np.array([(c0 + c1 - 1) / 2, (r0 + r1 - 1) / 2])
            rows.append(dict(region=region, direction=direction, side=side, mtf50=e["mtf50"],
                             slant_deg=e["slant_deg"],
                             radius=float(np.linalg.norm(center - fc) / half_diag),
                             peak=float(roi.max() / span), paper_level=float(hi / span),
                             drift_px=drift, reliable=drift >= MIN_EDGE_DRIFT_PX,
                             roi=(r0, r1, c0, c1), **{k: e[k] for k in ("freq", "mtf", "esf_x", "esf", "lsf")}))
    return rows


# ----------------------------------------------------------------- series and summaries

def series_folders(folder):
    """Group the raw files of a slanted-edge folder into series.

    Each immediate subfolder holding raw files is one series (one target position); raw
    files directly in the folder form a series named after the folder.

    Parameters
    ----------
    folder : str or Path

    Returns
    -------
    dict
        {series name: sorted list of Path}.
    """
    folder = Path(folder)
    out = {}
    own = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in meta.RAW_SUFFIXES)
    if own:
        out[folder.name] = own
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in meta.RAW_SUFFIXES)
        if files:
            out[sub.name] = files
    return out


def summarize_series(frames):
    """Mean +/- sample sd of MTF50 over the frames of one series, per region.

    The math. For each frame, a region's MTF50 is the mean of its two edge directions
    (or the one available; edges that do not drift enough, see edge_drift, are left
    out); over the n frames of the series, value = mean and its
    uncertainty = sample sd (ddof = 1), as in notebook 04 section 4. Each direction is
    also summarized on its own, so a lens that resolves the two directions differently
    off-axis is visible.

    Parameters
    ----------
    frames : list of list of dict
        analyze_frame's rows for each frame of the series.

    Returns
    -------
    dict
        {region: dict(mean, sd (None for one frame), n, radius (mean), by_direction
        {direction: dict(mean, sd, n)}, peak (max), paper_level (mean))}.
    """
    out = {}
    for region in REGIONS:
        per_frame, by_dir, radius, peak, paper = [], {d: [] for d in DIRECTIONS}, [], [], []
        for rows in frames:
            mine = [r for r in rows if r["region"] == region and np.isfinite(r["mtf50"])
                    and r.get("reliable", True)]
            if not mine:
                continue
            per_frame.append(np.mean([r["mtf50"] for r in mine]))
            for r in mine:
                by_dir[r["direction"]].append(r["mtf50"])
            radius += [r["radius"] for r in mine]; peak += [r["peak"] for r in mine]; paper += [r["paper_level"] for r in mine]
        if not per_frame:
            continue
        v = np.array(per_frame)
        stat = lambda a: dict(mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else None, n=len(a))
        out[region] = dict(**stat(v), radius=float(np.mean(radius)),
                           by_direction={d: stat(a) for d, a in by_dir.items() if a},
                           peak=float(np.max(peak)), paper_level=float(np.mean(paper)))
    return out


def datasheet_series(names, pattern=None):
    """The series allowed to supply datasheet values: names matching a glob pattern.

    Center and corners must come from one capture distance, or "corners vs center"
    compares different setups (and, if the print limits the near shots, different
    blurs). Series names that encode the distance make this a pattern, e.g. "*_501mm".

    Parameters
    ----------
    names : iterable of str
        All series names.
    pattern : str, optional
        fnmatch-style pattern ("*" any text, "?" one character); None allows all.

    Returns
    -------
    list of str
        The matching names, in the given order.
    """
    from fnmatch import fnmatch
    return [n for n in names if pattern is None or fnmatch(n, pattern)]


def pick_center_series(names):
    """The center series among the datasheet series: the only one, or the only one whose
    name contains "center". None if that is ambiguous.

    Parameters
    ----------
    names : list of str
        Datasheet-eligible series names (datasheet_series).

    Returns
    -------
    str or None
    """
    if len(names) == 1:
        return names[0]
    centers = [n for n in names if "center" in n.lower()]
    return centers[0] if len(centers) == 1 else None


def datasheet_values(summaries, center_series, eligible=None):
    """The datasheet's MTF50 numbers from the per-series summaries.

    Center: the center region of center_series. Each corner: from the eligible series
    whose square for that corner sat farthest out (largest mean radius), with at least
    two frames.

    Systematic uncertainty. A series' sd covers only frame-to-frame noise within one
    setup; re-placing the target (tilt, flatness, focus at that spot, lighting) can move
    the result much more. When other eligible series measured the same corner at a
    comparable position, they are repeats of the setup. A series qualifies as a repeat
    when it is eligible (a datasheet series), measured the corner in at least two
    reliable frames, and its radius for that corner is within RADIUS_MATCH of the
    farthest radius any eligible series reached for it (the square sat in essentially
    the same place in the frame). With k >= 2 qualifying series of means m_1 ... m_k,
    value and uncertainty come from the same series:
        value    = (m_1 + ... + m_k) / k,
        sd_sys   = sample sd of m_1 ... m_k (ddof = 1),
        sd_total = max(sd_sys, sd_within),
    where sd_within is the root-mean-square of the series' own within-series sds. The
    spread of the series means already contains each series' frame noise, so the two
    are not added; the max guards against a chance agreement of two noisy series. Like
    the color datasheet (plan Q24), this is the sd of a repeat, not the standard error
    of the mean. (On Sat Nov 7 the bottom-left corner read 0.265 and 0.160 at radius
    0.71 and 0.69: value 0.213, sd 0.074, about ten times the within-series sd.) With
    one qualifying series, the value is its mean, sd_sys is not known, and the corner
    keeps its within-series sd; the report should say so.

    Corner mean: the average of the four corners, uncertainty sqrt(sum sd_total_i^2) / 4
    (independent corners). Worst corner: the lowest corner value with its own sd_total.

    Parameters
    ----------
    summaries : dict
        {series name: summarize_series result}.
    center_series : str
    eligible : list of str, optional
        Series allowed to supply corners (datasheet_series); None allows all. Use it to
        keep center and corners at one capture distance.

    Returns
    -------
    dict
        center (dict(mean, sd, n, radius, series)), corners ({corner: dict(mean, sd
        (= sd_total), sd_within, sd_sys (None without a repeat), repeats (series names
        used for sd_sys), n, radius, series)}), corner_mean (mean, sd), corner_worst
        (mean, sd, corner), missing (corners not found).
    """
    if center_series not in summaries or "center" not in summaries[center_series]:
        raise SystemExit(f"no center measurement in series {center_series!r}")
    eligible = list(summaries) if eligible is None else eligible
    c = summaries[center_series]["center"]
    out = dict(center=dict(mean=c["mean"], sd=c["sd"], n=c["n"], radius=c["radius"], series=center_series),
               corners={}, missing=[])
    for corner in CORNERS:
        cands = [(name, summaries[name][corner]) for name in eligible
                 if name in summaries and corner in summaries[name] and summaries[name][corner]["n"] >= 2]
        if not cands:
            out["missing"].append(corner)
            continue
        name, v = max(cands, key=lambda kv: kv[1]["radius"])
        repeats = [(n_, s_) for n_, s_ in cands if s_["radius"] >= v["radius"] - RADIUS_MATCH]
        if len(repeats) >= 2:
            means = [s_["mean"] for _, s_ in repeats]
            withins = [s_["sd"] for _, s_ in repeats if s_["sd"] is not None]
            sd_within = float(np.sqrt(np.mean(np.square(withins)))) if withins else None
            sd_sys = float(np.std(means, ddof=1))
            out["corners"][corner] = dict(
                mean=float(np.mean(means)), sd=max(sd_sys, sd_within or 0.0), sd_within=sd_within,
                sd_sys=sd_sys, repeats=[n_ for n_, _ in repeats], n=int(sum(s_["n"] for _, s_ in repeats)),
                radius=float(np.mean([s_["radius"] for _, s_ in repeats])), series=name)
        else:
            out["corners"][corner] = dict(mean=v["mean"], sd=v["sd"], sd_within=v["sd"], sd_sys=None,
                                          repeats=[], n=v["n"], radius=v["radius"], series=name)
    if out["corners"]:
        vals = list(out["corners"].values())
        sds = [v["sd"] for v in vals if v["sd"] is not None]
        out["corner_mean"] = dict(mean=float(np.mean([v["mean"] for v in vals])),
                                  sd=float(np.sqrt(np.sum(np.square(sds))) / len(vals)) if len(sds) == len(vals) else None)
        worst_name, worst = min(out["corners"].items(), key=lambda kv: kv[1]["mean"])
        out["corner_worst"] = dict(mean=worst["mean"], sd=worst["sd"], corner=worst_name)
    return out


def unit_conversions(cy_per_green_px, pitch_um):
    """One spatial frequency in several units.

    The math. A green-plane sample spans two raw pixels, so cycles per raw pixel =
    cycles per green sample / 2, and one raw pixel is pitch_um micrometers wide, so
    cycles per micrometer = cycles per green sample / (2 pitch_um).

    Parameters
    ----------
    cy_per_green_px : float
        Cycles per green-plane sample (green Nyquist 0.5).
    pitch_um : float
        Raw (binned output) pixel pitch in micrometers.

    Returns
    -------
    dict
        cy_per_green_px, cy_per_raw_px, cy_per_um, cy_per_mm, and frac_of_green_nyquist.
    """
    per_um = cy_per_green_px / (2.0 * pitch_um)
    return dict(cy_per_green_px=cy_per_green_px, cy_per_raw_px=cy_per_green_px / 2.0,
                cy_per_um=per_um, cy_per_mm=per_um * 1000.0, frac_of_green_nyquist=cy_per_green_px / 0.5)


def two_distance(near, far):
    """The professor's test (HW2 feedback, item 16): is the print limiting the MTF?

    The math. Moving twice as far shrinks the print's blur on the sensor by half but
    leaves the camera's own blur unchanged. If the print limits the measurement, MTF50
    in cycles per sample rises at the longer distance; if the camera limits it, MTF50
    stays about the same. The difference is judged against its combined sd.

    Parameters
    ----------
    near, far : dict
        Center summaries (mean, sd) at the usual and the longer distance.

    Returns
    -------
    dict
        ratio (far / near), difference, sd_difference, z (difference / sd), and verdict.
    """
    diff = far["mean"] - near["mean"]
    sd = np.hypot(near["sd"] or 0.0, far["sd"] or 0.0)
    z = diff / sd if sd > 0 else float("inf") * np.sign(diff)
    verdict = ("print-limited: MTF50 rises at the longer distance" if z > 2 else
               "camera-limited: no significant change with distance" if abs(z) <= 2 else
               "MTF50 falls at the longer distance: check focus at that distance")
    return dict(ratio=far["mean"] / near["mean"], difference=diff, sd_difference=float(sd), z=float(z), verdict=verdict)


def results_entries(values, decimals=4):
    """The MTF keys of results.json (cycles per green-plane sample; see unit_conversions).

    Parameters
    ----------
    values : dict
        datasheet_values result.
    decimals : int, default 4

    Returns
    -------
    dict
        mtf50_center_cy_per_px, and mtf50_corner_mean_cy_per_px and
        mtf50_corner_worst_cy_per_px when corners were measured; each [value, sd].
    """
    pair = lambda v: [round(v["mean"], decimals), None if v["sd"] is None else round(v["sd"], decimals)]
    out = {"mtf50_center_cy_per_px": pair(values["center"])}
    if "corner_mean" in values:
        out["mtf50_corner_mean_cy_per_px"] = pair(values["corner_mean"])
        out["mtf50_corner_worst_cy_per_px"] = pair(values["corner_worst"])
    return out


# ----------------------------------------------------------------- the measurement

def plot_series(name, frames, out_png):
    """Mean MTF curve per region and direction for one series, with green Nyquist marked.

    Parameters
    ----------
    name : str
        Series name.
    frames : list of list of dict
        analyze_frame rows per frame.
    out_png : str or Path

    Returns
    -------
    Path
        The path written.
    """
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for region in REGIONS:
        for direction, style in zip(DIRECTIONS, ("-", "--")):
            curves = [np.interp(grid, r["freq"], r["mtf"]) for rows in frames for r in rows
                      if r["region"] == region and r["direction"] == direction]
            if curves:
                ax.plot(grid, np.mean(curves, axis=0), style, lw=1.3, label=f"{region}, {direction} ({len(curves)})")
    ax.axvline(0.5, color="0.5", ls=":", lw=1); ax.axhline(0.5, color="0.5", ls=":", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
    ax.set_xlabel("cycles per green-plane sample (green Nyquist 0.5)"); ax.set_ylabel("MTF")
    ax.set_title(f"Series {name}: mean MTF per region (solid vertical edge, dashed horizontal)", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "no edges measured in this series", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def measure_mtf(folder, output_dir, center_series=None, datasheet_pattern=None, two_distance_pair=None,
                pitch_um=None):
    """Slanted-edge MTF over every series in folder; print, plot and summarize.

    Runs the lens_id assertion and series identity over all files, analyzes every frame
    (analyze_frame), summarizes each series (summarize_series), flags clipping and low
    exposure, picks the datasheet values from the series matching datasheet_pattern
    (datasheet_values), and runs the two-distance comparison if two_distance_pair is
    given. Writes mtf_frames.csv (one row per edge per frame) and mtf_<series>.png
    (mean curves) to output_dir.

    Parameters
    ----------
    folder : str or Path
        Slanted-edge captures, one subfolder per series (see series_folders).
    output_dir : str or Path
    center_series : str, optional
        Series for the center value. Default: pick_center_series among the datasheet
        series (the only one, or the only one whose name contains "center").
    datasheet_pattern : str, optional
        Glob pattern of the series allowed to supply datasheet values (center and
        corners), to keep them at one capture distance, e.g. "*_501mm". None: all.
    two_distance_pair : (str, str), optional
        (near, far) center series for the two-distance test, e.g. ("center_250mm",
        "center_501mm"); independent of the datasheet series.
    pitch_um : float, optional
        Raw pixel pitch for unit conversions; default from EXIF (gm.exif_geometry).

    Returns
    -------
    dict
        series ({name: summarize_series}), values (datasheet_values), two_distance
        (or None), pitch_um, units (unit_conversions of the center value).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = series_folders(folder)
    if not groups:
        raise SystemExit(f"no raw files in {folder}")
    files = [p for paths in groups.values() for p in paths]
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(files)} slanted-edge files:")
    meta.assert_one_camera(files)
    ident, _ = meta.series_identity(files)

    summaries, table = {}, []
    for name, paths in groups.items():
        frames = []
        for p in paths:
            plane, span = green_plane(p)
            rows = analyze_frame(plane, span)
            frames.append(rows)
            if not rows:
                print(f"  {p.name}: no target squares found, frame not used (is the whole sheet in "
                      "frame, with paper around every square?)")
            table += [dict(series=name, file=p.name, **{k: r[k] for k in
                           ("region", "direction", "side", "mtf50", "slant_deg", "drift_px", "reliable",
                            "radius", "peak", "paper_level")})
                      for r in rows]
            for r in rows:
                if not r["reliable"]:
                    print(f"  {p.name} {r['region']} {r['direction']} edge: slant {r['slant_deg']:+.2f} deg drifts "
                          f"only {r['drift_px']:.1f} px (< {MIN_EDGE_DRIFT_PX:.0f}); excluded. Rotate the target "
                          "a few degrees more for this edge.")
        summaries[name] = summarize_series(frames)
        plot_series(name, frames, output_dir / f"mtf_{name}.png")
        print(f"\nSeries {name} ({len(paths)} frames), MTF50 in cycles per green-plane sample, mean +/- sd:")
        for region, v in summaries[name].items():
            dirs = ", ".join(f"{d} {s['mean']:.3f}" for d, s in v["by_direction"].items())
            flags = (" CLIPPED" if v["peak"] > CLIPPED_PEAK else "") + \
                    (f" UNDEREXPOSED (paper at {100 * v['paper_level']:.0f}% of span)" if v["paper_level"] < UNDEREXPOSED_PAPER_LEVEL else "")
            sd = f" +/- {v['sd']:.3f}" if v["sd"] is not None else ""
            print(f"  {region:13s} {v['mean']:.3f}{sd} (n {v['n']}, radius {v['radius']:.2f}; {dirs}){flags}")
    pd.DataFrame(table).to_csv(output_dir / "mtf_frames.csv", index=False)

    eligible = datasheet_series(summaries, datasheet_pattern)
    if not eligible:
        raise SystemExit(f"no series match {datasheet_pattern!r}; series are {list(summaries)}")
    if center_series is None:
        center_series = pick_center_series(eligible)
        if center_series is None:
            raise SystemExit(f"cannot tell the center series among {eligible}; name it "
                             "(--mtf-center-series) or narrow --mtf-datasheet-series")
    values = datasheet_values(summaries, center_series, eligible)
    print(f"\nDatasheet series: {', '.join(eligible)} (center: {center_series})")
    if pitch_um is None:
        size = (ident["size_rawpy_visible"][0], ident["size_rawpy_visible"][1])
        pitch_um = gm.exif_geometry(files[0], frame_size=size)["pitch_um"]
    units = unit_conversions(values["center"]["mean"], pitch_um)

    print(f"\nDatasheet MTF50 (cycles per green-plane sample; green Nyquist 0.5):")
    c = values["center"]
    print(f"  center       {c['mean']:.3f} +/- {c['sd'] or 0:.3f}  (series {c['series']}, n {c['n']}) = "
          f"{units['cy_per_raw_px']:.3f} cy/raw px = {units['cy_per_mm']:.0f} cy/mm on the sensor "
          f"(pitch {pitch_um:.3f} um); {100 * units['frac_of_green_nyquist']:.0f}% of green Nyquist")
    for corner, v in values["corners"].items():
        sys_txt = (f"; within {v['sd_within'] or 0:.3f}, between series {v['sd_sys']:.3f} "
                   f"from {', '.join(v['repeats'])}" if v["sd_sys"] is not None else "; no repeat at this radius")
        print(f"  {corner:12s} {v['mean']:.3f} +/- {v['sd'] or 0:.3f}  (series {v['series']}, "
              f"radius {v['radius']:.2f}{sys_txt})")
    if "corner_mean" in values:
        cm_, w = values["corner_mean"], values["corner_worst"]
        print(f"  corner mean  {cm_['mean']:.3f} +/- {cm_['sd'] or 0:.3f}; worst {w['corner']} {w['mean']:.3f}; "
              f"corners average {100 * (cm_['mean'] / c['mean'] - 1):+.0f}% vs center")
    if values["missing"]:
        print(f"  not measured: {', '.join(values['missing'])}")

    td = None
    if two_distance_pair:
        near_name, far_name = two_distance_pair
        missing = [n for n in two_distance_pair if n not in summaries or "center" not in summaries[n]]
        if missing:
            print(f"\nTwo-distance test skipped: no center measurement in {missing}")
        else:
            near, far = summaries[near_name]["center"], summaries[far_name]["center"]
            td = two_distance(near, far)
            print(f"\nTwo-distance test: center {near['mean']:.3f} ({near_name}) vs {far['mean']:.3f} "
                  f"({far_name}); difference {td['difference']:+.3f} +/- {td['sd_difference']:.3f} "
                  f"(z {td['z']:+.1f}): {td['verdict']}")
    return dict(series=summaries, values=values, two_distance=td, pitch_um=pitch_um, units=units)
