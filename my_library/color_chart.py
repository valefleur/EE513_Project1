# color_chart.py
"""
Locate the 24 patches of a ColorChecker Classic and measure them in linear light.

Credits
-------
- Middle-half sampling rule (ph//4 .. 3*ph//4 of each patch in each direction) follows
  Prof. James McNames, Lecture 7 demo, 07_build_an_isp.ipynb, section 4 "Now reorder it".
- Patch order and names (index 18 is white) are Prof. McNames' psu_imaging.COLORCHECKER_NAMES,
  the same order as `cam` in his Lecture 6 demo, 06_color_matrix.ipynb, section 2.
- Raw -> linear light -> quarter-res RGB reuses my_library.linear_light (HW4 Problem 2).
- Patch-size estimate uses the edge-spread idea from Prof. McNames' Lecture 4 demo,
  04_measure_your_camera.ipynb.
- Detector wrapper, homography, patch-size estimate and checks written with Claude.

Works with OpenCV 4.x contrib and OpenCV 5.x (the mcc API changed between them).
"""
import numpy as np
import cv2 as cv
import rawpy
import matplotlib.pyplot as plt

import psu_imaging as psu
import my_library.linear_light as ll

N_ROWS, N_COLS = 4, 6
# Chart coordinates: center of patch k (0-based, row-major from "dark skin") is (col, row),
# so neighbouring centers are exactly one unit (one "pitch") apart.
CANON_CENTERS = np.float32([[c, r] for r in range(N_ROWS) for c in range(N_COLS)])
CORNER_IDX = [0, 5, 23, 18]     # patches 1, 6, 24, 19: dark skin, bluish green, black, white
WHITE_IDX, BLACK_IDX = 18, 23
NEUTRAL_IDX = list(range(18, 24))   # patches 19..24, white to black

# A patch whose mean is this close to saturation in any channel is treated as clipped
# even if no site reached saturation_dn exactly: a sensor's response can bend just
# below saturation, and noise means some sites in the patch are at the ceiling.
NEAR_CEILING = 0.95

# Display color for clipped tiles in previews. Saturated pure blue: no chart patch or
# ordinary highlight looks like it, and it cannot be mistaken for a white-balance cast.
CLIP_COLOR = (0, 0, 255)

# Linear reflectance of the chart patches (relative), from Prof. McNames' published
# sRGB values. Only used to scale previews so that patch 19 lands at display white.
_REF_LINEAR = psu.srgb_to_linear(np.asarray(psu.COLORCHECKER_SRGB, float))


# ----------------------------------------------------------------------------- loading
def load_chart_quarter_rgb(path, saturation_dn, black_level=None):
    """Raw file -> (quarter-res linear RGB, quarter-res clip mask). Values are NOT
    clipped: slightly negative pixels near black are kept for unbiased averages.

    black_level: DN to subtract. None (default) uses the file's declared black level.

    Linear RGB comes from ll.raw_to_linear_light + ll.quarter_rgb (Problem 2).
    The clip mask is True where ANY of the four CFA sites of a 2x2 tile reached
    saturation_dn. It has to be computed from the raw DN: after averaging the two
    greens, a tile with one clipped green no longer reads 1.0.
    """
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible
        if black_level is None:
            blacks = set(raw.black_level_per_channel)
            assert len(blacks) == 1, f"per-channel black levels differ: {blacks}"
            black_level = blacks.pop()
        # clip=False: keep the noise below black so dark patch means are unbiased.
        lin = ll.raw_to_linear_light(mosaic, black_level, saturation_dn, clip=False)
        rgb = ll.quarter_rgb(lin, raw.raw_pattern, raw.color_desc)
        h, w = rgb.shape[:2]
        hit = mosaic >= saturation_dn
        clipped = (hit[0:2*h:2, 0:2*w:2] | hit[0:2*h:2, 1:2*w:2] |
                   hit[1:2*h:2, 0:2*w:2] | hit[1:2*h:2, 1:2*w:2])
    return rgb, clipped


def preview_8bit(rgb, srgb=False, gains=None, white_level=None, clipped=None,
                 clip_color=CLIP_COLOR, pct=99.5):
    """8-bit RGB image for display or for the detector. Nothing is measured on it.

    Balance and exposure:
      gains, white_level given: multiply by per-channel gains (white balance), then
          divide by white_level so that level becomes display white. measure_chart()
          gets both from the chart's own neutral patches; see neutral_balance().
      otherwise: rough automatic version. Each channel is divided by its median and
          the `pct` percentile becomes display white. Fine for the detector, but it
          gives color casts when the frame is not mostly neutral or anything clipped.
    srgb=False stays linear (midtones look dark); srgb=True sRGB-encodes for a screen.
    clipped: optional boolean mask (same height and width). Those pixels are painted
    clip_color, so clipping is visible instead of showing up as a false color.
    """
    x = np.clip(rgb, 0, None)
    if gains is not None and white_level is not None:
        x = x * np.asarray(gains, float) / white_level
    else:
        x = x / np.maximum(np.median(x, axis=(0, 1)), 1e-9)
        x = x / np.percentile(x, pct)
    x = np.clip(x, 0, 1)
    if srgb:
        x = psu.linear_to_srgb(x)
    out = (255 * x + 0.5).astype(np.uint8)                              # RGB order
    if clipped is not None:
        out[clipped] = clip_color
    return out


# ----------------------------------------------------------------------------- locating
def detect_centers(preview_rgb):
    """24 patch centers (x, y) in standard order from OpenCV's mcc detector, or None.

    The detector identifies patches by color, so a chart that is rotated in the
    frame still comes back in standard order (tested at 90, 180 and 270 degrees).
    """
    bgr = cv.cvtColor(preview_rgb, cv.COLOR_RGB2BGR)
    det = cv.mcc.CCheckerDetector_create()
    if cv.__version__.startswith("4."):
        found = det.process(bgr, cv.mcc.MCC24, 1)          # 4.x: chart type per call
    else:
        det.setColorChartType(cv.mcc.MCC24)                 # 5.x: chart type is a setting
        found = det.process(bgr)
    if not found:
        return None
    quads = np.array(det.getBestColorChecker().getColorCharts()).reshape(24, 4, 2)
    return quads.mean(axis=1).astype(np.float32)


def homography_from_centers(img_centers, canon_centers=CANON_CENTERS):
    """Least-squares homography, chart coordinates -> image pixels (4 or more points)."""
    H, _ = cv.findHomography(np.float32(canon_centers), np.float32(img_centers), 0)
    return H


def homography_from_clicks(four_xy):
    """Fallback when the detector fails. four_xy: image (x, y) of the CENTERS of
    patch 1 (dark skin), 6 (bluish green), 24 (black), 19 (white), in that order,
    in quarter-res pixel coordinates. Identify them by color, not by position."""
    return homography_from_centers(four_xy, CANON_CENTERS[CORNER_IDX])


def estimate_patch_halfside(rgb, H, px_per_pitch=200):
    """Patch half-width in pitch units, measured from where the patch edges are.

    The chart is straightened, then from every patch center the brightness profile
    is followed outward in four directions. Each profile is an edge spread function
    (Prof. McNames, 04_measure_your_camera.ipynb): flat on the patch, a blurred step,
    flat on the black separator. The edge is where it crosses halfway between the
    patch level and the separator level. The median over all usable profiles is the
    answer; patches too close to the separator's brightness (black, dark neutrals,
    dim captures) are skipped because their crossing is mostly noise.
    """
    p = px_per_pitch
    S = np.array([[p, 0, p / 2], [0, p, p / 2], [0, 0, 1]], float)
    rect = cv.warpPerspective(np.asarray(rgb, np.float32).sum(axis=2), H @ np.linalg.inv(S),
                              (N_COLS * p, N_ROWS * p),
                              flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    band, half = max(1, p // 20), p // 2
    offs = (np.arange(half) + 0.5) / p                 # 0 .. 0.5 pitch from the center
    used, crossings = [], []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cy, cx = r * p + half, c * p + half
            rows = rect[cy - band:cy + band]
            cols = rect[:, cx - band:cx + band]
            for prof in (rows[:, cx:cx + half].mean(axis=0),            # right
                         rows[:, cx - half:cx][:, ::-1].mean(axis=0),   # left
                         cols[cy:cy + half].mean(axis=1),               # down
                         cols[cy - half:cy][::-1].mean(axis=1)):        # up
                level = np.median(prof[:int(0.15 * p)])                 # on the patch
                sep = np.median(prof[int(0.46 * p):])                   # on the separator
                if level < 3 * max(sep, 0) + 1e-3:                      # too little contrast
                    continue
                norm = (prof - sep) / (level - sep)                     # 1 on patch, 0 on separator
                below = np.nonzero(norm < 0.5)[0]
                if below.size == 0 or below[0] == 0:
                    continue
                k = below[0]                                            # interpolate the crossing
                t = (norm[k - 1] - 0.5) / (norm[k - 1] - norm[k])
                crossings.append(offs[k - 1] + t * (offs[k] - offs[k - 1]))
                used.append(norm)
    if len(crossings) < 8:
        raise RuntimeError(f"only {len(crossings)} usable patch edges; is the chart in view?")
    return float(np.median(crossings)), offs, np.mean(used, axis=0), np.array(crossings)


# ----------------------------------------------------------------------------- sampling
def sampling_polygons(H, halfside, fraction=0.5):
    """Image-space quads covering the middle `fraction` of each patch in each direction."""
    h = halfside * fraction
    sq = np.float32([[-h, -h], [h, -h], [h, h], [-h, h]])
    canon = (CANON_CENTERS[:, None, :] + sq[None]).reshape(-1, 1, 2)
    return cv.perspectiveTransform(canon, H).reshape(24, 4, 2)


def sample_patches(rgb, polys, clipped):
    """Per patch: mean and std of linear RGB, std after sRGB encoding to 8 bits,
    pixel count, clipped-tile count."""
    means, stds, stds_srgb8, counts, nclip = [], [], [], [], []
    for q in polys:
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv.fillPoly(mask, [np.round(q).astype(np.int32)], 1)
        m = mask.astype(bool)
        px = rgb[m]
        means.append(px.mean(axis=0))
        stds.append(px.std(axis=0, ddof=1))
        # Each pixel sRGB-encoded to 0..255 first, then the spread. linear_to_srgb clips
        # to [0, 1], which is right here: this mimics what an 8-bit file would hold.
        stds_srgb8.append((255 * psu.linear_to_srgb(px)).std(axis=0, ddof=1))
        counts.append(int(m.sum()))
        nclip.append(int(clipped[m].sum()))
    return (np.array(means), np.array(stds), np.array(stds_srgb8),
            np.array(counts), np.array(nclip))


def usable_patches(means, nclip, ceiling=NEAR_CEILING):
    """True for patches whose means are valid linear measurements: no clipped tiles
    and no channel mean at or above `ceiling`."""
    return (np.asarray(nclip) == 0) & (np.asarray(means).max(axis=1) < ceiling)


def neutral_balance(means, usable):
    """White-balance gains from the brightest usable neutral patch (19..23).

    Von Kries scaling as in Prof. McNames' 06_color_matrix.ipynb section 2,
    gains = cam[18].max() / cam[18], but using the brightest gray patch that did not
    clip instead of always the white one. Black (24) is never used: it is so close to
    zero that flare and noise dominate its color.
    Returns (gains, patch index, estimated level of patch 19 after the gains).
    The last value is extrapolated through the chart's published reflectances, so it
    exists even when patch 19 itself clipped. It is for setting preview brightness
    only: on this camera's unclipped LED charts it came out about 15% high (flare
    and the published values not matching this sensor's green channel exactly).
    """
    for k in NEUTRAL_IDX[:-1]:
        if usable[k]:
            gray = means[k]
            gains = gray.max() / gray
            white_level = gray.max() * _REF_LINEAR[WHITE_IDX, 1] / _REF_LINEAR[k, 1]
            return gains, k, float(white_level)
    return None, None, None


def sanity_checks(means, stds, counts, nclip, ceiling=NEAR_CEILING):
    """Catch wrong numbering, a box touching an edge, and clipping.

    The numbering checks use only usable (unclipped) patches, because clipping
    forces bright patches to the same value: a clipped white and a clipped yellow
    tie, and either can come out "brightest".
      1. The darkest patch is #24 (black). Black never clips, so this always works.
      2. Usable neutrals darken in order from 19 to 24.
      3. Usable neutrals 19..23 share one color: the same R/G and B/G within 15%.
         A misnumbered chart would put colored patches in these slots.
      4. If #19 is usable, it is the brightest usable patch.
    """
    msgs, g = [], means[:, 1]
    usable = usable_patches(means, nclip, ceiling)
    if np.argmin(g) != BLACK_IDX:
        msgs.append(f"WARNING darkest patch is #{np.argmin(g) + 1}, expected #24 (black): "
                    "numbering is probably wrong")
    neut = [k for k in NEUTRAL_IDX if usable[k]]
    if not np.all(np.diff(g[neut]) < 0):
        msgs.append(f"WARNING usable neutrals {[k + 1 for k in neut]} do not darken in order")
    grays = [k for k in neut if k != BLACK_IDX]
    if len(grays) >= 2:
        ratios = means[grays][:, [0, 2]] / means[grays][:, [1]]
        spread = np.abs(ratios / np.median(ratios, axis=0) - 1).max()
        if spread > 0.15:
            msgs.append(f"WARNING neutrals {[k + 1 for k in grays]} differ in color by up to "
                        f"{100 * spread:.0f}%: numbering may be wrong")
    if usable[WHITE_IDX] and np.argmax(np.where(usable, g, -np.inf)) != WHITE_IDX:
        msgs.append("WARNING #19 (white) is unclipped but is not the brightest usable patch")
    cv_g = stds[:WHITE_IDX + 1, 1] / np.maximum(means[:WHITE_IDX + 1, 1], 1e-9)
    worst = int(np.argmax(cv_g))
    msgs.append(f"largest G std/mean (patches 1-19): #{worst + 1} = {cv_g[worst]:.3f}, "
                f"median {np.median(cv_g):.3f}")
    clipped = [int(k) + 1 for k in np.nonzero(np.asarray(nclip) > 0)[0]]
    near = [int(k) + 1 for k in np.nonzero(~usable & (np.asarray(nclip) == 0))[0]]
    if clipped:
        msgs.append(f"WARNING clipped tiles in patches {clipped}")
    if near:
        msgs.append(f"WARNING mean at or above {ceiling} of saturation in patches {near}")
    if clipped or near:
        msgs.append(f"usable patches: {int(usable.sum())} of 24 "
                    "(exclude the others from everything that follows)")
    msgs.append(f"pixels per patch: {counts.min()} to {counts.max()}")
    return msgs


# ----------------------------------------------------------------------------- one chart
def measure_chart(path, saturation_dn, clicks=None, black_level=None):
    """Everything for one capture. Returns a dict, or None if the chart was not found.
    black_level: see load_chart_quarter_rgb (None = the file's declared value)."""
    rgb, clipped = load_chart_quarter_rgb(path, saturation_dn, black_level)
    if clicks is not None:
        H, how = homography_from_clicks(clicks), "manual clicks"
    else:
        # The detector gets the rough automatic sRGB preview, like an ordinary photo.
        centers = detect_centers(preview_8bit(rgb, srgb=True))
        if centers is None:
            return None
        H, how = homography_from_centers(centers), "mcc detector"
    halfside, offs, profile, crossings = estimate_patch_halfside(rgb, H)
    polys = sampling_polygons(H, halfside, fraction=0.5)
    means, stds, stds_srgb8, counts, nclip = sample_patches(rgb, polys, clipped)
    usable = usable_patches(means, nclip)
    gains, wb_idx, white_level = neutral_balance(means, usable)
    # Display previews, balanced on the chart's own gray and with clipping painted.
    preview = {flag: preview_8bit(rgb, srgb=flag, gains=gains, white_level=white_level,
                                  clipped=clipped)
               for flag in (False, True)}
    return dict(path=path, preview=preview, stds_srgb8=stds_srgb8, H=H, how=how,
                halfside=halfside, edge_offsets=offs, edge_profile=profile,
                edge_crossings=crossings, polys=polys, means=means, stds=stds,
                counts=counts, nclip=nclip, usable=usable, wb_gains=gains,
                wb_patch=None if wb_idx is None else wb_idx + 1,
                n_clipped_frame=int(clipped.sum()),
                checks=sanity_checks(means, stds, counts, nclip))


# ----------------------------------------------------------------------------- outputs
def plot_regions(result, title, save_path=None, SHOW=False, srgb=False):
    """Deliverable figure: the sampled regions drawn on the chart.
    srgb=False shows the chart in linear light; srgb=True sRGB-encodes it.
    Solid outline: usable patch. Dashed outline: clipped or near the ceiling."""
    polys = result["polys"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(result["preview"][srgb])
    for k, q in enumerate(polys):
        qq = np.vstack([q, q[:1]])
        style = "-" if result["usable"][k] else "--"               # dashed: excluded patch
        ax.plot(qq[:, 0], qq[:, 1], style, color="black", lw=2.6)  # under-stroke: visible on yellow
        ax.plot(qq[:, 0], qq[:, 1], style, color="yellow", lw=1.2)
        ax.text(*q.mean(axis=0), str(k + 1), color="white", fontsize=7, ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    pts = polys.reshape(-1, 2)
    pad = 0.15 * np.ptp(pts, axis=0).max()
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].max() + pad, pts[:, 1].min() - pad)
    balance = (f"balanced on #{result['wb_patch']}" if result["wb_patch"]
               else "auto balance (no usable gray)")
    ax.set_title(f"{title}\nmiddle half of each patch ({result['how']}, "
                 f"patch half-width {result['halfside']:.3f} pitch)\n"
                 f"{'sRGB-encoded' if srgb else 'linear light'} preview, {balance}, "
                 f"clipped tiles in blue ({result['n_clipped_frame']} in frame)", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if SHOW:
        plt.show()
    plt.close(fig)


def plot_edge_profile(result, save_path=None, SHOW=False):
    """Diagnostic: the average patch-to-separator edge and the half-width found from it."""
    hs = result["halfside"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
    ax[0].plot(result["edge_offsets"], result["edge_profile"])
    ax[0].axhline(0.5, ls=":", color="gray")
    ax[0].axvline(hs, ls=":", color="k", label=f"patch edge {hs:.3f}")
    ax[0].axvspan(0, hs / 2, alpha=0.15, label="sampled (middle half)")
    ax[0].set_xlabel("distance from patch center (pitches)")
    ax[0].set_ylabel("normalized brightness")
    ax[0].legend(fontsize=8)
    ax[1].hist(result["edge_crossings"], bins=20)
    ax[1].axvline(hs, ls=":", color="k")
    ax[1].set_xlabel("edge position, each patch side (pitches)")
    ax[1].set_title(f"{len(result['edge_crossings'])} edges used", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if SHOW:
        plt.show()
    plt.close(fig)


def patch_table(result, **labels):
    """Long-format DataFrame, one row per patch.

    Columns: R, G, B and std_* are linear light (0 = black, 1 = saturation).
    std_*_lin8 is the same linear std times 255. std_*_srgb8 is the spread of the
    patch's pixels after sRGB encoding to 0..255, as an 8-bit image file would hold them. Extra keyword args (round=..., chart=...)
    become leading columns so many charts can be concatenated."""
    import pandas as pd
    m, s, s8 = result["means"], result["stds"], result["stds_srgb8"]
    cols = {"patch": np.arange(1, 25), "name": psu.COLORCHECKER_NAMES}
    for j, c in enumerate("RGB"):
        cols[c] = m[:, j]                              # mean, linear light (the deliverable)
    for j, c in enumerate("RGB"):
        cols[f"std_{c}"] = s[:, j]                     # std, linear light
    for j, c in enumerate("RGB"):
        cols[f"std_{c}_lin8"] = 255 * s[:, j]          # std, linear light in 8-bit levels
    for j, c in enumerate("RGB"):
        cols[f"std_{c}_srgb8"] = s8[:, j]              # std after sRGB encoding, 8-bit levels
    cols["n_px"], cols["n_clipped"] = result["counts"], result["nclip"]
    cols["usable"] = result["usable"]                  # False: clipped or >= NEAR_CEILING
    df = pd.DataFrame(cols)
    for k, v in reversed(list(labels.items())):
        df.insert(0, k, v)
    return df
