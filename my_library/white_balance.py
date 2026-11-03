# white_balance.py
"""HW4 Part 2: ColorChecker reference data and white balance gains.

Von Kries gains (one multiplier per channel) follow Prof. James McNames,
06_color_matrix.ipynb section 2, but are set from an unclipped mid gray instead of white.
"""
import numpy as np
import colour

REFERENCE = "ColorChecker24 - After November 2014"
# Preferences are PATCH NUMBERS (1-based, as printed on the chart), not array indices.
# Gains: mid grays first, far from both clipping and noise: 22 neutral 5, 21 neutral 6.5,
# 23 neutral 3.5, 20 neutral 8.
GRAY_PREFERENCE = (22, 21, 23, 20)
NEUTRAL_IDX = list(range(18, 24))       # 0-based indices of patches 19..24


def load_reference():
    """Patch names, XYZ (24 x 3, Y of white about 0.9) and the reference illuminant xy."""
    cc = colour.CCS_COLOURCHECKERS[REFERENCE]
    names = list(cc.data.keys())
    xyY = np.array(list(cc.data.values()), float)
    return names, colour.xyY_to_XYZ(xyY), np.asarray(cc.illuminant, float)


def check_numbering(ref_names, our_names):
    """Our patch k must be the reference's patch k. Reference names carry a density
    suffix on the neutrals ("white 9.5 (.05 D)"), so compare by prefix."""
    bad = [(k + 1, o, r) for k, (o, r) in enumerate(zip(our_names, ref_names))
           if not r.lower().startswith(o.lower().split(" (")[0])]
    assert len(ref_names) == len(our_names) == 24 and not bad, f"numbering mismatch: {bad}"
    return f"patch 1 = {ref_names[0]}, patch 19 = {ref_names[18]}, all 24 names match"


def pick_gray(usable, preference=GRAY_PREFERENCE):
    """First usable patch in `preference` (patch numbers). Returns its 0-based index, or None."""
    return next((p - 1 for p in preference if usable[p - 1]), None)


def gains_from_gray(gray_rgb):
    """Gains that make this gray R = G = B, normalized so the green gain is 1."""
    gray_rgb = np.asarray(gray_rgb, float)
    return gray_rgb[1] / gray_rgb


def neutral_residual(balanced, usable):
    """Largest |R/G - 1| and |B/G - 1| over usable neutrals 19..23 after balancing
    (black is left out: too close to zero for a ratio to mean much)."""
    ks = [k for k in NEUTRAL_IDX[:-1] if usable[k]]
    if not ks:
        return np.nan, []
    r = balanced[ks][:, [0, 2]] / balanced[ks][:, [1]]
    return float(np.abs(r - 1).max()), [k + 1 for k in ks]


# ----------------------------------------------------------------------------- Problem 6+
# Linear sRGB (no transfer curve) -> CIE XYZ, from the sRGB primaries and D65 white.
SRGB_TO_XYZ = np.asarray(colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ, float)
# CIELAB reference white: patch 19 (white), then the brightest neutrals 20..23.
WHITE_PREFERENCE = (19, 20, 21, 22, 23)


def pick_white(usable, preference=WHITE_PREFERENCE):
    """Patch 19 if usable, otherwise the brightest usable neutral (patch numbers in
    `preference`). Returns its 0-based index, or None."""
    return next((p - 1 for p in preference if usable[p - 1]), None)


def lab_with_white(XYZ, white_XYZ):
    """CIELAB using `white_XYZ` as the reference white (any scale)."""
    w = np.asarray(white_XYZ, float)
    return colour.XYZ_to_Lab(np.asarray(XYZ, float) / w[1], illuminant=colour.XYZ_to_xy(w))


def delta_e_vs_reference(cam_balanced, ref_XYZ, usable, M=SRGB_TO_XYZ):
    """CIEDE2000 between measured patches (camera RGB -> XYZ through M) and the reference.

    Both sides are converted to CIELAB with the SAME neutral patch k as white: patch 19
    if usable, else the brightest usable neutral. That neutral is scaled to white's level
    by the reference's own ratio Y_19 / Y_k, so L* keeps its usual 0..100 range; for
    k = 19 the scale is 1. Returns (dE for all 24 patches, k).
    """
    print(f"Shape of cam_balanced: {cam_balanced.shape}, shape of color correcting matrix: {M.shape}.")
    k = pick_white(usable)

    # Multiply by M, the standard sRGB to XYZ matrix.
    XYZ_meas = np.asarray(cam_balanced, float) @ np.asarray(M, float).T
    scale = ref_XYZ[18, 1] / ref_XYZ[k, 1]
    lab_meas = lab_with_white(XYZ_meas, XYZ_meas[k] * scale)
    lab_ref = lab_with_white(ref_XYZ, ref_XYZ[k] * scale)
    return np.asarray(colour.delta_E(lab_meas, lab_ref, method="CIE 2000")), k
