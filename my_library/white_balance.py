# white_balance.py
"""HW4 Part 2: ColorChecker reference data and white balance gains.

Von Kries gains (one multiplier per channel) follow Prof. James McNames,
06_color_matrix.ipynb section 2, but are set from an unclipped mid gray instead of white.
"""
import numpy as np
import colour

REFERENCE = "ColorChecker24 - After November 2014"
# 0-based patch indices, in order of preference: 22 neutral 5, 21 neutral 6.5,
# 23 neutral 3.5, 20 neutral 8. Mid grays first: far from clipping and from noise.
GRAY_PREFERENCE = (21, 20, 22, 19)
NEUTRAL_IDX = list(range(18, 24))       # patches 19..24


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
    """First usable patch in `preference` (0-based index), or None."""
    return next((k for k in preference if usable[k]), None)


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
