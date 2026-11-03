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


# ----------------------------------------------------------------------------- Problem 7
def scale_to_reference(cam_balanced, ref_XYZ, usable):
    """Exposure only: scale camera RGB so the green of the reference-white patch k equals
    that patch's reference Y. Then a matrix that preserves neutrals has a Y row summing
    to about 1, and its X and Z rows sum to the reference white's X/Y and Z/Y."""
    k = pick_white(usable)
    return np.asarray(cam_balanced, float) * (ref_XYZ[k, 1] / cam_balanced[k, 1]), k


def fit_matrix(cam, ref_XYZ, usable):
    """Least squares 3x3 M with ref_XYZ ~= cam @ M.T, over the usable patches only.
    Each column of lstsq's solution is one output (X, Y or Z), so transpose to get M."""
    sol, *_ = np.linalg.lstsq(np.asarray(cam)[usable], np.asarray(ref_XYZ)[usable], rcond=None)
    return sol.T


def white_xyz_y1(ref_XYZ):
    """Reference white patch (19) XYZ normalized to Y = 1: what M @ [1, 1, 1] should be."""
    return ref_XYZ[18] / ref_XYZ[18, 1]


def matrix_latex(M, row_sums, target, caption):
    """M as an equation with its row sums and the neutral-preserving target beside it."""
    rows = " \\\\\n".join(" & ".join(f"{v:.4f}" for v in r) for r in M)
    sums = " \\\\\n".join(f"{v:.4f}" for v in row_sums)
    targ = " \\\\\n".join(f"{v:.4f}" for v in target)
    return ("% " + caption + "\n\\begin{equation*}\nM = \\begin{bmatrix}\n" + rows +
            "\n\\end{bmatrix},\\quad\nM\\begin{bmatrix}1\\\\1\\\\1\\end{bmatrix} = "
            "\\begin{bmatrix}\n" + sums + "\n\\end{bmatrix}\\quad\\text{vs. white }"
            "\\begin{bmatrix}\n" + targ + "\n\\end{bmatrix}\n\\end{equation*}\n")


# ----------------------------------------------------------------------------- comparison figure
def _xyz_d50_to_srgb8(XYZ, illuminant_xy):
    """Reference-style XYZ (D50) -> display sRGB, adapted to D65 (Bradford), 0..1."""
    rgb = colour.XYZ_to_sRGB(np.asarray(XYZ, float), illuminant=illuminant_xy,
                             chromatic_adaptation_transform="Bradford")
    return np.clip(rgb, 0, 1)


def _linear_to_display(rgb_linear):
    return np.clip(colour.cctf_encoding(np.clip(rgb_linear, 0, 1), function="sRGB"), 0, 1)


def comparison_figure(cam_raw, cam_balanced, M, ref_XYZ, illuminant_xy, usable, title, save_path):
    """Three charts side by side: camera RGB, white balanced (= naive sRGB), fitted matrix.
    Each patch: left half = pipeline, right half = published reference. Clipped patches: x.
    Every panel is scaled so its reference-white patch matches the reference's level."""
    import matplotlib.pyplot as plt
    k = pick_white(usable)
    ref_disp = _xyz_d50_to_srgb8(ref_XYZ, illuminant_xy)
    raw_s, _ = scale_to_reference(cam_raw, ref_XYZ, usable)
    bal_s, _ = scale_to_reference(cam_balanced, ref_XYZ, usable)
    panels = [("Camera RGB, no white balance", _linear_to_display(raw_s)),
              ("White balanced (= naive sRGB, Problems 5-6)", _linear_to_display(bal_s)),
              ("Fitted 3x3 matrix (Problem 7)", _xyz_d50_to_srgb8(bal_s @ M.T, illuminant_xy))]
    s, gap = 40, 8
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (name, disp) in zip(axes, panels):
        img = np.ones((4 * (s + gap) + gap, 6 * (s + gap) + gap, 3)) * 0.12
        for p in range(24):
            r, c = divmod(p, 6)
            y, x = gap + r * (s + gap), gap + c * (s + gap)
            img[y:y + s, x:x + s // 2] = disp[p]
            img[y:y + s, x + s // 2:x + s] = ref_disp[p]
        ax.imshow(img)
        for p in range(24):
            r, c = divmod(p, 6)
            y, x = gap + r * (s + gap), gap + c * (s + gap)
            ax.text(x + 2, y + 9, str(p + 1), fontsize=6, color="white",
                    bbox=dict(fc="black", alpha=0.5, lw=0, pad=0.5))
            if not usable[p]:
                ax.plot([x, x + s], [y, y + s], "w-", [x, x + s], [y + s, y], "w-", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title}   (each patch: left = pipeline, right = reference; "
                 f"x = clipped; scaled to patch {k + 1})", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
