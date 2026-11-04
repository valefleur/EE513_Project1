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
    # print(f"Shape of cam_balanced: {cam_balanced.shape}, shape of color correcting matrix: {M.shape}.")
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


# ----------------------------------------------------------------------------- Problem 8
def plot_de_before_after(rows, names, title, save_path):
    """rows: list of (chart label, dE before, dE after, usable). One panel per chart,
    before and after bars per patch; clipped patches are left out and marked."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.6 * len(rows) + 0.8),
                             sharex=True, squeeze=False)
    x = np.arange(24)
    for ax, (label, before, after, usable) in zip(axes[:, 0], rows):
        b = np.where(usable, before, np.nan)
        a = np.where(usable, after, np.nan)
        ax.bar(x - 0.2, b, 0.4, color="#999999", label="before (no matrix, sRGB primaries)")
        ax.bar(x + 0.2, a, 0.4, color="#0072B2", label="after (fitted 3x3 matrix)")
        for p in np.nonzero(~usable)[0]:
            ax.text(p, 0.3, "clipped", rotation=90, ha="center", va="bottom", fontsize=7, color="#D55E00")
        ax.set_ylabel("$\\Delta E_{00}$")
        ax.set_title(f"{label}: mean {np.nanmean(b):.2f} -> {np.nanmean(a):.2f}, "
                     f"worst {np.nanmax(b):.2f} -> {np.nanmax(a):.2f}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([f"{k + 1} {n}" for k, n in enumerate(names)], rotation=60,
                                ha="right", fontsize=8)
    fig.suptitle(title + " (metric: CIEDE2000)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- Problem 9
def column_comparison(matrices):
    """Compare matrices column by column as DIRECTIONS.

    matrices: {label: 3x3 M}. Column j of M is the XYZ direction camera channel j maps to.
    For each pair of matrices and each column: the angle between the two columns in
    degrees, and the ratio of their lengths. Neither blows up when an element is near
    zero, unlike element-by-element relative differences.
    """
    import itertools
    import pandas as pd
    rows = []
    for (la, A), (lb, B) in itertools.combinations(matrices.items(), 2):
        row = {"pair": f"{la} vs {lb}"}
        for j, ch in enumerate("RGB"):
            a, b = A[:, j], B[:, j]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            row[f"angle_{ch}"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            row[f"length_{ch}"] = float(np.linalg.norm(b) / np.linalg.norm(a))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Part 4
def plot_transfer_per_patch(series, names, title, save_path):
    """series: list of (label, per-patch mean dE over charts, NaN = clipped everywhere).
    Grouped bars, one group per patch, one bar per illuminant/matrix combination."""
    import matplotlib.pyplot as plt
    colors = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(24)
    w = 0.8 / len(series)
    for i, (label, de) in enumerate(series):
        ax.bar(x - 0.4 + w * (i + 0.5), de, w, color=colors[i % len(colors)],
               label=f"{label} (mean {np.nanmean(de):.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k + 1} {n}" for k, n in enumerate(names)], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta E_{00}$ (mean over charts)")
    ax.set_title(title + " (metric: CIEDE2000; missing bar = clipped)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def transfer_view_figure(panels, ref_XYZ, illuminant_xy, title, save_path):
    """Charts side by side, each rendered through its own matrix.

    panels: list of (panel title, scaled camera RGB 24x3, M, usable, dE 24).
    Each patch: left half = camera RGB through M, right half = published reference,
    with that patch's CIEDE2000 printed on it. Clipped patches are crossed out.
    """
    import matplotlib.pyplot as plt
    ref_disp = _xyz_d50_to_srgb8(ref_XYZ, illuminant_xy)
    s, gap = 46, 8
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4))
    for ax, (name, cam, M, usable, de) in zip(np.atleast_1d(axes), panels):
        disp = _xyz_d50_to_srgb8(np.asarray(cam) @ np.asarray(M).T, illuminant_xy)
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
            label = f"{p + 1}: {de[p]:.1f}" if usable[p] else f"{p + 1}: clip"
            ax.text(x + 2, y + s - 4, label, fontsize=6, color="white",
                    bbox=dict(fc="black", alpha=0.55, lw=0, pad=0.6))
            if not usable[p]:
                ax.plot([x, x + s], [y, y + s], "w-", [x, x + s], [y + s, y], "w-", lw=0.8)
        ax.set_title(f"{name}\nmean $\\Delta E_{{00}}$ {np.mean(de[usable]):.2f}, "
                     f"worst {np.max(de[usable]):.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}   (each patch: left = corrected camera, right = reference; "
                 f"number = CIEDE2000)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
