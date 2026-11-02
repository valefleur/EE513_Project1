# shading.py

import warnings
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import psu_imaging as psu

# TODO have a "css"-like module for colors, line styles, etc. to keep them consistent across all plots
# Or, a winter break project could be to replot everything in Seaborn... Been meaning to learn that.
CHANNEL_COLORS = ("#D7263D", "#00A676", "#003B73")   # R, G, B (analyze.COLORS palette)

def center_mean(img, frac=0.10):
    """Per-channel mean of a centered box covering `frac` of each dimension."""
    h, w = img.shape[:2]
    dh, dw = max(1, int(h * frac / 2)), max(1, int(w * frac / 2))
    cy, cx = h // 2, w // 2
    return img[cy - dh:cy + dh, cx - dw:cx + dw].mean(axis=(0, 1))


def corner_mean(img, frac=0.10, all_corners=True):
    """Per-channel mean of corner box(es), each `frac` of each dimension."""
    h, w = img.shape[:2]
    dh, dw = max(1, int(h * frac)), max(1, int(w * frac))
    boxes = [img[:dh, :dw], img[:dh, -dw:], img[-dh:, :dw], img[-dh:, -dw:]]
    if not all_corners:
        boxes = boxes[:1]
    return np.mean([b.mean(axis=(0, 1)) for b in boxes], axis=0)


def shading_map_from_white(white_rgb, center_frac=0.10, smooth_sigma=0, max_center=0.98):
    """White sheet divided by its own per-channel center resulting in 1.0 at center, <1 toward the
    corners.  Measures lens AND lamp falloff together; don't assume which until profiles are
    compared.  Optional smoothing keeps sensor noise in the flat from being copied into every
    image divided by it."""
    c = center_mean(white_rgb, center_frac)
    if (c > max_center).any():
        warnings.warn(f"white-sheet center is near the ceiling {np.round(c, 3)}; "
                      "the center is pinned and the map will look too flat")
    m = white_rgb / c
    return cv.GaussianBlur(m, (0, 0), smooth_sigma) if smooth_sigma > 0 else m


def remove_shading(img_rgb, shading_map, floor=0.05):
    """Divide out shading. `floor` avoids dividing by ~0 in dark corners."""
    return img_rgb / np.maximum(shading_map, floor)


def corner_to_center_ratio(img, frac=0.10):
    """Per-channel corner/center ratio (1.0 means no shading)."""
    return corner_mean(img, frac) / center_mean(img, frac)


def horizontal_profile(img_rgb, band_frac=0.05):
    """Mean of a thin band of rows through the center, per channel."""
    h = img_rgb.shape[0]
    d = max(1, int(h * band_frac / 2))
    return img_rgb[h // 2 - d:h // 2 + d].mean(axis=0)        # (w, 3)


def radial_profile(img_rgb, n_bins=80):
    """Average all pixels at the same distance from center. Binning pattern after
    McNames, 04_measure_your_camera.ipynb section 3 (np.digitize / np.bincount)."""
    h, w = img_rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - (w - 1) / 2, y - (h - 1) / 2).ravel()
    edges = np.linspace(0, r.max(), n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    prof = np.stack([np.bincount(idx, weights=img_rgb[..., c].ravel(), minlength=n_bins) / np.maximum(count, 1)
                     for c in range(3)], axis=1)
    return 0.5 * (edges[:-1] + edges[1:]), prof


def cos4_model(h, w, camera_spec, bin_factor=2):
    """Natural-vignetting model, 1.0 at center. (h, w) must be the binned image size."""
    f_px = camera_spec.focal_px(bin_factor)
    y, x = np.mgrid[0:h, 0:w]
    r2 = (x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2
    return (1 + r2 / f_px ** 2) ** -2

def plot_white_and_map(white_rgb, shading_map, title, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    # 1. The measurement as a picture. Encode for display; linear data looks too dark.
    axes[0].imshow(psu.linear_to_srgb(white_rgb))
    axes[0].set_title("white sheet (camera RGB, no white balance)")
    # 2. The map, one channel per panel, same scale, so you compare channels honestly.
    lo, hi = np.percentile(shading_map, [0.5, 99.5])      # robust to single noisy pixels
    for c, ax in enumerate(axes[1:]):
        im = ax.imshow(shading_map[..., c], vmin=lo, vmax=hi, cmap="viridis")
        ax.set_title(f"shading map, {'RGB'[c]}")
    fig.colorbar(im, ax=axes[1:], fraction=0.02, label="relative to center")
    fig.suptitle(title)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def shading_ratios(round_name, white_rgb, shading_map, model, frac=0.10):
    """Corner divided by center ratios on the white sheet itself, before and after removal."""
    before = corner_to_center_ratio(white_rgb, frac)
    after = corner_to_center_ratio(remove_shading(white_rgb, shading_map), frac)
    row = {"round": round_name}
    for c, n in enumerate("RGB"):
        row[f"{n}_before"], row[f"{n}_after"] = before[c], after[c]
    row["cos4_model"] = corner_mean(model, frac) / center_mean(model, frac)   # 2D input -> scalar
    return row

# Okabe-Ito colorblind-safe palette (black left out; it is used for the model).
# Rounds get these colors, so a round's color never looks like a channel's R, G or B.
ROUND_COLORS = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442")
ROUND_STYLES = ("-", "--", "-.", ":")


def plot_profiles(profiles: dict, model_profile=None, title="", save_path=None, layout="by_channel"):
    """Central horizontal profiles of every round, normalized to center.

    profiles: {round: (w, 3) array}.
    layout="by_channel" (default): one panel per color channel, sharing the y axis.
        Every round is drawn on the same axis within each panel, colored by round,
        so rounds are compared directly instead of hunting through nine overlaid
        lines. The cos^4 model is the same in every panel.
    layout="single": the original single axis, color = channel, line style = round.
    """
    if layout == "single":
        return _plot_profiles_single(profiles, model_profile, title, save_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for c, ax in enumerate(axes):
        for i, (name, prof) in enumerate(profiles.items()):
            x = np.arange(len(prof)) - (len(prof) - 1) / 2      # distance from center, px
            ax.plot(x, prof[:, c], color=ROUND_COLORS[i % len(ROUND_COLORS)],
                    ls=ROUND_STYLES[i % len(ROUND_STYLES)], lw=1.4, label=str(name))
        if model_profile is not None:
            xm = np.arange(len(model_profile)) - (len(model_profile) - 1) / 2
            ax.plot(xm, model_profile, color="k", lw=2, alpha=0.6, label="cos⁴ model")
        ax.axhline(1.0, color="gray", lw=0.6)
        # Panel title in the channel's own color, so the channel is still obvious.
        ax.set_title(f"{'RGB'[c]} channel", color=CHANNEL_COLORS[c], fontweight="bold")
        ax.set_xlabel("distance from image center (quarter-res px)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("relative to center")
    axes[-1].legend(fontsize=8, loc="lower center")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_profiles_single(profiles, model_profile=None, title="", save_path=None):
    """Original layout: one axis, color = channel, line style = round."""
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (name, prof), ls in zip(profiles.items(), styles * 2):
        x = np.arange(len(prof)) - (len(prof) - 1) / 2          # distance from center, px
        # TODO over winter break: Wse a different plotting library that can do interactive legends (click to hide/show).
        for c in range(3):
            ax.plot(x, prof[:, c], ls=ls, color=CHANNEL_COLORS[c], lw=1.3, alpha=0.5)
    if model_profile is not None:
        ax.plot(x, model_profile, color="k", lw=2, alpha=0.6)
    ax.set_xlabel("distance from image center (quarter-res px)")
    ax.set_ylabel("relative to center")
    ax.set_title(title)
    ch = [Line2D([], [], color=c, label=n) for c, n in zip(CHANNEL_COLORS, "RGB")]
    rd = [Line2D([], [], color="gray", ls=ls, label=n) for (n, _), ls in zip(profiles.items(), styles * 2)]
    md = [Line2D([], [], color="k", lw=2, alpha=0.6, label="cos⁴ model")] if model_profile is not None else []
    ax.legend(handles=ch + rd + md, fontsize=8, ncol=2)
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
