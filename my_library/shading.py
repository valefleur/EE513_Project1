# shading.py

import warnings
import numpy as np
import cv2 as cv

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
    """White sheet / its own per-channel center. 1.0 at center, <1 toward the corners.
    Measures lens AND lamp falloff together; don't assume which until profiles are compared."""
    c = center_mean(white_rgb, center_frac)
    if (c > max_center).any():
        warnings.warn(f"white-sheet center is near the ceiling {np.round(c, 3)}; "
                      "the center is pinned and the map will look too flat")
    m = white_rgb / c
    return cv.GaussianBlur(m, (0, 0), smooth_sigma) if smooth_sigma > 0 else m


def remove_shading(img_rgb, shading_map, floor=0.05):
    return img_rgb / np.maximum(shading_map, floor)


def horizontal_profile(img_rgb, band_frac=0.05):
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