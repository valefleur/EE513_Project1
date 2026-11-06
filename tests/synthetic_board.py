"""Synthetic checkerboard captures with a known camera, for calibration tests.

synthetic_views() is adapted from Prof. James McNames' synthetic() in
02_calibrate_your_camera.ipynb: a capture session with varied tilts and sub-pixel
corner noise, returned as object and image points. render_view() draws the board as
the camera would see it, for tests that must start from pixels.
"""
import cv2 as cv
import numpy as np

from my_library.geometry_measurement import PATTERN_SIZE, board_object_points

SQUARE_MM = 20.0


def synthetic_views(n=15, tilt_deg=40.0, noise_px=0.20, seed=513,
                    K=None, dist=(-0.12, 0.08, 0.0, 0.0, 0.0), size=(4032, 3024),
                    z_range=(320, 900), margin=60):
    rng = np.random.default_rng(seed)
    w, h = size
    if K is None:
        K = np.array([[2795.7, 0, (w - 1) / 2], [0, 2795.7, (h - 1) / 2], [0, 0, 1]])
    dist = np.asarray(dist, float)
    objp, obj, img, poses = board_object_points(SQUARE_MM), [], [], []
    span = PATTERN_SIZE[0] * SQUARE_MM
    while len(img) < n:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        rvec = ax * np.radians(rng.uniform(0.35, 1.0) * tilt_deg)
        tvec = np.array([rng.uniform(-0.75, 0.15) * span,
                         rng.uniform(-0.75, 0.15) * span * 0.8,
                         rng.uniform(*z_range)])
        p, _ = cv.projectPoints(objp, rvec, tvec, K, dist)
        p = p.reshape(-1, 2)
        if (p[:, 0].min() > margin and p[:, 0].max() < w - margin and
                p[:, 1].min() > margin and p[:, 1].max() < h - margin):
            obj.append(objp)
            img.append((p + rng.normal(0, noise_px, p.shape)).astype(np.float32).reshape(-1, 1, 2))
            poses.append((rvec, tvec))
    return obj, img, size, K, dist, poses


def render_view(K, rvec, tvec, size, px_per_mm=8, background=128):
    """8-bit grayscale image of the board (no lens distortion) seen from pose rvec, tvec.

    The board texture has one extra square on every side of the inner-corner grid,
    plus a white quiet zone, like a printed target.
    """
    cols, rows = PATTERN_SIZE
    sq = int(SQUARE_MM * px_per_mm)
    quiet = sq
    tex = np.full(((rows + 1) * sq + 2 * quiet, (cols + 1) * sq + 2 * quiet), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = quiet + r * sq, quiet + c * sq
                tex[y:y + sq, x:x + sq] = 0
    # texture pixel -> board mm (inner corner (0,0) is at texture quiet + sq)
    c0 = quiet + sq - 0.5     # pixel centers sit at integer indices, edges at half-integers
    T = np.array([[1 / px_per_mm, 0, -c0 / px_per_mm],
                  [0, 1 / px_per_mm, -c0 / px_per_mm], [0, 0, 1]])
    R, _ = cv.Rodrigues(np.asarray(rvec, float))
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float)]) @ T
    return cv.warpPerspective(tex, H, size, flags=cv.INTER_AREA,
                              borderMode=cv.BORDER_CONSTANT, borderValue=background)
