"""Project 1, measurement 1: intrinsics and lens distortion from checkerboard captures.

Detect the board's inner corners in every capture, refine them to sub-pixel, and fit
the pinhole-plus-distortion model with cv2.calibrateCameraExtended, which also returns
the standard deviation of every parameter (the uncertainties the datasheet needs).

Method and several helpers follow Prof. James McNames' 02_calibrate_your_camera.ipynb
(from_real_images, synthetic, the residual plot) and 02_projection_and_intrinsics.ipynb
(geometry_from_exif); each adapted function says so.

Images are read straight from the Bayer mosaic (raw_gray): no demosaic, no white balance
from metadata, no gamma, no denoise, no flip. Pixel coordinates are therefore those of
rawpy's raw_image_visible, the same array the noise and MTF measurements use.

Pixel coordinate convention (OpenCV's, and numpy indexing): pixel (row i, column j) has
its center at x = j, y = i; its edges are at half-integers. A frame w pixels wide spans
x = -0.5 ... w - 0.5 and its geometric center is at x = (w - 1) / 2. frame_center() and
frame_corners() encode this; nothing in this module should use w / 2.
"""
import json
from fractions import Fraction
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy
import tifffile

import my_library.analyze as analyze
import my_library.image_metadata as meta

PATTERN_SIZE = analyze.PATTERN_SIZE        # inner corners (across, down)
VIEW_RMS_FLAG_PX = 1.5                     # per-view RMS above this is reported

# Corner detectors. OpenCV 5 docs: https://docs.opencv.org/5.0/main_modules/objdetect_qrcode.html
DETECTORS = ("sb", "classic")
# sb: findChessboardCornersSB (sector-based, Duda & Frese, BMVC 2018), which returns
# sub-pixel corners itself. EXHAUSTIVE searches harder for hard images; ACCURACY
# up-samples the image before locating corners. On rendered test boards ACCURACY cut
# mean corner error by about a third at about 4x the run time; NORMALIZE_IMAGE made
# it slightly worse; ADAPTIVE_THRESH is a classic-only flag that SB rejects in 5.0.
SB_FLAGS = cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
# classic: findChessboardCorners with its documented default flags, then cornerSubPix.
CLASSIC_FLAGS = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
# cornerSubPix settings, as in HW1 and Prof. McNames' from_real_images():
#   window half-size 11 -> a 23 x 23 px search window; no dead zone (-1, -1);
#   stop after 30 iterations or when a corner moves less than 0.001 px in one iteration.
SUBPIX_HALF_WINDOW = (11, 11)
SUBPIX_ZERO_ZONE = (-1, -1)
SUBPIX_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# estimateChessboardSharpness: OpenCV's docs say the black-to-white transition width
# "should be below ~3.0 pixels" for a board sharp enough to calibrate from.
SHARPNESS_LIMIT_PX = 3.0
# Prof. McNames, HW2 feedback: "a phone's main camera has a few percent at most" of
# barrel distortion. Fits implying more than this are flagged as suspect.
DISTORTION_FLAG_PERCENT = 5.0
# Published pixel pitch of the binned 12 MP output (48 MP sensor, 1.22 um photodiodes,
# 2x2 binned), used only as an independent check on the EXIF-route pitch:
#   GSMArena review, "Wide (main): 48MP (1/1.28", 1.22um-2.44um)":
#     https://www.gsmarena.com/apple_iphone_16_pro_max-review-2750p5.php
#   9to5Mac, "Main lens sensor: 2.44 um quad pixel":
#     https://9to5mac.com/iphone-16-pro-vs-13-pro/
PUBLISHED_PITCH_UM = 2.44
# The same sources give the 48 MP sensor's photodiode pitch as 1.22 um ("1.22um-2.44um"):
# 2 x 2 binning makes each 12 MP output pixel two photodiodes wide.
PUBLISHED_PHOTODIODE_UM = 1.22
SENSOR_MEGAPIXELS = 48.0
# Full-frame (35 mm film) image area, 36 x 24 mm; "35 mm" is the width of the film strip
# including sprocket holes. Its diagonal, hypot(36, 24) = 43.27 mm, is the reference for
# the 35 mm-equivalent focal length (Prof. McNames, 02_projection_and_intrinsics.ipynb,
# "Predict f_x from EXIF").
FULL_FRAME_MM = (36.0, 24.0)
# raw_gray(): percentile of each Bayer phase, inside the board, taken as that phase's
# white-paper level (white squares are about half the board, so the 90th percentile
# lands on white paper while ignoring a few hot pixels).
WHITE_PERCENTILE_ON_BOARD = 90


# ----------------------------------------------------------------- pixel convention

def frame_center(size):
    """Geometric center (x, y) of a frame of size (width, height): ((w-1)/2, (h-1)/2)."""
    return (size[0] - 1) / 2.0, (size[1] - 1) / 2.0


def frame_corners(size):
    """Outer corners (x, y) of the frame: the edges of the corner pixels, at -0.5 and
    w - 0.5 (and likewise in y)."""
    w, h = size
    return [(-0.5, -0.5), (w - 0.5, -0.5), (-0.5, h - 0.5), (w - 0.5, h - 0.5)]


# ----------------------------------------------------------------- reading raw

def _to_uint8(values, full_scale):
    """Scale linear values so full_scale maps to 255, round, and clip to 8 bits."""
    return np.clip(np.rint(values / full_scale * 255.0), 0, 255).astype(np.uint8)


def raw_gray(path, pattern=PATTERN_SIZE):
    """Full-resolution 8-bit grayscale straight from the Bayer mosaic, for a neutral
    (black and white) target. Returns (image, info).

    The four Bayer phases of a neutral target see the same pattern, each scaled by its
    own constant (that color channel's response to the light). Dividing each phase by
    its own white-paper level equalizes them. Only per-phase linear scaling is applied
    after black-level subtraction; no pixel is interpolated or moved.

    The white levels are measured on the board itself, because a bright colored object
    elsewhere in the frame would bias whole-frame levels. A rough pass finds the board
    in one green phase of the mosaic (green_phase_to_full maps its corners back), and
    each phase's level is its WHITE_PERCENTILE_ON_BOARD percentile inside the board's
    outline. The rough pass only builds that mask; the corners used for calibration are
    found later, in the full-resolution balanced image. If the board is not found in
    the green phase, whole-frame 99th percentiles are used and info["gain_source"] says so.
    """
    with rawpy.imread(str(path)) as raw:          # closed at the end of this block
        mosaic = raw.raw_image_visible.astype(np.float64)
        black_levels = np.asarray(raw.black_level_per_channel, float)
        bayer = np.asarray(raw.raw_pattern)
        desc = raw.color_desc.decode()
    for r in (0, 1):
        for c in (0, 1):
            mosaic[r::2, c::2] -= black_levels[bayer[r, c]]

    # Rough pass on a single green phase: one color, no pixels combined, no gains needed.
    # It could instead use a 2x2-binned image (the mean of each B, G, G, R tile), which is
    # less noisy and also needs no gains, since every binned pixel has the same color mix;
    # its pixel j is centered at full-resolution 2j + 0.5. The single phase is used so
    # that nothing in this step mixes or rescales light.
    gr, gc = green_phase(bayer, desc)
    green = mosaic[gr::2, gc::2]
    ok, rough = cv.findChessboardCornersSB(_to_uint8(green, np.percentile(green, 99.5)),
                                           pattern, flags=cv.CALIB_CB_EXHAUSTIVE)
    if ok:
        full = green_phase_to_full(rough.reshape(-1, 2), gr, gc)
        mask = np.zeros(mosaic.shape, np.uint8)
        cv.fillConvexPoly(mask, cv.convexHull(full.astype(np.float32)).astype(np.int32), 1)
        mask = mask.astype(bool)
        level = lambda phase, m: np.percentile(phase[m], WHITE_PERCENTILE_ON_BOARD)
        source = "board"
    else:
        mask = np.ones(mosaic.shape, bool)
        level = lambda phase, m: np.percentile(phase, 99)
        source = "whole frame (board not found at half resolution)"

    gray = np.empty_like(mosaic)
    levels = {}
    for r in (0, 1):
        for c in (0, 1):
            lv = level(mosaic[r::2, c::2], mask[r::2, c::2])
            gray[r::2, c::2] = mosaic[r::2, c::2] / lv
            levels[(r, c)] = float(lv)
    image = _to_uint8(gray, 1.0)
    return image, dict(gain_source=source, white_levels_dn=levels,
                       rough_pass=f"green phase at tile row {gr}, column {gc}")


def green_phase(bayer, desc):
    """(row, column) of the first green site in the 2x2 Bayer tile, in raster order.
    bayer is rawpy's raw_pattern; desc its color_desc (e.g. 'RGBG'). BGGR gives (0, 1)."""
    for r in (0, 1):
        for c in (0, 1):
            if desc[bayer[r, c]] == "G":
                return r, c
    raise ValueError(f"no green site in Bayer tile {bayer.tolist()} ({desc})")


def green_phase_to_full(points, r, c):
    """Map (x, y) in a single-phase plane mosaic[r::2, c::2] to full-resolution (x, y).

    Plane pixel (row i, column j) is full-resolution pixel (2i + r, 2j + c), and pixel
    centers sit at integer coordinates, so x_full = 2 x + c and y_full = 2 y + r.
    """
    pts = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([2.0 * pts[:, 0] + c, 2.0 * pts[:, 1] + r])


# ----------------------------------------------------------------- pure functions

def board_object_points(square_mm, pattern=PATTERN_SIZE):
    """Corner positions on the flat board, in mm (z = 0).
    Adapted from Prof. McNames, 02_calibrate_your_camera.ipynb."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    return objp


def detect_corners(gray, pattern=PATTERN_SIZE, detector="sb"):
    """Inner corners of the board in an 8-bit grayscale image, or None if not found.

    sb (default): findChessboardCornersSB with SB_FLAGS; its corners are already sub-pixel.
    classic:      findChessboardCorners with CLASSIC_FLAGS, then cornerSubPix (HW1's and
                  Prof. McNames' from_real_images() method).
    """
    if detector == "sb":
        ok, corners = cv.findChessboardCornersSB(gray, pattern, flags=SB_FLAGS)
        return corners.astype(np.float32) if ok else None
    if detector == "classic":
        ok, corners = cv.findChessboardCorners(gray, pattern, CLASSIC_FLAGS)
        if not ok:
            return None
        return cv.cornerSubPix(gray, corners, SUBPIX_HALF_WINDOW, SUBPIX_ZERO_ZONE, SUBPIX_CRITERIA)
    raise ValueError(f"detector must be one of {DETECTORS}")


def board_sharpness(gray, corners, pattern=PATTERN_SIZE):
    """Mean black-to-white transition width across the board's cells, in pixels.

    cv.estimateChessboardSharpness: smaller is sharper; OpenCV's docs give about 3 px
    as the limit for calibration (SHARPNESS_LIMIT_PX). Returns (width_px,
    dark_square, light_square): the mean brightness of the board's dark and light
    squares (8-bit), so dim or low-contrast views can be spotted too. These are the
    board's square levels, not the sensor's black level.
    """
    stats, _ = cv.estimateChessboardSharpness(gray, pattern, np.asarray(corners, np.float32))
    return float(stats[0]), float(stats[1]), float(stats[2])


def calibrate(obj_points, img_points, image_size, fix_k3=True):
    """cv2.calibrateCameraExtended with its standard deviations, as a dict.

    image_size is (width, height). fix_k3 (default) holds k3 at zero: with it free,
    k2 and k3 can grow large with opposite signs and nearly cancel, the unstable
    five-coefficient fit Prof. McNames identified in HW2. The template reports only
    k1, k2, p1, p2.
    """
    flags = cv.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs, sd_int, _, per_view = cv.calibrateCameraExtended(
        obj_points, img_points, image_size, None, None, flags=flags)
    sd = sd_int.ravel()
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    values = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist.ravel()[:5])
    return dict(rms=float(rms), K=K, dist=dist.ravel(), rvecs=rvecs, tvecs=tvecs,
                per_view=per_view.ravel(), image_size=tuple(image_size), fix_k3=fix_k3,
                params={n: (float(v), float(s)) for n, v, s in zip(names, values, sd[:9])})


def radial_term_percent(k1, k2, k3, r_n):
    """100 * (k1 r^2 + k2 r^4 + k3 r^6) for an undistorted normalized radius r_n.

    OpenCV's radial model moves a point at normalized radius r (r^2 = x_n^2 + y_n^2,
    x_n = X/Z) to r_d = r (1 + k1 r^2 + k2 r^4 + k3 r^6). The relative change,
    (r_d - r) / r, is the bracket minus one, which this returns as a percent.
    Negative is barrel (points pulled toward the center).
    """
    r2 = np.asarray(r_n, float) ** 2
    return 100.0 * (k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)


def radial_distortion_percent(cal, points_px):
    """Distortion at pixel locations in the captured image, as a percent of radius.

    For each point: its distorted normalized radius is read directly from the pixel
    position, x_d = (u - c_x) / f_x and y_d = (v - c_y) / f_y (pixels = f x normalized,
    so normalized = pixels / f). Its undistorted radius comes from inverting the full
    fitted model (radial and tangential) with cv.undistortPoints, given tight iteration
    criteria (OpenCV 5 accepts them directly; 4.x needed undistortPointsIter). The result
    is 100 (r_d - r_u) / r_u; with no tangential terms it equals radial_term_percent(r_u).
    """
    K, dist = cal["K"], cal["dist"]
    pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
    und = cv.undistortPoints(pts, K, dist, None, None, None,
                             (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1e-12))
    r_u = np.linalg.norm(und.reshape(-1, 2), axis=1)
    xd = (pts[:, 0, 0] - K[0, 2]) / K[0, 0]
    yd = (pts[:, 0, 1] - K[1, 2]) / K[1, 1]
    r_d = np.hypot(xd, yd)
    return 100.0 * (r_d - r_u) / r_u


def residuals(cal, obj_points, img_points):
    """(positions, residuals) stacked over all views; residual = projected - detected,
    the sign convention of Prof. McNames' notebook."""
    res, pos = [], []
    for op, ip, rv, tv in zip(obj_points, img_points, cal["rvecs"], cal["tvecs"]):
        proj, _ = cv.projectPoints(op, rv, tv, cal["K"], cal["dist"])
        res.append(proj.reshape(-1, 2) - ip.reshape(-1, 2))
        pos.append(ip.reshape(-1, 2))
    return np.vstack(pos), np.vstack(res)


def parse_exposure(text):
    """'1/125', '0.008' -> seconds as float."""
    return float(Fraction(text))


def focal_mm(f_px, pitch_um):
    return f_px * pitch_um / 1000.0


def exif_geometry(path, frame_size=None):
    """Focal length and pixel pitch predicted from EXIF alone.

    Adapted from Prof. McNames' geometry_from_exif(), 02_projection_and_intrinsics.ipynb:
    the 35 mm-equivalent focal length gives the crop factor, hence the sensor diagonal,
    hence the pixel pitch. FocalLengthIn35mmFilm is stored as a whole number of mm, so
    the pitch carries roughly +/-0.5/equiv relative uncertainty from rounding alone.

    frame_size (width, height): the frame the calibration was done in, so the predicted
    f is in the same pixels. Without it, the size comes from IFD0 as in the original,
    which in some DNG layouts is a small preview rather than the raw image; a warning
    is printed in that case.
    """
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]
        ex = pg.tags["ExifTag"].value
        fl = ex["FocalLength"]
        f = fl[0] / fl[1]
        equiv = float(ex["FocalLengthIn35mmFilm"])
        crop_tag = pg.tags.get("DefaultCropSize")
        nx, ny = (crop_tag.value if crop_tag
                  else (pg.tags["ImageWidth"].value, pg.tags["ImageLength"].value))
        subfile = pg.tags.get("NewSubfileType")
        is_preview = bool(subfile is not None and int(subfile.value) & 1)
    if frame_size is not None:
        nx, ny = frame_size
    else:
        print(f"  WARNING: exif_geometry() is using IFD0's size {int(nx)} x {int(ny)} "
              + ("which is a reduced-resolution preview (NewSubfileType = 1); "
                 if is_preview else "")
              + "pass frame_size to predict f in the calibrated frame's pixels")
    nx, ny = int(nx), int(ny)
    diag = np.hypot(*FULL_FRAME_MM) / (equiv / f)       # sensor diagonal = 43.27 mm / crop factor
    pitch_mm = diag * nx / np.hypot(nx, ny) / nx
    rounding_rel = 0.5 / equiv
    return dict(f_mm=f, equiv_mm=equiv, nx=nx, ny=ny, pitch_um=pitch_mm * 1000,
                f_px=f / pitch_mm, rounding_rel=rounding_rel,
                # FocalLengthIn35mmFilm is a whole number, so the true value lies within
                # +/-0.5 mm of it. Treated as a rectangular distribution, the standard
                # uncertainty is the half-width / sqrt(3) (the usual GUM rule).
                pitch_rel_sd=rounding_rel / np.sqrt(3))


def mean_focal_px(cal):
    """(f, sd): the mean of f_x and f_y, as HW2 item 4 defines f_px.

    calibrateCameraExtended gives each one's standard deviation but not their
    covariance, and f_x and f_y are strongly positively correlated (both scale with
    the same lens-to-sensor distance). The sd of the mean is therefore bounded above
    by the fully correlated case, (sd_x + sd_y) / 2, which is what is reported.
    """
    (fx, sx), (fy, sy) = cal["params"]["fx"], cal["params"]["fy"]
    return 0.5 * (fx + fy), 0.5 * (sx + sy)


def focal_mm_with_uncertainty(cal, pitch_um, pitch_rel_sd):
    """(f_mm, sd) = mean f_px x pitch, with relative uncertainties added in quadrature."""
    f_px, s_px = mean_focal_px(cal)
    f = focal_mm(f_px, pitch_um)
    return f, f * np.hypot(s_px / f_px, pitch_rel_sd)


def binning_check(pitch_um, frame_size, photodiode_um=PUBLISHED_PHOTODIODE_UM,
                  sensor_mp=SENSOR_MEGAPIXELS, tolerance=0.1):
    """Is this a 2 x 2 binned image from a higher-resolution sensor? Two independent tests.

    The math. Binning combines a 2 x 2 group of photodiodes into one output pixel, so
    (1) the output pixel pitch is twice the photodiode pitch, and (2) the output has a
    quarter of the sensor's pixels. Test 1 uses the EXIF-route pitch (measured from this
    file's own focal lengths and frame size) against the published photodiode pitch;
    test 2 uses the frame's pixel count against the published sensor resolution. Both
    must agree within `tolerance` (relative). The published values are for the main
    camera (GSMArena: 48 MP, 1.22 um photodiodes), so the check only applies to it.

    Parameters
    ----------
    pitch_um : float
        Output pixel pitch from the EXIF route (exif_geometry).
    frame_size : (width, height)
        The raw frame's size in pixels.
    photodiode_um : float, default PUBLISHED_PHOTODIODE_UM
    sensor_mp : float, default SENSOR_MEGAPIXELS
    tolerance : float, default 0.1

    Returns
    -------
    dict
        pitch_ratio (output pitch / photodiode pitch, 2 if binned), pixel_ratio (sensor
        pixels / output pixels, 4 if binned), binned (both within tolerance), and
        consistent (both tests give the same answer).
    """
    pitch_ratio = pitch_um / photodiode_um
    pixel_ratio = sensor_mp * 1e6 / (frame_size[0] * frame_size[1])
    by_pitch = abs(pitch_ratio / 2 - 1) < tolerance
    by_pixels = abs(pixel_ratio / 4 - 1) < tolerance
    return dict(pitch_ratio=float(pitch_ratio), pixel_ratio=float(pixel_ratio),
                binned=bool(by_pitch and by_pixels), consistent=bool(by_pitch == by_pixels))


def rms_jackknife(obj_points, img_points, image_size, fix_k3=True):
    """Uncertainty of the RMS reprojection error by leaving one view out at a time.

    The math. OpenCV reports the RMS but no uncertainty for it. The jackknife asks how
    much the result depends on which views happened to be shot: recalibrate n times,
    each time without one view, giving RMS_i; with mean RMS_bar, the jackknife standard
    error is
        sd = sqrt( (n - 1) / n * sum_i (RMS_i - RMS_bar)^2 ).
    The (n - 1) / n factor (instead of 1 / (n - 1)) accounts for the n leave-one-out
    results sharing all but one view each, so they scatter far less than independent
    repeats would. Checked against the actual scatter of the RMS over repeated simulated
    calibrations in tests/test_geometry_measurement.py.

    Parameters
    ----------
    obj_points, img_points : lists
        As passed to calibrate (one entry per view).
    image_size : (width, height)
    fix_k3 : bool, default True
        As in calibrate.

    Returns
    -------
    (sd, rms_i) : tuple
        The jackknife standard error (px) and the n leave-one-out RMS values.
    """
    n = len(img_points)
    rms_i = np.array([calibrate([o for k, o in enumerate(obj_points) if k != i],
                                [p for k, p in enumerate(img_points) if k != i],
                                image_size, fix_k3=fix_k3)["rms"] for i in range(n)])
    return float(np.sqrt((n - 1) / n * np.sum((rms_i - rms_i.mean()) ** 2))), rms_i


def results_entries(cal, n_used, pitch_um, pitch_rel_sd, square_mm, square_mm_sd=None,
                    decimals=4, rms_sd=None, binned=None):
    """The geometry keys of results.json.

    focal_length_px is the mean of f_x and f_y (see mean_focal_px). focal_length_mm
    uses the EXIF-route pixel pitch and its rounding uncertainty. The RMS uncertainty is
    the jackknife standard error over views (rms_jackknife), if given. binned is the
    binning_check verdict, if given.
    """
    p = cal["params"]
    pair = lambda v, s=None: [round(float(v), decimals), None if s is None else round(float(s), decimals)]
    out = {
        "focal_length_px": pair(*mean_focal_px(cal)),
        "focal_length_mm": pair(*focal_mm_with_uncertainty(cal, pitch_um, pitch_rel_sd)),
        "principal_point_x_px": pair(*p["cx"]),
        "principal_point_y_px": pair(*p["cy"]),
        "distortion_k1": pair(*p["k1"]),
        "distortion_k2": pair(*p["k2"]),
        "distortion_p1": pair(*p["p1"]),
        "distortion_p2": pair(*p["p2"]),
        "reprojection_rms_px": pair(cal["rms"], rms_sd),
        "n_calibration_images_used": int(n_used),
        "checkerboard_square_mm_measured": pair(square_mm, square_mm_sd),
    }
    if binned is not None:
        out["binned_from_higher_resolution_sensor"] = bool(binned)
    return out


# ----------------------------------------------------------------- figures

def plot_residuals(cal, obj_points, img_points, out_png, arrow_px=150):
    """Residual arrows by position and per-view RMS. Adapted from Prof. McNames'
    02_calibrate_your_camera.ipynb. Arrows are scaled so a residual equal to the
    overall RMS is drawn arrow_px long (his fixed 500x suits sub-pixel residuals but
    turns a 2 px calibration into a tangle); the title states the scale used."""
    pos, res = residuals(cal, obj_points, img_points)
    drawn = arrow_px / max(cal["rms"], 1e-9)
    w, h = cal["image_size"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.quiver(pos[:, 0], pos[:, 1], res[:, 0], res[:, 1], np.linalg.norm(res, axis=1),
              cmap="viridis", angles="xy", scale_units="xy", scale=1 / drawn, width=0.003)
    a1.add_patch(plt.Rectangle((-0.5, -0.5), w, h, fill=False, ec="#213921", lw=1.2))
    a1.set_xlim(-200, w + 200); a1.set_ylim(h + 200, -200); a1.set_aspect("equal")
    a1.set_title(f"residuals by position, drawn {drawn:.0f}x", fontsize=11)
    a1.xaxis.tick_top()
    a2.plot(np.arange(1, len(cal["per_view"]) + 1), cal["per_view"], "o-",
            color="#6d8d24", lw=1.5, ms=5)
    a2.axhline(cal["rms"], color="#213921", ls="--", lw=1.3, label=f"overall {cal['rms']:.3f} px")
    a2.set_xlabel("view"); a2.set_ylabel("per-view RMS (px)")
    a2.set_title("which capture is dragging the fit", fontsize=11); a2.legend(fontsize=9)
    plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    worst = int(np.argmax(np.linalg.norm(res, axis=1)))
    return dict(drawn=drawn, worst_index=worst, worst_pos=pos[worst], worst_res=res[worst])


def _edge_for_display(corners, image_size):
    """Pick the outer corner column nearer a frame edge (more distortion to show)."""
    cols, rows = PATTERN_SIZE
    grid = np.asarray(corners, np.float32).reshape(rows, cols, 2)
    first, last = grid[:, 0, 0].mean(), grid[:, -1, 0].mean()
    cx = frame_center(image_size)[0]
    return "first_col" if abs(first - cx) >= abs(last - cx) else "last_col"


def plot_undistort(path, cal, corners, out_png, alpha=1.0):
    """Original and undistorted frames side by side, with a line fitted through the
    board's outer column of corners and its deviation from straight in each.
    alpha is getOptimalNewCameraMatrix's: 1 keeps every source pixel (black border).
    The raw mosaic is read here (raw_gray) and released when this function returns."""
    rgb, _ = raw_gray(path)
    h, w = rgb.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(cal["K"], cal["dist"], (w, h), alpha)
    und = cv.undistort(rgb, cal["K"], cal["dist"], None, newK)
    pts = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    pts_und = cv.undistortPoints(pts, cal["K"], cal["dist"], P=newK).reshape(-1, 2)
    side = _edge_for_display(corners, (w, h))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats = {}
    for ax, img, p, label in ((axes[0], rgb, pts.reshape(-1, 2), "as captured"),
                              (axes[1], und, pts_und, "undistorted")):
        edge, (vx, vy, x0, y0), dev = analyze.edge_deviation(p, side=side)
        t = (edge - [x0, y0]) @ [vx, vy]
        p0 = np.array([x0, y0]) + t.min() * np.array([vx, vy])
        p1 = np.array([x0, y0]) + t.max() * np.array([vx, vy])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "c-", lw=1.5)
        ax.scatter(edge[:, 0], edge[:, 1], s=40, facecolors="none", edgecolors="cyan")
        stats[label] = (float(np.abs(dev).max()), float(np.sqrt(np.mean(dev ** 2))))
        ax.set_title(label); ax.xaxis.tick_top()
        ax.set_xlabel(f"edge deviation from straight: max {stats[label][0]:.2f} px, "
                      f"RMS {stats[label][1]:.2f} px")
    plt.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return stats


# ----------------------------------------------------------------- the measurement

def save_corner_overlay(gray, corners, out_png, label, scale=0.25):
    """Save a downscaled copy of the capture with the detected corners drawn on it.

    A mid-calculation sanity check: cv.drawChessboardCorners draws the corners in their
    detected order (colored row by row, joined by a line), so a misplaced corner, a
    flipped grid or a detection on the wrong object is visible at a glance. When no
    full grid was found, the frame is saved with that label so the cause (board off the
    frame edge, glare, blur) can be seen.

    Parameters
    ----------
    gray : ndarray
        8-bit grayscale capture (raw_gray), full resolution.
    corners : ndarray or None
        Detected corners (N x 1 x 2, full-resolution pixels), or None.
    out_png : str or Path
        Where to save the image.
    label : str
        Text drawn in the top-left corner (e.g. file name and status).
    scale : float, default 0.25
        Output size relative to the capture.

    Returns
    -------
    Path
        The path written.
    """
    bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    if corners is not None:
        thickness = max(2, int(round(2 / scale)))
        cv.drawChessboardCorners(bgr, PATTERN_SIZE, corners.astype(np.float32), True)
        for x, y in corners.reshape(-1, 2):            # thicker markers survive downscaling
            cv.circle(bgr, (int(round(x)), int(round(y))), thickness * 3, (0, 255, 255), thickness)
    small = cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    color = (40, 200, 40) if corners is not None else (40, 40, 230)
    cv.putText(small, label, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(small, label, (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_png), small)
    return out_png


def _detect_in_file(path, detector, overlay_dir=None):
    """Read one capture's raw mosaic, find the board, and return only small results;
    the image goes out of scope when this function returns. With overlay_dir, also
    save the corner overlay (save_corner_overlay) as <overlay_dir>/<file stem>.png."""
    gray, info = raw_gray(path)
    size = gray.shape[::-1]
    corners = detect_corners(gray, detector=detector)
    if overlay_dir is not None:
        status = f"{len(corners)} corners" if corners is not None else f"NO full {PATTERN_SIZE} grid"
        save_corner_overlay(gray, corners, Path(overlay_dir) / f"{Path(path).stem}.png",
                            f"{Path(path).name}: {status}")
    out = dict(size=size, corners=corners, transition_px=None, dark_square=None, light_square=None,
               max_radius_px=None, gain_source=info["gain_source"])
    if corners is not None:
        out["transition_px"], out["dark_square"], out["light_square"] = board_sharpness(gray, corners)
        out["max_radius_px"] = float(np.linalg.norm(
            corners.reshape(-1, 2) - np.array(frame_center(size)), axis=1).max())
    return out


def select_files(folder, exposure=None, exclude=()):
    """Raw files under folder, optionally only one exposure time (EXIF, within 1%)
    and minus excluded file names. Returns (paths, table of file and exposure)."""
    paths = meta.get_image_paths(folder)
    rows = [dict(path=p, file=p.name, exposure_s=meta.raw_metadata(p)["exposure_s"]) for p in paths]
    table = pd.DataFrame(rows, columns=["path", "file", "exposure_s"])
    if exposure is not None:
        table = table[np.isclose(table["exposure_s"].astype(float), exposure, rtol=0.01)]
    if exclude:
        table = table[~table["file"].isin(set(exclude))]
    return table.reset_index(drop=True)


def measure_geometry(folder, output_dir, square_mm, exposure=None, exclude=(),
                     detector="sb", fix_k3=True, undistort_file=None):
    """Calibrate from every usable capture in folder. Writes figures, a per-view CSV,
    calibration.json, and a corner overlay for every capture (corners/<file>.png,
    including the ones with no full grid) to output_dir. Also computes the RMS's
    jackknife uncertainty (rms_jackknife) and the binning check (binning_check).
    Returns a summary dict (see results_entries)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = select_files(folder, exposure, exclude)
    if table.empty:
        raise SystemExit(f"no calibration files in {folder}"
                         + (f" at exposure {exposure:g} s" if exposure else ""))
    paths = table["path"].tolist()
    print(f"lens_id assertion (Prof. McNames, Project 1 page) over {len(paths)} calibration files:")
    meta.assert_one_camera(paths)
    ident, n = meta.series_identity(paths)
    meta.print_identity(ident, n)

    objp = board_object_points(square_mm)
    obj, img, kept, sizes, frames = [], [], [], set(), {}
    for row in table.itertuples():
        view = _detect_in_file(row.path, detector, overlay_dir=output_dir / "corners")
        sizes.add(view["size"])
        if view["corners"] is None:
            print(f"  no full {PATTERN_SIZE} grid in {row.file}; not used")
            continue
        obj.append(objp); img.append(view["corners"])
        kept.append(dict(file=row.file, exposure_s=row.exposure_s,
                         transition_px=view["transition_px"],
                         dark_square_dn=view["dark_square"], light_square_dn=view["light_square"],
                         max_radius_px=view["max_radius_px"], gain_source=view["gain_source"]))
        frames[row.file] = row.path
    if len(sizes) != 1:
        raise SystemExit(f"calibration frames differ in size: {sizes}")
    size = sizes.pop()
    print(f"{len(img)} of {len(table)} captures gave a full {PATTERN_SIZE} grid; "
          f"raw frame size {size[0]} x {size[1]}")
    if len(img) < 3:
        raise SystemExit("too few usable captures to calibrate")
    if size != ident["size_rawpy_visible"]:
        print(f"  NOTE: calibration frame {size} differs from rawpy's visible raw "
              f"{ident['size_rawpy_visible']}")
    whole = [k for k in kept if not k["gain_source"].startswith("board")]
    if whole:
        print(f"  note: {len(whole)} view(s) used whole-frame Bayer phase levels "
              "(board not found at half resolution)")

    cal = calibrate(obj, img, size, fix_k3=fix_k3)
    views = pd.DataFrame(kept)
    views["rms_px"] = cal["per_view"]
    views.insert(0, "view", range(1, len(views) + 1))
    views.to_csv(output_dir / "geometry_per_view.csv", index=False)

    p = cal["params"]
    print(f"\nRMS reprojection error  {cal['rms']:.4f} px over {len(img)} views")
    for name in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"):
        v, s = p[name]
        print(f"  {name:3s} {v:12.5f} +/- {s:.5f}")
    fc = frame_center(size)
    print(f"  principal point vs frame center ({fc[0]:.1f}, {fc[1]:.1f}): "
          f"({p['cx'][0] - fc[0]:+.1f}, {p['cy'][0] - fc[1]:+.1f}) px")
    (fx, sx), (fy, sy) = p["fx"], p["fy"]
    print(f"  f_y - f_x = {fy - fx:+.2f} px (f_y/f_x = {fy / fx:.5f}; square pixels give 0); "
          f"its sd lies between {abs(sx - sy):.2f} (f_x, f_y fully correlated) and "
          f"{np.hypot(sx, sy):.2f} px (independent)")
    for r in views.itertuples():
        notes = []
        if r.rms_px > VIEW_RMS_FLAG_PX:
            notes.append(f"RMS {r.rms_px:.3f} px > {VIEW_RMS_FLAG_PX}")
        if r.transition_px > SHARPNESS_LIMIT_PX:
            notes.append(f"edge transition {r.transition_px:.2f} px > {SHARPNESS_LIMIT_PX} (soft)")
        if notes:
            print(f"  view {r.view} {r.file}: " + "; ".join(notes))
    print(f"  edge transition width (estimateChessboardSharpness): median "
          f"{views['transition_px'].median():.2f} px, worst {views['transition_px'].max():.2f} px")
    if len(views) > 2:
        rho = views["transition_px"].corr(views["rms_px"], method="spearman")
        reading = ("blurrier views have larger errors" if rho >= 0.5 else
                   "no strong link between blur and error" if rho > -0.5 else
                   "sharper views have larger errors (unexpected; look closer)")
        print(f"  transition width vs per-view RMS: Spearman rho = {rho:+.2f}, {reading}")

    exif = exif_geometry(paths[0], frame_size=size)
    f_mean, s_mean = mean_focal_px(cal)
    print(f"\nEXIF route (geometry_from_exif): {exif['f_mm']} mm, {exif['equiv_mm']:.0f} mm "
          f"equivalent, {exif['nx']} x {exif['ny']} frame -> pitch {exif['pitch_um']:.4f} um "
          f"+/- {100 * exif['pitch_rel_sd']:.2f}% (35 mm rounding, half-width "
          f"{100 * exif['rounding_rel']:.1f}%)")
    print(f"  published pitch {PUBLISHED_PITCH_UM} um (GSMArena, 9to5Mac): EXIF route differs by "
          f"{100 * (exif['pitch_um'] - PUBLISHED_PITCH_UM) / PUBLISHED_PITCH_UM:+.2f}%")
    print(f"  mean f = {f_mean:.2f} +/- {s_mean:.2f} px (fully correlated bound) vs EXIF-predicted "
          f"{exif['f_px']:.1f} px: {100 * (f_mean - exif['f_px']) / exif['f_px']:+.2f}% "
          "(calibration sees the lens-to-sensor distance at the focus distance)")
    f_mm, s_mm = focal_mm_with_uncertainty(cal, exif["pitch_um"], exif["pitch_rel_sd"])
    print(f"  focal length {f_mm:.4f} +/- {s_mm:.4f} mm vs EXIF FocalLength {exif['f_mm']} mm")

    resid = plot_residuals(cal, obj, img, output_dir / "geometry_residuals.png")
    worst_view = int(np.searchsorted(np.cumsum([len(c) for c in img]), resid["worst_index"], side="right"))
    print(f"  largest single-corner residual: view {worst_view + 1} ({views.file[worst_view]}) at "
          f"{np.round(resid['worst_pos'], 1)}, {np.linalg.norm(resid['worst_res']):.2f} px")

    show = undistort_file or views.loc[views["max_radius_px"].idxmax(), "file"]
    k = views.index[views["file"] == show]
    if len(k) != 1:
        raise SystemExit(f"--undistort-file {show} is not among the calibrated views")
    cx, cy = p["cx"][0], p["cy"][0]
    all_corners = np.vstack([c.reshape(-1, 2) for c in img])
    radii = np.linalg.norm(all_corners - [cx, cy], axis=1)
    outer_pt = all_corners[np.argmax(radii)]
    corner_pts = np.array(frame_corners(size), float)
    corner_r = np.linalg.norm(corner_pts - [cx, cy], axis=1)
    frame_pt = corner_pts[np.argmax(corner_r)]
    r_data, r_corner = float(radii.max()), float(corner_r.max())
    d_data, d_corner = radial_distortion_percent(cal, [outer_pt, frame_pt])
    print(f"\nImplied radial distortion: {d_data:+.2f}% at {r_data:.0f} px (outermost calibrated "
          f"corner), {d_corner:+.2f}% at {r_corner:.0f} px (frame corner"
          f"{', extrapolated beyond the data' if r_corner > r_data * 1.02 else ''})")
    if max(abs(d_data), abs(d_corner)) > DISTORTION_FLAG_PERCENT:
        print(f"  WARNING: more than {DISTORTION_FLAG_PERCENT}% implied; a phone's main camera has a "
              "few percent at most. Suspect the fit (unstable k2/k3, or no views near the corners).")
    straight = plot_undistort(frames[show], cal, img[k[0]], output_dir / "geometry_undistort.png")
    print(f"  undistort figure from {show}: edge deviation max/RMS "
          f"{straight['as captured'][0]:.2f}/{straight['as captured'][1]:.2f} px as captured, "
          f"{straight['undistorted'][0]:.2f}/{straight['undistorted'][1]:.2f} px undistorted")
    if straight["undistorted"][1] > straight["as captured"][1]:
        print("  WARNING: undistorting made this edge LESS straight; the distortion model "
              "is not describing this lens well (check k2/k3, corner coverage, outliers).")

    (output_dir / "calibration.json").write_text(json.dumps(dict(
        rms_px=cal["rms"], image_size=list(size), K=cal["K"].tolist(), dist=cal["dist"].tolist(),
        params=p, fix_k3=fix_k3, detector=detector, square_mm=square_mm,
        implied_distortion_percent=dict(at_outermost_corner=float(d_data), at_frame_corner=float(d_corner)),
        exposure_s=exposure, files=views["file"].tolist()), indent=2) + "\n")
    rms_sd, rms_i = rms_jackknife(obj, img, size, fix_k3=fix_k3)
    print(f"\nRMS reprojection error {cal['rms']:.4f} +/- {rms_sd:.4f} px (jackknife over {len(img)} views; "
          f"leave-one-out RMS {rms_i.min():.4f} to {rms_i.max():.4f})")
    binning = binning_check(exif["pitch_um"], size)
    print(f"Binning check: output pitch / photodiode pitch = {binning['pitch_ratio']:.2f} (2 if 2x2 binned), "
          f"sensor pixels / output pixels = {binning['pixel_ratio']:.2f} (4 if binned) -> "
          f"{'binned' if binning['binned'] else 'NOT binned'}"
          + ("" if binning["consistent"] else "  WARNING: the two tests disagree (a different camera?)"))
    return dict(cal=cal, n_used=len(img), ident=ident, views=views, exif=exif,
                distortion_percent=(float(d_data), float(d_corner)), straightness=straight,
                rms_sd=rms_sd, binning=binning)
