"""Tests for my_library/geometry_measurement.py, against cameras with known answers."""
import json

import cv2 as cv
import numpy as np
import pytest

import my_library.geometry_measurement as gm
import my_library.results as res
from tests.synthetic_board import SQUARE_MM, render_view, synthetic_views
from tests.synthetic_dng import write_dng

SMALL_K = np.array([[830.0, 0, 599.5], [0, 830.0, 449.5], [0, 0, 1]])


def small_session(n=10, seed=7, noise_px=0.0):
    return synthetic_views(n=n, size=(1200, 900), K=SMALL_K, z_range=(260, 520),
                           dist=(0, 0, 0, 0, 0), noise_px=noise_px, seed=seed)


def test_calibrate_recovers_known_camera_within_its_own_uncertainty():
    obj, img, size, K, dist, _ = synthetic_views(n=15, noise_px=0.2)
    cal = gm.calibrate(obj, img, size)
    fx, sfx = cal["params"]["fx"]
    assert abs(fx - K[0, 0]) < 4 * sfx                  # truth inside the reported spread
    assert abs(cal["params"]["k1"][0] - dist[0]) < 4 * cal["params"]["k1"][1]
    assert cal["rms"] == pytest.approx(0.2 * np.sqrt(2), rel=0.25)   # 2-D noise of 0.2 px/axis


def test_k3_is_held_at_zero_by_default():
    obj, img, size, *_ = synthetic_views(n=12)
    assert gm.calibrate(obj, img, size)["params"]["k3"][0] == 0.0
    assert gm.calibrate(obj, img, size, fix_k3=False)["params"]["k3"][0] != 0.0


def test_radial_term_percent_analytic():
    """k1 = -0.12, k2 = 0.08 at r = 0.5: 100 (-0.12 * 0.25 + 0.08 * 0.0625) = -2.5 %."""
    assert gm.radial_term_percent(-0.12, 0.08, 0.0, 0.5) == pytest.approx(-2.5)
    assert gm.radial_term_percent(-0.12, 0.08, 0.0, 0.0) == pytest.approx(0.0)


def _cal_from(K, dist):
    return {"K": np.asarray(K, float), "dist": np.asarray(dist, float)}


@pytest.mark.parametrize("direction", [(1, 0), (0, 1), (0.6, 0.8)])
def test_radial_distortion_percent_at_a_pixel_is_exact(direction):
    """Place a point at undistorted normalized radius 0.5, project it through the model
    to a pixel, and recover the analytic percent from that pixel alone."""
    K = [[2800.0, 0, 2015.5], [0, 2800.0, 1511.5], [0, 0, 1]]
    dist = [-0.12, 0.08, 0.0, 0.0, 0.0]
    point = np.array([[[*(0.5 * np.array(direction)), 1.0]]])
    pixel, _ = cv.projectPoints(point, np.zeros(3), np.zeros(3), np.asarray(K), np.asarray(dist))
    got = gm.radial_distortion_percent(_cal_from(K, dist), pixel.reshape(1, 2))[0]
    assert got == pytest.approx(gm.radial_term_percent(-0.12, 0.08, 0.0, 0.5), abs=1e-6)


def test_calibration_recovers_the_true_distortion_percent():
    obj, img, size, K, dist, _ = synthetic_views(n=20, noise_px=0.2)
    cal = gm.calibrate(obj, img, size)
    corner = np.array([[3800.0, 2900.0]])
    truth = gm.radial_distortion_percent(_cal_from(K, dist), corner)[0]
    assert gm.radial_distortion_percent(cal, corner)[0] == pytest.approx(truth, abs=0.3)


def test_frame_center_and_corners_follow_the_pixel_center_convention():
    assert gm.frame_center((4, 2)) == (1.5, 0.5)          # pixels at x = 0..3, y = 0..1
    assert (-0.5, -0.5) in gm.frame_corners((4, 2)) and (3.5, 1.5) in gm.frame_corners((4, 2))


def _colored_scene_dng(path, seed=7, red_patch=True):
    """A board under unequal channel gains (BGGR: B 0.70, G 1.0, R 0.55), optionally with
    a bright red patch elsewhere in the frame. Returns (path, true corners)."""
    _, img, size, K, _, poses = small_session(n=1, seed=seed)
    gray = render_view(K, *poses[0], size).astype(float) / 255
    gains = [[0.70, 1.0], [1.0, 0.55]]
    mosaic = np.zeros_like(gray)
    for r in (0, 1):
        for c in (0, 1):
            mosaic[r::2, c::2] = gray[r::2, c::2] * gains[r][c]
            if red_patch:
                mosaic[20 + r:200:2, 20 + c:300:2] = 1.0 if (r, c) == (1, 1) else 0.2
    write_dng(path, raw_data=(100 + mosaic * 3900).astype(np.uint16), crop=size)
    return path, img[0], gray


def test_raw_gray_equalizes_bayer_phases_on_the_board(tmp_path):
    path, truth, gray = _colored_scene_dng(tmp_path / "c.dng")
    image, info = gm.raw_gray(path)
    assert info["gain_source"] == "board"
    white = (gray > 0.95)
    white[20:200, 20:300] = False
    levels = [np.median(image[r::2, c::2][white[r::2, c::2]]) for r in (0, 1) for c in (0, 1)]
    assert max(levels) - min(levels) <= 2                 # out of 255: phases match


def test_raw_gray_moves_no_pixels(tmp_path):
    """Corners found in the raw gray image land on the true corners: no resampling."""
    path, truth, _ = _colored_scene_dng(tmp_path / "c.dng")
    image, _ = gm.raw_gray(path)
    found = gm.detect_corners(image)
    nearest = np.linalg.norm(found.reshape(-1, 1, 2) - truth.reshape(1, -1, 2), axis=2).min(axis=1)
    assert nearest.mean() < 0.1
    assert image.shape == (900, 1200)


def test_residuals_vanish_for_noiseless_views():
    obj, img, size, *_ = synthetic_views(n=12, noise_px=0.0)
    cal = gm.calibrate(obj, img, size)
    _, r = gm.residuals(cal, obj, img)
    assert np.abs(r).max() < 1e-3


@pytest.mark.parametrize("detector", gm.DETECTORS)
def test_detect_corners_on_rendered_board(detector):
    _, img, size, K, _, poses = small_session(n=4)
    for (rvec, tvec), truth in zip(poses, img):
        found = gm.detect_corners(render_view(K, rvec, tvec, size), detector=detector)
        assert found is not None and len(found) == len(truth)
        nearest = np.linalg.norm(found.reshape(-1, 1, 2) - truth.reshape(1, -1, 2), axis=2).min(axis=1)
        assert nearest.mean() < 0.4 and nearest.max() < 1.0


def test_detect_corners_returns_none_without_a_board():
    assert gm.detect_corners(np.full((900, 1200), 128, np.uint8)) is None


def test_blur_widens_edge_transition_past_the_documented_limit():
    _, img, size, K, _, poses = small_session(n=1)
    sharp = render_view(K, *poses[0], size)
    blurred = cv.GaussianBlur(sharp, (0, 0), 2.5)
    sharp_px = gm.board_sharpness(sharp, img[0])[0]
    blurred_px = gm.board_sharpness(blurred, img[0])[0]
    assert sharp_px < gm.SHARPNESS_LIMIT_PX < blurred_px


def test_sb_is_the_default_and_more_accurate_than_classic_on_rendered_boards():
    _, img, size, K, _, poses = small_session(n=6, seed=11)
    err = {d: [] for d in gm.DETECTORS}
    for (rvec, tvec), truth in zip(poses, img):
        gray = render_view(K, rvec, tvec, size)
        for d in gm.DETECTORS:
            c = gm.detect_corners(gray, detector=d)
            err[d].append(np.linalg.norm(c.reshape(-1, 1, 2) - truth.reshape(1, -1, 2), axis=2).min(axis=1).mean())
    assert gm.detect_corners.__defaults__[-1] == "sb"
    assert np.mean(err["sb"]) < np.mean(err["classic"])


def test_sb_rejects_classic_only_flag():
    """Documents an OpenCV 5 behavior: ADAPTIVE_THRESH is not a valid SB flag."""
    _, _, size, K, _, poses = small_session(n=1)
    gray = render_view(K, *poses[0], size)
    with pytest.raises(cv.error):
        cv.findChessboardCornersSB(gray, gm.PATTERN_SIZE, flags=gm.CLASSIC_FLAGS)


def test_parse_exposure():
    assert gm.parse_exposure("1/125") == pytest.approx(0.008)
    assert gm.parse_exposure("0.0166") == pytest.approx(0.0166)


def test_results_entries_fit_the_template():
    obj, img, size, *_ = synthetic_views(n=12)
    cal = gm.calibrate(obj, img, size)
    entries = gm.results_entries(cal, 12, pitch_um=2.42, pitch_rel_sd=0.012, square_mm=20.01)
    results, template = res.load()
    results.update(entries)
    assert res.validate(results, template) == []
    f_mean = 0.5 * (cal["params"]["fx"][0] + cal["params"]["fy"][0])
    assert entries["focal_length_px"][0] == pytest.approx(f_mean, abs=1e-4)
    assert entries["focal_length_mm"][0] == pytest.approx(f_mean * 2.42e-3, abs=1e-4)
    assert entries["focal_length_mm"][1] > 0 and entries["reprojection_rms_px"][1] is None
    assert entries["checkerboard_square_mm_measured"] == [20.01, None]


def test_mean_focal_and_focal_mm_uncertainty_analytic():
    """f = (2800 + 2810)/2 = 2805 px; sd bound = (3 + 5)/2 = 4 px. With pitch 2.42 um
    +/- 1.2 %: f_mm = 6.7881 mm, relative sd = hypot(4/2805, 0.012)."""
    cal = {"params": {"fx": (2800.0, 3.0), "fy": (2810.0, 5.0)}}
    assert gm.mean_focal_px(cal) == (2805.0, 4.0)
    f, s = gm.focal_mm_with_uncertainty(cal, 2.42, 0.012)
    assert f == pytest.approx(2805 * 2.42e-3)
    assert s / f == pytest.approx(np.hypot(4 / 2805, 0.012))


def test_exif_route_pitch_with_exact_focal_length(tmp_path):
    """The synthetic DNG's EXIF is 6.765 mm at 24 mm equivalent: the route used in
    HW2 item 4, with the exact focal length, gives 2.420 um at 4032 x 3024."""
    path = write_dng(tmp_path / "a.dng")
    g = gm.exif_geometry(path, frame_size=(4032, 3024))
    assert g["pitch_um"] == pytest.approx(43.2666 / (24 / 6.765) / 5040 * 1000, rel=1e-4)
    assert g["pitch_um"] == pytest.approx(2.420, abs=0.001)
    assert g["pitch_rel_sd"] == pytest.approx(0.5 / 24 / np.sqrt(3))


def test_measure_geometry_end_to_end_on_synthetic_dngs(tmp_path, capsys):
    """Render the board through a known camera into DNG files, then run the whole step."""
    _, _, size, K, _, poses = small_session(n=10)
    folder = tmp_path / "Checkerboard"
    folder.mkdir()
    for i, (rvec, tvec) in enumerate(poses):
        gray = render_view(K, rvec, tvec, size).astype(np.uint16)
        write_dng(folder / f"IMG_{i:02d}.dng", raw_data=100 + gray * 12, crop=size)

    summary = gm.measure_geometry(folder, tmp_path / "out", SQUARE_MM)
    assert abs(summary["distortion_percent"][0]) < 0.5            # this camera has none
    assert (summary["views"]["transition_px"] < gm.SHARPNESS_LIMIT_PX).all()
    fx, sfx = summary["cal"]["params"]["fx"]
    assert summary["n_used"] == 10
    assert abs(fx - 830.0) < max(4 * sfx, 0.01 * 830)
    assert summary["exif"]["nx"] == 1200                      # predicted in the calibrated frame
    out = capsys.readouterr().out
    assert "all images from the same camera" in out           # the lens_id assertion
    for name in ("calibration.json", "geometry_per_view.csv",
                 "geometry_residuals.png", "geometry_undistort.png"):
        assert (tmp_path / "out" / name).exists()
    saved = json.loads((tmp_path / "out" / "calibration.json").read_text())
    assert saved["K"][0][0] == pytest.approx(fx)


def test_exclude_and_exposure_filter(tmp_path):
    for i in range(3):
        write_dng(tmp_path / f"IMG_{i}.dng")
    table = gm.select_files(tmp_path, exclude=["IMG_1.dng"])
    assert table["file"].tolist() == ["IMG_0.dng", "IMG_2.dng"]


def test_exif_geometry_warns_when_falling_back_to_a_preview(tmp_path, capsys):
    path = write_dng(tmp_path / "a.dng")                 # IFD0 is a 16 x 12 preview
    gm.exif_geometry(path)
    assert "reduced-resolution preview" in capsys.readouterr().out
    gm.exif_geometry(path, frame_size=(64, 48))
    assert "WARNING" not in capsys.readouterr().out


# ------------------------------------------------ real data (opt-in, see conftest.py)

def test_real_detectors_agree(real_calibration_images, real_calibration_exposure):
    """Two independent corner detectors should land on nearly the same points."""
    exposure = gm.parse_exposure(real_calibration_exposure) if real_calibration_exposure else None
    table = gm.select_files(real_calibration_images, exposure)
    compared = []
    for path in table["path"].head(5):
        sb = gm._detect_in_file(path, "sb")["corners"]
        classic = gm._detect_in_file(path, "classic")["corners"]
        if sb is None or classic is None:
            continue
        d = np.linalg.norm(sb.reshape(-1, 1, 2) - classic.reshape(1, -1, 2), axis=2).min(axis=1)
        compared.append(d.mean())
        print(f"{path.name}: mean SB-classic distance {d.mean():.3f} px, max {d.max():.3f} px")
    if not compared:
        pytest.skip("no image where both detectors found the board")
    assert np.mean(compared) < 1.0


def test_real_calibration_is_plausible(real_calibration_images, real_calibration_exposure, tmp_path):
    """No ground truth, so check against independent expectations: the EXIF focal
    prediction (allowing 2.1% EXIF rounding plus up to ~4% focus breathing at close
    range, so 8%), and a principal point near the frame center."""
    exposure = gm.parse_exposure(real_calibration_exposure) if real_calibration_exposure else None
    s = gm.measure_geometry(real_calibration_images, tmp_path, 20.0, exposure=exposure)
    fx = s["cal"]["params"]["fx"][0]
    w, h = s["cal"]["image_size"]
    assert s["n_used"] >= 3
    assert abs(fx - s["exif"]["f_px"]) / s["exif"]["f_px"] < 0.08
    cx0, cy0 = gm.frame_center((w, h))
    assert abs(s["cal"]["params"]["cx"][0] - cx0) < 0.05 * w
    assert abs(s["cal"]["params"]["cy"][0] - cy0) < 0.05 * h


def test_green_phase_location_for_each_bayer_order():
    desc = "RGBG"                                        # rawpy's color_desc for these sensors
    assert gm.green_phase(np.array([[2, 3], [1, 0]]), desc) == (0, 1)   # BGGR (iPhone)
    assert gm.green_phase(np.array([[0, 1], [3, 2]]), desc) == (0, 1)   # RGGB
    assert gm.green_phase(np.array([[1, 0], [2, 3]]), desc) == (0, 0)   # GRBG


def test_green_phase_to_full_follows_the_pixel_center_convention():
    """Plane pixel (row 3, column 5) of mosaic[0::2, 1::2] is full-res (row 6, column 11)."""
    assert gm.green_phase_to_full([[5.0, 3.0]], r=0, c=1).tolist() == [[11.0, 6.0]]
    assert gm.green_phase_to_full([[5.0, 3.0]], r=1, c=0).tolist() == [[10.0, 7.0]]


def test_rough_pass_corners_map_back_onto_the_full_resolution_corners(tmp_path):
    """Corners found in the green phase alone, mapped back, land near the true
    full-resolution corners (well enough to outline the board)."""
    path, truth, _ = _colored_scene_dng(tmp_path / "c.dng")
    import rawpy
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image_visible.astype(float)
        r, c = gm.green_phase(np.asarray(raw.raw_pattern), raw.color_desc.decode())
    green = mosaic[r::2, c::2]
    found = gm.detect_corners(gm._to_uint8(green, np.percentile(green, 99.5)), detector="sb")
    full = gm.green_phase_to_full(found, r, c)
    nearest = np.linalg.norm(full.reshape(-1, 1, 2) - truth.reshape(1, -1, 2), axis=2).min(axis=1)
    assert nearest.mean() < 0.5
    _, info = gm.raw_gray(path)
    assert info["rough_pass"] == "green phase at tile row 0, column 1"
