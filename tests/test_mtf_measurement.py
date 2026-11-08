"""Tests for my_library/mtf_measurement.py, on synthetic targets with known blur."""
import numpy as np
import pytest

import my_library.mtf_measurement as mtf
import my_library.results as res
from tests.synthetic_target import expected_mtf50, render_target, write_target_dngs


def _plane(**kw):
    """One green phase (BGGR: row 0, column 1) of a rendered target, as the step sees it."""
    return (render_target(**kw) * 3000.0)[0::2, 1::2]


def test_every_region_and_direction_recovers_the_known_blur():
    rows = mtf.analyze_frame(_plane(psf_sigma_px=1.5), 3995.0)
    found = {(r["region"], r["direction"]) for r in rows}
    assert found == {(g, d) for g in mtf.REGIONS for d in mtf.DIRECTIONS}
    expected = expected_mtf50(1.5)
    for r in rows:
        assert r["mtf50"] == pytest.approx(expected, rel=0.04), (r["region"], r["direction"])
        assert 3 < abs(r["slant_deg"]) < 7                     # the 5 degree tilt


def test_corner_regions_are_far_from_the_center_and_the_center_is_not():
    rows = mtf.analyze_frame(_plane(psf_sigma_px=1.5), 3995.0)
    radius = {r["region"]: r["radius"] for r in rows}
    assert radius["center"] < 0.3
    assert all(radius[c] > 0.6 for c in mtf.CORNERS)


def test_double_image_puts_an_mtf_zero_at_one_over_twice_the_offset():
    """Prof. McNames' suggested test (HW2 feedback): two Gaussians offset by d across
    vertical edges multiply their MTF by |cos(pi f d)|, first zero at 1 / (2 d). Here
    d = 2.4 full-res px = 1.2 green-plane samples, so the zero is at 0.417. Horizontal
    edges do not see a horizontal offset."""
    rows = mtf.analyze_frame(_plane(psf_sigma_px=0.6, psf_offset=2.4), 3995.0)
    for r in rows:
        f, m = r["freq"], r["mtf"]
        sel = (f > 0.25) & (f < 0.6)
        if r["direction"] == "vertical":
            assert f[sel][np.argmin(m[sel])] == pytest.approx(1 / 2.4, abs=0.03)
            assert m[sel].min() < 0.1
    v = np.mean([r["mtf50"] for r in rows if r["direction"] == "vertical"])
    h = np.mean([r["mtf50"] for r in rows if r["direction"] == "horizontal"])
    assert h > 1.5 * v


def test_edge_drift_and_a_nearly_straight_edge_is_excluded():
    """Tilt 0.5 deg: over ~90 rows the edge drifts < 1 px, too little to sample every
    sub-pixel phase; those edges are flagged unreliable and left out of the statistics."""
    assert mtf.edge_drift(143, 0.57) == pytest.approx(1.42, abs=0.02)
    assert mtf.edge_drift(143, 5.0) == pytest.approx(12.5, abs=0.1)
    rows = mtf.analyze_frame(_plane(psf_sigma_px=1.5, tilt_deg=0.5), 3995.0)
    assert rows and all(not r["reliable"] for r in rows)
    assert mtf.summarize_series([rows]) == {}


def test_log_threshold_finds_squares_under_strong_uneven_light():
    """Light falling to 35% across the sheet (left dim, right bright), as on APC_0500:
    a linear threshold merges the dim side's squares with the background; the log
    threshold still finds all five."""
    img = render_target(psf_sigma_px=1.5)
    x = np.linspace(0.35, 1.0, img.shape[1])[None, :]
    plane = (img * x * 3000.0)[0::2, 1::2]
    assert set(mtf.find_squares(plane)) == set(mtf.REGIONS)


def test_esf_crop_keeps_a_noisy_known_answer_edge_stable():
    """HW2's known-answer edge (5 deg tilt, Gaussian PSF sigma 1.0 px, analytic MTF50
    0.1874) with noise: with the ESF always cropped, 100 noisy trials stay within 4% of
    the answer and scatter by less than 0.005 (uncropped, they scattered by ~0.1)."""
    from scipy.special import erf
    import my_library.mtf_analysis as mtfa
    rows, cols, ss = 256, 128, 8
    y, x = np.mgrid[0:rows * ss, 0:cols * ss] / ss
    d = (x - cols / 2) - np.tan(np.radians(5)) * (y - rows / 2)
    edge = (0.5 * (1 + erf(d / np.sqrt(2)))).reshape(rows, ss, cols, ss).mean(axis=(1, 3))
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(100):
        f, m, *_ = mtfa.slanted_edge_mtf(edge * 1000 + rng.normal(0, 10, edge.shape))
        vals.append(mtfa.mtf50(f, m))
    truth = mtfa.analytic_gaussian_mtf50(1.0)
    assert np.mean(vals) == pytest.approx(truth, rel=0.04)
    assert np.std(vals) < 0.005


def test_unit_conversions():
    u = mtf.unit_conversions(0.35, 2.42)
    assert u["cy_per_raw_px"] == pytest.approx(0.175)
    assert u["cy_per_um"] == pytest.approx(0.35 / 4.84)
    assert u["cy_per_mm"] == pytest.approx(1000 * 0.35 / 4.84)
    assert u["frac_of_green_nyquist"] == pytest.approx(0.7)


def test_two_distance_verdicts():
    near = dict(mean=0.30, sd=0.01)
    assert mtf.two_distance(near, dict(mean=0.36, sd=0.01))["verdict"].startswith("print-limited")
    assert mtf.two_distance(near, dict(mean=0.305, sd=0.01))["verdict"].startswith("camera-limited")


def test_datasheet_values_pick_the_farthest_corner_and_propagate_uncertainty():
    s = lambda mean, sd, radius: dict(mean=mean, sd=sd, n=5, radius=radius, by_direction={}, peak=0.5, white=0.5)
    summaries = {
        "center_25cm": {"center": s(0.33, 0.01, 0.1), "top left": s(0.30, 0.02, 0.6),
                        "top right": s(0.29, 0.02, 0.6), "bottom left": s(0.28, 0.02, 0.6),
                        "bottom right": s(0.27, 0.02, 0.6)},
        "corners_out": {"top left": s(0.22, 0.03, 0.9), "bottom right": s(0.20, 0.04, 0.85)},
        "center_50cm": {"center": s(0.34, 0.01, 0.1), "top left": s(0.40, 0.01, 0.95)},
    }
    v = mtf.datasheet_values(summaries, "center_25cm", eligible=["center_25cm", "corners_out"])
    assert v["corners"]["top left"]["series"] == "corners_out"          # farther out wins
    assert v["corners"]["top right"]["series"] == "center_25cm"
    assert v["corner_mean"]["mean"] == pytest.approx(np.mean([0.22, 0.29, 0.28, 0.20]))
    assert v["corner_mean"]["sd"] == pytest.approx(np.sqrt(0.03**2 + 0.02**2 + 0.02**2 + 0.04**2) / 4)
    assert v["corner_worst"]["corner"] == "bottom right"
    entries = mtf.results_entries(v)
    results, template = res.load()
    results.update(entries)
    assert res.validate(results, template) == []


def test_series_folders_groups_by_subfolder(tmp_path):
    write_target_dngs(tmp_path / "slanted" / "a", n=2)
    write_target_dngs(tmp_path / "slanted" / "b", n=1)
    groups = mtf.series_folders(tmp_path / "slanted")
    assert {k: len(v) for k, v in groups.items()} == {"a": 2, "b": 1}


def test_measure_mtf_end_to_end_with_a_two_distance_series(tmp_path, capsys):
    root = tmp_path / "slanted"
    write_target_dngs(root / "center_25cm", n=3, psf_sigma_px=1.5)
    write_target_dngs(root / "center_50cm", n=3, psf_sigma_px=1.0, seed=1)
    out = mtf.measure_mtf(root, tmp_path / "out", datasheet_pattern="*_25cm",
                          two_distance_pair=("center_25cm", "center_50cm"))
    assert out["values"]["center"]["mean"] == pytest.approx(expected_mtf50(1.5), rel=0.04)
    assert out["two_distance"]["verdict"].startswith("print-limited")
    for name in ("mtf_frames.csv", "mtf_center_25cm.png", "mtf_center_50cm.png"):
        assert (tmp_path / "out" / name).exists()
    assert "all images from the same camera" in capsys.readouterr().out


def test_clipped_and_underexposed_series_are_flagged(tmp_path, capsys):
    write_target_dngs(tmp_path / "s" / "center_hot", n=2, level=4200)
    write_target_dngs(tmp_path / "t" / "center_dim", n=2, level=300)
    mtf.measure_mtf(tmp_path / "s", tmp_path / "o1")
    mtf.measure_mtf(tmp_path / "t", tmp_path / "o2")
    printed = capsys.readouterr().out
    assert "CLIPPED" in printed and "UNDEREXPOSED" in printed


def test_ambiguous_center_series_asks_to_be_named(tmp_path):
    write_target_dngs(tmp_path / "x" / "pos_a", n=2)
    write_target_dngs(tmp_path / "x" / "pos_b", n=2, seed=3)
    with pytest.raises(SystemExit):
        mtf.measure_mtf(tmp_path / "x", tmp_path / "o")



# ------------------------------------------------ real data (opt-in, see conftest.py)

def test_real_slanted_edges_give_plausible_mtf50(real_slanted_edges, tmp_path):
    """No ground truth: every series finds the center square, every MTF50 lies between 0
    and green Nyquist's double (the method can report past Nyquist), and frames within
    a series agree to within 25% of their mean."""
    groups = mtf.series_folders(real_slanted_edges)
    for name, paths in groups.items():
        frames = [mtf.analyze_frame(*mtf.green_plane(p)) for p in paths]
        summary = mtf.summarize_series(frames)
        assert "center" in summary, name
        for region, v in summary.items():
            assert 0 < v["mean"] < 1.0, (name, region)
            if v["sd"] is not None:
                assert v["sd"] < 0.25 * v["mean"], (name, region)



def test_mtf_analysis_is_silent_unless_asked(capsys):
    """Importing mtf_analysis and calling slanted_edge_mtf print nothing; the diffraction
    summary and the slope print only when asked (hw2.py asks)."""
    import importlib
    import my_library.mtf_analysis as mtfa
    importlib.reload(mtfa)
    plane = (render_target(psf_sigma_px=1.5) * 3000.0)[0::2, 1::2]
    roi = plane[200:300, 380:440]
    mtfa.slanted_edge_mtf(roi)
    assert capsys.readouterr().out == ""
    mtfa.slanted_edge_mtf(roi, verbose=True)
    mtfa.print_diffraction_summary()
    printed = capsys.readouterr().out
    assert "slanted edge slope" in printed and "diffraction cutoff" in printed



def test_datasheet_pattern_keeps_center_and_corners_at_one_distance(tmp_path, capsys):
    """The Sat Nov 7 layout: complete sets at 250 mm and 501 mm. With '*_501mm' the
    center and every corner come from 501 mm series even though a 250 mm series has a
    square farther out; the two-distance test compares the two center series."""
    root = tmp_path / "slanted"
    write_target_dngs(root / "center_250mm", n=2, psf_sigma_px=1.5)
    write_target_dngs(root / "left_250mm", n=2, psf_sigma_px=1.5, offset=(-60, 0), seed=1)
    write_target_dngs(root / "center_501mm", n=2, psf_sigma_px=1.2, seed=2)
    write_target_dngs(root / "left_top_501mm", n=2, psf_sigma_px=1.2, offset=(-30, -20), seed=3)
    out = mtf.measure_mtf(root, tmp_path / "out", datasheet_pattern="*_501mm",
                          two_distance_pair=("center_250mm", "center_501mm"))
    v = out["values"]
    assert v["center"]["series"] == "center_501mm"
    assert all(c["series"].endswith("_501mm") for c in v["corners"].values())
    assert out["two_distance"] is not None
    assert "Datasheet series: center_501mm, left_top_501mm" in capsys.readouterr().out


def test_datasheet_series_and_center_pick():
    names = ["center_250mm", "center_501mm", "left_250mm", "left_top_501mm"]
    assert mtf.datasheet_series(names, "*_501mm") == ["center_501mm", "left_top_501mm"]
    assert mtf.datasheet_series(names, None) == names
    assert mtf.pick_center_series(["center_501mm", "left_top_501mm"]) == "center_501mm"
    assert mtf.pick_center_series(names) is None                      # two centers: ambiguous



def test_repeats_at_a_comparable_radius_add_a_systematic_uncertainty():
    """The Sat Nov 7 bottom-left case: 0.265 +/- 0.008 (radius 0.71) and 0.160 +/- 0.024
    (radius 0.69) are repeats of one position; their spread, sd 0.074, is added in
    quadrature. A series farther than RADIUS_MATCH away is not a repeat."""
    s = lambda mean, sd, radius: dict(mean=mean, sd=sd, n=5, radius=radius, by_direction={}, peak=0.5, white=0.5)
    summaries = {"center_501mm": {"center": s(0.369, 0.004, 0.15)},
                 "left_top_501mm": {"bottom left": s(0.265, 0.008, 0.71), "top left": s(0.310, 0.016, 0.76)},
                 "left_bottom_501mm": {"bottom left": s(0.160, 0.024, 0.69), "top left": s(0.167, 0.004, 0.58)}}
    v = mtf.datasheet_values(summaries, "center_501mm")
    bl, tl = v["corners"]["bottom left"], v["corners"]["top left"]
    sd_sys = np.std([0.265, 0.160], ddof=1)
    assert bl["mean"] == 0.265 and bl["sd_sys"] == pytest.approx(sd_sys)
    assert bl["sd"] == pytest.approx(np.hypot(0.008, sd_sys))
    assert sorted(bl["repeats"]) == ["left_bottom_501mm", "left_top_501mm"]
    assert tl["sd_sys"] is None and tl["sd"] == 0.016          # 0.58 is not within 0.05 of 0.76
    assert v["corner_mean"]["sd"] == pytest.approx(np.hypot(bl["sd"], tl["sd"]) / 2)
