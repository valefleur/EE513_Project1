"""Tests for my_library/color_measurement.py using cases with known answers."""
import numpy as np
import pytest

import my_library.color_measurement as cm
import my_library.white_balance as wb

_, REF_XYZ, _ = wb.load_reference()
# A made-up camera: its RGB is an exact linear mix of XYZ, with unequal channel gains.
CAMERA = np.array([[0.9, 0.3, -0.1], [-0.2, 1.1, 0.1], [0.05, -0.15, 0.8]]) * [[1.4], [1.0], [0.6]]


def synthetic_chart(camera=CAMERA, exposure=0.37):
    return exposure * REF_XYZ @ np.linalg.inv(camera).T


def test_exact_linear_camera_is_corrected_to_zero_error():
    usable = np.ones(24, bool)
    c = cm.correct_chart(synthetic_chart(), usable, REF_XYZ)
    assert c["de_after"].max() < 1e-6
    assert c["de_before"].mean() > 1.0            # the uncorrected camera really is wrong
    assert c["gray_idx"] + 1 == wb.GRAY_PREFERENCE[0]   # first-choice gray when all usable


def test_unusable_patch_is_excluded_from_the_fit():
    means, usable = synthetic_chart(), np.ones(24, bool)
    means[2] *= [3.0, 0.2, 1.0]                   # corrupt patch 3 (as clipping might)
    usable[2] = False
    c = cm.correct_chart(means, usable, REF_XYZ)
    assert c["de_after"][usable].max() < 1e-6     # fit unaffected by the bad patch
    assert c["de_after"][2] > 1.0                 # and the bad patch really is bad


def test_no_usable_gray_returns_none():
    usable = np.ones(24, bool)
    usable[19:23] = False                         # patches 20-23, every gray choice
    assert cm.correct_chart(synthetic_chart(), usable, REF_XYZ) is None


def _chart(before_mean, after_values, usable=None):
    """Minimal per-chart result for summarize_round tests. All patches usable unless
    a usable mask is given."""
    after = np.asarray(after_values, float)
    usable = np.ones(24, bool) if usable is None else np.asarray(usable, bool)
    return dict(de_before=np.full(24, before_mean), de_after=after, usable=usable)


def test_summarize_round_mean_and_sample_std_across_charts():
    a = np.ones(24); a[5] = 4.0                   # chart 1: worst patch 6
    b = np.ones(24) * 2; b[5] = 6.0               # chart 2: worst patch 6
    s = cm.summarize_round([_chart(10.0, a), _chart(14.0, b)])
    assert s["mean_before"]["mean"] == pytest.approx(12.0)
    assert s["mean_before"]["std"] == pytest.approx(np.std([10.0, 14.0], ddof=1))
    assert s["worst_after"]["mean"] == pytest.approx(5.0)
    assert s["worst_patch"].startswith("6 ") and s["worst_patch_count"] == 2


def test_summarize_round_ignores_unusable_patches():
    a = np.ones(24); a[3] = 50.0                  # a clipped patch with a huge error
    usable = np.ones(24, bool); usable[3] = False
    s = cm.summarize_round([_chart(10.0, a, usable), _chart(10.0, np.ones(24) * 3)])
    assert s["mean_after"]["mean"] == pytest.approx(2.0)     # (1 + 3) / 2, patch 4 left out
    assert s["worst_after"]["mean"] == pytest.approx(2.0)
    assert not s["worst_patch"].startswith("4 ")


def test_one_chart_has_no_uncertainty():
    s = cm.summarize_round([_chart(10.0, np.ones(24))])
    assert s["mean_after"]["std"] is None
    assert cm.results_entries(s, matrix="per chart")["delta_e_mean_after_ccm"] == [1.0, None]


def test_results_entries_keys_and_shape():
    s = cm.summarize_round([_chart(10.0, np.ones(24)), _chart(12.0, np.ones(24) * 3)])
    e = cm.results_entries(s, matrix="per chart")
    assert e["delta_e_metric"] == "CIEDE2000"
    assert e["colorchecker_reference_set"] == wb.REFERENCE
    assert e["delta_e_mean_after_ccm"] == [2.0, round(np.std([1.0, 3.0], ddof=1), 2)]


def test_chart_catalog_drops_white_sheets_and_numbers_charts(tmp_path, capsys):
    from tests.synthetic_dng import write_dng
    for rnd in ("a", "b"):
        (tmp_path / rnd).mkdir()
        for i in range(2):
            write_dng(tmp_path / rnd / f"IMG_{i}.dng")
        write_dng(tmp_path / rnd / "IMG_9_white.dng")
    catalog, ident, n = cm.chart_catalog({"a": tmp_path / "a", "b": tmp_path / "b"}, 4095)
    assert n == 4 and not catalog["file"].str.contains("_white").any()
    assert catalog.groupby("round")["chart"].apply(list).to_dict() == {"a": [1, 2], "b": [1, 2]}
    assert "all images from the same camera" in capsys.readouterr().out   # lens_id assertion


# ------------------------------------------------ HW4 feedback: exposure from six grays

def _gray_de(c):
    return c["de_after"][19:23].mean()                  # patches 20-23


def test_white_reading_low_is_detected_and_six_gray_exposure_absorbs_it():
    """An exact linear camera whose white patch reads 12% low (the HW4 situation):
    the gray ratios all come out ~+13.6% high, the white is flagged, and exposure from
    the six grays gives much smaller gray errors than exposure from patch 19."""
    means = synthetic_chart()
    means[18] *= 0.88
    usable = np.ones(24, bool)
    white = cm.correct_chart(means, usable, REF_XYZ, exposure="white")
    grays = cm.correct_chart(means, usable, REF_XYZ, exposure="grays")
    for p, r in grays["gray_ratios"].items():
        assert r["deviation"] == pytest.approx(1 / 0.88 - 1, abs=0.01), p   # green ~ Y to ~1%
    assert grays["white_reads"] < -0.05
    assert _gray_de(white) > 1.0
    assert _gray_de(grays) < 0.5 * _gray_de(white)


def test_exact_camera_is_exact_with_either_exposure_method():
    usable = np.ones(24, bool)
    for method in cm.EXPOSURE_METHODS:
        c = cm.correct_chart(synthetic_chart(), usable, REF_XYZ, exposure=method)
        assert c["de_after"].max() < 1e-6, method
        assert all(abs(r["deviation"]) < 0.01 for r in c["gray_ratios"].values())   # green ~ Y


def test_exposure_scale_from_grays_is_the_relative_least_squares_solution():
    """With green_i / Y_i = r_i the scale is sum(r) / sum(r^2); for r all equal to 2
    that is 1/2, whatever the brightness of each neutral."""
    balanced = np.zeros((24, 3))
    balanced[18:24, 1] = 2 * REF_XYZ[18:24, 1]
    assert cm.exposure_scale(balanced, REF_XYZ, np.ones(24, bool), "grays") == pytest.approx(0.5)


def test_pooled_matrix_equals_per_chart_for_identical_charts_and_recovers_the_camera():
    usable = np.ones(24, bool)
    charts = [dict(cm.correct_chart(synthetic_chart(), usable, REF_XYZ), usable=usable) for _ in range(3)]
    pooled = cm.fit_pooled_matrix(charts, REF_XYZ)
    assert np.allclose(pooled, charts[0]["M"], atol=1e-9)


def test_summary_reports_standard_error_of_the_mean():
    s = cm.summarize_round([_chart(10.0, np.ones(24)), _chart(12.0, np.ones(24) * 3),
                            _chart(11.0, np.ones(24) * 2)])
    sd = s["mean_after"]["std"]
    assert s["mean_after"]["sem"] == pytest.approx(sd / np.sqrt(3))



# ------------------------------------------------ flare, per-patch table, pooled datasheet values

def test_flare_fit_recovers_a_known_offset_and_a_low_white():
    """Add flare of 2% of white to every patch, then make the white read 8% low: the fit
    finds the 2% offset and the white's -8% beyond it."""
    usable = np.ones(24, bool)
    c = cm.correct_chart(synthetic_chart(), usable, REF_XYZ)
    balanced = c["scaled"].copy()
    white_level = balanced[18, 1]
    balanced += 0.02 * white_level
    balanced[18] *= 0.92
    f = cm.flare_fit(balanced, REF_XYZ, usable)
    assert f["flare"] == pytest.approx(0.02, abs=0.005)            # the offset, as a fraction of white
    assert f["white_residual"] == pytest.approx(-0.08, abs=0.01)   # the white's own error


def test_flare_fit_is_zero_for_an_exact_camera_and_needs_three_neutrals():
    usable = np.ones(24, bool)
    c = cm.correct_chart(synthetic_chart(), usable, REF_XYZ)
    assert abs(c["flare"]["flare"]) < 0.01 and abs(c["flare"]["white_residual"]) < 0.01
    few = usable.copy(); few[20:24] = False                         # only patch 20 left of 20-24
    assert cm.flare_fit(c["scaled"], REF_XYZ, few) is None


def test_neutral_diagnostics_when_white_is_clipped_on_every_chart():
    """The fluorescent case: no ratios to white, no warning, no nan in the white reading."""
    usable = np.ones(24, bool); usable[18] = False
    charts = [dict(cm.correct_chart(synthetic_chart(), usable, REF_XYZ), usable=usable) for _ in range(3)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = cm.neutral_diagnostics(charts)
    assert d["white_reads"] is None and d["deviation"] == {} and d["n_with_white"] == 0
    assert np.isfinite(d["flare"]) and np.isnan(d["white_residual"])


def test_per_patch_table_has_24_rows_and_skips_unusable_patches():
    usable = np.ones(24, bool); usable[5] = False
    charts = [dict(cm.correct_chart(synthetic_chart(), usable, REF_XYZ), usable=usable) for _ in range(3)]
    pooled = cm.fit_pooled_matrix(charts, REF_XYZ)
    t = cm.per_patch_table(charts, REF_XYZ, pooled)
    assert len(t) == 24 and t.loc[5, "n"] == 0 and t.loc[0, "n"] == 3
    assert t["dE_after_pooled_mean"].drop(index=5).max() < 1e-6        # exact camera


def test_results_entries_use_the_pooled_matrix_by_default():
    s = cm.summarize_round([_chart(10.0, np.ones(24)), _chart(12.0, np.ones(24) * 3)])
    s.update(pooled_after=dict(mean=2.5, std=0.1), pooled_worst=dict(mean=6.0, std=0.4))
    e = cm.results_entries(s)
    assert e["delta_e_mean_after_ccm"] == [2.5, 0.1] and e["delta_e_worst_after_ccm"] == [6.0, 0.4]
    assert e["delta_e_mean_before_ccm"][0] == pytest.approx(11.0)
