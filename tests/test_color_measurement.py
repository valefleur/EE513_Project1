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
    assert cm.results_entries(s)["delta_e_mean_after_ccm"] == [1.0, None]


def test_results_entries_keys_and_shape():
    s = cm.summarize_round([_chart(10.0, np.ones(24)), _chart(12.0, np.ones(24) * 3)])
    e = cm.results_entries(s)
    assert e["delta_e_metric"] == "CIEDE2000"
    assert e["colorchecker_reference_set"] == wb.REFERENCE
    assert e["delta_e_mean_after_ccm"] == [2.0, round(np.std([1.0, 3.0], ddof=1), 2)]


def test_chart_catalog_drops_white_sheets_and_numbers_charts(tmp_path):
    from tests.synthetic_dng import write_dng
    for rnd in ("a", "b"):
        (tmp_path / rnd).mkdir()
        for i in range(2):
            write_dng(tmp_path / rnd / f"IMG_{i}.dng")
        write_dng(tmp_path / rnd / "IMG_9_white.dng")
    catalog, ident, n = cm.chart_catalog({"a": tmp_path / "a", "b": tmp_path / "b"}, 4095)
    assert n == 4 and not catalog["file"].str.contains("_white").any()
    assert catalog.groupby("round")["chart"].apply(list).to_dict() == {"a": [1, 2], "b": [1, 2]}
