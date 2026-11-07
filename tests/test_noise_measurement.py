"""Tests for my_library/noise_measurement.py: dynamic range from one photon transfer fit."""
import numpy as np
import pytest

import my_library.noise_measurement as nm
import my_library.results as res
from tests.synthetic_dng import write_dng

K_TRUE, READ_E_TRUE = 0.2567, 6.99           # DN per electron, electrons (HW3 values)
BLACK, WHITE = 100, 4095                      # the synthetic DNG's levels


# ------------------------------------------------ analytic cases

def test_professors_numbers():
    """HW3 feedback: 3567 / 0.2567 = 13,900 e- and 11.0 stops."""
    span, c = 3567.0, (K_TRUE * READ_E_TRUE) ** 2
    cap, _ = nm.saturation_capacity_e(span, K_TRUE, 0.0)
    dr, _ = nm.dynamic_range_from_fit(span, c, 0.0)
    assert cap == pytest.approx(13896, abs=1)
    assert dr == pytest.approx(10.96, abs=0.01)


@pytest.mark.parametrize("K", [0.1, 0.2567, 1.0])
def test_gain_cancels(K):
    """log2(capacity / read noise) in electrons equals log2(span / sqrt(c)) for any K."""
    span, c = 3567.0, 3.22
    cap, _ = nm.saturation_capacity_e(span, K, 0.0)
    read_e = np.sqrt(c) / K
    assert nm.dynamic_range_from_fit(span, c, 0.0)[0] == pytest.approx(np.log2(cap / read_e))


def test_uncertainty_matches_a_numerical_derivative():
    span, c, sc = 3567.0, 3.22, 0.4
    h = 1e-6
    slope = (nm.dynamic_range_from_fit(span, c + h, 0)[0] - nm.dynamic_range_from_fit(span, c - h, 0)[0]) / (2 * h)
    assert nm.dynamic_range_from_fit(span, c, sc)[1] == pytest.approx(abs(slope) * sc, rel=1e-6)


@pytest.mark.parametrize("target_snr", [1.0, 10.0])
def test_signal_at_snr_solves_its_equation(target_snr):
    K, c = 0.2567, 3.22
    S = nm.signal_at_snr(K, c, target_snr=target_snr)
    assert S / np.sqrt(K * S + c) == pytest.approx(target_snr)


@pytest.mark.parametrize("target_snr", [1.0, 10.0])
def test_dn_and_electron_forms_agree(target_snr):
    """The DN form (variance K S + c) and the electron form (variance mu + sigma_r^2)
    give the same floor: S / K equals the electron root."""
    K, read_e = 0.2567, 6.99
    c = (K * read_e) ** 2
    mu = (target_snr ** 2 + np.sqrt(target_snr ** 4 + 4 * target_snr ** 2 * read_e ** 2)) / 2
    assert nm.signal_at_snr(K, c, target_snr) / K == pytest.approx(mu)
    assert mu / np.sqrt(mu + read_e ** 2) == pytest.approx(target_snr)


def test_exact_snr1_floor_is_above_the_read_noise():
    """With shot noise counted, SNR = 1 needs a little more signal than sqrt(c), so the
    exact DR is slightly smaller; with no shot noise (K -> 0) the two agree."""
    span, K, c = 3567.0, 0.2567, 3.22
    assert nm.dynamic_range_at_snr(span, K, c) < nm.dynamic_range_from_fit(span, c, 0)[0]
    assert nm.dynamic_range_at_snr(span, 1e-12, c) == pytest.approx(nm.dynamic_range_from_fit(span, c, 0)[0])


def test_decibels_and_stops_describe_the_same_ratio():
    for ratio in (2.0, 1988.0, 13896 / 6.99):
        assert nm.stops_to_db(np.log2(ratio)) == pytest.approx(20 * np.log10(ratio))
    assert nm.DB_PER_STOP == pytest.approx(6.0206, abs=1e-4)


def test_table_rows_and_where_each_floor_sits():
    """HW3 numbers: knee at sigma_r^2 = 48.9 e- (SNR 4.94 there). The exact SNR = 1 floor
    (7.5 e-) is below the knee; SNR = 10 (136 e-) is above it."""
    K, read_e, span = 0.2567, 6.99, 3567.0
    c = (K * read_e) ** 2
    rows, knee = nm.dynamic_range_table(span, K, c, 0.1)
    assert knee["knee_e"] == pytest.approx(read_e ** 2)
    assert knee["snr_at_knee"] == pytest.approx(read_e / np.sqrt(2))
    floor, exact, ten = rows
    assert floor["floor_e"] == pytest.approx(read_e) and floor["sd_stops"] is not None
    assert exact["floor_e"] == pytest.approx((1 + np.sqrt(1 + 4 * read_e ** 2)) / 2)
    assert exact["snr_at_floor"] == pytest.approx(1.0) and ten["snr_at_floor"] == pytest.approx(10.0)
    assert ten["floor_e"] == pytest.approx((100 + np.sqrt(100 ** 2 + 400 * read_e ** 2)) / 2)
    assert (exact["region"], ten["region"]) == ("below the knee", "above the knee")
    for r in rows:
        assert r["db"] == pytest.approx(20 * np.log10(r["ratio"]))
    assert floor["stops"] > exact["stops"] > ten["stops"]


def test_non_positive_intercept_gives_no_dynamic_range():
    assert np.isnan(nm.dynamic_range_from_fit(3567.0, -11.3, 7.2)[0])
    summary = dict(dr=float("nan"), sd_dr=float("nan"), exact_snr1=float("nan"), dr_db=float("nan"))
    assert nm.results_entries(summary) == {"dynamic_range_stops": None}


# ------------------------------------------------ simulated sensor, end to end

def _simulated_flat_fields(folder, levels_dn, size=320, iso=64, prnu=0.01, seed=5):
    """Pairs of flat-field DNGs from a Poisson-Gaussian sensor with known K and read noise.
    A fixed 1% pixel gain pattern (PRNU) is the same in both frames of every pair."""
    rng = np.random.default_rng(seed)
    gain_pattern = 1 + prnu * rng.standard_normal((size, size))
    folder.mkdir(parents=True, exist_ok=True)
    for i, level in enumerate(levels_dn):
        for frame in (0, 1):
            electrons = rng.poisson(level / K_TRUE * gain_pattern)
            dn = BLACK + K_TRUE * electrons + rng.normal(0, K_TRUE * READ_E_TRUE, (size, size))
            raw = np.clip(np.rint(dn), 0, WHITE).astype(np.uint16)
            write_dng(folder / f"IMG_{i:02d}_{frame}.dng", raw_data=raw, crop=(size, size),
                      exposure_s=(i + 1) / 1000, iso=iso)
    return folder


LEVELS = [8, 15, 30, 60, 110, 180, 260, 340, 800, 1600, 3000]   # DN above black


def test_measure_ptc_recovers_the_simulated_sensor(tmp_path, capsys):
    folder = _simulated_flat_fields(tmp_path / "flats", LEVELS)
    result = nm.measure_ptc(folder, WHITE, BLACK)
    fit = result["fit"]
    assert result["n_used"] == 8 and result["iso"] == 64
    assert abs(fit["K"] - K_TRUE) < 4 * fit["sK"]
    # Rounding to whole DN adds 1/12 DN^2 to the read-noise variance.
    c_true = (K_TRUE * READ_E_TRUE) ** 2 + 1 / 12
    summary = nm.report_dynamic_range(result)
    dr_true = np.log2((WHITE - BLACK) / np.sqrt(c_true))
    assert abs(summary["dr"] - dr_true) < 4 * summary["sd_dr"]
    out = capsys.readouterr().out
    assert "all images from the same camera" in out and "K cancels" in out
    results, template = res.load()
    results.update(nm.results_entries(summary))
    assert res.validate(results, template) == []


def test_measure_ptc_rejects_mixed_iso(tmp_path):
    _simulated_flat_fields(tmp_path / "flats", LEVELS[:3], iso=64)
    _simulated_flat_fields(tmp_path / "flats" / "z", LEVELS[:2], iso=320, seed=6)
    with pytest.raises(SystemExit):
        nm.measure_ptc(tmp_path / "flats", WHITE, BLACK)


# ------------------------------------------------ real data (opt-in, see conftest.py)

def test_real_flat_fields_give_a_plausible_dynamic_range(real_flat_fields):
    """No ground truth: a positive intercept below the cutoff, a gain and read noise in
    a physical range, and a dynamic range typical of a phone sensor."""
    result = nm.measure_ptc(real_flat_fields, 4095, 528)
    summary = nm.report_dynamic_range(result)
    assert summary["c"] > 0
    assert 0.05 < summary["K"] < 2.0 and 0.5 < summary["read_e"] < 50
    assert 8 < summary["dr"] < 14
