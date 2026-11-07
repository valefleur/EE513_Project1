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


def test_weighted_fit_recovers_an_exact_line():
    S = np.array([8, 15, 30, 60, 110, 180, 260, 340, 800.0])
    var = 0.2567 * S + 3.22
    fit = nm.fit_ptc_weighted(S, var, 3567.0, 0.10, 16384)
    assert fit["K"] == pytest.approx(0.2567) and fit["c"] == pytest.approx(3.22)
    assert fit["used"].sum() == 8 and fit["chi2_red"] < 1e-6


def test_weighted_fit_inflates_uncertainty_when_scatter_exceeds_the_model():
    """Points scattered 10x more than sampling error allows: chi2_red >> 1, and the
    reported uncertainties grow by sqrt(chi2_red) instead of staying optimistic."""
    rng = np.random.default_rng(0)
    S = np.linspace(8, 340, 12)
    true = 0.2567 * S + 3.22
    clean = true * (1 + rng.normal(0, np.sqrt(2 / 16383), S.size))
    noisy = true * (1 + rng.normal(0, 10 * np.sqrt(2 / 16383), S.size))
    f_clean = nm.fit_ptc_weighted(S, clean, 3567.0, 0.10, 16384)
    f_noisy = nm.fit_ptc_weighted(S, noisy, 3567.0, 0.10, 16384)
    assert f_noisy["chi2_red"] > 20
    assert f_noisy["sK"] > 4 * f_clean["sK"]


def test_non_positive_intercept_gives_no_dynamic_range():
    assert np.isnan(nm.dynamic_range_from_fit(3567.0, -11.3, 7.2)[0])
    summary = dict(dr=float("nan"), sd_dr=float("nan"), exact_snr1=float("nan"), dr_db=float("nan"))
    assert nm.results_entries(summary) == {"dynamic_range_stops": None}


# ------------------------------------------------ simulated sensor, end to end

def _simulated_flat_fields(folder, levels_dn, size=320, iso=64, prnu=0.01, seed=5,
                           drift=0.0, shading=0.0, ramp=0.0, noise_profile=None):
    """Pairs of flat-field DNGs from a Poisson-Gaussian sensor with known K and read noise.

    A fixed pixel gain pattern (PRNU, relative sd prnu) and an optional radial shading
    (relative fall-off `shading` at the patch corner) are the same in both frames of
    every pair. drift: the second frame of each pair gets (1 - drift) of the light (a
    uniform change). ramp: the second frame's light also changes by a gradient across
    x, from -ramp/2 to +ramp/2 (a non-uniform change, like HW3's).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = ((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2) / ((size - 1) / 2) ** 2 / 2
    gain_pattern = (1 + prnu * rng.standard_normal((size, size))) * (1 - shading * r2)
    folder.mkdir(parents=True, exist_ok=True)
    for i, level in enumerate(levels_dn):
        for frame in (0, 1):
            light = level * (1 - drift * frame) * (1 + frame * ramp * (xx / (size - 1) - 0.5))
            electrons = rng.poisson(light / K_TRUE * gain_pattern)
            dn = BLACK + K_TRUE * electrons + rng.normal(0, K_TRUE * READ_E_TRUE, (size, size))
            raw = np.clip(np.rint(dn), 0, WHITE).astype(np.uint16)
            write_dng(folder / f"IMG_{i:02d}_{frame}.dng", raw_data=raw, crop=(size, size),
                      exposure_s=(i + 1) / 1000, iso=iso, noise_profile=noise_profile)
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


# ------------------------------------------------ noise step: simulated, known answers

def _flat_pair(S, size=256, prnu=0.01, shading=0.0, drift=0.0, ramp=0.0, seed=0):
    """One pair of black-subtracted-free patches (black = 0) with known physics.
    ramp: an extra light gradient in the second frame only (fractional change across x)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = ((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2) / ((size - 1) / 2) ** 2 / 2
    pattern = (1 + prnu * rng.standard_normal((size, size))) * (1 - shading * r2)
    def frame(scale):
        e = rng.poisson(S / K_TRUE * pattern * scale)
        return K_TRUE * e + rng.normal(0, K_TRUE * READ_E_TRUE, (size, size))
    return frame(1.0), frame((1 - drift) * (1 + ramp * (xx / (size - 1) - 0.5)))


def test_a_uniform_drift_barely_changes_the_pair_variance():
    """0.6% uniform drift, 1% PRNU, 5% shading at S = 2000 DN: the residual eps x pattern is
    ~0.01% of S, so plain and gain-matched variances agree with the true K S + c."""
    a, b = _flat_pair(2000.0, prnu=0.01, shading=0.05, drift=0.006)
    expected = K_TRUE * a.mean() + (K_TRUE * READ_E_TRUE) ** 2
    plain = np.var(a - b) / 2
    matched, eps = nm.gain_matched_pair_variance(a, b, 0.0)
    assert eps == pytest.approx(0.006 / 0.994, abs=0.0005)
    assert plain == pytest.approx(expected, rel=0.02) and matched == pytest.approx(expected, rel=0.02)


def test_gain_matching_matters_only_with_a_strong_pattern():
    """The math's limit case: a 30% shading pattern and a 5% change leave eps x pattern
    ~ 1.5% of S in the difference; matching removes it."""
    a, b = _flat_pair(2000.0, prnu=0.0, shading=0.6, drift=0.05, seed=1)
    expected = K_TRUE * 0.5 * (a.mean() + b.mean()) + (K_TRUE * READ_E_TRUE) ** 2
    matched, _ = nm.gain_matched_pair_variance(a, b, 0.0)
    assert np.var(a - b) / 2 > 1.1 * expected                # measured: about 19% high
    assert matched == pytest.approx(expected, rel=0.03)


def test_plane_removal_fixes_a_gradient_change_and_matching_does_not():
    """A 2% light gradient across the patch in the second frame (the HW3 kind of drift)."""
    a, b = _flat_pair(2000.0, ramp=0.02, prnu=0.0, seed=5)
    expected = K_TRUE * 0.5 * (a.mean() + b.mean()) + (K_TRUE * READ_E_TRUE) ** 2
    assert np.var(a - b) / 2 > 1.08 * expected               # measured: about 12% high
    assert nm.gain_matched_pair_variance(a, b, 0.0)[0] > 1.08 * expected
    assert nm.plane_removed_pair_variance(a, b) == pytest.approx(expected, rel=0.03)


def test_plane_removal_does_not_bias_pure_noise():
    a, b = _flat_pair(500.0, prnu=0.01, seed=8)
    assert nm.plane_removed_pair_variance(a, b) == pytest.approx(np.var(a - b) / 2, rel=0.01)


def test_difference_structure_tells_noise_uniform_gradient_and_blotches_apart():
    noise = nm.difference_structure(*_flat_pair(2000.0, prnu=0.01, seed=3), black=0.0)
    assert 0.8 < noise["R_plain"] < 1.25
    uniform = nm.difference_structure(*_flat_pair(2000.0, drift=0.01, prnu=0.01, shading=0.1, seed=4), black=0.0)
    assert uniform["R_plain"] < 1.25                       # a uniform change barely shows
    ramp = nm.difference_structure(*_flat_pair(2000.0, ramp=0.02, prnu=0.0, seed=5), black=0.0)
    assert ramp["R_plain"] > 1.5 and ramp["R_plane"] < 1.25
    gx, _ = ramp["gradient_dn"]
    assert gx * 255 == pytest.approx(-0.02 * 2000, rel=0.15)   # the 2% ramp, in DN
    a, b = _flat_pair(2000.0, prnu=0.0, seed=6)
    yy, xx = np.mgrid[0:256, 0:256]
    b = b + 60 * np.exp(-((xx - 90) ** 2 + (yy - 160) ** 2) / (2 * 20 ** 2))   # a blotch
    blotch = nm.difference_structure(a, b, 0.0)
    assert blotch["R_plane"] > 1.5                         # survives plane removal


def test_single_frame_bias_recovers_the_pattern_and_steepens_the_slope(tmp_path):
    folder = _simulated_flat_fields(tmp_path / "flats", LEVELS, prnu=0.01)
    result = nm.measure_ptc(folder, WHITE, BLACK)
    single = nm.single_frame_bias(result)
    assert single["p"] == pytest.approx(0.01, rel=0.2)
    assert single["K_ratio"] > 1.0


def test_vendor_noise_profile_converts_back_to_k_and_read_noise(tmp_path):
    """Build S = K / span and O = (K sigma_r / span)^2; the conversion must return K and sigma_r."""
    span = WHITE - BLACK
    S, O = K_TRUE / span, (K_TRUE * READ_E_TRUE / span) ** 2
    path = write_dng(tmp_path / "np.dng", noise_profile=(S, O), exposure_s=0.01, iso=64)
    v = nm.vendor_noise_profile(path, K_TRUE, READ_E_TRUE)
    assert v["green"]["K"] == pytest.approx(K_TRUE) and v["green"]["read"] == pytest.approx(READ_E_TRUE)
    assert v["K_ratio"] == pytest.approx(1.0) and v["read_ratio"] == pytest.approx(1.0)
    assert nm.vendor_noise_profile(write_dng(tmp_path / "none.dng"), K_TRUE, READ_E_TRUE) is None


def _simulated_dark_frames(folder, rate_e_s=20.0, exposures=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0), seed=3):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for t in exposures:
        for k in range(2):
            e = rng.poisson(rate_e_s * t, (320, 320))
            raw = np.clip(np.rint(BLACK + K_TRUE * e + rng.normal(0, K_TRUE * READ_E_TRUE, (320, 320))), 0, WHITE)
            write_dng(folder / f"D_{t:05.2f}_{k}.dng", raw_data=raw.astype(np.uint16),
                      crop=(320, 320), exposure_s=t, iso=64)
    return folder


def test_dark_current_and_zero_exposure_black(tmp_path, capsys):
    dark = nm.measure_dark_current(_simulated_dark_frames(tmp_path / "dark"), K_TRUE, 0.001)
    assert abs(dark["rate_e_s"] - 20.0) < 4 * dark["sd_rate_e_s"]
    assert dark["black_measured_dn"] == pytest.approx(BLACK, abs=0.05)   # intercept, not the 1/10 s mean
    assert "all images from the same camera" in capsys.readouterr().out


def test_report_noise_end_to_end_with_a_gradient_change(tmp_path):
    """A sensor whose second frames see a 2% light gradient (HW3-like drift), fitted up to
    70% of the span: the plain pair fit's K is pulled up; the plane-removed fit recovers
    the true K. Writes every figure and valid results.json entries."""
    folder = _simulated_flat_fields(tmp_path / "flats", LEVELS, prnu=0.01, shading=0.05, ramp=0.02,
                                    noise_profile=(K_TRUE / (WHITE - BLACK), (K_TRUE * READ_E_TRUE / (WHITE - BLACK)) ** 2))
    result = nm.measure_ptc(folder, WHITE, BLACK, cutoff=0.7, method="plane-removed")
    plain, plane = result["fits"]["pair difference"], result["fits"]["plane-removed"]
    assert abs(plane["K"] - K_TRUE) < 4 * plane["sK"]
    assert plain["K"] > plane["K"] + 4 * plane["sK"]
    summary = nm.report_noise(result, tmp_path / "out", dark_folder=_simulated_dark_frames(tmp_path / "dark"))
    assert summary["structure"]["R_plain"] > 1.5 and summary["structure"]["R_plane"] < 1.5
    assert summary["spectrum"]["n_pairs"] >= 10
    for name in ("noise_snr.png", "noise_difference_image.png", "noise_ptc_methods.png",
                 "noise_dark_current.png", "noise_pairs.csv", "noise_power_spectrum.png"):
        assert (tmp_path / "out" / name).exists(), name
    entries = nm.noise_results_entries(summary)
    results, template = res.load()
    results.update(entries)
    assert res.validate(results, template) == []
    assert entries["vendor_noiseprofile_present"] is True and entries["iso_of_noise_measurement"] == 64


def test_real_noise_measurement_is_plausible(real_flat_fields, request, tmp_path):
    """Opt-in: --flat-fields (and optionally --dark-frames). Positive intercept, physical
    gain and read noise, knee between 1 and 1000 electrons."""
    result = nm.measure_ptc(real_flat_fields, 4095, 528)
    dark = request.config.getoption("dark_frames")
    summary = nm.report_noise(result, tmp_path, dark_folder=dark)
    assert summary["c"] > 0 and 0.05 < summary["K"] < 2.0 and 1 < summary["knee_e"] < 1000



# ------------------------------------------------ neighbor correlation (sharpening clue)

def test_neighbor_correlation_is_zero_for_independent_noise():
    d = np.random.default_rng(0).normal(0, 3, (256, 256))
    rx, ry = nm.neighbor_correlation(d)
    assert abs(rx) < 0.02 and abs(ry) < 0.02
    assert nm.predicted_block_ratio(d) == pytest.approx(1.0, abs=0.05)


def test_autocorrelation_center_is_one_and_symmetric():
    d = np.random.default_rng(1).normal(0, 1, (128, 128))
    rho = nm.autocorrelation(d, 3)
    assert rho.shape == (7, 7) and rho[3, 3] == pytest.approx(1.0)
    assert np.allclose(rho, rho[::-1, ::-1])


def test_sharpening_gives_negative_correlation_and_r_below_one():
    """Apply the same sharpening to both frames (as a camera pipeline would): the noise
    in the difference becomes anti-correlated, R drops below 1, and the correlations
    predict the measured R."""
    import cv2 as cv
    a, b = _flat_pair(2000.0, prnu=0.0, seed=9)
    kernel = -0.12 * np.ones((3, 3)); kernel[1, 1] = 1 + 8 * 0.12     # unsharp-like, sums to 1
    a, b = (cv.filter2D(x, -1, kernel, borderType=cv.BORDER_REFLECT) for x in (a, b))
    st = nm.difference_structure(a, b, 0.0)
    rx, ry = st["rho"]
    assert rx < -0.05 and ry < -0.05
    assert st["R_plane"] < 0.9
    assert st["R_predicted"] == pytest.approx(st["R_plane"], abs=0.05)


def test_smoothing_gives_positive_correlation():
    import cv2 as cv
    a, b = _flat_pair(2000.0, prnu=0.0, seed=10)
    a, b = (cv.GaussianBlur(x, (0, 0), 0.7) for x in (a, b))
    rx, ry = nm.difference_structure(a, b, 0.0)["rho"]
    assert rx > 0.1 and ry > 0.1


# ------------------------------------------------ noise power spectrum

def _filtered_noise(kernel, n_images=30, size=64, seed=0):
    """White noise images passed through `kernel` with circular (FFT) convolution,
    and the exact normalized |H|^2 of that filter on the same frequency grid."""
    rng = np.random.default_rng(seed)
    k = np.zeros((size, size))
    kh, kw = kernel.shape
    k[:kh, :kw] = kernel
    k = np.roll(k, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    H2 = np.abs(np.fft.fft2(k)) ** 2
    images = [np.real(np.fft.ifft2(np.fft.fft2(rng.normal(0, 1, (size, size))) * np.fft.fft2(k)))
              for _ in range(n_images)]
    return images, np.fft.fftshift(H2 / H2.mean())


SHARPEN = -0.12 * np.ones((3, 3)); SHARPEN[1, 1] = 1 + 8 * 0.12


def test_spectrum_of_independent_noise_is_flat_at_one():
    images, _ = _filtered_noise(np.ones((1, 1)))
    spec = nm.noise_power_spectrum(images)
    assert spec["nps"].mean() == pytest.approx(1.0)              # Parseval normalization
    assert spec["ratio"] == pytest.approx(1.0, abs=0.08)
    assert np.allclose(spec["radial"][1], 1.0, atol=0.25)


def test_spectrum_recovers_a_known_sharpening_filter():
    """The averaged spectrum of filtered white noise is the filter's normalized |H|^2."""
    images, H2 = _filtered_noise(SHARPEN)
    spec = nm.noise_power_spectrum(images)
    f, measured = spec["radial"]
    fr = np.hypot(*np.meshgrid(spec["fx"], spec["fy"]))
    exact = [H2[(fr >= lo) & (fr < lo + 1 / 64)].mean() for lo in f - 0.5 / 64]
    assert np.allclose(measured, exact, rtol=0.12)
    assert spec["ratio"] > 1.3


def test_spectrum_and_autocorrelation_agree():
    """Wiener-Khinchin: the lag-1 correlation recovered from the spectrum equals the one
    measured directly in the image."""
    images, _ = _filtered_noise(SHARPEN, n_images=1, size=128)
    spec = nm.noise_power_spectrum(images)
    FX, FY = np.meshgrid(spec["fx"], spec["fy"])
    rho_x_from_spectrum = np.mean(spec["nps"] * np.cos(2 * np.pi * FX))
    rho_y_from_spectrum = np.mean(spec["nps"] * np.cos(2 * np.pi * FY))
    rho_x, rho_y = nm.neighbor_correlation(images[0])
    assert rho_x_from_spectrum == pytest.approx(rho_x, abs=0.01)
    assert rho_y_from_spectrum == pytest.approx(rho_y, abs=0.01)


def test_smoothing_makes_the_spectrum_fall():
    images, _ = _filtered_noise(np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16)
    assert nm.noise_power_spectrum(images)["ratio"] < 0.5


# ------------------------------------------------ dark-frame spectrum and ISO check

def _sharpen(x):
    import cv2 as cv
    return cv.filter2D(x, -1, SHARPEN, borderType=cv.BORDER_REFLECT)


def _dark_frames_filtered(folder, where=None, iso=64, n_exposures=4, per_exposure=4, size=320, seed=11):
    """Dark DNGs (read noise only). where=None: no filtering; "plane": sharpen each Bayer
    color plane separately (same-color neighbors coupled); "mosaic": sharpen the full
    mosaic (adjacent, different-color pixels coupled)."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n_exposures):
        t = 0.01 * (i + 1)
        for k in range(per_exposure):
            noise = rng.normal(0, K_TRUE * READ_E_TRUE, (size, size))
            if where == "mosaic":
                noise = _sharpen(noise)
            elif where == "plane":
                for r in (0, 1):
                    for c in (0, 1):
                        noise[r::2, c::2] = _sharpen(noise[r::2, c::2].copy())
            raw = np.clip(np.rint(BLACK + 50 + noise), 0, WHITE).astype(np.uint16)
            write_dng(folder / f"D_{i}_{k}.dng", raw_data=raw, crop=(size, size), exposure_s=t, iso=iso)
    return folder


def test_dark_spectrum_is_flat_without_filtering(tmp_path):
    spec = nm.dark_difference_spectrum(_dark_frames_filtered(tmp_path / "d"))
    assert spec["n_pairs"] == 8 and spec["iso"] == [64]
    assert spec["ratio"] == pytest.approx(1.0, abs=0.12)


def test_dark_spectrum_sees_filtering_within_a_color_plane(tmp_path):
    spec = nm.dark_difference_spectrum(_dark_frames_filtered(tmp_path / "d", where="plane"))
    assert spec["ratio"] > 1.3
    assert spec["rho"][0] < -0.05 and spec["rho"][1] < -0.05


def test_full_mosaic_sharpening_barely_shows_in_one_green_plane(tmp_path):
    """The clue in the real data: same-color neighbors are 2 full-res pixels apart, and a
    3x3 kernel on the full mosaic couples them only through k(-1) k(+1), a small positive
    term. So a green plane looks nearly independent; anti-correlated green-plane noise
    means the filtering acts within each color plane."""
    spec = nm.dark_difference_spectrum(_dark_frames_filtered(tmp_path / "d", where="mosaic"))
    assert 0.85 < spec["ratio"] < 1.15
    assert abs(spec["rho"][0]) < 0.04 and abs(spec["rho"][1]) < 0.04


def test_dark_current_is_skipped_when_iso_differs(tmp_path, capsys):
    folder = _simulated_flat_fields(tmp_path / "flats", LEVELS)
    result = nm.measure_ptc(folder, WHITE, BLACK)
    dark = _dark_frames_filtered(tmp_path / "dark", iso=320)
    summary = nm.report_noise(result, tmp_path / "out", dark_folder=dark)
    assert summary["dark"] is None
    assert "Dark current skipped" in capsys.readouterr().out
    assert summary["dark_spectrum"] is not None                       # the shape check still runs
    assert (tmp_path / "out" / "noise_power_spectrum_dark_vs_flat.png").exists()
    assert "dark_current_e_per_s" not in nm.noise_results_entries(summary)


def test_remove_plane_removes_an_exact_plane():
    yy, xx = np.mgrid[0:32, 0:32]
    assert np.abs(nm.remove_plane(3 + 0.2 * xx - 0.1 * yy)).max() < 1e-9
