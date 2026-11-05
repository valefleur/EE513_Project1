"""Tests for the camera identity functions in my_library/image_metadata.py."""
import pytest

import my_library.image_metadata as meta
from tests.synthetic_dng import write_dng


@pytest.mark.parametrize("cfa", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_cfa_from_tags_all_orders(cfa):
    from tests.synthetic_dng import CFA_CODES
    assert meta.cfa_from_tags((2, 2), CFA_CODES[cfa]) == cfa


def test_cfa_from_tags_rejects_non_bayer_tile():
    with pytest.raises(ValueError):
        meta.cfa_from_tags((6, 6), tuple(range(36)))


@pytest.mark.parametrize("cfa", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_identity_cfa_agrees_with_rawpy(tmp_path, cfa):
    ident = meta.camera_identity(write_dng(tmp_path / "a.dng", cfa=cfa))
    assert ident["cfa_pattern"] == cfa          # camera_identity raises if rawpy disagrees


def test_identity_reads_raw_ifd_not_preview(tmp_path):
    ident = meta.camera_identity(write_dng(tmp_path / "a.dng", size=(64, 48), crop=(56, 40)))
    assert ident["raw_ifd_where"] == "IFD0>SubIFD"
    assert ident["size_raw_ifd"] == (64, 48)    # not the 16 x 12 preview in IFD0
    assert ident["size_default_crop"] == (56, 40)
    assert ident["size_rawpy_visible"][0] <= 64
    assert ident["make"] == "TestMake" and ident["software"] == "TestApp 1.0"


def test_identity_rejects_linear_raw(tmp_path):
    with pytest.raises(ValueError, match="LinearRaw"):
        meta.camera_identity(write_dng(tmp_path / "pro.dng", photometric=34892))


def test_missing_lens_model_is_none_and_blocks_results_entries(tmp_path):
    ident = meta.camera_identity(write_dng(tmp_path / "a.dng"))
    assert ident["lens_model"] is None
    with pytest.raises(ValueError, match="LensModel"):
        meta.identity_results_entries(ident)


def test_results_entries_size_source(tmp_path):
    ident = meta.camera_identity(write_dng(tmp_path / "a.dng", crop=(56, 40)))
    ident["lens_model"] = "Test lens 1mm f/2"   # stand-in for the Exif value
    e = meta.identity_results_entries(ident, size_from="default_crop")
    assert e == {"camera": "Test lens 1mm f/2", "cfa_pattern": "BGGR",
                 "raw_width_px": 56, "raw_height_px": 40}
    with pytest.raises(ValueError):
        meta.identity_results_entries(ident, size_from="exif_guess")


def test_series_identity_accepts_matching_and_rejects_mixed(tmp_path):
    a = write_dng(tmp_path / "a.dng", cfa="BGGR")
    b = write_dng(tmp_path / "b.dng", cfa="BGGR")
    c = write_dng(tmp_path / "c.dng", cfa="RGGB")
    ident, n = meta.series_identity([a, b])
    assert n == 2 and ident["cfa_pattern"] == "BGGR"
    with pytest.raises(SystemExit):
        meta.series_identity([a, b, c])


def test_read_tiff_tags_unchanged_for_noise_profile(tmp_path):
    path = write_dng(tmp_path / "n.dng", noise_profile=(8.858e-5, 9.874e-7))
    assert meta.noise_profile(path) == [(8.858e-5, 9.874e-7)]
    vals, where = meta.read_tiff_tags(path)[meta.TAG_NOISE_PROFILE]
    assert where == "IFD0"
