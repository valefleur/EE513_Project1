"""Tests for project1.py's command line and results writing (no images needed)."""
import json

import pytest

import project1


@pytest.fixture
def fake_steps(monkeypatch):
    """Replace the real measurement steps with ones that record their arguments."""
    calls = []
    def color(args):
        calls.append(args)
        return {"delta_e_mean_after_ccm": [1.23, 0.04]}
    def geometry(args):
        calls.append(args)
        return {"focal_length_px": [2800.0, 3.0]}
    monkeypatch.setattr(project1, "MEASUREMENTS", {"geometry": geometry, "color": color})
    return calls


def parse(argv):
    parser = project1.build_parser()
    args = parser.parse_args(argv)
    project1.resolve_datasheet_illuminant(parser, args)
    return args


def test_no_flags_selects_every_measurement():
    assert project1.selected_steps(parse([])) == list(project1.MEASUREMENTS)


def test_color_flag_selects_only_color():
    assert project1.selected_steps(parse(["--color"])) == ["color"]


def test_geometry_flag_and_step_order():
    assert project1.selected_steps(parse(["--geometry"])) == ["geometry"]
    assert project1.selected_steps(parse(["--color", "--geometry"])) == ["geometry", "color"]


def test_geometry_option_defaults():
    args = parse([])
    assert args.corner_detector == "sb" and not args.free_k3
    assert args.square_mm == pytest.approx(20.01) and args.square_mm_sd is None
    assert args.size_from == "rawpy_visible" and args.calibration_images is None


def test_unknown_illuminant_is_rejected():
    with pytest.raises(SystemExit):
        parse(["--datasheet-illuminant", "moonlight"])


def test_datasheet_illuminant_defaults_to_constant():
    assert parse([]).datasheet_illuminant == project1.DATASHEET_ILLUMINANT


def test_explicit_datasheet_illuminant_must_be_measured():
    with pytest.raises(SystemExit):
        parse(["--illuminant-rounds", "led", "--datasheet-illuminant", "shade"])


def test_default_datasheet_illuminant_not_measured_is_skipped(monkeypatch):
    monkeypatch.setattr(project1, "DATASHEET_ILLUMINANT", "led")
    assert parse(["--illuminant-rounds", "shade"]).datasheet_illuminant is None


def test_raw_level_defaults_and_override():
    args = parse([])
    assert args.black_level == 528 and args.white_level == 4095
    assert parse(["--black-level", "527.46"]).black_level == pytest.approx(527.46)


def test_main_writes_step_and_raw_level_entries(tmp_path, fake_steps):
    out = tmp_path / "results.json"
    assert project1.main(["--results", str(out), "--black-level", "527.46"]) == 0
    written = json.loads(out.read_text())
    assert written["delta_e_mean_after_ccm"] == [1.23, 0.04]
    assert written["black_level_dn"] == [527.46, None]     # no uncertainty yet: stays open
    assert written["white_level_dn"] == 4095
    assert fake_steps[0].datasheet_illuminant == project1.DATASHEET_ILLUMINANT


def test_no_results_leaves_file_untouched(tmp_path, fake_steps):
    out = tmp_path / "results.json"
    project1.main(["--results", str(out), "--no-results"])
    assert len(fake_steps) == 2 and not out.exists()


def test_percent_complete_is_printed(tmp_path, fake_steps, capsys):
    project1.main(["--results", str(tmp_path / "r.json")])
    import re
    assert re.search(r"\d+ of \d+ entries complete \(\d+%\)", capsys.readouterr().out)
