"""pytest options to run the core-algorithm tests on real data as well as simulated.

Every core-algorithm test runs on simulated data (a synthetic camera, rendered
boards, synthetic DNGs) with known answers. These options add tests that run the same
algorithms on your own captures, where the answer is not known but must be plausible.
Run from the repository root:

    uv run pytest --calibration-images images/iPhone16/Lightroom/Checkerboard
    uv run pytest --calibration-images <folder> --calibration-exposure 1/125
    uv run pytest --sample-dng <one DNG from the camera>
    uv run pytest --flat-fields images/iPhone16/Lightroom/ISO64_Sunset

Without an option, its tests are skipped.

Why this file is at the repository root and not in tests/: pytest reads the command
line before it collects tests, so options must be registered by a conftest.py that it
loads first. When an option is not yet known, "--sample-dng images/x.dng" is read as an
unknown flag followed by a test path, and pytest then loads only conftests on that
path's way up to the root; a conftest inside tests/ is never among them, so the option
stays "unrecognized". The root is above every path in the repository, so here it is
always loaded. For a path outside the repository, write the option as one token,
--sample-dng=/path/to/file.dng. (pytest docs: pytest_addoption "should be implemented
only in plugins or conftest.py files situated at the tests root directory".)
The import path for the tests still comes from tests/__init__.py.
"""
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("ee513", "EE 513 real-data tests")
    group.addoption("--calibration-images", type=Path, default=None,
                    help="folder of real checkerboard DNGs for the geometry tests")
    group.addoption("--calibration-exposure", default=None,
                    help="only use calibration images at this exposure, e.g. 1/125")
    group.addoption("--sample-dng", type=Path, default=None,
                    help="one real DNG for the camera-identity tests")
    group.addoption("--flat-fields", type=Path, default=None,
                    help="folder of real flat-field pairs for the dynamic-range test")


def _option_or_skip(request, name):
    value = request.config.getoption(name)
    if value is None:
        pytest.skip(f"real-data test: pass --{name.replace('_', '-')} to run it")
    if not Path(value).exists():
        pytest.fail(f"--{name.replace('_', '-')} {value} does not exist")
    return Path(value)


@pytest.fixture
def real_calibration_images(request):
    return _option_or_skip(request, "calibration_images")


@pytest.fixture
def real_calibration_exposure(request):
    return request.config.getoption("calibration_exposure")


@pytest.fixture
def real_sample_dng(request):
    return _option_or_skip(request, "sample_dng")


@pytest.fixture
def real_flat_fields(request):
    return _option_or_skip(request, "flat_fields")
