"""
Marks the tests/ directory as a package.  Do not delete this file!

By default, pytest is in "pretent" import mode, which walks up from a package to the first
non-package folder and adds that to sys.path.  This is the repo root.  This is what lets the tests
import my_library and tests.synthetic_dng without updating PATH or needing a conftest.py.  Test
collection fails without this file.  Leave it as it is.
See https://docs.pytest.org/en/stable/goodpractices.html#test-discovery for more information.
"""
