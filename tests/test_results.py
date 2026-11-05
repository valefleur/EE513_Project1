"""Tests for my_library/results.py: results.json must keep the template's shape."""
import json

import pytest

import my_library.results as res


def test_committed_results_json_matches_template():
    results, template = res.load()
    assert res.validate(results, template) == []


def test_keys_are_in_template_order():
    results, template = res.load()
    assert list(results) == list(template)


def test_missing_and_extra_keys_are_reported():
    results, template = res.load()
    broken = dict(results)
    del broken["camera"]
    broken["not_a_key"] = 1
    errors = res.validate(broken, template)
    assert "missing key: camera" in errors
    assert "unexpected key: not_a_key" in errors


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], ["x", 0.1], [1.0, -0.1],
                                 [float("nan"), 0.1], [1.0, float("inf")], 1.0])
def test_bad_pairs_are_rejected(bad):
    results, template = res.load()
    assert res.validate(dict(results, read_noise_e=bad), template)


PAIR_KEYS = [k for k, v in json.loads(res.TEMPLATE.read_text()).items() if isinstance(v, list)]


@pytest.mark.parametrize("key", PAIR_KEYS)
def test_value_without_uncertainty_is_valid_but_listed_as_open(key):
    results, template = res.load()
    r = dict(results, **{key: [7.0, None]})
    assert res.validate(r, template) == []
    assert f"{key}: value 7.0 has no uncertainty yet" in res.todo(r)


@pytest.mark.parametrize("key", PAIR_KEYS)
def test_complete_pair_is_not_listed_as_open(key):
    results, _ = res.load()
    r = dict(results, **{key: [7.0, 0.1]})
    assert not any(line.startswith(f"{key}:") for line in res.todo(r))


def test_every_null_key_is_listed_as_open():
    results, template = res.load()
    all_null = {k: (v if k == "_README" else None) for k, v in template.items()}
    assert len(res.todo(all_null)) == len(template) - 1


def test_bool_is_not_accepted_as_int():
    results, template = res.load()
    assert res.validate(dict(results, white_level_dn=True), template)


def test_update_round_trip(tmp_path):
    out = tmp_path / "results.json"
    res.update({"read_noise_e": res.pair(6.99, 0.32)}, path=out)
    written = json.loads(out.read_text())
    assert written["read_noise_e"] == [6.99, 0.32]
    assert written["camera"] is None                  # untouched keys start as null
    assert list(written) == list(json.loads(res.TEMPLATE.read_text()))


def test_update_rejects_unknown_key(tmp_path):
    with pytest.raises(KeyError):
        res.update({"focal_lenght_px": [1.0, 0.1]}, path=tmp_path / "r.json")


def test_update_refuses_to_write_invalid_file(tmp_path):
    out = tmp_path / "r.json"
    with pytest.raises(ValueError):
        res.update({"read_noise_e": [7.0, -1.0]}, path=out)
    assert not out.exists()


def test_progress_counts_open_pairs_as_incomplete():
    results, template = res.load()
    all_null = {k: (v if k == "_README" else None) for k, v in template.items()}
    assert res.progress(all_null) == (0, len(template) - 1, 0.0)
    one = dict(all_null, read_noise_e=[7.0, 0.3], black_level_dn=[528.0, None])
    done, total, pct = res.progress(one)
    assert done == 1 and pct == pytest.approx(100 / total)
