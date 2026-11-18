"""Project 1 results.json: one file, every template key, [value, uncertainty] pairs.

The template (results.template.json, from Prof. James McNames, EE 513 Project 1) is the
schema: every key it has must be present in results.json, no others, and each list entry
is a [value, uncertainty] pair in the units named by the key. Anything not measured is
null. A measured value whose uncertainty is not yet defensible is [value, null], so it
shows up in todo() instead of being forgotten.

Each measurement script calls update() with only its own keys, so results.json is
written by the code that measured the numbers, never edited by hand.

    python -m my_library.results      # validate results.json and list what is still open
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "results.template.json"
RESULTS = ROOT / "results.json"


def _read(path):
    return json.loads(Path(path).read_text())


def pair(value, uncertainty=None):
    """[value, uncertainty] as plain floats; uncertainty None until it can be defended."""
    return [float(value), None if uncertainty is None else float(uncertainty)]


def load(path=RESULTS, template=TEMPLATE):
    """Return (results, template). A missing results file starts as all null."""
    t = _read(template)
    if Path(path).exists():
        return _read(path), t
    return {k: (v if k == "_README" else None) for k, v in t.items()}, t


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(results, template):
    """List of problems; empty means results matches the template's shape."""
    errors = [f"missing key: {k}" for k in template if k not in results]
    errors += [f"unexpected key: {k}" for k in results if k not in template]
    for k, example in template.items():
        v = results.get(k)
        if k not in results or v is None or k == "_README":
            continue
        if isinstance(example, list):
            if not (isinstance(v, list) and len(v) == 2):
                errors.append(f"{k}: expected [value, uncertainty], got {v!r}")
            elif not _is_number(v[0]):
                errors.append(f"{k}: value must be a finite number, got {v[0]!r}")
            elif v[1] is not None and not (_is_number(v[1]) and v[1] >= 0):
                errors.append(f"{k}: uncertainty must be null or a finite number >= 0, got {v[1]!r}")
        elif isinstance(example, bool):
            if not isinstance(v, bool):
                errors.append(f"{k}: expected true/false, got {v!r}")
        elif isinstance(example, int):
            if not (isinstance(v, int) and not isinstance(v, bool)):
                errors.append(f"{k}: expected an integer, got {v!r}")
        elif isinstance(example, str):
            if not isinstance(v, str):
                errors.append(f"{k}: expected a string, got {v!r}")
    return errors


def todo(results):
    """Keys that are still null, or pairs whose uncertainty is still null."""
    open_keys = []
    for k, v in results.items():
        if k == "_README":
            continue
        if v is None:
            open_keys.append(f"{k}: not measured yet")
        elif isinstance(v, list) and len(v) == 2 and v[1] is None:
            open_keys.append(f"{k}: value {v[0]} has no uncertainty yet")
    return open_keys


def progress(results):
    """(complete, total, percent): entries with a value and, for pairs, an uncertainty."""
    total = sum(1 for k in results if k != "_README")
    done = total - len(todo(results))
    return done, total, 100.0 * done / total if total else 0.0


def print_todo(results):
    """Print the open entries with an overall percent-complete line."""
    remaining = todo(results)
    done, total, pct = progress(results)
    print(f"results.json: {done} of {total} entries complete ({pct:.0f}%), "
          f"{len(remaining)} still open:")
    for line in remaining:
        print(f"  {line}")


def update(entries, path=RESULTS, template=TEMPLATE):
    """Merge entries into results.json, validate, and write it in template key order."""
    results, t = load(path, template)
    unknown = [k for k in entries if k not in t]
    if unknown:
        raise KeyError(f"not in the template: {unknown}")
    results.update(entries)
    errors = validate(results, t)
    if errors:
        raise ValueError("results.json would be invalid:\n  " + "\n  ".join(errors))
    ordered = {k: results[k] for k in t}
    Path(path).write_text(json.dumps(ordered, indent=2) + "\n")
    return ordered


if __name__ == "__main__":
    results, t = load()
    errors = validate(results, t)
    print("results.json is valid" if not errors else "PROBLEMS:\n  " + "\n  ".join(errors))
    print()
    print_todo(results)
