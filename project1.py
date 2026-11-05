"""EE 513 Project 1: characterize a camera and write its datasheet to results.json.

    uv run python project1.py            # run every implemented measurement (the one command)
    uv run python project1.py --color    # run only the color measurement
    uv run python project1.py --help     # all options

Each measurement step returns a dict of results.json entries. They are merged and
written once at the end, through my_library.results, which refuses to write anything
that does not match results.template.json. Pass --no-results to compute and print
without touching results.json.
"""
import argparse
import sys
from pathlib import Path

import my_library.color_measurement as cm
import my_library.results as res
from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

DEFAULT_IMAGES = Path("images/iPhone16/Lightroom")
DEFAULT_OUTPUT = Path("output/project1")

# Which illuminant round supplies the datasheet's color numbers.
# --datasheet-illuminant overrides this; None writes no Delta E values.
DATASHEET_ILLUMINANT = "led"


def run_color(args):
    """Measurement 5: ColorChecker, least-squares CCM, CIEDE2000 before and after."""
    names = args.illuminant_rounds or list(cm.COLOR_ROUNDS)
    rounds = {r: args.images / cm.COLOR_ROUNDS[r] for r in names}
    summaries = cm.measure_color(rounds, args.white_level, args.output / "color",
                                 black_level=args.black_level)

    entries = {"delta_e_metric": cm.METRIC, "colorchecker_reference_set": cm.wb.REFERENCE}
    if args.datasheet_illuminant is None:
        print("\nNo datasheet illuminant chosen (DATASHEET_ILLUMINANT / --datasheet-illuminant), "
              "so the Delta E values are not written to results.json.")
    elif args.datasheet_illuminant not in summaries:
        raise SystemExit(f"--datasheet-illuminant {args.datasheet_illuminant}: "
                         f"no usable charts in that round")
    else:
        entries = cm.results_entries(summaries[args.datasheet_illuminant])
        print(f"\nDatasheet color values from the {args.datasheet_illuminant} round:")
        cm.compare_with_hw4(args.datasheet_illuminant, entries)
    return entries


# Order here is the order steps run in. Add geometry, mtf, noise, dynamic range here.
MEASUREMENTS = {
    "color": run_color,
}


def build_parser():
    p = argparse.ArgumentParser(
        description="EE 513 Project 1: measure a camera and write results.json.",
        epilog="With no measurement flags, every implemented measurement runs.")
    steps = p.add_argument_group("measurements (choose any; default is all)")
    steps.add_argument("--color", action="store_true",
                       help="ColorChecker: color correction matrix and Delta E before and after")

    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                   help=f"root folder of the raw captures (default: {DEFAULT_IMAGES})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"folder for figures and tables (default: {DEFAULT_OUTPUT})")
    p.add_argument("--results", type=Path, default=res.RESULTS,
                   help="results.json to update (default: the repository's results.json)")
    p.add_argument("--no-results", action="store_true",
                   help="compute and print, but do not write results.json")

    raw = p.add_argument_group("raw levels (used by every step)")
    raw.add_argument("--black-level", type=float, default=cam_spec.black_level_dn, metavar="DN",
                     help=f"black level to subtract (default: {cam_spec.black_level_dn}, the "
                          "value declared in the DNGs)")
    raw.add_argument("--white-level", type=int, default=cam_spec.saturation_dn, metavar="DN",
                     help=f"saturation level (default: {cam_spec.saturation_dn}, measured "
                          "in HW4 Problem 1)")

    color = p.add_argument_group("color options")
    color.add_argument("--illuminant-rounds", nargs="+", choices=list(cm.COLOR_ROUNDS),
                       metavar="ILLUMINANT",
                       help="illuminant rounds to measure (default: all of "
                            f"{', '.join(cm.COLOR_ROUNDS)})")
    color.add_argument("--datasheet-illuminant", choices=list(cm.COLOR_ROUNDS),
                       metavar="ILLUMINANT", default=None,
                       help="illuminant round whose values go into results.json "
                            f"(default: DATASHEET_ILLUMINANT = {DATASHEET_ILLUMINANT})")
    return p


def resolve_datasheet_illuminant(parser, args):
    """Fill args.datasheet_illuminant from DATASHEET_ILLUMINANT unless it was given.

    Given on the command line but not among --illuminant-rounds: an error. Taken from
    DATASHEET_ILLUMINANT but not among --illuminant-rounds (a quick run of other
    rounds): no Delta E values are written this run.
    """
    measured = args.illuminant_rounds or list(cm.COLOR_ROUNDS)
    if args.datasheet_illuminant is not None:
        if args.datasheet_illuminant not in measured:
            parser.error(f"--datasheet-illuminant {args.datasheet_illuminant} is not among "
                         f"--illuminant-rounds {' '.join(measured)}")
    elif DATASHEET_ILLUMINANT in measured:
        args.datasheet_illuminant = DATASHEET_ILLUMINANT
    elif DATASHEET_ILLUMINANT is not None:
        print(f"note: DATASHEET_ILLUMINANT {DATASHEET_ILLUMINANT} is not being measured "
              "this run, so no Delta E values will be written to results.json")


def selected_steps(args):
    """The steps named on the command line, or all of them if none were."""
    chosen = [name for name in MEASUREMENTS if getattr(args, name)]
    return chosen or list(MEASUREMENTS)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolve_datasheet_illuminant(parser, args)

    # Raw levels used by every step. The black level has no defended uncertainty yet,
    # so it stays [value, null] and shows up in the todo list.
    entries = {"black_level_dn": res.pair(args.black_level),
               "white_level_dn": int(args.white_level)}
    for name in selected_steps(args):
        print(f"\n=================== {name} ===================")
        entries.update(MEASUREMENTS[name](args))

    if args.no_results:
        print("\n--no-results: results.json not written")
        results, _ = res.load(args.results)
        results.update(entries)
    else:
        results = res.update(entries, path=args.results)
        print(f"\nWrote {len(entries)} entries to {args.results}")
    res.print_todo(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
