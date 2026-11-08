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
import my_library.geometry_measurement as gm
import my_library.mtf_measurement as mtf
import my_library.noise_measurement as nm
import my_library.image_metadata as meta
import my_library.results as res
from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

DEFAULT_IMAGES = Path("images/iPhone16/Lightroom")
DEFAULT_OUTPUT = Path("output/project1")
# Every input folder must come from this camera (LensModel in the DNG). The lens_id
# assertion inside each step only checks that a series agrees with itself; this checks
# it is the camera the datasheet describes. --expected-lens overrides ("any" disables).
EXPECTED_LENS = "iPhone 16 Pro Max back camera 6.765mm f/1.78"
CALIBRATION_SUBDIR = "hw1/EXP125th"      # under --images; override with --calibration-images
FLAT_FIELD_SUBDIR = "ISO64_Sunset"       # under --images; the HW3 ISO 64 pairs
DARK_SUBDIR = "ISO64_Dark"               # under --images; the HW3 lens-capped frames
SLANTED_SUBDIR = "hw2_slanted/Nov_07"    # under --images; one subfolder per series (target position)
MTF_DATASHEET_SERIES = "*_501mm"         # datasheet center and corners from one distance (print-limited at 250 mm)
MTF_TWO_DISTANCE = ("center_250mm", "center_501mm")   # print-vs-camera test; skipped if either is missing

# Which illuminant round supplies the datasheet's color numbers.
# --datasheet-illuminant overrides this; None writes no Delta E values.
DATASHEET_ILLUMINANT = "led"


def run_geometry(args):
    """Measurement 1: intrinsics and lens distortion from the checkerboard captures.
    Also writes the camera-description keys, read from the calibration DNGs."""
    folder = args.calibration_images or args.images / CALIBRATION_SUBDIR
    exposure = gm.parse_exposure(args.calibration_exposure) if args.calibration_exposure else None
    summary = gm.measure_geometry(
        folder, args.output / "geometry", args.square_mm, exposure=exposure,
        exclude=args.exclude or (), detector=args.corner_detector, fix_k3=not args.free_k3,
        undistort_file=args.undistort_file)
    exif = summary["exif"]
    entries = gm.results_entries(summary["cal"], summary["n_used"], exif["pitch_um"],
                                 exif["pitch_rel_sd"], args.square_mm, args.square_mm_sd)
    entries.update(meta.identity_results_entries(summary["ident"], size_from=args.size_from))
    print(f"\nfocal_length_mm uses the EXIF-route pitch {exif['pitch_um']:.4f} um "
          f"(camera_specs.pitch_um {cam_spec.pitch_um} is kept for the homework scripts)")
    return entries


def run_mtf(args):
    """Measurement 2: slanted-edge MTF50 at the center and the four corners."""
    folder = args.slanted_edges or args.images / SLANTED_SUBDIR
    out = mtf.measure_mtf(folder, args.output / "mtf", center_series=args.mtf_center_series,
                          datasheet_pattern=args.mtf_datasheet_series,
                          two_distance_pair=args.mtf_two_distance)
    return mtf.results_entries(out["values"])


def photon_transfer(args):
    """The photon transfer fit shared by the noise and dynamic-range steps, computed once
    per run (cached on args), so both report from the same fit."""
    if getattr(args, "_ptc", None) is None:
        folder = args.flat_fields or args.images / FLAT_FIELD_SUBDIR
        args._ptc = nm.measure_ptc(folder, args.white_level, args.black_level,
                                   args.ptc_cutoff, method=args.pair_method)
    return args._ptc


def run_noise(args):
    """Measurement 3: conversion gain, read noise, SNR and its knee, the single-frame
    bias, the vendor NoiseProfile, and dark current."""
    dark = args.dark_frames or args.images / DARK_SUBDIR
    summary = nm.report_noise(photon_transfer(args), args.output / "noise", dark_folder=dark)
    return nm.noise_results_entries(summary)


def run_dynamic_range(args):
    """Measurement 4: dynamic range from the photon transfer fit (measurement 3's data)."""
    result = photon_transfer(args)
    summary = nm.report_dynamic_range(result)
    if summary["table"]:
        path = nm.write_dynamic_range_table(summary, args.output / "noise" / "dynamic_range_table.csv")
        print(f"  table written to {path}")

    if "noise" in getattr(args, "steps", ()):
        return nm.results_entries(summary, args.ptc_cutoff)    # same fit, written together
    # Run alone: the noise keys must come from this same fit; flag it if results.json disagrees.
    current, _ = res.load(args.results)
    for key, value in (("conversion_gain_dn_per_e", summary["K"]), ("read_noise_e", summary["read_e"])):
        held = current.get(key)
        if held and held[0] and abs(held[0] - value) > 0.005 * abs(value):
            print(f"  NOTE: results.json {key} = {held[0]} differs from this fit's {value:.4g}; "
                  "the noise step should write both from the same fit")
    if current.get("iso_of_noise_measurement") not in (None, result["iso"]):
        print(f"  NOTE: results.json iso_of_noise_measurement is "
              f"{current['iso_of_noise_measurement']}, these flat fields are ISO {result['iso']}")
    return nm.results_entries(summary, args.ptc_cutoff)


def run_color(args):
    """Measurement 5: ColorChecker, least-squares CCM, CIEDE2000 before and after."""
    names = args.illuminant_rounds or list(cm.COLOR_ROUNDS)
    rounds = {r: args.images / cm.COLOR_ROUNDS[r] for r in names}
    summaries = cm.measure_color(rounds, args.white_level, args.output / "color",
                                 black_level=args.black_level, exposure=args.color_exposure)

    entries = {"delta_e_metric": cm.METRIC, "colorchecker_reference_set": cm.wb.REFERENCE}
    if args.datasheet_illuminant is None:
        print("\nNo datasheet illuminant chosen (DATASHEET_ILLUMINANT / --datasheet-illuminant), "
              "so the Delta E values are not written to results.json.")
    elif args.datasheet_illuminant not in summaries:
        raise SystemExit(f"--datasheet-illuminant {args.datasheet_illuminant}: "
                         f"no usable charts in that round")
    else:
        summary = summaries[args.datasheet_illuminant]
        entries = cm.results_entries(summary, matrix="pooled")
        t = summary["per_patch"]
        print(f"\nPer-patch {cm.METRIC}, {args.datasheet_illuminant} round, mean +/- sd over "
              f"{summary['n_charts']} charts (after = the pooled matrix, the CCM):")
        for r in t.itertuples():
            print(f"  {r.patch:2d} {r.name:15s} before {r.dE_before_mean:5.2f} +/- {r.dE_before_sd:4.2f}"
                  f"   after {r.dE_after_pooled_mean:5.2f} +/- {r.dE_after_pooled_sd:4.2f}")
        print(f"\nDatasheet color values from the {args.datasheet_illuminant} round (after = pooled CCM):")
        if args.color_exposure == "white":
            cm.compare_with_hw4(args.datasheet_illuminant, entries)
        else:
            print("  (HW4 used --color-exposure white; run with it to compare against HW4)")
    return entries


# Order here is the order steps run in. Add geometry, mtf, noise, dynamic range here.
MEASUREMENTS = {
    "geometry": run_geometry,
    "mtf": run_mtf,
    "noise": run_noise,
    "dynamic_range": run_dynamic_range,
    "color": run_color,
}


def build_parser():
    p = argparse.ArgumentParser(
        description="EE 513 Project 1: measure a camera and write results.json.",
        epilog="With no measurement flags, every implemented measurement runs.")
    steps = p.add_argument_group("measurements (choose any; default is all)")
    steps.add_argument("--geometry", action="store_true",
                       help="checkerboard calibration: intrinsics, distortion, undistort figure")
    steps.add_argument("--mtf", action="store_true",
                       help="slanted-edge MTF50: center and four corners, both edge directions")
    steps.add_argument("--noise", action="store_true",
                       help="noise: gain, read noise, SNR and knee, single-frame bias, "
                            "NoiseProfile, dark current")
    steps.add_argument("--dynamic-range", action="store_true",
                       help="dynamic range from the photon transfer fit of the flat-field pairs")
    steps.add_argument("--color", action="store_true",
                       help="ColorChecker: color correction matrix and Delta E before and after")

    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                   help=f"root folder of the raw captures (default: {DEFAULT_IMAGES})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"folder for figures and tables (default: {DEFAULT_OUTPUT})")
    p.add_argument("--results", type=Path, default=res.RESULTS,
                   help="results.json to update (default: the repository's results.json)")
    p.add_argument("--expected-lens", default=EXPECTED_LENS, metavar="LENSMODEL",
                   help=f"LensModel every input must have (default: {EXPECTED_LENS!r}; 'any' disables)")
    p.add_argument("--no-results", action="store_true",
                   help="compute and print, but do not write results.json")

    raw = p.add_argument_group("raw levels (used by every step)")
    raw.add_argument("--black-level", type=float, default=cam_spec.black_level_dn, metavar="DN",
                     help=f"black level to subtract (default: {cam_spec.black_level_dn}, the "
                          "value declared in the DNGs)")
    raw.add_argument("--white-level", type=int, default=cam_spec.saturation_dn, metavar="DN",
                     help=f"saturation level (default: {cam_spec.saturation_dn}, measured "
                          "in HW4 Problem 1)")

    geo = p.add_argument_group("geometry options")
    geo.add_argument("--calibration-images", type=Path, metavar="PATH",
                     help=f"checkerboard captures (default: --images/{CALIBRATION_SUBDIR})")
    geo.add_argument("--calibration-exposure", metavar="SECONDS",
                     help="use only captures at this exposure time, e.g. 1/125")
    geo.add_argument("--exclude", nargs="+", metavar="FILE",
                     help="calibration files to leave out (state why in the report)")
    geo.add_argument("--corner-detector", choices=gm.DETECTORS, default="sb",
                     help="sb: findChessboardCornersSB (default); "
                          "classic: findChessboardCorners + cornerSubPix")
    geo.add_argument("--free-k3", action="store_true",
                     help="also fit k3 (default holds it at zero, per HW2 feedback: a free k3 "
                          "trades off against k2 and destabilizes the fit)")
    geo.add_argument("--square-mm", type=float, default=gm.analyze.SQUARE_MM, metavar="MM",
                     help=f"measured checkerboard square size (default: {gm.analyze.SQUARE_MM})")
    geo.add_argument("--square-mm-sd", type=float, metavar="MM",
                     help="uncertainty of --square-mm (default: none yet)")
    geo.add_argument("--undistort-file", metavar="FILE",
                     help="view for the undistort figure (default: the one reaching "
                          "farthest from the frame center)")
    geo.add_argument("--size-from", choices=meta.SIZE_SOURCES, default="rawpy_visible",
                     help="which image size goes in raw_width_px/raw_height_px "
                          "(default: rawpy_visible)")

    mtf_opts = p.add_argument_group("MTF options")
    mtf_opts.add_argument("--slanted-edges", type=Path, metavar="PATH",
                     help=f"slanted-edge captures, one subfolder per series (default: --images/{SLANTED_SUBDIR})")
    mtf_opts.add_argument("--mtf-center-series", metavar="NAME",
                     help="series for the center value (default: among the datasheet series, the "
                          "only one, or the only one whose name contains 'center')")
    mtf_opts.add_argument("--mtf-datasheet-series", metavar="PATTERN", default=MTF_DATASHEET_SERIES,
                          help="glob pattern of the series that supply the datasheet values, to keep "
                               f"center and corners at one distance (default: {MTF_DATASHEET_SERIES}; "
                               "'*' for all)")
    mtf_opts.add_argument("--mtf-two-distance", nargs=2, metavar=("NEAR", "FAR"),
                          default=list(MTF_TWO_DISTANCE),
                          help="center series at two distances for the print-vs-camera test "
                               f"(default: {' '.join(MTF_TWO_DISTANCE)}; skipped if either is missing)")

    noise = p.add_argument_group("noise and dynamic range options")
    noise.add_argument("--flat-fields", type=Path, metavar="PATH",
                       help=f"flat-field pairs, consecutive by file name (default: "
                            f"--images/{FLAT_FIELD_SUBDIR})")
    noise.add_argument("--dark-frames", type=Path, metavar="PATH",
                       help=f"lens-capped frames at several exposures (default: --images/{DARK_SUBDIR})")
    noise.add_argument("--pair-method", choices=nm.PAIR_METHODS, default="pair difference",
                       metavar="METHOD",
                       help="variance estimate the results use: 'pair difference' (default), "
                            "'gain-matched' or 'plane-removed'; all three are always printed")
    noise.add_argument("--ptc-cutoff", type=float, default=nm.LOW_SIGNAL_CUTOFF, metavar="FRACTION",
                       help=f"fit only pairs below this fraction of the saturation span "
                            f"(default: {nm.LOW_SIGNAL_CUTOFF}, per HW3 feedback)")

    color = p.add_argument_group("color options")
    color.add_argument("--color-exposure", choices=cm.EXPOSURE_METHODS, default="grays",
                       help="exposure and CIELAB white: 'grays' fits all six neutrals (default, "
                            "HW4 feedback); 'white' uses patch 19 alone, as HW4 did")
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


def step_folders(args):
    """{step name: list of input folders} for the selected steps, as the steps resolve them."""
    images = args.images
    folders = {
        "geometry": [args.calibration_images or images / CALIBRATION_SUBDIR],
        "mtf": [args.slanted_edges or images / SLANTED_SUBDIR],
        "noise": [args.flat_fields or images / FLAT_FIELD_SUBDIR, args.dark_frames or images / DARK_SUBDIR],
        "dynamic_range": [args.flat_fields or images / FLAT_FIELD_SUBDIR],
        "color": [images / cm.COLOR_ROUNDS[r] for r in (args.illuminant_rounds or cm.COLOR_ROUNDS)],
    }
    return {name: folders[name] for name in args.steps if name in folders}


def check_expected_camera(args):
    """Stop before any step runs if an input folder came from another camera.

    Reads the LensModel of the first raw file in every input folder (and every series
    subfolder) of the selected steps. Each step's own lens_id assertion confirms its
    files agree with each other; this confirms they are the camera the datasheet
    describes. (On Sat Nov 7 a checkerboard set from the ultra-wide passed every
    per-series check and wrote its geometry into results.json.)
    """
    if args.expected_lens.lower() == "any":
        return
    wrong = []
    for step, folders in step_folders(args).items():
        for folder in folders:
            folder = Path(folder)
            if not folder.is_dir():
                continue
            subdirs = [d for d in sorted(folder.iterdir()) if d.is_dir()] or [folder]
            for d in [folder] + subdirs:
                files = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in meta.RAW_SUFFIXES]
                if files:
                    lens = meta.camera_identity(files[0])["lens_model"]
                    if lens != args.expected_lens:
                        wrong.append(f"  {step}: {d} ({files[0].name}) is {lens!r}")
    if wrong:
        raise SystemExit("Input from a different camera than EXPECTED_LENS "
                         f"({args.expected_lens!r}):\n" + "\n".join(wrong)
                         + "\nNothing was written. Use --expected-lens to measure that camera on purpose "
                           "(with its own --results and --output), or fix the folder.")


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
    args.steps = selected_steps(args)
    check_expected_camera(args)
    for name in args.steps:
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
