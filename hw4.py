# hw4.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rawpy
import psu_imaging as psu
import matplotlib.pyplot as plt

import my_library.analyze as analyze
import my_library.image_metadata as img
import my_library.photon_transfer_curve as ptc
import my_library.linear_light as ll
import my_library.generate as gen
import my_library.latex_formatter as tex
import my_library.shading as shading
import my_library.catalog as datalog
import my_library.color_chart as chart
import my_library.white_balance as wb

from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

def plot_two_images(first, second, first_title, second_title, save_path, overall_title=None,first_color=None, second_color=None, SHOW=False):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].set_title(first_title)
    if first_color:
        ax[0].imshow(first, cmap=first_color)
    else:
        ax[0].imshow(first)
    ax[0].xaxis.tick_top()
    ax[0].set_xlabel('Columns')
    ax[0].set_ylabel('Rows')

    ax[1].set_title(second_title)
    if second_color:
        ax[1].imshow(second, cmap=second_color)
    else:
        ax[1].imshow(second)
    ax[1].xaxis.tick_top()
    ax[1].set_xlabel('Columns')
    ax[1].set_ylabel('Rows')

    if overall_title:
        fig.suptitle(overall_title)

    plt.tight_layout()
    if SHOW:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = False
    TESTING = False

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'
    hw = 'hw4_color_charts/'

    flor_path = f'{camera_path}{lightroom}{hw}florescent/'
    led_path = f'{camera_path}{lightroom}{hw}led/'
    incandesent_path = f'{camera_path}{lightroom}{hw}incandesent/'
    cloudy_path = f'{camera_path}{lightroom}{hw}outdoors_cloudy/'
    shade_path = f'{camera_path}{lightroom}{hw}outdoors_shade/'

    if TESTING:
        color_sets = [flor_path, led_path]
        rounds = {'florescent': flor_path, 'led': led_path}
    else:
        color_sets = [flor_path, led_path, incandesent_path, f'{cloudy_path}/set1', f'{cloudy_path}/set2', shade_path]
        rounds = {'florescent': flor_path, 'led': led_path, 'incandesent': incandesent_path,
            'cloudy1': f'{cloudy_path}/set1', 'cloudy2': f'{cloudy_path}/set2', 'shade': shade_path}

    output_path = 'output/hw4/'

    catalog = datalog.add_clip_stats(datalog.build_catalog(rounds), cam_spec.saturation_dn) # pyright: ignore[reportArgumentType]

    # HW4-specific data
    # Differentiate between a color chart and a white flat
    catalog["kind"] = np.where(catalog["file"].str.contains("_white"), "white", "chart")
    is_chart = catalog["kind"] == "chart"
    catalog["chart"] = pd.Series(pd.NA, index=catalog.index, dtype="Int64")
    catalog.loc[is_chart, "chart"] = catalog[is_chart].groupby("round").cumcount() + 1

    datalog.print_inventory(catalog)
    datalog.check_constant_within(catalog, "round", ["iso", "exposure_s", "black_levels"])
    img.assert_one_camera(catalog["path"].tolist())        # the professor's function, his output

    # Problem 1 deliverables that cover every round, so they are produced once here.
    # (The per-round contact sheets and histograms are made inside the loop below.)
    problem = "01"
    print(f"\n**** Problem {problem}: saturation and inventory, all rounds ****")
    catalog, measured_sat = datalog.add_saturation_stats(catalog)
    found = [v for v in measured_sat.values() if v is not None]
    declared = sorted(set(catalog["white_level_dn"]))
    print(f"DELIVERABLE {problem}: raw value at which the files actually saturate, per CFA site "
          f"(declared white level {declared}):")
    for ch, v in measured_sat.items():
        n_frames = int((catalog[f"spike_{ch}"] >= datalog.SPIKE_THRESHOLD).sum())
        print(f"\t{ch}: {v if v is not None else 'not measurable (never clipped)'}"
              f"{f' (spike in {n_frames} of {len(catalog)} frames)' if v is not None else ''}")
    if found and min(found) != cam_spec.saturation_dn:
        print(f"\tWARNING measured saturation {min(found)} differs from "
              f"cam_spec.saturation_dn = {cam_spec.saturation_dn}. Update camera_specs.py, "
              "because linear light and every clip mask downstream use that value.")
    # DELIVERABLE 01: the file list as a LaTeX table, columns in the order the deliverable
    # names them. Per-CFA-site fractions (sat_<site>_pct) are in catalog.csv.
    print(f"DELIVERABLE {problem}: files with round, chart, ISO, exposure time and fraction "
          f"at saturation, written to {output_path}{problem}_inventory.txt")
    chart_label = [str(c) if pd.notna(c) else "white" for c in catalog["chart"]]
    tex.generate_table([catalog["file"].tolist(), catalog["round"].tolist(), chart_label,
                        catalog["iso"].tolist(), catalog["exposure"].tolist(),
                        [f"{v:.3f}" for v in catalog["sat_frame_pct"]]],
                       labels=["File", "Round", "Chart", "ISO", "Exposure time",
                               "Fraction at saturation (%)"],
                       caption=f"Captures from the {cam_spec.name}: fraction of each frame "
                               "at or above the measured saturation",
                       output_path=f"{output_path}{problem}_inventory.txt",
                       escape=True, VERBOSE=False)
    sites = list(measured_sat)
    tex.generate_table([sites,
                        [str(measured_sat[ch]) if measured_sat[ch] is not None else "not clipped"
                         for ch in sites],
                        [str(int((catalog[f"spike_{ch}"] >= datalog.SPIKE_THRESHOLD).sum()))
                         for ch in sites]],
                       labels=["CFA site", "Saturation (DN)", "Frames clipped"],
                       caption=f"Measured saturation per CFA site (declared white level {declared})",
                       output_path=f"{output_path}{problem}_saturation.txt",
                       escape=True, VERBOSE=False)
    catalog.to_csv(f"{output_path}catalog.csv", index=False)


    # Since all captures were from the same camera, we only benefit from looking at the Bayer mosaic once.
    a_directory = Path(color_sets[0])
    an_image_file = next((f for f in a_directory.iterdir() if f.is_file() and f.suffix.lower() in img.RAW_SUFFIXES), None)
    print(f"Printing the Bayer mosaic for {an_image_file}...")
    analyze.bayer_mosaic(an_image_file)

    # Problem 4 state. MANUAL_CLICKS: only for files the detector misses. Keys are file names,
    # values are quarter-res (x, y) centers of patches 1, 6, 24, 19 (dark skin, bluish green,
    # black, white), in that order.
    MANUAL_CLICKS = {}
    patch_tables = []
    patch_means = {}

    for lighting_type, path_to_images in rounds.items():
        print(f"\n\n~*~*~*~*~*~*~*~*~  Now looking at images taken with {lighting_type} lighting. ~*~*~*~*~*~*~*~")
        output_path_for_lighting = f"{output_path}{lighting_type}/"

        round_df = catalog[catalog["round"] == lighting_type]
        image_paths = round_df["path"].tolist()
        exposure_times = round_df["exposure"].tolist()
        raws = [str(p) for p in image_paths]

        # One metadata table for all rounds is written before this loop (01_inventory.txt).

        # Problem 1
        problem = "01"
        print(f"\n**** Problem {problem} ****")
        '''
        Inventory your captures and confirm that every one of them is a Bayer raw file from the same
        camera, and that the ISO and exposure time are identical within each round. Then check every
        frame for clipping, which is the one defect that cannot be repaired afterwards. Do not take
        the white level recorded in the file on trust: histogram the raw values and look for a spike
        at the top of the range, because a file that saturates below its own declared white level will
        hand you patches pinned to a ceiling that look perfectly reasonable in every other respect. Look
        only at the very top of the histogram. Raw files are often quantized coarsely at high signal,
        which scatters smaller spikes through the upper half, and a search that starts from mid scale
        will find one of those instead. Any clipped patch has to be excluded from everything that
        follows, and the outdoor round is where you are most likely to find one.

        Deliverable: A list of your files with the round, chart, ISO, and exposure time of each, the
        lens_id assertion from the Project 1 page passing across all of them, the raw value at which
        your files actually saturate, and the fraction of each frame that reaches it.
        '''
        analyze.contact_sheet(raws, camera=cam, output=output_path_for_lighting, question=problem, labels=exposure_times)
        VERBOSE = False
        for i in image_paths:
            # print(f"\nHistogram analysis for {os.path.basename(i)}:")
            analyze.histogram(i, output_path_for_lighting, question=problem, VERBOSE=VERBOSE)

        # Saturation value and fraction of each frame at saturation: measured for all rounds
        # before this loop (DELIVERABLE 01 printout, 01_saturation.txt, 01_inventory.txt).


        # Problem 2
        problem = "02"
        print(f"\n**** Problem {problem} ****")
        print(f'HEAD:\n{round_df.head()}')
        '''
        Convert one capture to linear color without demosaicing it. Subtract the black level, divide
        by the span from black up to the saturation value you measured in item 1, and then take each
        2 by 2 block of the mosaic to one pixel of a quarter resolution image, using the red site,
        the mean of the two green sites, and the blue site. No interpolation belongs in a measurement,
        and this costs you nothing because a patch is thousands of pixels across. Read the mosaic
        order from the file rather than assuming it is RGGB; phones differ, and a rotated pattern
        will swap your red and blue channels without producing any other sign that something is wrong.

        Deliverable: The quarter resolution image displayed, and a statement of the CFA pattern, the
        black level, and the saturation value you used.
        '''

        first_img = 0
        with rawpy.imread(str(raws[first_img])) as raw:
            raw_img = raw.raw_image_visible
            # 1. Convert to linear light
            black = round_df.iloc[first_img]['black_levels'][0]
            # saturation = round_df.iloc[first_img]['max_brightness'] # Error: don't use max brightness
            # linear = ll.raw_to_linear_light(raw_img, black_levels=round_df.iloc[first_img]['black_levels'], saturation_level=round_df.iloc[first_img]['white_level']) # Error: use saturation level instead of white level
            linear = ll.raw_to_linear_light(raw_img, black_level=black, saturation_level=cam_spec.saturation_dn)

            # 2. Resize down 25% by sampling the 2x2 blocks to one pixel
            # using the red site, the mean of the two green sites, and the blue site
            # my camera's CFA pattern comes out to BGGR, so the red site is at (1, 1), the green sites are at (0, 1) and (1, 0), and the blue site is at (0, 0)
            quarter = ll.quarter_rgb(linear, pattern=raw.raw_pattern, color_desc=raw.color_desc)

            plot_two_images(linear, psu.linear_to_srgb(quarter), f'Original image in linear light from {round_df.iloc[first_img]["file"]}', f'Quarter resolution linear light image from {round_df.iloc[first_img]["file"]}', f"{output_path_for_lighting}{problem}_{lighting_type}_quarter_comparison.png", overall_title=f'Full res LL to quarter-res RGB from {round_df.iloc[first_img]["file"]}', first_color='gray', second_color=None, SHOW=False)


        # Problem 4
        problem = "04"
        print(f"\n**** Problem {problem} ****")
        '''
        Use OpenCV4 to locate the 24 patches of flat color in the quarter resolution image.  Sample
        no more than the middle 50% of each patch to avoid the black borders and any other edge
        effects. Measure the mean and standard deviation of each patch. The standard deviation is a
        measure of how flat the patch is, and the mean is a measure of its color. The mean should
        be reported in linear light, not sRGB. The standard deviation should be reported in 8-bit
        levels and linear light.
        '''
        # Patch location and middle-half sampling: see my_library/color_chart.py for credits
        # (sampling rule from Prof. James McNames, 07_build_an_isp.ipynb section 4).
        charts_this_round = catalog[(catalog["round"] == lighting_type) & (catalog.kind == "chart")]
        for _, row in charts_this_round.iterrows():
            label = f"{lighting_type}_chart{row.chart}"
            result = chart.measure_chart(row.path, cam_spec.saturation_dn, clicks=MANUAL_CLICKS.get(row.file))
            if result is None:
                print(f"{row.file}: detector found no chart. Add its four corner-patch centers "
                      f"to MANUAL_CLICKS (see chart.homography_from_clicks) and rerun.")
                continue
            print(f"\n{label} ({row.file}), located with {result['how']}:")
            for msg in result["checks"]:
                print(f"\t{msg}")

            print(f"DELIVERABLE {problem}a: Plot sampled patch regions...")
            chart.plot_regions(result, f"{label}: {row.file}",
                               save_path=f"{output_path_for_lighting}{problem}_{label}_regions.png")
            chart.plot_edge_profile(result, save_path=f"{output_path_for_lighting}{problem}_{label}_edge_profile.png")
            df = chart.patch_table(result, round=lighting_type, chart=row.chart, file=row.file)
            patch_tables.append(df)
            patch_means[(lighting_type, row.chart)] = result["means"]      # 24 x 3, for later problems

            print(f"DELIVERABLE {problem}b: Generating LaTeX table of 24 by 3 patch means...")
            tex.generate_table([df["patch"].tolist(), df["name"].tolist()] +
                               [[f"{v:.4f}" for v in df[c]] for c in ("R", "G", "B")],
                               labels=["Patch", "Name", "R", "G", "B"],
                               caption=f"Mean linear camera RGB, {label}",
                               output_path=f"{output_path_for_lighting}{problem}_{label}_patch_means.txt")

    if patch_tables:
        all_patches = pd.concat(patch_tables, ignore_index=True)
        all_patches.to_csv(f"{output_path}04_patch_means.csv", index=False)
        print(f"\nProblem 4: wrote {len(all_patches)} patch rows to {output_path}04_patch_means.csv")


    # Problem 3
    problem = "03"
    print(f"\n**** Problem {problem} ****")
    maps: dict[str, np.ndarray] = dict()
    profiles: dict[str, np.ndarray] = dict()
    corner_center_ratios: list[dict] = []
    FRAC = 0.10

    first_white = next(iter(catalog[catalog.kind == "white"]["path"]))
    h, w = ll.load_linear_quarter_rgb(first_white, cam_spec.saturation_dn).shape[:2]
    model = shading.cos4_model(h, w, cam_spec, bin_factor=2)      # same for every round, computed once

    for name, white_path in catalog[catalog.kind == "white"].set_index("round")["path"].items():
        print(f"\nWhite sheet, {str(name).upper()}: {white_path.name}")
        white_rgb = ll.load_linear_quarter_rgb(white_path, cam_spec.saturation_dn)
        shading_map = shading.shading_map_from_white(white_rgb, center_frac=FRAC)
        maps[str(name)], profiles[str(name)] = shading_map, shading.horizontal_profile(shading_map)

        clipped = (white_rgb >= 0.999).astype(float)
        print("clipped fraction, whole frame (R,G,B):", np.round(clipped.mean(axis=(0, 1)), 4))
        print("clipped fraction, center box  (R,G,B):", np.round(shading.center_mean(clipped, FRAC), 4))
        print("map percentiles [1, 50, 99] per channel:\n",
            np.round(np.percentile(shading_map, [1, 50, 99], axis=(0, 1)).T, 3))

        # plot a linearlized white flat image next to a normalized white flat image
        # Error: Plotting the shading map is not realistic.  It's a map of the per-channel gain.  It's not an image, per se.
        # To be more honest and useful, show the shading map per color channel and compare to the linearized white flat in quarter RGB.
        # plot_two_images(white_flat_rgb, map_of_white_flat, f'White flat in linear light from {img_data['file'][len(img_data)-1]}', f'Normalized white flat in linear light from {img_data['file'][len(img_data)-1]}', f"{output_path_for_lighting}{problem}_white_flat_normalized.png", first_color=None, second_color=None, SHOW=False)
        shading.plot_white_and_map(white_rgb, shading_map, f"White sheet, {name}",
                                f"{output_path}{problem}_{name}_white_sheet.png")
        corner_center_ratios.append(shading.shading_ratios(name, white_rgb, shading_map, model, FRAC))

        print(f"DELIVERABLE {problem}a: The quarter-resolution RGB image before and after removing the shading map...")
        plot_two_images(psu.linear_to_srgb(white_rgb), psu.linear_to_srgb(shading.remove_shading(white_rgb, shading_map)),
                        f'White sheet in linear light from {name}', f'White sheet after removing shading map from {name}', f"{output_path}{problem}a_{name}_white_sheet_corrected.png",
                        overall_title=f'White sheet correction from {name}', first_color=None, second_color=None, SHOW=False)

    print(f"DELIVERABLE {problem}b: The ratio of the corner region to the center region on a white flat both before and after dividing out the shading map, per color channel...")
    # print(pd.DataFrame(corner_center_ratios).set_index("round").round(3))
    ratios_df = pd.DataFrame(corner_center_ratios)
    print(ratios_df.set_index("round").round(3))
    # One table for every round, so the file name has no round in it.
    tex.generate_table([ratios_df["round"].tolist()] +
                       [[f"{v:.3f}" for v in ratios_df[f"{c}_{when}"]]
                        for c in "RGB" for when in ("before", "after")] +
                       [[f"{v:.3f}" for v in ratios_df["cos4_model"]]],
                       labels=["Round", "R before", "R after", "G before", "G after",
                               "B before", "B after", "cos$^4$ model"],
                       caption="Corner-to-center ratio on the white flat, per color channel, "
                               "before and after dividing out the shading map",
                       output_path=f"{output_path}{problem}b_corner_to_center_ratios.txt",
                       VERBOSE=False)

    print(f"DELIVERABLE {problem}c: Plotting the central horizontal profiles from each round on the same axis...")
    shading.plot_profiles(profiles, model_profile=model[h // 2],
                        title="Horizontal profile, normalized to center",
                        save_path=f"{output_path}{problem}c_horizontal_profiles.png")


    # Problem 5: white balance gains from an unclipped mid gray
    problem = "05"
    print(f"\n**** Problem {problem} ****")
    ref_names, ref_XYZ, ref_illuminant_xy = wb.load_reference()    # used through Part 2
    print(f"Reference: {wb.REFERENCE}. Numbering check: "
          f"{wb.check_numbering(ref_names, psu.COLORCHECKER_NAMES)}")
    print("Orientation of every chart was confirmed by the Problem 4 checks "
          "(darkest patch is #24, usable neutrals darken in order and share one color).")

    gain_rows = []
    balanced_means = {}                      # (round, chart) -> 24 x 3, for later problems
    for (rnd, chart_n), g in all_patches.groupby(["round", "chart"], sort=False):
        means = g[["R", "G", "B"]].to_numpy()
        usable = g["usable"].to_numpy()
        k = wb.pick_gray(usable)
        if k is None:
            print(f"{rnd} chart {chart_n}: no usable mid gray, skipped")
            continue
        gains = wb.gains_from_gray(means[k])
        balanced = means * gains
        balanced_means[(rnd, chart_n)] = balanced
        resid, used = wb.neutral_residual(balanced, usable)
        gain_rows.append(dict(round=rnd, chart=chart_n, gray=f"{k + 1} {psu.COLORCHECKER_NAMES[k]}",
                              gR=gains[0], gG=gains[1], gB=gains[2], resid=resid,
                              neutrals=", ".join(map(str, used))))

        round_dir = f"{output_path}{rnd}/"
        os.makedirs(round_dir, exist_ok=True)
        print(f"DELIVERABLE {problem}b: patch means after balancing, {rnd} chart {chart_n} "
              f"-> {round_dir}{problem}b_{rnd}_chart{chart_n}_balanced_means.txt")
        tex.generate_table([list(range(1, 25)), list(psu.COLORCHECKER_NAMES)] +
                           [[f"{v:.4f}" for v in balanced[:, j]] for j in range(3)] +
                           [["yes" if u else "no" for u in usable]],
                           labels=["Patch", "Name", "R", "G", "B", "Usable"],
                           caption=f"Patch means after white balance (gains from patch {k + 1}, "
                                   f"{psu.COLORCHECKER_NAMES[k]}), {rnd} chart {chart_n}",
                           output_path=f"{round_dir}{problem}b_{rnd}_chart{chart_n}_balanced_means.txt",
                           escape=True, VERBOSE=False)

    gains_df = pd.DataFrame(gain_rows)
    print(f"DELIVERABLE {problem}a: white balance gains normalized to green "
          f"-> {output_path}{problem}a_white_balance_gains.txt")
    print(gains_df.to_string(index=False, float_format="{:.3f}".format))
    tex.generate_table([gains_df["round"].tolist(), gains_df["chart"].tolist(), gains_df["gray"].tolist()] +
                       [[f"{v:.3f}" for v in gains_df[c]] for c in ("gR", "gG", "gB", "resid")],
                       labels=["Round", "Chart", "Gray patch used", "R gain", "G gain", "B gain",
                               "Max neutral error"],
                       caption="White balance gains normalized to green, set from an unclipped mid gray. "
                               "Max neutral error: largest $|R/G-1|$ or $|B/G-1|$ over the other usable "
                               "neutrals 19-23 after balancing.",
                       output_path=f"{output_path}{problem}a_white_balance_gains.txt",
                       escape=True, VERBOSE=False)


    # Problem 6: error with no color matrix (balanced camera RGB treated as linear sRGB)
    problem = "06"
    print(f"\n**** Problem {problem} ****")
    METRIC = "CIEDE2000"
    de_rows, de_patches = [], []
    for (rnd, chart_n), balanced in balanced_means.items():
        usable = all_patches[(all_patches["round"] == rnd) &
                             (all_patches["chart"] == chart_n)]["usable"].to_numpy()
        # Pass in the white-balanced samples from problem 5, naively pretending that the
        # camera's R, G and B channels match that of sRGB's R, G and B channels.
        de, k_white = wb.delta_e_vs_reference(balanced, ref_XYZ, usable)   # sRGB primaries
        worst = int(np.argmax(np.where(usable, de, -np.inf)))
        de_rows.append(dict(round=rnd, chart=chart_n,
                            white=f"{k_white + 1} {psu.COLORCHECKER_NAMES[k_white]}",
                            n=int(usable.sum()), mean=float(de[usable].mean()),
                            worst=float(de[worst]),
                            worst_patch=f"{worst + 1} {psu.COLORCHECKER_NAMES[worst]}"))
        de_patches.append(pd.DataFrame({"round": rnd, "chart": chart_n, "patch": range(1, 25),
                                        "name": psu.COLORCHECKER_NAMES, "dE00_srgb": de,
                                        "usable": usable}))
    de_df = pd.DataFrame(de_rows)
    pd.concat(de_patches).to_csv(f"{output_path}{problem}_dE_per_patch_srgb.csv", index=False)
    print(f"DELIVERABLE {problem}a: mean and worst {METRIC} over usable patches, no color matrix "
          f"(sRGB primaries) -> {output_path}{problem}a_dE_srgb_baseline.txt")
    print(de_df.to_string(index=False, float_format="{:.2f}".format))
    tex.generate_table([de_df["round"].tolist(), de_df["chart"].tolist(), de_df["white"].tolist(),
                        de_df["n"].tolist()] +
                       [[f"{v:.2f}" for v in de_df[c]] for c in ("mean", "worst")] +
                       [de_df["worst_patch"].tolist()],
                       labels=["Round", "Chart", "Reference white", "Patches",
                               f"Mean $\\Delta E_{{00}}$", f"Worst $\\Delta E_{{00}}$", "Worst patch"],
                       caption=f"Error before any color matrix: white-balanced camera RGB treated as "
                               f"linear sRGB. Metric: {METRIC}, over usable (unclipped) patches.",
                       output_path=f"{output_path}{problem}a_dE_srgb_baseline.txt",
                       escape=False, VERBOSE=False)


    # Problem 7: least squares 3x3 matrix, white balanced camera RGB -> reference XYZ
    problem = "07"
    print(f"\n**** Problem {problem} ****")
    white_target = wb.white_xyz_y1(ref_XYZ)              # M @ [1, 1, 1] if neutrals are preserved
    color_matrices = {}                                  # round -> M, for later problems
    for rnd in dict.fromkeys(r for r, _ in balanced_means):
        round_dir = f"{output_path}{rnd}/"
        os.makedirs(round_dir, exist_ok=True)
        charts = [(c, balanced_means[(r, c)]) for r, c in balanced_means if r == rnd]
        cams, refs, use = [], [], []
        for chart_n, balanced in charts:
            usable = all_patches[(all_patches["round"] == rnd) &
                                 (all_patches["chart"] == chart_n)]["usable"].to_numpy()
            scaled, _ = wb.scale_to_reference(balanced, ref_XYZ, usable)
            cams.append(scaled); refs.append(ref_XYZ); use.append(usable)
        M = wb.fit_matrix(np.vstack(cams), np.vstack(refs), np.concatenate(use))
        color_matrices[rnd] = M
        row_sums = M.sum(axis=1)
        print(f"DELIVERABLE {problem}a: {rnd} color matrix (camera RGB -> XYZ), fit to "
              f"{int(np.concatenate(use).sum())} usable patches from {len(charts)} charts "
              f"-> {round_dir}{problem}a_{rnd}_color_matrix.txt")
        print(np.array2string(M, precision=4, suppress_small=True))
        chroma_err = (row_sums / row_sums[1]) / white_target - 1     # brightness divided out
        print(f"DELIVERABLE {problem}b: {rnd} row sums {np.round(row_sums, 4)} vs reference white "
              f"(Y = 1) {np.round(white_target, 4)}. Brightness (Y row sum): "
              f"{100 * (row_sums[1] - 1):+.1f} %. Neutral color, X and Z after dividing by the "
              f"Y row sum: {100 * chroma_err[0]:+.1f} %, {100 * chroma_err[2]:+.1f} %")
        with open(f"{round_dir}{problem}a_{rnd}_color_matrix.txt", "w") as f:
            f.write(wb.matrix_latex(M, row_sums, white_target,
                                    f"Problem 7, {rnd}: least squares color matrix"))

        # Not a deliverable: how the matrix does per chart, and the comparison figure.
        for chart_n, balanced in charts:
            g = all_patches[(all_patches["round"] == rnd) & (all_patches["chart"] == chart_n)]
            usable = g["usable"].to_numpy()
            scaled, _ = wb.scale_to_reference(balanced, ref_XYZ, usable)
            de, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable, M=M)
            print(f"\t{rnd} chart {chart_n}: CIEDE2000 with this matrix, mean "
                  f"{de[usable].mean():.2f}, worst {de[usable].max():.2f}")
            wb.comparison_figure(g[["R", "G", "B"]].to_numpy(), balanced, M, ref_XYZ,
                                 ref_illuminant_xy, usable, f"{rnd} chart {chart_n}",
                                 f"{round_dir}{problem}_{rnd}_chart{chart_n}_comparison.png")


    # Problem 8: error after the matrix, per patch, and which patches stay worst
    problem = "08"
    print(f"\n**** Problem {problem} ****")
    summary_rows = []
    for rnd, M in color_matrices.items():
        round_dir = f"{output_path}{rnd}/"
        plot_rows, after_all = [], []
        for (r, chart_n), balanced in balanced_means.items():
            if r != rnd:
                continue
            usable = all_patches[(all_patches["round"] == rnd) &
                                 (all_patches["chart"] == chart_n)]["usable"].to_numpy()
            scaled, _ = wb.scale_to_reference(balanced, ref_XYZ, usable)
            before, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable)        # sRGB primaries
            after, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable, M=M)    # fitted matrix
            plot_rows.append((f"{rnd} chart {chart_n}", before, after, usable))
            after_all.append(np.where(usable, after, np.nan))
            worst3 = np.argsort(np.where(usable, after, -np.inf))[::-1][:3]
            summary_rows.append(dict(round=rnd, chart=chart_n, n=int(usable.sum()),
                                     mean_before=before[usable].mean(), worst_before=before[usable].max(),
                                     mean_after=after[usable].mean(), worst_after=after[usable].max(),
                                     worst3=", ".join(f"{k + 1} {psu.COLORCHECKER_NAMES[k]}" for k in worst3)))
        print(f"DELIVERABLE {problem}a: per-patch CIEDE2000 before and after, {rnd} "
              f"-> {round_dir}{problem}a_{rnd}_dE_before_after.png")
        wb.plot_de_before_after(plot_rows, psu.COLORCHECKER_NAMES,
                                f"Per-patch error before and after the color matrix, {rnd}",
                                f"{round_dir}{problem}a_{rnd}_dE_before_after.png")
        # Worst patches for the round: mean over its charts, ignoring clipped values.
        stack = np.vstack(after_all)
        n_ok = np.sum(~np.isnan(stack), axis=0)          # patches clipped on every chart: 0
        round_mean = np.where(n_ok > 0, np.nansum(stack, axis=0) / np.maximum(n_ok, 1), np.nan)
        worst_round = np.argsort(np.nan_to_num(round_mean, nan=-1))[::-1][:3]
        print(f"DELIVERABLE {problem}c: worst patches after the matrix, {rnd} (mean over charts): " +
              ", ".join(f"{k + 1} {psu.COLORCHECKER_NAMES[k]} ({round_mean[k]:.2f})" for k in worst_round))

    summary = pd.DataFrame(summary_rows)
    print(f"DELIVERABLE {problem}b: mean and worst CIEDE2000 before and after the matrix "
          f"-> {output_path}{problem}b_dE_before_after.txt")
    print(summary.to_string(index=False, float_format="{:.2f}".format))
    tex.generate_table([summary["round"].tolist(), summary["chart"].tolist(), summary["n"].tolist()] +
                       [[f"{v:.2f}" for v in summary[c]] for c in
                        ("mean_before", "worst_before", "mean_after", "worst_after")] +
                       [summary["worst3"].tolist()],
                       labels=["Round", "Chart", "Patches", "Mean before", "Worst before",
                               "Mean after", "Worst after", "Three worst after"],
                       caption="CIEDE2000 over usable patches before (balanced camera RGB treated as "
                               "linear sRGB) and after the fitted 3x3 color matrix.",
                       output_path=f"{output_path}{problem}b_dE_before_after.txt",
                       escape=True, VERBOSE=False)
    print(f"DELIVERABLE {problem}d: written explanation of the residual (in the report)")


    # Problem 9 (Part 3): items 5-8 independently for each chart of a round
    problem = "09"
    print(f"\n**** Problem {problem} ****")
    chart_matrices = {}                       # (round, chart) -> M fit to that chart alone
    per_chart_results = {}                    # round -> Problem 9 table, for Problem 10
    for rnd in dict.fromkeys(r for r, _ in balanced_means):
        round_dir = f"{output_path}{rnd}/"
        rows, eqs, Ms = [], [], {}
        for (r, chart_n), balanced in balanced_means.items():
            if r != rnd:
                continue
            g = all_patches[(all_patches["round"] == rnd) & (all_patches["chart"] == chart_n)]
            usable, file = g["usable"].to_numpy(), g["file"].iloc[0]
            scaled, _ = wb.scale_to_reference(balanced, ref_XYZ, usable)
            M = wb.fit_matrix(scaled, ref_XYZ, usable)             # this chart only
            chart_matrices[(rnd, chart_n)] = M
            Ms[f"chart {chart_n}"] = M
            de, _ = wb.delta_e_vs_reference(scaled, ref_XYZ, usable, M=M)
            worst = int(np.argmax(np.where(usable, de, -np.inf)))
            rows.append(dict(chart=chart_n, file=file, n=int(usable.sum()), mean=de[usable].mean(),
                             worst=de[worst], worst_patch=f"{worst + 1} {psu.COLORCHECKER_NAMES[worst]}"))
            eqs.append(wb.matrix_latex(M, M.sum(axis=1), wb.white_xyz_y1(ref_XYZ),
                                       f"Problem 9, {rnd} chart {chart_n} ({file})")
                       .replace("M = ", f"M_{{{chart_n}}} = ")
                       .replace("M\\begin", f"M_{{{chart_n}}}\\begin"))
            print(f"\t{rnd} chart {chart_n} ({file}): matrix fit to this chart alone\n"
                  f"{np.array2string(M, precision=4, suppress_small=True)}")
        df9 = pd.DataFrame(rows)
        per_chart_results[rnd] = df9
        print(f"DELIVERABLE {problem}a: {rnd} mean and worst CIEDE2000 per chart, each with its own "
              f"matrix -> {round_dir}{problem}a_{rnd}_per_chart_dE.txt")
        print(df9.to_string(index=False, float_format="{:.2f}".format))
        tex.generate_table([df9["chart"].tolist(), df9["file"].tolist(), df9["n"].tolist(),
                            [f"{v:.2f}" for v in df9["mean"]], [f"{v:.2f}" for v in df9["worst"]],
                            df9["worst_patch"].tolist()],
                           labels=["Chart", "File", "Patches", "Mean dE00", "Worst dE00", "Worst patch"],
                           caption=f"Each chart under {rnd} light fit independently (white balance, "
                                   f"matrix and error). Metric: CIEDE2000, usable patches.",
                           output_path=f"{round_dir}{problem}a_{rnd}_per_chart_dE.txt",
                           escape=True, VERBOSE=False)
        print(f"DELIVERABLE {problem}b: {rnd} the three matrices -> {round_dir}{problem}b_{rnd}_matrices.txt")
        with open(f"{round_dir}{problem}b_{rnd}_matrices.txt", "w") as f:
            f.write("\n".join(eqs))
        cols = wb.column_comparison(Ms)
        print(f"DELIVERABLE {problem}c: {rnd} matrix columns compared as directions "
              f"-> {round_dir}{problem}c_{rnd}_column_angles.txt")
        print(cols.to_string(index=False, float_format="{:.3f}".format))
        tex.generate_table([cols["pair"].tolist()] +
                           [[f"{v:.2f}" for v in cols[f"angle_{c}"]] for c in "RGB"] +
                           [[f"{v:.3f}" for v in cols[f"length_{c}"]] for c in "RGB"],
                           labels=["Pair", "R angle (deg)", "G angle (deg)", "B angle (deg)",
                                   "R length ratio", "G length ratio", "B length ratio"],
                           caption=f"Columns of the per-chart matrices under {rnd} light compared as "
                                   f"directions: angle between corresponding columns, and ratio of "
                                   f"their lengths.",
                           output_path=f"{round_dir}{problem}c_{rnd}_column_angles.txt",
                           escape=True, VERBOSE=False)


    # Problem 10: chart-to-chart spread as the uncertainty; the color entries of results.json
    problem = "10"
    print(f"\n**** Problem {problem} ****")
    import json
    for rnd, df9 in per_chart_results.items():
        round_dir = f"{output_path}{rnd}/"
        before = de_df[de_df["round"] == rnd].set_index("chart")["mean"].reindex(df9["chart"]).to_numpy()
        after = df9["mean"].to_numpy()
        # Uncertainty = sample standard deviation over the charts (ddof=1): the scatter of one
        # chart's result. With three charts this is itself only a rough estimate.
        stats = {name: dict(mean=v.mean(), std=v.std(ddof=1), lo=v.min(), hi=v.max())
                 for name, v in (("before", before), ("after", after))}
        worst_counts = df9["worst_patch"].value_counts()
        worst_patch = worst_counts.index[0]
        worst_vals = df9.loc[df9["worst_patch"] == worst_patch, "worst"].to_numpy()
        a = stats["after"]
        print(f"DELIVERABLE {problem}a: {rnd} mean CIEDE2000 after correction = "
              f"{a['mean']:.2f} +/- {a['std']:.2f} (sample std over {len(after)} charts; "
              f"range {a['lo']:.2f} to {a['hi']:.2f}). Before correction: "
              f"{stats['before']['mean']:.2f} +/- {stats['before']['std']:.2f}. "
              f"-> {round_dir}{problem}a_{rnd}_dE_uncertainty.txt")
        tex.generate_table([["Before correction (no matrix)", "After correction (per-chart matrix)"]] +
                           [[f"{stats[k][c]:.2f}" for k in ("before", "after")]
                            for c in ("mean", "std", "lo", "hi")],
                           labels=["Mean $\\Delta E_{00}$", "Mean over charts", "Std (charts)", "Min", "Max"],
                           caption=f"Mean CIEDE2000 under {rnd} light across {len(after)} nominally "
                                   f"identical charts. Uncertainty: sample standard deviation across charts.",
                           output_path=f"{round_dir}{problem}a_{rnd}_dE_uncertainty.txt",
                           escape=False, VERBOSE=False)
        # Keys and format from the Project 1 results.template.json: [value, uncertainty],
        # with the uncertainty taken as the sample std across this round's charts.
        def pair(v):
            return [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2)]
        color_entries = {
            "delta_e_metric": "CIEDE2000",
            "colorchecker_reference_set": wb.REFERENCE,
            "delta_e_mean_before_ccm": pair(before),
            "delta_e_mean_after_ccm": pair(after),
            "delta_e_worst_after_ccm": pair(df9["worst"].to_numpy()),
        }
        with open(f"{round_dir}{problem}b_{rnd}_results_color.json", "w") as f:
            json.dump(color_entries, f, indent=2)
        print(f"DELIVERABLE {problem}b: {rnd} color entries for results.json, "
              f"[value, uncertainty] -> {round_dir}{problem}b_{rnd}_results_color.json")
        print(json.dumps(color_entries, indent=2))
        print(f"\tworst patch after correction was {worst_patch} on {worst_counts.iloc[0]} of "
              f"{len(df9)} charts, CIEDE2000 {worst_vals.mean():.2f} +/- "
              f"{(worst_vals.std(ddof=1) if len(worst_vals) > 1 else 0):.2f}")





        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


    cols = ["round", "kind", "chart", "file", "iso", "exposure"]
    tex.generate_table([catalog[c].tolist() for c in cols], labels=cols, caption="All data", output_path=f"{output_path}catalog_table.txt")