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

from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

def plot_two_images(first, second, first_title, second_title, save_path, first_color=None, second_color=None, SHOW=False):
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

    plt.tight_layout()
    if SHOW:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = False
    TESTING = True

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
    print(f"DELIVERABLE {problem}: fraction of each frame at or above saturation:")
    print(catalog[["round", "kind", "chart", "file", "sat_frame_pct"] +
                  [f"sat_{ch}_pct" for ch in measured_sat]].to_string(
                      index=False, float_format="{:.3f}".format))
    # TODO NOW: This should be a LaTeX table

    chart_label = [str(c) if pd.notna(c) else "white" for c in catalog["chart"]]
    tex.generate_table([catalog["round"].tolist(), chart_label, catalog["file"].tolist(),
                        catalog["iso"].tolist(), catalog["exposure"].tolist(),
                        [f"{v:.3f}" for v in catalog["sat_frame_pct"]]],
                       labels=["Round", "Chart", "File", "ISO", "Exposure", "% at saturation"],
                       caption=f"Captures from the {cam_spec.name}, all rounds",
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

            plot_two_images(linear, psu.linear_to_srgb(quarter), f'Original image in linear light from {round_df.iloc[first_img]["file"]}', f'Quarter resolution linear light image from {round_df.iloc[first_img]["file"]}', f"{output_path_for_lighting}{problem}_{lighting_type}_quarter_comparison.png", first_color='gray', second_color=None, SHOW=False)


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

    print(f"DELIVERABLE {problem}b: The ratio of the corner region to the center region on a white flat both before and after dividing out the shading map, per color channel...")
    print(pd.DataFrame(corner_center_ratios).set_index("round").round(3))
    print(f"DELIVERABLE {problem}c: Plotting the central horizontal profiles from each round on the same axis...")
    shading.plot_profiles(profiles, model_profile=model[h // 2],
                        title="Horizontal profile, normalized to center",
                        save_path=f"{output_path}{problem}_horizontal_profiles.png")





        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


    cols = ["round", "kind", "chart", "file", "iso", "exposure"]
    tex.generate_table([catalog[c].tolist() for c in cols], labels=cols, caption="All data", output_path=f"{output_path}catalog_table.txt")