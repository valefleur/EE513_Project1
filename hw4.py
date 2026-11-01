# hw4.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rawpy
import psu_imaging as psu
import matplotlib.pyplot as plt

import opencv4 as cv4

import my_library.analyze as analyze
import my_library.image_metadata as img
import my_library.photon_transfer_curve as ptc
import my_library.linear_light as ll
import my_library.generate as gen
import my_library.latex_formatter as tex
import my_library.shading as shading
import my_library.catalog as datalog

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
    incadesent_path = f'{camera_path}{lightroom}{hw}incadesent/'
    cloudy_path = f'{camera_path}{lightroom}{hw}outdoors_cloudy/'
    shade_path = f'{camera_path}{lightroom}{hw}outdoors_shade/'

    if TESTING:
        color_sets = [flor_path, led_path]
        rounds = {'florescent': flor_path, 'led': led_path}
    else:
        color_sets = [flor_path, led_path, incadesent_path, f'{cloudy_path}/set1', f'{cloudy_path}/set2', shade_path]
        rounds = {'florescent': flor_path, 'led': led_path, 'incadesent': incadesent_path,
            'cloudy1': f'{cloudy_path}/set1', 'cloudy2': f'{cloudy_path}/set2', 'shade': shade_path}

    output_path = 'output/hw4/'

    flats = {}

    catalog = datalog.build_catalog(rounds, cam_spec.saturation_dn)
    datalog.print_inventory(catalog)
    datalog.check_rounds(catalog)
    catalog.to_csv(f"{output_path}catalog.csv", index=False)


    # Assert one camera type for all images in all color_sets
    raws = []
    for path in color_sets:
        raws.extend(analyze._list_raw_files(path))
    img.assert_one_camera(raws)

    # Since all captures were from the same camera, we only benefit from looking at the Bayer mosaic once.
    a_directory = Path(color_sets[0])
    an_image_file = next((f for f in a_directory.iterdir() if f.is_file() and f.suffix.lower() in img.RAW_SUFFIXES), None)
    print(f"Printing the Bayer mosaic for {an_image_file}...")
    analyze.bayer_mosaic(an_image_file)

    for lighting_type, path_to_images in rounds.items():
        print(f"\n\n~*~*~*~*~*~*~*~*~  Now looking at images taken with {lighting_type} lighting. ~*~*~*~*~*~*~*~")
        output_path_for_lighting = f"{output_path}{lighting_type}/"

        img_data = pd.DataFrame(columns=['file', 'camera', 'exposure_s', 'iso', 'black_levels', 'white_level', 'bayer', 'shape'])

        # TODO clean up this pre-problems section to make it more useful & concise
        rows, img_df = img.print_raw_metadata(path_to_images, img_data)
        # rows = img.print_raw_metadata(path_to_images)
        # print(f"First row: {rows[0]}")
        file_names = []
        exposure_times = []
        isos = []
        for row in rows:
            file_names.append(row['file'])
            exposure_times.append(img._format_exposure(row['exposure_s']))
            isos.append(row['iso'])
        # labels should be every other exposure time, so that the contact sheet is not too cluttered
        labels = ["File Name", "Exposure Time", "ISO"]
        tex.generate_table([file_names, exposure_times, isos], labels=labels, output_path=f"{output_path_for_lighting}01_exposure_ISO.txt", caption=f"Metadata for {cam} images")

        raws = analyze._list_raw_files(path_to_images)

        image_paths = img.get_image_paths(path_to_images) # TODO ensure it returns the order expected by the dataframe, or sort it to match raws

        first_img = 0

        # From the output of the histogram data, we know the saturation level is 4095.
        '''
        TODO rewrite the loop logic so we look at the saturation levels of all images before
        starting in on the problems and their calculations to use a global saturation level for
        the camera.
        '''
        SATURATION = 4095

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
        # max_brightness_per_img = [] # TODO is max_brightness worth keeping?
        VERBOSE = False
        for i in image_paths:
            # print the fraction of each frame at saturation
            clip_report = analyze.clip_report(i, sat=SATURATION)
            print(f"\nHistogram analysis for {os.path.basename(i)}:")
            print(f"\tPercentage of pixels at saturation:")
            if VERBOSE:
                for k, v in clip_report.items():
                    if k != 'spike_ratio' and isinstance(v, float):
                        print(f"\t\t{k}: {v*100:.3f}%")
            print(f"\t\tSpike ratio: {clip_report['spike_ratio']:.3f}")
            analyze.histogram(i, output_path_for_lighting, question=problem, VERBOSE=VERBOSE)
            # max_brightness_point = analyze.histogram(i, set_output_path, question=problem)
            # max_brightness_per_img.append(max_brightness_point)

        # add the max_brightness to the dataframe
        # img_df['max_brightness'] = max_brightness_per_img

        # histogram the raw values to look for a spike in the upper half

        # Problem 2
        problem = "02"
        print(f"\n**** Problem {problem} ****")
        print(f'HEAD:\n{img_df.head()}')
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
        with rawpy.imread(str(raws[first_img])) as raw:
            raw_img = raw.raw_image_visible
            # 1. Convert to linear light
            black = img_df['black_levels'][first_img][0]
            # saturation = img_df['max_brightness'][first_img] # Error: don't use max brightness
            # linear = ll.raw_to_linear_light(raw_img, black_levels=img_df['black_levels'][first_img], saturation_level=img_df['white_level'][first_img]) # WRONG, use saturation level instead of white level
            linear = ll.raw_to_linear_light(raw_img, black_level=black, saturation_level=SATURATION)

            # 2. Resize down 25% by sampling the 2x2 blocks to one pixel
            # using the red site, the mean of the two green sites, and the blue site
            # my camera's CFA pattern comes out to BGGR, so the red site is at (1, 1), the green sites are at (0, 1) and (1, 0), and the blue site is at (0, 0)
            quarter = ll.quarter_rgb(linear, pattern=raw.raw_pattern, color_desc=raw.color_desc)

            plot_two_images(linear, psu.linear_to_srgb(quarter), f'Original image in linear light from {img_df['file'][first_img]}', f'Quarter resolution linear light image from {img_df['file'][first_img]}', f"{output_path_for_lighting}{problem}_linear_light.png", first_color='gray', second_color=None, SHOW=False)


        # # Problem 4
        # problem = "04"
        # print(f"\n**** Problem {problem} ****")
        '''
        Use OpenCV4 to locate the 24 patches of flat color in the quarter resolution image.  Sample
        no more than the middle 50% of each patch to avoid the black borders and any other edge
        effects. Measure the mean and standard deviation of each patch. The standard deviation is a
        measure of how flat the patch is, and the mean is a measure of its color. The mean should
        be reported in linear light, not sRGB. The standard deviation should be reported in 8-bit
        levels and linear light.
        '''



    # Problem 3
    problem = "03"
    print(f"\n**** Problem {problem} ****")
    for name, white_path in catalog[catalog.kind == "white"].set_index("round").path.items():
        #print name as all upper case
        print(f"\nProcessing white image: {str(name).upper()} - {os.path.basename(white_path)}")

        # Grab the image of the white sheet, which is the last one in the list of raws, and convert it to linear light
        white_flat_rgb = ll.load_linear_quarter_rgb(white_path, saturation_dn=SATURATION)
        map_of_white_flat = shading.shading_map_from_white(white_flat_rgb)
        flats[name] = map_of_white_flat
        # plot a linearlized white flat image next to a normalized white flat image
        plot_two_images(white_flat_rgb, map_of_white_flat, f'White flat in linear light from {img_df['file'][len(img_df)-1]}', f'Normalized white flat in linear light from {img_df['file'][len(img_df)-1]}', f"{output_path_for_lighting}{problem}_white_flat_normalized.png", first_color=None, second_color=None, SHOW=False)

        analytic_lens_shading = shading.cos4_model(white_flat_rgb.shape[0], white_flat_rgb.shape[1], camera_spec=cam_spec, bin_factor=2)
        # print(f"Analytic lens shading mean: {analytic_lens_shading.mean(axis=(0,1))}")

        fraction = 0.10
        print(f"Percentage of pixels per RGB channel with value >= 0.999, i.e. clipping: {(white_flat_rgb >= 0.999).mean(axis=(0,1))}")
        print(f"The per RGB channel mean of the center {fraction*100:.0f}% of the shading map of the white flat: {shading.center_mean(map_of_white_flat, frac=fraction)}")

        '''
        with rawpy.imread(str(raws[-1])) as raw:
            raw_img = raw.raw_image_visible
            black = img_df['black_levels'][len(img_df)-1][0]
            # saturation = img_df['max_brightness'][len(img_df)-1] # Error: Don't use max brightness
            white_flat_linear = ll.raw_to_linear_light(raw_img, black_level=black, saturation_level=SATURATION)

            # Determine the center (ideally we'd use the optical axis center principle point, but we don't have that info)
            height, width = white_flat_linear.shape
            print(f'White flat in linear light height: {height}, width: {width}')
            center_x, center_y = width // 2, height // 2
            x = np.arange(width) - center_x
            y = np.arange(height) - center_y

            # Calculate the distance squarted from center for each pixel
            xv, yv = np.meshgrid(x, y)
            distance_squared = xv**2 + yv**2

            # Normalize the flat field to its own center and divide, treating each of the three channels separately rather than applying one gray correction.
            # white_flat_linear = white_flat_linear / white_flat_linear[center_y, center_x]
            # white_flat_linear_normalized = ll.normalize_flat(white_flat_linear) # Error: Use the quarter-res RGB image so we get the three channels separately rather than applying one gray correction.
            white_flat_rgb = ll.quarter_rgb(white_flat_linear, pattern=raw.raw_pattern, color_desc=raw.color_desc)
            white_flat_linear_normalized = ll.normalize_flat(white_flat_rgb)

            # Plot white_flat_linear next to white_flat_linear_normalized
            plot_two_images(white_flat_rgb, white_flat_linear_normalized, f'White flat in linear light from {img_df['file'][len(img_df)-1]}', f'Normalized white flat in linear light from {img_df['file'][len(img_df)-1]}', f"{set_output_path}{problem}_white_flat_normalized.png", first_color='gray', second_color='gray', SHOW=True)

            # Get an analytical calculate cos(theta) as the distance from the center of the image
            # to each pixel divided by the focal length in pixels, and raise it to the fourth
            # power to get cos^4(theta) for each pixel.
            # This is the the theoretical lens shading of the camera lens for comparison with the
            # measured shading map from the white flat image.  A shading map will include real world
            # effects such as shading from the lamps, improperly placed lens hoods, etc. that are
            # not part of the theoretical lens shading.
            analytic_lens_shading = gen.cos4_map(white_flat_rgb.shape[0], white_flat_rgb.shape[1], bin_factor=2)
        '''


        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


        # # Problem 3
        # problem = "03"
        # print(f"\n**** Problem {problem} ****")


cols = ["round", "kind", "chart", "file", "iso", "exposure"]
tex.generate_table([catalog[c].tolist() for c in cols], labels=cols, caption="All data", output_path=f"{output_path}catalog_table.txt")