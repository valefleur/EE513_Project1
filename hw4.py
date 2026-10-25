# hw4.py
import os

import numpy as np
import pandas as pd
import psu_imaging as psu
import my_library.analyze as analyze
import my_library.image_metadata as img
import my_library.photon_transfer_curve as ptc
import my_library.linear_light as ll
import my_library.latex_formatter as tex

if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = False
    TESTING = True

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'
    hw = 'hw4_color_charts/'

    light_types = {'flor': 'florescent/', 'led': 'led/', 'inc': 'incadesent/', 'cloudy': 'outdoors_cloudy/', 'shade': 'outdoors_shade/'}

    flor_path = f'{camera_path}{lightroom}{hw}{light_types["flor"]}'
    led_path = f'{camera_path}{lightroom}{hw}{light_types["led"]}'
    incadesent_path = f'{camera_path}{lightroom}{hw}{light_types["inc"]}'
    cloudy_path = f'{camera_path}{lightroom}{hw}{light_types["cloudy"]}'
    shade_path = f'{camera_path}{lightroom}{hw}{light_types["shade"]}'

    on_cloudy_set = 0

    if TESTING:
        color_sets = [flor_path, led_path]
    else:
        color_sets = [flor_path, led_path, incadesent_path, f'{cloudy_path}/set1', f'{cloudy_path}/set2', shade_path]

    output_path = 'output/hw4/'

    for path_to_images, lighting in zip(color_sets, light_types.keys()):
        print(f"\n\n~*~*~*~*~*~*~*~  Now looking at images taken with {lighting} lighting. ~*~*~*~*~*~*~*~")

        if lighting == 'cloudy':
            on_cloudy_set = on_cloudy_set + 1
        cloudy_set = f'set{on_cloudy_set}/' if (lighting == 'cloudy') else ''

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
        tex.generate_table([file_names, exposure_times, isos], labels=labels, output_path=f"{output_path}{light_types[lighting]}{cloudy_set}_01_exposure_ISO.txt", caption=f"Metadata for {cam} images")

        raws = analyze._list_raw_files(path_to_images)

        image_paths = img.get_image_paths(path_to_images)


        # Problem 1
        print(f"\n**** Problem 1 ****")
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
        analyze.contact_sheet(raws, camera=cam, output=output_path, question="01", labels=exposure_times)
        analyze.bayer_mosaic(raws[0])
        img.assert_one_camera(raws)
        for i in image_paths:
            analyze.histogram(i, output_path, question='01', camera=cam)

        # histogram the raw values to look for a spike in the upper half

        # # Problem 2
        # print(f"\n**** Problem 2 ****")
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
        # 1. Convert to linear light
        # linear = img.to_linear_light(raws[0])
        # 2. Resize down 25% by sampling the 2x2 blocks to one pixel

        # # Problem 3
        # print(f"\n**** Problem 3 ****")
        #The black level per channel from the iso64_dark files, the mean of your shortest dark frames in digital numbers
        # dark_raws = analyze._list_raw_files(iso64_dark)
        # dark_metadata = [img.raw_metadata(r) for r in dark_raws]
        # black_levels = [m['black_levels'] for m in dark_metadata]
        # tex.generate_table([black_levels], labels=[f"Black Levels (per channel)"], output_path=f"{output_path}03_black_levels.txt", caption=f"Black Levels for {cam} images")
        # mean_black_levels = [sum(b) / len(b) for b in zip(*black_levels)]
        # print(f"Mean black levels: {mean_black_levels}")

        # ptc.dark_frame_stats(iso64_dark, K=0.256)


