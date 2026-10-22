# hw4.py
import os

import numpy as np
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
    LOW_ISO = True

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'
    hw = 'hw4_color_charts/'

    flor_path = f'{camera_path}{lightroom}{hw}florescent/'
    led_path = f'{camera_path}{lightroom}{hw}led/'
    incadesent_path = f'{camera_path}{lightroom}{hw}incadesent/'
    cloudy_path = f'{camera_path}{lightroom}{hw}outdoors_cloudy/'
    shade_path = f'{camera_path}{lightroom}{hw}outdoors_shade/'

    color_sets = [flor_path, led_path, incadesent_path, f'{cloudy_path}/set1', f'{cloudy_path}/set2', shade_path]

    output_path = 'output/hw4/'

    # path_to_images = led_path


    # if not SPEED_UP:

    for path_to_images in color_sets:
        print(f"\n\n~*~*~*~*~*~*~*~  Now looking at {path_to_images}  ~*~*~*~*~*~*~*~")

        # TODO clean up this pre-problems section to make it more useful & concise
        rows = img.print_raw_metadata(path_to_images)
        print(f"First row: {rows[0]}")
        file_names = []
        exposure_times = []
        isos = []
        for row in rows:
            file_names.append(row['file'])
            exposure_times.append(img._format_exposure(row['exposure_s']))
            isos.append(row['iso'])
        # labels should be every other exposure time, so that the contact sheet is not too cluttered
        labels = ["File Name", "Exposure Time", "ISO"]
        tex.generate_table([file_names, exposure_times, isos], labels=labels, output_path=f"{output_path}01_exposure_ISO.txt", caption=f"Metadata for {cam} images")

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
            analyze.histogram(i, output_path, camera=cam)

        # histogram the raw values to look for a spike in the upper half

        # # Problem 2
        # print(f"\n**** Problem 2 ****")

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


