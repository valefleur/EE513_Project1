# hw3.py
import os

import numpy as np
import my_library.analyze as analyze
import my_library.image_metadata as img
import my_library.photon_transfer_curve as ptc
import my_library.latex_formatter as tex
import hw3_part2_sim as sim

if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = True
    LOW_ISO = True

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'

    iso320_path = f'{camera_path}{lightroom}/ISO320/'

    iso64_path = f'{camera_path}{lightroom}/ISO64/'
    iso64_sunset = f'{camera_path}{lightroom}/ISO64_Sunset/'

    iso64_dark = f'{camera_path}{lightroom}/ISO64_Dark/'

    output_path = 'output/hw3/'

    if LOW_ISO:
        # image_path = iso64_path
        image_path = iso64_sunset
        iso = 64
    else:
        image_path = iso320_path
        iso = 320



    rows = img.print_raw_metadata(image_path)
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

    raws = analyze._list_raw_files(image_path)

    if not SPEED_UP:
        # Problem 1
        print(f"\n**** Problem 1 ****")
        analyze.contact_sheet(raws, camera=cam, output=output_path, labels=exposure_times)

        # Problem 2
        print(f"\n**** Problem 2 ****")
        img.assert_one_camera(raws)

        # Problem 3
        print(f"\n**** Problem 3 ****")
        #The black level per channel from the iso64_dark files, the mean of your shortest dark frames in digital numbers
        # dark_raws = analyze._list_raw_files(iso64_dark)
        # dark_metadata = [img.raw_metadata(r) for r in dark_raws]
        # black_levels = [m['black_levels'] for m in dark_metadata]
        # tex.generate_table([black_levels], labels=[f"Black Levels (per channel)"], output_path=f"{output_path}03_black_levels.txt", caption=f"Black Levels for {cam} images")
        # mean_black_levels = [sum(b) / len(b) for b in zip(*black_levels)]
        # print(f"Mean black levels: {mean_black_levels}")

        ptc.dark_frame_stats(iso64_dark, K=0.256)


        # Problem 4
        #Before fitting any of your own data, run your photon transfer code on the simulated sensor in the lecture 5 demo notebook, which reports the true gain and read noise it used to generate the frames.
        # Deliverable: Your recovered gain and read noise beside the true values, as percentage errors.
        print(f"\n**** Problem 4 ****")
        sim.main()


    # Problem 5
    print(f"\n**** Problem 5 ****")
    fit, fit1, fig = ptc.run_real_ptc(image_path, iso=iso, output=output_path, title=f"ISO {iso}", plot_path=f"{output_path}06_ptc_iso{iso}.png")
    # # cross-check with the other green pixel of the tile:
    pairs = ptc.load_flat_pairs(image_path, which=1)
    rows = ptc.analyze_pairs(pairs)

    # ptc.plot_residuals(rows, fracs=(0.7, 0.1), saturation_dn=4077,
    #                title=f"ISO {iso}", plot_path=f"{output_path}06_resid_iso{iso}.png")
    # ptc.fit_range_table(rows, saturation_dn=4077)
    # ptc.report_ptc(rows, title=f"ISO {iso}, low signal", plot_path=f"{output_path}06_ptc_iso{iso}_low.png", saturation_dn=4077)



    # Problem 9
    print(f"\n**** Problem 9 ****")
    mean = np.array([r["mean"] for r in rows])
    var = np.array([r["var"] for r in rows])
    span = 4077 - rows[0]["black"]            # ADC clip point, not the nominal 4095
    fit10 = ptc.fit_ptc(mean, var, span, 0.1)
    knee, fig = ptc.plot_snr(rows, fit10, saturation_dn=4077, fit_frac=0.1,
                            plot_path=f"{output_path}09_snr_iso64.png", title="ISO 64")

    # Problem 10
    print("\n**** Problem 10 ****")
    first_iso64_file = f"{iso64_sunset}"
    first_iso64_file += sorted([f for f in os.listdir(first_iso64_file) if f.lower().endswith(tuple(img.RAW_SUFFIXES))])[0]
    print(f"First ISO 64 file: {first_iso64_file}")
    img.compare_noise_profile(first_iso64_file, K=fit10["K"], read=fit10["read"],
                            s_K=fit10["sK"], s_read=fit10["s_read"])

    first_iso320_file = f"{iso320_path}"
    first_iso320_file += sorted([f for f in os.listdir(first_iso320_file) if f.lower().endswith(tuple(img.RAW_SUFFIXES))])[0]
    print(f"\nFirst ISO 320 file: {first_iso320_file}")
    k_320 = 1.2178713783922652
    read_320 = 2.8559242580638777
    s_K_320 = 0.01692223091547653
    s_read_320 = 0.28043630579517187
    # print(f" *~*~*~* 320 vars: K={fit10["K"]}, read={fit10["read"]}, s_K={fit10["sK"]}, s_read={fit10["s_read"]}")
    img.compare_noise_profile(first_iso320_file, K=fit10["K"], read=fit10["read"],
                            s_K=fit10["sK"], s_read=fit10["s_read"])


    # Problem 11
    print("\n**** Problem 11 ****")
    dark_rows = ptc.dark_frame_stats(iso64_dark, K=fit10["K"])   # or reuse the Problem 3 result
    dc = ptc.dark_current_fit(dark_rows, fit10["K"], fit10["sK"],
                            plot_path=f"{output_path}11_dark_current.png", title="Dark frames, ISO 64")