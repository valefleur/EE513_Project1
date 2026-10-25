# hw2.py
import my_library.analyze as analyze
import my_library.mtf_analysis as mtf
import my_library.image_metadata as img

if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    SPEED_UP = True
    TEST = False

    camera_path = 'images/iPhone16/'
    calib_path = 'all_calib_images/'
    lightroom = 'Lightroom/'

    slanted_path = f'{camera_path}{lightroom}hw2_slanted/'
    good_slanted_dir = f'{slanted_path}good_lighting/'
    bad_slanted_dir  = f'{slanted_path}bad_lighting/'

    output = 'output/hw2/'
    if TEST:
        calib_path = 'test/'

    calib_imgs_path = f'{camera_path}{calib_path}'

    # straight_lines_img = f'{calib_imgs_path}D1044335-EB95-4F60-8709-7B942E0B6825.DNG'
    straight_lines_img = f'{calib_imgs_path}3FB78461-4AFA-46EE-834B-9E01675900C4.DNG'
    # straight_lines_img = f'{calib_imgs_path}A56046EA-FEEA-4284-A51A-2DAF03198E01.DNG'
    # straight_lines_img = f'{calib_imgs_path}3D244010-6AE0-40AD-8050-317C858BC691.DNG'


    # K, distortion, corners = analyze.calibrate_camera(calib_imgs_path, oneoff=straight_lines_img, camera=cam)

    # original, undistorted = analyze.undistort(straight_lines_img, K, distortion, corners, cam)

    if not SPEED_UP:
        # Problem 7
        problem_num = "07"
        print(f"\n**** Problem {problem_num} ****")
        analytical_mtf50 = mtf.main(output)


        # Problem 8
        problem_num = "08"
        print(f"\n**** Problem {problem_num} ****")
        analytical_mtf50_in_um = analytical_mtf50 / 1000
        analytical_mtf50_in_pixels = analytical_mtf50_in_um * mtf.PITCH_UM * 2 # multiply by 2 since we are using the green-plane
        print(f"Analytical system MTF50 is:\n\t{analytical_mtf50_in_um:.4f} cycles/um\n\t{analytical_mtf50:.1f}  cycles/mm\n\t{analytical_mtf50_in_pixels:.4f} cycles/pixel (green-plane)")


    if not SPEED_UP:
        # Problem 12
        problem_num = "12"
        print(f"\n**** Problem {problem_num} ****")
        raw_slant_paths = analyze._list_raw_files(good_slanted_dir)
        analyze.contact_sheet(raw_slant_paths, camera=cam, output=output, question=problem_num)

        # Problem 13
        problem_num = "13"
        print(f"\n**** Problem {problem_num} ****")
        img.assert_one_camera(raw_slant_paths)