# hw1.py
import my_library.analyze as analyze

if __name__ == "__main__":
    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'

    K, distortion = analyze.calibrate_camera('images/iPhone16/all_calib_images/', camera=cam)
    # K, distortion = analyze.calibrate_camera('images/iPhone16/test/', camera=cam)

    # analyze.undistort('images/iPhone16/all_calib_images/D1044335-EB95-4F60-8709-7B942E0B6825.DNG', camera=cam)

    # straigh_lines_img = 'images/iPhone16/Straight_Lines/A56046EA-FEEA-4284-A51A-2DAF03198E01.DNG'
    straigh_lines_img = 'images/iPhone16/Straight_Lines/3D244010-6AE0-40AD-8050-317C858BC691.DNG'
    analyze.undistort(straigh_lines_img, K, distortion, cam)
