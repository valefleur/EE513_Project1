# hw1.py
import my_library.analyze as analyze

if __name__ == "__main__":
    sony = 'images/a7sii/DSC04632.ARW'
    sony_ordinary = 'images/a7sii/RAW_and_JPEG/A7S04652.JPG'
    sony_both = 'images/a7sii/RAW_and_JPEG/A7S04652'

    iphone_path = 'images/iPhone16/'
    not_lr = 'not_lightroom/'
    lightroom = 'Lightroom/'

    # First time using Lightroom app; didn't understand the manual focus so
    # everything is too out of focus to use
    # lightroom_checkerboard = f'{iphone_path}{lightroom}/Checkerboard/'
    # checkerboard_img = f'{lightroom_checkerboard}APC_0192.dng'

    not_lr_calib_path = f'{iphone_path}{not_lr}iphone16_calibration/'
    lr_calib_path = f'{iphone_path}{lightroom}hw1/hw1_checkerboard2/'
    exp125th_calib_path = f'{iphone_path}{lightroom}hw1/EXP125th/'
    nov_07_calib_path = f'{iphone_path}{lightroom}hw1/Nov_07/'

    iphone = f'{iphone_path}Terry.DNG'
    iphone_ordinary = f'{iphone_path}RAW_and_HEIC/IMG_6980.HEIC'
    iphone_both = f'{iphone_path}RAW_and_HEIC/IMG_6980'


    # print(f"Sony alpha 7s II:")
    cam = 'sony'
    # analyze.bayer_mosaic(sony)
    # analyze.metadata(sony)
    # analyze.metadata(sony_ordinary)
    # sony_raw_img, sony_raw_name = analyze._open_image(f'{sony_both}.ARW')
    # sony_jpg_img, sony_jpg_name = analyze._open_image(f'{sony_both}.JPG')
    # analyze.zoomed_way_in(sony_raw_img, sony_jpg_img, sony_raw_name, camera=cam)
    # analyze.calibrate_camera('images/a7sii/a7sii_calibration/', camera=cam)
    # analyze.calibrate_camera('images/a7sii/test/', camera=cam)

    # analyze.generate_side_by_side(f'{sony_both}.ARW', f'{sony_both}.JPG')

    print(f"------------------")

    print(f"iPhone 16 Pro Max:")
    cam = 'iphone'
    # analyze.bayer_mosaic(iphone)
    # analyze.metadata(iphone)
    # analyze.metadata(iphone_ordinary)
    # iphone_raw_img, iphone_raw_name = analyze._open_image(f'{iphone_both}.DNG')
    # iphone_heic_img, iphone_heic_name = analyze._open_image(f'{iphone_both}.HEIC')
    # analyze.zoomed_way_in(iphone_raw_img, iphone_heic_img, iphone_raw_name, camera=cam)
    analyze.calibrate_camera(nov_07_calib_path, camera=cam)
    # analyze.calibrate_camera('images/iPhone16/test/', camera=cam)

    # analyze.generate_side_by_side(f'{iphone_both}.DNG', f'{iphone_both}.HEIC')

