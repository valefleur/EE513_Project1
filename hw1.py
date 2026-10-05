# hw1.py
import my_library.analyze as analyze

if __name__ == "__main__":
    sony = 'images/a7sii/DSC04632.ARW'
    sony_ordinary = 'images/a7sii/RAW_and_JPEG/A7S04652.JPG'
    sony_both = 'images/a7sii/RAW_and_JPEG/A7S04652'
    iphone = 'images/iPhone16/Terry.DNG'
    iphone_ordinary = 'images/iPhone16/RAW_and_HEIC/IMG_6980.HEIC'
    iphone_both = 'images/iPhone16/RAW_and_HEIC/IMG_6980'

    print(f"Sony alpha 7s II:")
    # analyze.bayer_mosaic(sony)
    # analyze.metadata(sony)
    # analyze.metadata(sony_ordinary)
    # sony_raw_img, sony_raw_name = analyze._open_image(f'{sony_both}.ARW')
    # sony_jpg_img, sony_jpg_name = analyze._open_image(f'{sony_both}.JPG')
    # analyze.zoomed_way_in(sony_raw_img, sony_jpg_img, sony_raw_name)
    # analyze.calibrate_camera('images/a7sii/a7sii_calibration/')
    # analyze.calibrate_camera('images/a7sii/test/')

    print(f"------------------")

    print(f"iPhone 16 Pro Max:")
    # analyze.bayer_mosaic(iphone)
    # analyze.metadata(iphone)
    # analyze.metadata(iphone_ordinary)
    iphone_raw_img, iphone_raw_name = analyze._open_image(f'{iphone_both}.DNG')
    iphone_heic_img, iphone_heic_name = analyze._open_image(f'{iphone_both}.HEIC')
    analyze.zoomed_way_in(iphone_raw_img, iphone_heic_img, iphone_raw_name)
    # analyze.calibrate_camera('images/iPhone16/iphone16_calibration/')
    # analyze.calibrate_camera('images/iPhone16/test/')
