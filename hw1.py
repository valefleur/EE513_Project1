# hw1.py
import my_library.analyze as analyze

if __name__ == "__main__":
    sony = 'images/a7sii/DSC04632.ARW'
    sony_ordinary = 'images/a7sii/RAW_and_JPEG/A7S04652.JPG'
    iphone = 'images/iPhone16/Terry.DNG'
    iphone_ordinary = 'images/iPhone16/RAW_and_HEIC/IMG_6980.HEIC'

    print(f"Sony alpha 7s II:")
    # analyze.bayer_mosaic(sony)
    # analyze.metadata(sony)
    analyze.metadata(sony_ordinary)

    print(f"iPhone 16 Pro Max:")
    # analyze.bayer_mosaic(iphone)
    analyze.metadata(iphone)
    # analyze.metadata(iphone_ordinary)