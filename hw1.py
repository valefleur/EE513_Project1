# hw1.py
import my_library.analyze as analyze

if __name__ == "__main__":
    sony = 'images/a7sii/DSC04632.ARW'
    iphone = 'images/iPhone16/Terry.DNG'

    print(f"Sony alpha 7s II:")
    analyze.bayer_mosaic(sony)
    print(f"iPhone 16 Pro Max:")
    analyze.bayer_mosaic(iphone)