"""Write small synthetic DNGs that rawpy (LibRaw) can open, for tests.

Layout follows a typical DNG: IFD0 is a reduced-resolution RGB preview carrying the
DNG-wide tags, and the full-resolution raw image sits in a SubIFD. There is no Exif
IFD, so LensModel is absent; tests of the LensModel path must use a real DNG.
"""
import numpy as np
import tifffile

CFA_CODES = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}


def write_dng(path, cfa="BGGR", size=(64, 48), crop=(56, 40), photometric=32803,
              noise_profile=None, software="TestApp 1.0"):
    w, h = size
    preview = np.zeros((12, 16, 3), np.uint8)
    ifd0_tags = [(271, "s", 0, "TestMake", True), (272, "s", 0, "Test Phone", True),
                 (50706, "B", 4, (1, 4, 0, 0), True), (50707, "B", 4, (1, 1, 0, 0), True),
                 (50708, "s", 0, "TestMake Test Phone", True),
                 (50721, "2i", 9, (10000, 10000, 0, 10000, 0, 10000, 0, 10000, 10000,
                                   10000, 0, 10000, 0, 10000, 0, 10000, 10000, 10000), True),
                 (50778, "H", 1, 21, True)]
    if noise_profile is not None:
        ifd0_tags.append((51041, "d", len(noise_profile), tuple(noise_profile), True))
    raw_tags = [(50714, "H", 1, 100, True), (50717, "H", 1, 4095, True),
                (50719, "I", 2, (4, 4), True), (50720, "I", 2, crop, True)]
    if photometric == 32803:
        data = (np.arange(h * w).reshape(h, w) % 4000 + 100).astype(np.uint16)
        raw_tags += [(33421, "H", 2, (2, 2), True), (33422, "B", 4, CFA_CODES[cfa], True),
                     (50710, "B", 3, (0, 1, 2), True)]
    else:                                   # LinearRaw: three samples per pixel
        data = np.zeros((h, w, 3), np.uint16)
    with tifffile.TiffWriter(path) as t:
        t.write(preview, photometric="rgb", subfiletype=1, subifds=1,
                software=software, extratags=ifd0_tags)
        t.write(data, photometric=photometric, subfiletype=0, extratags=raw_tags)
    return path
