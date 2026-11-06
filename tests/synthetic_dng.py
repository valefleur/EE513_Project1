"""Write small synthetic DNGs that rawpy (LibRaw) can open, for tests.

Layout follows a typical DNG: IFD0 is a reduced-resolution RGB preview carrying the
DNG-wide tags, and the full-resolution raw image sits in a SubIFD. An Exif IFD with
LensModel and focal lengths is added afterwards by add_exif(), because tifffile will
not write the ExifTag pointer itself.
"""
import struct

import numpy as np
import tifffile

DEFAULT_LENS = "Test Phone back camera 6.765mm f/1.78"

CFA_CODES = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}


def write_dng(path, cfa="BGGR", size=(64, 48), crop=(56, 40), photometric=32803,
              noise_profile=None, software="TestApp 1.0", lens_model=DEFAULT_LENS,
              raw_data=None):
    """Write a small DNG. lens_model=None leaves out the Exif IFD entirely.
    raw_data: optional uint16 mosaic (height x width) to store instead of the ramp."""
    if raw_data is not None:
        size = (raw_data.shape[1], raw_data.shape[0])
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
        data = (raw_data if raw_data is not None
                else (np.arange(h * w).reshape(h, w) % 4000 + 100)).astype(np.uint16)
        raw_tags += [(33421, "H", 2, (2, 2), True), (33422, "B", 4, CFA_CODES[cfa], True),
                     (50710, "B", 3, (0, 1, 2), True)]
    else:                                   # LinearRaw: three samples per pixel
        data = np.zeros((h, w, 3), np.uint16)
    with tifffile.TiffWriter(path) as t:
        t.write(preview, photometric="rgb", subfiletype=1, subifds=1,
                software=software, extratags=ifd0_tags)
        t.write(data, photometric=photometric, subfiletype=0, extratags=raw_tags)
    if lens_model is not None:
        add_exif(path, lens_model)
    return path


def add_exif(path, lens_model, focal_mm=(6765, 1000), focal_35mm=24):
    """Append an Exif IFD and a copy of IFD0 that points to it.

    IFD0 is rewritten at the end of the file with one more entry (ExifTag), and the
    header is pointed at the copy. Every other entry is copied byte for byte, so the
    data it points to stays valid.
    """
    data = bytearray(open(path, "rb").read())
    assert data[:2] == b"II", "little-endian TIFF expected"
    old = struct.unpack("<I", data[4:8])[0]
    n = struct.unpack("<H", data[old:old + 2])[0]
    entries = [bytes(data[old + 2 + 12 * i:old + 14 + 12 * i]) for i in range(n)]
    next_ifd = bytes(data[old + 2 + 12 * n:old + 6 + 12 * n])

    def word_align():
        if len(data) % 2:
            data.extend(b"\0")

    # Exif IFD: FocalLength (RATIONAL), FocalLengthIn35mmFilm (SHORT), LensModel (ASCII)
    word_align()
    exif_off = len(data)
    lens = lens_model.encode("ascii") + b"\0"
    values_off = exif_off + 2 + 12 * 3 + 4
    data += struct.pack("<H", 3)
    data += struct.pack("<HHII", 37386, 5, 1, values_off)
    data += struct.pack("<HHIHH", 41989, 3, 1, focal_35mm, 0)
    data += struct.pack("<HHII", 42036, 2, len(lens), values_off + 8)
    data += struct.pack("<I", 0) + struct.pack("<II", *focal_mm) + lens

    # New IFD0: old entries plus ExifTag, sorted by tag number as TIFF requires
    entries.append(struct.pack("<HHII", 34665, 4, 1, exif_off))
    entries.sort(key=lambda e: struct.unpack("<H", e[:2])[0])
    word_align()
    new_ifd0 = len(data)
    data += struct.pack("<H", len(entries)) + b"".join(entries) + next_ifd
    data[4:8] = struct.pack("<I", new_ifd0)
    open(path, "wb").write(bytes(data))
