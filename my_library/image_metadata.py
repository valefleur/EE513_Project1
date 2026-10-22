from pathlib import Path
import exifread
import rawpy
import tifffile
import struct
import numpy as np

# Written by Claude Sonnet 5.5

RAW_SUFFIXES = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"}


TAG_NOISE_PROFILE = 51041
TAG_NOISE_REDUCTION = 50935
_TYPES = {1: ("B", 1), 2: ("c", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8),
          9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8)}


def read_tiff_tags(path, wanted=(TAG_NOISE_PROFILE, TAG_NOISE_REDUCTION)):
    """Find DNG/TIFF tags by number, searching every IFD, including SubIFDs and Exif.

    Returns {tag: (values, where)}. Needed because exifread and rawpy do not expose
    NoiseProfile. Rationals come back as floats.
    """
    found = {}
    with open(path, "rb") as f:
        data = f.read()
    end = {b"II": "<", b"MM": ">"}.get(data[:2])
    if end is None or struct.unpack(end + "H", data[2:4])[0] != 42:
        raise ValueError(f"{path} is not a TIFF/DNG file")

    def values(typ, count, field_off):
        fmt, size = _TYPES.get(typ, ("B", 1))
        n = count * size
        if n > 4:
            off = struct.unpack(end + "I", data[field_off:field_off + 4])[0]
            raw = data[off:off + n]
        else:
            raw = data[field_off:field_off + n]
        out = struct.unpack(end + fmt[0] * (count * len(fmt)), raw)
        if typ in (5, 10):
            out = tuple(a / b if b else float("nan") for a, b in zip(out[0::2], out[1::2]))
        return out

    seen = set()

    def walk(ifd_off, where):
        while ifd_off and ifd_off not in seen and ifd_off + 2 <= len(data):
            seen.add(ifd_off)
            n = struct.unpack(end + "H", data[ifd_off:ifd_off + 2])[0]
            for i in range(n):
                e = ifd_off + 2 + 12 * i
                tag, typ, count = struct.unpack(end + "HHI", data[e:e + 8])
                if tag in wanted:
                    found.setdefault(tag, (values(typ, count, e + 8), where))
                if tag in (330, 34665) and typ in (4, 13):   # SubIFDs, Exif IFD
                    for sub in values(4, count, e + 8):
                        walk(sub, f"{where}>{'SubIFD' if tag == 330 else 'Exif'}")
            ifd_off = struct.unpack(end + "I", data[ifd_off + 2 + 12 * n:ifd_off + 6 + 12 * n])[0]
            where = "IFD+"

    walk(struct.unpack(end + "I", data[4:8])[0], "IFD0")
    return found


def noise_profile(path):
    """The DNG NoiseProfile tag as a list of (S, O) pairs, or None if the file has none."""
    hit = read_tiff_tags(path).get(TAG_NOISE_PROFILE)
    if hit is None:
        return None
    v, _ = hit
    return [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]


def compare_noise_profile(path, K=None, read=None, s_K=None, s_read=None):
    """Print the vendor NoiseProfile converted to K (DN/e-) and read noise (e-).

    Var[x] = S*x + O in normalized units x = (D - B)/(W - B), so with span = W - B:
    K = S * span, sigma_r = sqrt(O) / S, knee = sigma_r^2. Prints one row per pair in
    the tag (one per color plane, in the file's plane order), and, if your K and read
    noise are given, the vendor/yours ratios. Prints a statement if the tag is absent.
    """
    tags = read_tiff_tags(path)
    meta = raw_metadata(path)
    span = meta["white_level"] - meta["black_levels"][0]
    nr = tags.get(TAG_NOISE_REDUCTION)
    print(f"{Path(path).name}: ISO {meta['iso']}, white {meta['white_level']}, "
          f"black {meta['black_levels'][0]} -> span {span} DN")
    if nr:
        print(f"NoiseReductionApplied tag: {nr[0][0]:g} (0 = none)")
    pairs = noise_profile(path)
    if pairs is None:
        print("NoiseProfile tag (51041): absent from this file")
        return None
    print(f"NoiseProfile tag (51041) found in {tags[TAG_NOISE_PROFILE][1]}: {len(pairs)} pair(s)")
    out = []
    for i, (S, O) in enumerate(pairs):
        Kv = S * span
        rv = np.sqrt(O) / S if O > 0 else float("nan")
        line = (f"  pair {i}: S = {S:.4e}  O = {O:.4e}  ->  K = {Kv:.4f} DN/e-, "
                f"sigma_r = {rv:.2f} e-, knee = {rv ** 2:.1f} e-")
        if K and read:
            line += f"   (vendor/yours: K {Kv / K:.2f}, read {rv / read:.2f})"
        print(line)
        out.append(dict(S=S, O=O, K=Kv, read=rv))
    if K:
        print(f"yours: K = {K:.4f}" + (f" +/- {s_K:.4f}" if s_K else "")
              + (f" DN/e-, sigma_r = {read:.2f}" if read else "")
              + (f" +/- {s_read:.2f} e-" if s_read else ""))
    return out


def lens_id(path):
    """The fingerprint of the camera that made this raw file.
    Originally written by Prof. James McNames.
    """
    print(f"path: {path}")
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        exif = p.tags["ExifTag"].value
        fl = exif["FocalLength"]
        return (exif.get("LensModel", "unknown"),
                round(fl[0] / fl[1], 3),
                exif.get("FocalLengthIn35mmFilm"),
                (p.tags["ImageWidth"].value, p.tags["ImageLength"].value))


def assert_one_camera(paths):
    """Originally written by Prof. James McNames."""
    ids = {lens_id(p) for p in paths}
    if len(ids) != 1:
        raise SystemExit(f"more than one camera in this series:\n  " +
                         "\n  ".join(map(str, ids)))
    else:
        print(f"all images from the same camera: {ids.pop()}")


def _to_float(value):
    """exifread Ratio, int or float to float. None if it cannot be converted."""
    try:
        if hasattr(value, "num"):
            return value.num / value.den if value.den else None
        return float(value)
    except (TypeError, ValueError):
        return None


def raw_metadata(path):
    """Return a dict of exposure time, ISO, black level and camera for one raw file.

    Keys: file, camera, exposure_s, iso, black_levels (per channel, as stored
    in the file), white_level, bayer (color order of the 2x2 tile), shape.
    A value that the file does not provide is None.
    """

    path = Path(path)
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    def first(*names):
        for name in names:
            if name in tags:
                return tags[name]
        return None

    exposure_tag = first("EXIF ExposureTime")
    iso_tag = first("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
    exposure = _to_float(exposure_tag.values[0]) if exposure_tag else None
    iso = _to_float(iso_tag.values[0]) if iso_tag else None

    make = str(tags.get("Image Make", "")).strip()
    model = str(tags.get("Image Model", "")).strip()
    lens = str(tags.get("EXIF LensModel", "")).strip()
    camera = " ".join(s for s in (make, model) if s)
    if lens:
        camera += " | " + lens

    with rawpy.imread(str(path)) as raw:
        black = [int(v) for v in raw.black_level_per_channel]
        white = int(raw.white_level)
        shape = tuple(raw.raw_image_visible.shape)
        try:
            colors = raw.color_desc.decode()
            bayer = "".join(colors[i] for i in raw.raw_pattern.flatten())
        except Exception:
            bayer = None

    return dict(file=path.name, camera=camera or None, exposure_s=exposure,
                iso=int(iso) if iso is not None else None,
                black_levels=black, white_level=white, bayer=bayer, shape=shape)


def _format_exposure(seconds):
    if seconds is None:
        return "?"
    if seconds < 1:
        return f"1/{round(1 / seconds)} s"
    return f"{seconds:g} s"


def get_image_paths(target, suffixes=RAW_SUFFIXES):
    target = Path(target)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*")
                       if p.is_file() and p.suffix.lower() in suffixes)
    else:
        paths = [target]
    if not paths:
        print(f"No raw files found in {target}")
        return []

    return paths


def print_raw_metadata(target, suffixes=RAW_SUFFIXES):
    """Print a table of metadata for one raw file or every raw file under a directory.

    Ends with a summary of the distinct (camera, ISO) combinations found, so a
    mixed-ISO folder is obvious. Returns the list of metadata dicts.
    """
    target = Path(target)
    if target.is_dir():
        paths = sorted(p for p in target.rglob("*")
                       if p.is_file() and p.suffix.lower() in suffixes)
    else:
        paths = [target]
    if not paths:
        print(f"No raw files found in {target}")
        return []

    rows = []
    print(f"{'file':28s} {'exposure':>10s} {'ISO':>5s}  {'black level (per channel)':26s} "
          f"{'white':>6s} {'bayer':>5s}  camera")
    for p in paths:
        try:
            m = raw_metadata(p)
        except Exception as exc:
            print(f"{p.name:28s} could not read: {exc}")
            continue
        rows.append(m)
        print(f"{m['file']:28s} {_format_exposure(m['exposure_s']):>10s} "
              f"{str(m['iso']) if m['iso'] is not None else '?':>5s}  "
              f"{str(m['black_levels']):26s} {m['white_level']:>6d} "
              f"{m['bayer'] or '?':>5s}  {m['camera'] or '?'}")

    print(f"Length of rows: {len(rows)}")

    combos = {}
    for m in rows:
        combos[(m["camera"], m["iso"])] = combos.get((m["camera"], m["iso"]), 0) + 1
    print(f"\n{len(rows)} file(s). Distinct (camera, ISO) combinations:")
    for (camera, iso), n in sorted(combos.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        print(f"  {n:3d} x ISO {iso if iso is not None else '?'}  {camera or '?'}")

    return rows

def green_patch(path, size=128, which=0):
    """Center size x size patch of one green plane of a raw file.

    Returns (patch in DN as float, black level of that plane). `which` picks the
    first (0) or second (1) green pixel of the 2x2 Bayer tile. The position is
    read from the file's own color pattern, so RGGB and BGGR both work.
    """

    with rawpy.imread(str(path)) as raw:
        image = raw.raw_image_visible.astype(float)
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode()
        greens = [(r, c) for r in (0, 1) for c in (0, 1) if desc[colors[r, c]] == "G"]
        r0, c0 = greens[which]
        black = float(raw.black_level_per_channel[colors[r0, c0]])
    plane = image[r0::2, c0::2]
    top = (plane.shape[0] - size) // 2
    left = (plane.shape[1] - size) // 2
    return plane[top:top + size, left:left + size], black


def channel_means(path):
    """Mean raw value (DN) of each color channel over the whole visible frame.

    Returns (means, labels). Channels are indexed as in the file's own color
    description, so the means line up with black_level_per_channel.
    """

    with rawpy.imread(str(path)) as raw:
        image = raw.raw_image_visible.astype(float)
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode()
    means = [float(image[colors == c].mean()) for c in range(4)]
    labels = [desc[c] if desc.index(desc[c]) == c else desc[c] + "2" for c in range(4)]
    return means, labels