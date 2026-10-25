import numpy as np

# Transfer functions from Prof. James McNames' psu_imaging.py (EE 513 Introduction to Image Processing).
import psu_imaging as psu


def green_plane_from_raw(raw, which_grn=0):
    """One green plane of a rawpy-like object as linear light in [0, 1] (black removed,
    divided by white - black). `which` picks the first (0) or second (1) green of the
    2x2 pattern in raster order."""
    desc = raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc
    pattern = np.asarray(raw.raw_pattern)
    greens = [(r, c) for r in range(2) for c in range(2) if desc[pattern[r, c]] == "G"]
    r, c = greens[which_grn]
    print(f'Extracting green plane from raw Bayer mosaic: {desc[pattern[r, c]]} at row {r}, col {c}')
    black = float(raw.black_level_per_channel[pattern[r, c]])
    print(f"Black level for this channel: {black}")
    plane = raw.raw_image_visible[r::2, c::2].astype(float)
    return np.clip((plane - black) / (raw.white_level - black), 0, 1)



def green_plane(path, which_grn=0):
    """Open a raw file with rawpy and return one green plane as linear light in [0, 1]."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        return green_plane_from_raw(raw, which_grn)


def reduce_two_ways(lin_plane, factor=8, exposure_scale=None, quantize=True):
    """Encode a linear plane to sRGB, then reduce by `factor` two ways.

    The plane is scaled so its 99.9th percentile is 1 (or by `exposure_scale`) so the
    sRGB image uses the display range, encoded, and (if `quantize`) rounded to 8 bits as
    an image file would be. Returns a dict with the 8-bit-scale source, the two reduced
    images (floats in 0..255), and their difference (linear-route minus direct-route).
    """
    scale = exposure_scale if exposure_scale is not None else 1.0 / np.percentile(lin_plane, 99.9)
    enc = np.clip(psu.linear_to_srgb(np.clip(lin_plane * scale, 0, 1)), 0, 1)
    src = np.round(enc * 255) if quantize else enc * 255
    h, w = (src.shape[0] // factor) * factor, (src.shape[1] // factor) * factor
    src = src[:h, :w]
    blocks = src.reshape(h // factor, factor, w // factor, factor)
    direct = blocks.mean(axis=(1, 3))                                   # average the encoded values
    lin = psu.srgb_to_linear(src / 255.0).reshape(blocks.shape)
    via_linear = 255.0 * psu.linear_to_srgb(lin.mean(axis=(1, 3)))      # average light, then encode
    return dict(source=src, direct=direct, via_linear=via_linear, diff=via_linear - direct,
                block_std=blocks.std(axis=(1, 3)), scale=scale, factor=factor)


def report_reduction(res, edge_std=12.0):
    """Print the mean absolute difference in 8-bit levels and where the error sits.

    A block is an 'edge' block if the standard deviation of its encoded source pixels
    exceeds `edge_std` levels (it holds both dark and bright pixels), 'flat' otherwise.
    """
    d = np.abs(res["diff"])
    edge = res["block_std"] > edge_std
    print(f"\nreduced image {d.shape[1]} x {d.shape[0]} (factor {res['factor']}, exposure scale {res['scale']:.3f})")
    print(f"mean abs. diff = {d.mean():.3f} levels of 255   (max {d.max():.1f})")
    print(f"signed mean (linear route - direct) = {res['diff'].mean():+.3f} levels")
    print(f"edge blocks: {edge.mean() * 100:.1f}% of the image, mean abs. diff "
          f"{d[edge].mean() if edge.any() else float('nan'):.2f}; "
          f"\n\tflat blocks: mean abs. diff {d[~edge].mean():.3f}")
    print(f"share of total absolute error in edge blocks: {d[edge].sum() / d.sum() * 100:.1f}%")
    return d.mean()


def plot_reduction(res, plot_path=None, title=""):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, img, t in zip(axes[:2], (res["direct"], res["via_linear"]),
                          ("average of encoded values", "average in linear light, then encode")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(t, fontsize=9.5)
    lim = max(1.0, np.abs(res["diff"]).max())
    im = axes[2].imshow(res["diff"], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axes[2].set_title(f"difference (linear $-$ direct), mean |d| = "
                      f"{np.abs(res['diff']).mean():.2f} levels", fontsize=9.5)
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="8-bit levels")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    if plot_path:
        fig.savefig(plot_path, dpi=150)
    return fig


def raw_to_linear_light(raw_img, black_level, saturation_level):
    """Convert a raw image to linear light in the range 0.0 to 1.0.

    Subtracts the black level, divides by the span from black to
    white, and returns the linearized image as a float array.
    """

    return np.clip((raw_img.astype(float) - black_level) / (saturation_level - black_level), 0, 1)


def quarter_rgb(linear, pattern, color_desc):
    """
    2x2 mosaic -> half-width, half-height RGB. No interpolation.
    Return a quarter-resolution RGB image from a 2x2 mosaic.
    """
    desc = color_desc.decode() if isinstance(color_desc, bytes) else color_desc
    h, w = (linear.shape[0] // 2) * 2, (linear.shape[1] // 2) * 2
    linear = linear[:h, :w]
    out = np.zeros((h // 2, w // 2, 3))
    greens = []
    for r in range(2):
        for c in range(2):
            plane = linear[r::2, c::2]
            ch = desc[pattern[r, c]]
            if ch == "R": out[..., 0] = plane
            elif ch == "B": out[..., 2] = plane
            else: greens.append(plane)
    out[..., 1] = np.mean(greens, axis=0)
    return out

