"""A synthetic slanted-edge target, rendered into DNG files with a known blur.

What the blur is, and is not. To test the measurement, the target is blurred by a
point-spread function whose MTF is known exactly: a Gaussian of standard deviation
psf_sigma_px (in full-resolution pixels). This is a test signal, like HW2 item 9's
Gaussian PSF with sigma = 1.0 px; it is not a model of the iPhone's lens. (An ideal
lens's diffraction MTF is 2/pi [arccos(s) - s sqrt(1 - s^2)], s = v / v_c, which
belongs to the prediction in HW2 Part 2, not to this test.) "sigma" here is the PSF's
width, not a noise level.

The expected MTF in cycles per green-plane sample, f, is the product of two stages:

    MTF(f) = exp(-2 pi^2 (psf_sigma_px / 2)^2 f^2) * |sinc(f / 2)|

- The Gaussian's MTF is exp(-2 pi^2 sigma^2 f^2) with sigma and 1/f in the same units;
  one green-plane sample spans 2 full-resolution pixels, so sigma in green samples is
  psf_sigma_px / 2.
- The pixel aperture: render_target averages the supersampled scene over each
  full-resolution pixel, an aperture 1 full-res pixel = 0.5 green samples wide, whose
  MTF is |sinc(0.5 f)| = |sinc(f / 2)| (numpy's sinc(x) = sin(pi x) / (pi x)). The /4
  from HW2 belongs to the green-plane Nyquist frequency, 1 / (4 p) in cycles per um,
  not to the aperture: the aperture is one pitch p wide whether or not the plane is
  subsampled. Check: at green Nyquist, f = 0.5, |sinc(0.25)| = 0.900, HW2 item 7's
  aperture MTF there.

expected_mtf50() solves MTF(f) = 0.5 numerically.
"""
import cv2 as cv
import numpy as np

from tests.synthetic_dng import write_dng


def _square(cx, cy, side, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * side / 2
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


def render_target(size=(1600, 1200), psf_sigma_px=1.5, tilt_deg=5.0, offset=(0, 0), center_side=400,
                  corner_side=180, supersample=3, psf_offset=None):
    """Grayscale target image (0..1) at full resolution.

    psf_sigma_px: the Gaussian PSF's standard deviation in full-resolution pixels.
    offset shifts the whole sheet (to move corner squares toward the frame corners).
    psf_offset: if given, the blur is two equal Gaussians offset by this many pixels
    horizontally (a double image), instead of one.
    """
    sigma = psf_sigma_px
    W, H = size
    S = supersample
    hi = np.full((H * S, W * S), 0.35, np.float32)
    ox, oy = offset
    sheet = np.array([[120, 80], [W - 120, 80], [W - 120, H - 80], [120, H - 80]]) + [ox, oy]
    cv.fillPoly(hi, [(sheet * S).astype(np.int32)], 1.0)
    cx, cy = W / 2 + ox, H / 2 + oy
    dx, dy = (W - 240) / 2 - corner_side * 0.9, (H - 160) / 2 - corner_side * 0.9
    for x, y, side in [(cx, cy, center_side), (cx - dx, cy - dy, corner_side), (cx + dx, cy - dy, corner_side),
                       (cx - dx, cy + dy, corner_side), (cx + dx, cy + dy, corner_side)]:
        cv.fillPoly(hi, [(_square(x, y, side, tilt_deg) * S).astype(np.int32)], 0.06)
    img = cv.resize(hi, (W, H), interpolation=cv.INTER_AREA).astype(float)
    if psf_offset is None:
        return cv.GaussianBlur(img, (0, 0), sigma)
    k = int(np.ceil(4 * sigma + abs(psf_offset)))
    x = np.arange(-k, k + 1)
    kern = np.exp(-(x - psf_offset / 2) ** 2 / (2 * sigma ** 2)) + np.exp(-(x + psf_offset / 2) ** 2 / (2 * sigma ** 2))
    img = cv.filter2D(img, -1, (kern / kern.sum())[None, :], borderType=cv.BORDER_REFLECT)
    return cv.GaussianBlur(img, (0, 0), sigmaX=0.01, sigmaY=sigma)


def write_target_dngs(folder, n=3, seed=0, noise_dn=2.0, level=3000, **render_kw):
    """n noisy DNG captures of the same rendered target (BGGR, black level 100)."""
    folder.mkdir(parents=True, exist_ok=True)
    img = render_target(**render_kw)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        raw = 100 + img * level + rng.normal(0, noise_dn, img.shape)
        paths.append(write_dng(folder / f"IMG_{i:02d}.dng", raw_data=np.clip(np.rint(raw), 0, 4095).astype(np.uint16),
                               crop=(img.shape[1], img.shape[0]), exposure_s=1 / 125, iso=64))
    return paths


def expected_mtf50(psf_sigma_px):
    """MTF50 in cycles per green-plane sample for a Gaussian PSF of psf_sigma_px full-res
    pixels plus the 1-pixel aperture (see the module docstring)."""
    f = np.linspace(0, 1, 200001)
    m = np.exp(-2 * np.pi ** 2 * (psf_sigma_px / 2) ** 2 * f ** 2) * np.abs(np.sinc(f / 2))
    return float(f[np.argmax(m < 0.5)])
