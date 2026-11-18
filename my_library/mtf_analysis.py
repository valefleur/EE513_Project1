import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import cv2 as cv
from my_library.camera_specs import IPHONE_16_PRO_MAX as cam_spec

# Note: All code in this file borrowed from Prof. James McNames' 03_mtf_cascade.ipynb and 04_measure_your_camera.ipynb and modified as needed for homeworks and Project 1.

plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

LAMBDA_UM = 0.550   # green light, micrometres
D_MM = cam_spec.focal_mm / cam_spec.f_number                       # aperture diameter
CUTOFF = 1.0 / (LAMBDA_UM * cam_spec.f_number)        # incoherent diffraction cutoff, cycles/um
NYQUIST = 1.0 / (2 * 2 * cam_spec.pitch_um)    # green-plane sensor Nyquist, cycles/um
AIRY_UM = 1.22 * LAMBDA_UM * cam_spec.f_number        # radius of the first dark ring

def print_diffraction_summary():
    """Print the diffraction numbers above (HW2 Part 2). These used to print at import
    time, which made every module that imported this one print them too."""
    print(f"aperture diameter D     {D_MM:8.2f} mm")
    print(f"diffraction cutoff      {CUTOFF * 1000:8.0f} cycles/mm")
    print(f"sensor Nyquist          {NYQUIST * 1000:8.0f} cycles/mm")
    print(f"cutoff / Nyquist        {CUTOFF / NYQUIST:8.2f}")
    print(f"Airy radius             {AIRY_UM:8.2f} um  ({AIRY_UM / cam_spec.pitch_um:.2f} pixels)")
    print()
    print("The cutoff depends on wavelength and f-number ALONE. Not on focal length,")
    print("not on sensor size, not on pixel count.")


def get_pixel_mtf(nu):
    '''A square pixel integrates over its area, so its MTF is |sinc(nu * pitch)|.
    Pixel aperature will always be based on the full pixel pitch since it has to do with the physical
    sensor.
    '''
    return np.abs(np.sinc(np.asarray(nu) * cam_spec.pitch_um))     # np.sinc is sin(pi x)/(pi x)


def get_diffraction_mtf(nu):
    '''Incoherent MTF of a clear circular aperture: the pupil autocorrelation.'''
    s = np.clip(np.asarray(nu, dtype=float) / CUTOFF, 0, 1)
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s ** 2))


def get_gaussian_mtf(nu, sigma_um):
    return np.exp(-2 * (np.pi * sigma_um * np.asarray(nu)) ** 2)


def analytic_gaussian_mtf50(sigma):
    """Return the MTF50 of a Gaussian PSF with standard deviation `sigma` in pixels."""
    return np.sqrt(np.log(2) / (2 * np.pi ** 2 * sigma ** 2))


def mtf50(freq, mtf):
    '''
    Get the scalar frequency at which the MTF drops to 50% of its original contrast.
    Parameters
    ----------
    freq : array_like
        The spatial frequencies corresponding to the MTF values.
    mtf : array_like
        The MTF values corresponding to the spatial frequencies.
    Returns
    -------
    float
        The spatial frequency at which the MTF drops to 50% contrast. Returns NaN if the MTF does not drop below 50%.
    '''
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    # Linearly interpolate to find the approximate frequency at which the MTF hits 50% contrast.
    return float(freq[i - 1] + (mtf[i - 1] - 0.5) * (freq[i] - freq[i - 1]) / (mtf[i - 1] - mtf[i]))


def edge_centroids(roi, window=8.0, smooth=1.5):
    """Sub-pixel edge position per row, from a windowed derivative centroid."""
    roi = np.asarray(roi, dtype=float)

    # Coarse location from a smoothed copy. Smoothing locates the edge; it never
    # touches the data we build the edge spread function from.
    smoothed = cv.GaussianBlur(roi, (0, 0), sigmaX=smooth, sigmaY=0.01)
    coarse = np.argmax(np.abs(np.diff(smoothed, axis=1)), axis=1) + 0.5

    deriv = np.abs(np.diff(roi, axis=1))
    x = np.arange(deriv.shape[1]) + 0.5
    weight = np.where(np.abs(x[None, :] - coarse[:, None]) <= window, deriv, 0.0)

    total = weight.sum(axis=1)
    good = total > 1e-9
    centers = np.full(roi.shape[0], np.nan)
    centers[good] = (weight[good] * x).sum(axis=1) / total[good]
    return centers


def slanted_edge_mtf(roi, bin_width=0.25, window=8.0, esf_half_width=12.0, verbose=False):
    """Slanted-edge MTF of one region with a near-vertical edge.

    The ESF is always cropped to +/- esf_half_width px around the edge before the LSF is
    taken, for synthetic and real data alike, so a known-answer test measures exactly the
    method used on real captures. Measured on HW2's known-answer edge (5 deg tilt,
    Gaussian PSF sigma 1.0 px, analytic MTF50 0.1874) with noise over 200 trials: no
    crop scattered MTF50 by sd 0.097 (about 50%); +/-12 px gave -2.9% bias, sd 0.0021;
    +/-16 px gave -4.0%, sd 0.0025 (more tail adds noise, not signal). The region given
    must extend somewhat beyond the crop on both sides of the edge.

    Parameters
    ----------
    roi : ndarray
        Region with a near-vertical edge (rows run along the edge).
    bin_width : float, default 0.25
        ESF bin width in pixels (4x oversampling).
    window : float, default 8.0
        Half-width in pixels of the derivative window used to locate the edge per row.
    esf_half_width : float, default 12.0
        Keep the ESF within this many pixels of the fitted edge.
    verbose : bool, default False
        Print the fitted edge's slope and intercept (as HW2 did).

    Returns
    -------
    tuple
        (freq in cycles per pixel of roi, mtf normalized to 1 at zero frequency, esf_x,
        esf, lsf, fitted edge position per row, slope in px/row, intercept).
    """
    roi = np.asarray(roi, dtype=float)
    rows = np.arange(roi.shape[0])

    centers = edge_centroids(roi, window=window)
    ok = np.isfinite(centers)
    # print(f"centers: {centers[ok]}")
    slope, intercept = np.polyfit(rows[ok], centers[ok], 1)
    if verbose:
        print(f"slanted edge slope = {slope:.4f} px/row, intercept = {intercept:.2f} px")
    fitted = slope * rows + intercept

    cols = np.arange(roi.shape[1])
    distance = (cols[None, :] - fitted[:, None]).ravel()
    values = roi.ravel()

    edges = np.arange(distance.min(), distance.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(distance, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    keep = count > 0
    esf = total[keep] / count[keep]
    esf_x = (edges[:-1] + bin_width / 2)[keep]

    # crop the ESF to the central +/-esf_half_width px of the edge to avoid edge effects
    crop = np.abs(esf_x) <= esf_half_width
    esf_x = esf_x[crop]
    esf = esf[crop]

    lsf = np.gradient(esf, esf_x)
    lsf = lsf - np.median(np.concatenate([lsf[:5], lsf[-5:]]))
    lsf = lsf * np.hanning(len(lsf))

    spectrum = np.abs(np.fft.rfft(lsf))
    return np.fft.rfftfreq(len(lsf), d=bin_width), spectrum / spectrum[0], esf_x, esf, lsf, fitted, slope, intercept


def get_analytic_mtf(freq, sigma=1.0):
    return np.exp(-2 * np.pi ** 2 * sigma ** 2 * freq ** 2)


def main(output=None):
    nu = np.linspace(0, 0.42, 600)    # cycles per micrometre
    lens_mtf = get_diffraction_mtf(nu)# * get_gaussian_mtf(nu, 0.62) # diffraction only
    pixel_mtf = get_pixel_mtf(nu)
    # shake = get_gaussian_mtf(nu, 0.45)
    '''
    System MTF is the product of the three stages in series.
    We can do this because the MTF is the Fourier transform of the PSF and convolution in space is
    multiplication in frequency.
    '''
    system_mtf = lens_mtf * pixel_mtf #* shake

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(nu * 1000, lens_mtf, lw=1.8, label="lens (diffraction)")
    ax.plot(nu * 1000, pixel_mtf, lw=1.8, label="pixel aperture, |sinc|")
    # ax.plot(nu * 1000, shake, lw=1.8, ls="--", label="hand shake")
    ax.plot(nu * 1000, system_mtf, lw=3.0, color="k", label="system = the product")

    # Show green-plane Nyquist line
    green_plane_nyquist_freq = NYQUIST * 1000
    ax.axvline(green_plane_nyquist_freq, color="#66A461", ls=":")
    ax.text(green_plane_nyquist_freq - 5, 0.76, f"Green-plane Nyquist = {green_plane_nyquist_freq:.1f} cycles/mm", rotation=90, ha="right", va="top", color="#66A461", fontsize=8)

    # Show full sensor Nyquist line
    full_nyquist_freq = green_plane_nyquist_freq * 2  # applies equally in both spatial dimensions
    ax.axvline(full_nyquist_freq, color="gray", ls=":")
    ax.text(full_nyquist_freq + 10, 0.76, f"Full sensor Nyquist = {full_nyquist_freq:.1f} cycles/mm", rotation=90, ha="right", va="top", color="gray", fontsize=8)

    # Show 50% contrast line
    ax.axhline(0.5, color="#DAA520", ls=":", alpha=0.7)
    ax.text(10, 0.5+ 0.02, f"50% Contrast", fontsize=8, color="#DAA520")

    # Drop a vertical line where the system MTF crosses 50% contrast
    mtf50_freq = mtf50(nu, system_mtf) * 1000  # convert to cycles/mm
    ax.axvline(mtf50_freq, ymax=0.5, color='black', ls=":", alpha=0.7)
    ax.text(mtf50_freq - 5, 0.5, f'MTF50 = {mtf50_freq:.1f} cycles/mm', rotation=90, ha="right", va="top", color="black", fontsize=8)

    ax.set_xlabel("spatial frequency (cycles/mm)"); ax.set_ylabel("contrast")
    margin = 1.02 # visual margin for the axes limits
    spatial_cutoff_freq = 1 / (LAMBDA_UM * cam_spec.f_number) * 1000 # because our frequency scale is in mm not um
    # ax.set_xlim(0, spatial_cutoff_freq * margin)
    ax.set_ylim(0, 1 * margin)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title(r"$\mathrm{MTF}$ of the iPhone 16 Imaging System")
    # add a subtitle with the MTF50 of the system
    ax.text(0.5, 0.94, f"Analytical System MTF50 = {mtf50_freq:.1f} cycles/mm", ha="center", va="bottom", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    plt.show()
    if output is not None:
        fig.savefig(f"{output}/07_mtf_cascade.png", dpi=150)
    plt.close(fig)

    at = lambda curve: np.interp(NYQUIST, nu, curve)
    system_contrast_at_green_nyquist = at(system_mtf)
    print(f"Remaining contrast at green-plane Nyquist ({NYQUIST * 1000:.2f} cycles/mm):")
    # print(f"  diffraction alone  {np.interp(NYQUIST, nu, diffraction_mtf(nu)):.3f}")
    print(f"  ideal lens         {at(lens_mtf):.3f}")
    print(f"  pixel aperture     {at(pixel_mtf):.3f}")
    # print(f"  hand shake         {at(shake):.3f}")
    print(f"  SYSTEM             {system_contrast_at_green_nyquist:.3f}   <- the product of the two above curves")

    return nu, system_mtf, mtf50_freq / 1000, system_contrast_at_green_nyquist