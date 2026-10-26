import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# Note: All code in this file borrowed from Prof. James McNames' 03_mtf_cascade.ipynb and 04_measure_your_camera.ipynb and modified as needed for homework 2.

plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

# ----- your camera -------------------------------------------------------
F_MM = 6.765        # focal length, mm
N = 1.78            # f-number
PITCH_UM = 2.42     # pixel pitch, micrometres
LAMBDA_UM = 0.550   # green light, micrometres
# -------------------------------------------------------------------------

D_MM = F_MM / N                       # aperture diameter
CUTOFF = 1.0 / (LAMBDA_UM * N)        # incoherent diffraction cutoff, cycles/um
NYQUIST = 1.0 / (2 * 2 * PITCH_UM)    # green-plane sensor Nyquist, cycles/um
AIRY_UM = 1.22 * LAMBDA_UM * N        # radius of the first dark ring

print(f"aperture diameter D     {D_MM:8.2f} mm")
print(f"diffraction cutoff      {CUTOFF * 1000:8.0f} cycles/mm")
print(f"sensor Nyquist          {NYQUIST * 1000:8.0f} cycles/mm")
print(f"cutoff / Nyquist        {CUTOFF / NYQUIST:8.2f}")
print(f"Airy radius             {AIRY_UM:8.2f} um  ({AIRY_UM / PITCH_UM:.2f} pixels)")
print()
print("The cutoff depends on wavelength and f-number ALONE. Not on focal length,")
print("not on sensor size, not on pixel count.")


def get_pixel_mtf(nu):
    '''A square pixel integrates over its area, so its MTF is |sinc(nu * pitch)|.'''
    return np.abs(np.sinc(np.asarray(nu) * PITCH_UM))     # np.sinc is sin(pi x)/(pi x)


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


def main(output=None):
    nu = np.linspace(0, 0.42, 600)                 # cycles per micrometre
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
    spatial_cutoff_freq = 1 / (LAMBDA_UM * N) * 1000 # because our frequency scale is in mm not um
    # ax.set_xlim(0, spatial_cutoff_freq * margin)
    ax.set_ylim(0, 1 * margin)
    ax.legend(fontsize=9, frameon=False)
    ax.set_title("$\mathrm{MTF}$ of the iPhone 16 Imaging System")
    # add a subtitle with the MTF50 of the system
    ax.text(0.5, 0.94, f"Analytical System MTF50 = {mtf50_freq:.1f} cycles/mm", ha="center", va="bottom", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    plt.show()
    if output is not None:
        fig.savefig(f"{output}/07_mtf_cascade.png", dpi=150)

    at = lambda curve: np.interp(NYQUIST, nu, curve)
    print(f"Remaining contrast at green-plane Nyquist ({NYQUIST * 1000:.0f} cycles/mm):")
    # print(f"  diffraction alone  {np.interp(NYQUIST, nu, diffraction_mtf(nu)):.3f}")
    print(f"  ideal lens         {at(lens_mtf):.3f}")
    print(f"  pixel aperture     {at(pixel_mtf):.3f}")
    # print(f"  hand shake         {at(shake):.3f}")
    print(f"  SYSTEM             {at(system_mtf):.3f}   <- the product of the two above curves")

    return mtf50_freq