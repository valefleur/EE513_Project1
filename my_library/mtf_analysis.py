import matplotlib.pyplot as plt
import numpy as np
from scipy import special

# Note: All code in this file borrowd from Prof. James McNames' 03_mtf_cascade.ipynb and modified as needed for homework 2.

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


def pixel_mtf(nu):
    '''A square pixel integrates over its area, so its MTF is |sinc(nu * pitch)|.'''
    return np.abs(np.sinc(np.asarray(nu) * PITCH_UM))     # np.sinc is sin(pi x)/(pi x)


def diffraction_mtf(nu):
    '''Incoherent MTF of a clear circular aperture: the pupil autocorrelation.'''
    s = np.clip(np.asarray(nu, dtype=float) / CUTOFF, 0, 1)
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1 - s ** 2))


def gaussian_mtf(nu, sigma_um):
    return np.exp(-2 * (np.pi * sigma_um * np.asarray(nu)) ** 2)

def main(output=None):
    nu = np.linspace(0, 0.42, 600)                 # cycles per micrometre
    lens = diffraction_mtf(nu)# * gaussian_mtf(nu, 0.62) # diffraction only
    pixel = pixel_mtf(nu)
    # shake = gaussian_mtf(nu, 0.45)
    system = lens * pixel #* shake

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(nu * 1000, lens, lw=1.8, label="lens (diffraction)")
    ax.plot(nu * 1000, pixel, lw=1.8, label="pixel aperture, |sinc|")
    # ax.plot(nu * 1000, shake, lw=1.8, ls="--", label="hand shake")
    ax.plot(nu * 1000, system, lw=3.0, color="k", label="system = the product")
    ax.axvline(NYQUIST * 1000, color="gray", ls=":")
    ax.text(NYQUIST * 1000 - 5, 0.76, f"Nyquist = {NYQUIST * 1000:.0f} cycles/mm", rotation=90, ha="right", va="top", color="gray")
    ax.set_xlabel("spatial frequency (cycles/mm)"); ax.set_ylabel("contrast")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 420); ax.legend(fontsize=9, frameon=False)
    ax.set_title("Every stage multiplies, so the worst stage wins")
    plt.tight_layout()
    plt.show()
    if output is not None:
        fig.savefig(f"{output}/07_mtf_cascade.png", dpi=150)

    at = lambda curve: np.interp(NYQUIST, nu, curve)
    print(f"at green-plane Nyquist ({NYQUIST * 1000:.0f} cycles/mm):")
    # print(f"  diffraction alone  {np.interp(NYQUIST, nu, diffraction_mtf(nu)):.3f}")
    print(f"  ideal lens         {at(lens):.3f}")
    print(f"  pixel aperture     {at(pixel):.3f}")
    # print(f"  hand shake         {at(shake):.3f}")
    print(f"  SYSTEM             {at(system):.3f}   <- the product of the three")