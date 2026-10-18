"""Part 2 on the simulated sensor from the Lecture 5 demo.

Adapted from 05_photon_transfer.ipynb by Professor James McNames.
Flat-field generation follows his flat_pair(); fitting and uncertainties live in ptc.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psu_imaging as psu
import my_library.photon_transfer_curve as ptc

PRNU = 0.012
SPAN = psu.WHITE_LEVEL - psu.BLACK_LEVEL
TRUE_K = SPAN / psu.FULL_WELL
TRUE_READ = psu.READ_NOISE


def flat_pair(electrons, level_index):
    patch = np.full((128, 128), electrons / psu.FULL_WELL)
    a = psu.expose_and_digitize(patch, np.random.default_rng(2 * level_index), prnu=PRNU)
    b = psu.expose_and_digitize(patch, np.random.default_rng(2 * level_index + 1), prnu=PRNU)
    return a.astype(float), b.astype(float)


def main():
    levels = np.logspace(np.log10(5), np.log10(psu.FULL_WELL * 0.92), 22)
    pairs = [flat_pair(e, i) for i, e in enumerate(levels)]

    means, var_pair, var_single = [], [], []
    for a, b in pairs:
        m, v, npix = ptc.pair_statistics(a, b, psu.BLACK_LEVEL)
        means.append(m)
        var_pair.append(v)
        var_single.append(ptc.single_frame_variance(a))
    means, var_pair, var_single = map(np.array, (means, var_pair, var_single))
    print(f"patch: {npix} pixels ({int(np.sqrt(npix))} x {int(np.sqrt(npix))})")

    fit = ptc.fit_ptc(means, var_pair, SPAN)
    fit1 = ptc.fit_ptc(means, var_single, SPAN)

    print(f"\ntrue K = {TRUE_K:.4f} DN/e-, true read noise = {TRUE_READ:.2f} e-, "
          f"true full well = {psu.FULL_WELL:.0f} e-")
    print(f"pair fit  : K = {fit['K']:.4f} +/- {fit['sK']:.4f}  "
          f"({ptc.pct_error(fit['K'], TRUE_K):+.2f} %)   "
          f"read = {fit['read']:.2f} +/- {fit['s_read']:.2f}  "
          f"({ptc.pct_error(fit['read'], TRUE_READ):+.2f} %)")
    print(f"single fit: K = {fit1['K']:.4f} +/- {fit1['sK']:.4f}  "
          f"({ptc.pct_error(fit1['K'], TRUE_K):+.2f} %)   read = {fit1['read']:.2f}")
    print(f"single vs pair gain difference: "
          f"{ptc.pct_error(fit1['K'], fit['K']):+.2f} %")

    fw, dr = ptc.full_well_and_dr(fit["K"], fit["read"], SPAN)
    print(f"full well = {fw:.0f} e-, dynamic range = {dr:.2f} stops "
          f"(true: {np.log2(psu.FULL_WELL / TRUE_READ):.2f})")
    print(f"points used: {fit['used'].sum()} of {len(means)}")

    print("\n  mean (DN)   var pair   var single")
    for m, vp, vs in zip(means, var_pair, var_single):
        print(f"  {m:9.1f}  {vp:9.2f}  {vs:10.2f}")

    used = fit["used"]
    xs = np.linspace(0, means.max() * 1.02, 100)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(means[used], var_pair[used], "o", label="fitted")
    ax.plot(means[~used], var_pair[~used], "o", mfc="none", label="excluded (> 70 % sat.)")
    ax.plot(xs, fit["K"] * xs + fit["c"], "-", label=f"K = {fit['K']:.4f}")
    ax.set_xlabel("mean above black (DN)")
    ax.set_ylabel(r"variance (DN$^2$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("ptc_sim.png", dpi=150)


if __name__ == "__main__":
    main()
