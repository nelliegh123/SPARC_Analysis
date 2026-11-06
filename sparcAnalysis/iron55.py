"""
Fits the Fe-55 peak in the ToT spectrum of every pixel in the sparc6 iron55 run and plots the spectra.

To use the results in another script (arrays are indexed [row, col], NaN where there is no fit):
    from iron55 import fit_all_pixels, load_hits
    mean_tot, sigma_tot = fit_all_pixels(load_hits())
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from sparc_analyzer import SPARCAnalyser

DATA_FILE = Path("/home/teleuser/Stephan/data/sparc6/iron55/data_vcasb300_ireset10_vcasn300_ibias100_idb200_run_0.raw")
OVERLAY_PIXELS = {(0, 0): "#2a9df4", (3, 3): "#e0479e", (6, 6): "#ffc000"}  # (col, row): color
MIN_PEAK_TOT = 15  # [µs] The Fe peak is searched above this, because the low ToT pile-up is higher than the peak
FIT_HALF_WIDTH = 3  # [µs] Only this close to the peak is fitted, because the peak has a tail on its low side


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


#Step 1: load the sparc6 iron55 data usong sparc analyser
def load_hits():
    analyser = SPARCAnalyser(DATA_FILE, use_cache=True)
    analyser.do_output = False
    analyser.analyze()
    return analyser.hits


#Step 2: fit the largest peak (to the right) and exctract the mean ToT (and sigma) of this fit.
#I want this script to returmn or to easily be able to access a list or arrray ccontaining the mean Fe-ToT of each pixel.
def fit_peak(tots):
    """Returns the fitted amplitude, mean [µs] and sigma [µs] of the Fe peak in the ToTs [µs] of one pixel"""
    counts, edges = np.histogram(tots, bins=120, range=(0, 60))
    centers = (edges[:-1] + edges[1:]) / 2

    above_pile_up = centers > MIN_PEAK_TOT
    peak = centers[above_pile_up][np.argmax(counts[above_pile_up])]
    window = np.abs(centers - peak) < FIT_HALF_WIDTH

    try:
        (amplitude, mean, sigma), _ = curve_fit(gaussian, centers[window], counts[window], p0=(counts[window].max(), peak, 2))
    except RuntimeError:  # The fit did not converge
        return np.nan, np.nan, np.nan

    # Reject fits that converged outside the fitted range
    if abs(mean - peak) > FIT_HALF_WIDTH:
        return np.nan, np.nan, np.nan
    return amplitude, mean, abs(sigma)


def fit_all_pixels(hits):
    """Returns the mean Fe-ToT and its sigma [µs] of every pixel, as arrays indexed [row, col]"""
    mean_tot = np.full((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), np.nan)
    sigma_tot = np.full((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), np.nan)
    for (row, col), pixel_hits in hits.groupby(["row", "column"])["tot"]:
        if pixel_hits.size >= 100:  # Too few hits to show a peak
            _, mean_tot[row, col], sigma_tot[row, col] = fit_peak(pixel_hits.to_numpy() * 1e-3)  # ns to µs
    return mean_tot, sigma_tot


if __name__ == "__main__":
    hits = load_hits()
    mean_tot, sigma_tot = fit_all_pixels(hits)
    print(f"Fitted {np.count_nonzero(~np.isnan(mean_tot))} pixels")
    print(f"Mean Fe-ToT: median {np.nanmedian(mean_tot):.1f} µs, spread between pixels {np.nanstd(mean_tot):.1f} µs")

    #Step 3: Plot the ToT histogram for all pixels.
    # Also plot the overlayed histogram (two or three pixels) like we do currently in sparcanalyzer, and include the gaussian fiot to the peak of these pixels
    plt.rcParams["font.size"] = 13

    plt.figure("ToT distribution", figsize=(8, 6))
    plt.hist(hits["tot"] * 1e-3, bins=120, range=(0, 60), color="#2a9df4")
    plt.grid(alpha=0.5)
    plt.xlabel("ToT [µs]")
    plt.ylabel("Entries")
    plt.title("sparc6 iron55, all pixels", fontsize=11)
    plt.tight_layout()

    plt.figure("Overlayed ToT distribution", figsize=(8, 6))
    x = np.linspace(0, 60, 1000)
    for (col, row), color in OVERLAY_PIXELS.items():
        tots = hits[(hits["column"] == col) & (hits["row"] == row)]["tot"].to_numpy() * 1e-3
        amplitude, mean, sigma = fit_peak(tots)
        plt.hist(tots, bins=120, range=(0, 60), color=color, alpha=0.5, label=f"Pixel ({col}, {row})")
        plt.plot(x, gaussian(x, amplitude, mean, sigma), color=color, lw=2, label=f"Fit: {mean:.1f} ± {sigma:.1f} µs")
    plt.grid(alpha=0.5)
    plt.xlabel("ToT [µs]")
    plt.ylabel("Entries")
    plt.title("sparc6 iron55", fontsize=11)
    plt.legend()
    plt.tight_layout()

    plt.show()
