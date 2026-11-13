"""
Fits the Fe-55 peak in the ToT spectrum of every pixel in iron55 run and plots the spectra.

To use the results in another script (arrays are indexed [col, row], NaN where there is no fit):
    from iron55 import fit_all_pixels, load_hits
    mean_tot, sigma_tot = fit_all_pixels(load_hits())
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from sparc_analyzer import SPARCAnalyser

# DATA_FILE = Path("/home/teleuser/Stephan/sparc-analysis/data/sparc6/iron55/data_vcasb300_ireset10_vcasn300_ibias100_idb200_run_0.raw")
# DATA_FILE = Path("/home/teleuser/Stephan/sparc-analysis/data/sparc7/iron55/data_vcasb360_ireset70_vcasn400_ibias700_idb300_run_0.raw")
DATA_FILE = Path("/home/teleuser/Stephan/data/sparc13/iron55/data_vcasb300_ireset7_vcasn300_ibias100_idb200_run_1.raw")

CHIP = 'Sparc 13'
XRANGE = (0, 65)  
TOT_BINS = np.linspace(0, 65, 121)  # [µs] The histogram bins that the Fe peak is fitted in

LOWER_LIMIT = 4 #Set this as a value in the "valley" between the left noise peak, and the actual iron peak
OVERLAY_PIXELS = {(3, 3): "#e0479e", (6, 6): "#ffc000"}  # (col, row): color (26, 26): "#2a9df4",


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def double_gaussian(x, amplitude1, mean1, sigma1, amplitude2, mean2, sigma2):
    return gaussian(x, amplitude1, mean1, sigma1) + gaussian(x, amplitude2, mean2, sigma2)


#Loads the iron55 data
def load_hits():
    analyser = SPARCAnalyser(DATA_FILE, use_cache=True)
    analyser.do_output = False
    analyser.analyze()
    return analyser.hits


#Fits the Iron55 spectra of a pixel
def fit_peak(tots, col, row):
    """Returns the fitted amplitude, mean [µs] and sigma [µs] of the Fe peak in the ToTs [µs] of one pixel"""
    counts, edges = np.histogram(tots, bins=TOT_BINS)
    centers = (edges[:-1] + edges[1:]) / 2

    peak = np.argmax(np.where(centers > LOWER_LIMIT, counts, -1))
    before_peak = (centers > LOWER_LIMIT) & (centers < centers[peak])
    valley = np.argmin(np.where(before_peak, counts, np.inf))
    start = centers[valley] if before_peak.any() else LOWER_LIMIT
    fit_range = centers >= start

    # Step 1: fit single Gaussian in region to the right of the valley (emptiest bin)
    try:
        (amplitude, mean, sigma), _ = curve_fit(gaussian, centers[fit_range], counts[fit_range], p0=(counts[peak], centers[peak], 2))
    except RuntimeError:  # The fit did not converge
        print(f"------------First (single) Gaussian fit did not converge: pixel ({col}, {row})---------")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    sigma = abs(sigma)

    # # Step 2: fit the single Gaussian again within a ±1 sigma range of the first mean
    single_range = np.abs(centers - mean) < 1 * sigma
    single_centers, single_counts = centers[single_range], counts[single_range]
    try:
        (amplitude, mean, sigma), _ = curve_fit(gaussian, single_centers, single_counts, p0=(amplitude, mean, sigma), bounds=([0, mean-sigma, 0], [np.inf, mean+sigma, 2*sigma]))
    except (RuntimeError, ValueError):
        print(f"------------2 sigma (single) Gaussian fit did not converge: pixel ({col}, {row})---------")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


    # Step 3: fit the double Gaussian
    double_range = centers > mean - 1 * sigma
    try:
        (amplitude1, mean1, sigma1, amplitude2, mean2, sigma2), _ = curve_fit(double_gaussian, centers[double_range], counts[double_range],
        p0=(amplitude, mean, sigma, amplitude/7, 1.1*mean, sigma), bounds=([0, mean-sigma, 0, 0, mean, 0], [np.inf, mean+sigma, 2*sigma, np.inf, 1.3*mean, 2*sigma]), maxfev=10000)
    except (RuntimeError, ValueError) as error:
        print(f"Double fit failed: pixel ({col}, {row}): {error}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, amplitude, mean, sigma

    #Step 4: fit double Gaussian again, not sure if this makes a difference?
    try:
        (amplitude1, mean1, sigma1, amplitude2, mean2, sigma2), _ = curve_fit(double_gaussian, centers[double_range], counts[double_range],
        p0=(amplitude1, mean1, sigma1, amplitude2, mean2, sigma2), bounds=([0, mean1-sigma1, 0, 0, mean2-sigma2, 0], [np.inf, mean1+sigma1, 2*sigma1, np.inf, mean2+sigma2, 2*sigma2]), maxfev=10000)
    except (RuntimeError, ValueError) as error:
        print(f"Double fit failed: pixel ({col}, {row}): {error}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, amplitude, mean, sigma


    return amplitude1, mean1, sigma1, amplitude2, mean2, sigma2, amplitude, mean, sigma


#Fits the Iron spectra of all pixels using "fit_peak"
def fit_all_pixels(hits):
    """Returns the mean Fe-ToT and its sigma [µs] of every pixel, as arrays indexed [col, row]"""
    mean_tot1, mean_tot2 = np.full((SPARCAnalyser.NCOLS, SPARCAnalyser.NROWS), np.nan), np.full((SPARCAnalyser.NCOLS, SPARCAnalyser.NROWS), np.nan)
    sigma_tot1, sigma_tot2 = np.full((SPARCAnalyser.NCOLS, SPARCAnalyser.NROWS), np.nan), np.full((SPARCAnalyser.NCOLS, SPARCAnalyser.NROWS), np.nan)
    for (col, row), pixel_hits in hits.groupby(["column", "row"])["tot"]:
        if pixel_hits.size >= 100:  # Too few hits to show a peak
            _, mean_tot1[col, row], sigma_tot1[col, row], _, mean_tot2[col, row], sigma_tot2[col, row], _, _, _ = fit_peak(pixel_hits.to_numpy() * 1e-3, col, row)  # ns to µs
    return mean_tot1, sigma_tot1, mean_tot2, sigma_tot2


#Caches result so we dont have to rerun everything all the time
FE_PEAKS_CACHE = Path(__file__).resolve().parent.parent / "sparcCache" / DATA_FILE.parent.parent.name / f"iron55_{DATA_FILE.stem}.npz"
def load_fe_peaks(hits, recompute=False):
    """Returns the same as fit_all_pixels(hits), but saves it to FE_PEAKS_CACHE the first time and reads it from there after that.
    Use recompute=True (or delete the file) after changing the data or the fits."""
    if FE_PEAKS_CACHE.exists() and not recompute:
        print(f"Loading the Fe-55 peak fits from {FE_PEAKS_CACHE}")
        with np.load(FE_PEAKS_CACHE) as cache:
            return cache["mean_tot1"], cache["sigma_tot1"], cache["mean_tot2"], cache["sigma_tot2"]
    mean_tot1, sigma_tot1, mean_tot2, sigma_tot2 = fit_all_pixels(hits)
    FE_PEAKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FE_PEAKS_CACHE, mean_tot1=mean_tot1, sigma_tot1=sigma_tot1, mean_tot2=mean_tot2, sigma_tot2=sigma_tot2)
    print(f"Saved the Fe-55 peak fits to {FE_PEAKS_CACHE}")
    return mean_tot1, sigma_tot1, mean_tot2, sigma_tot2


if __name__ == "__main__":
    hits = load_hits()
    mean_tot1, sigma_tot1, mean_tot2, sigma_tot2 = load_fe_peaks(hits)
    print()
    print(f"Fitted {np.count_nonzero(~np.isnan(mean_tot1))} pixels")
    print(f"Mean Fe-ToT: median {np.nanmedian(mean_tot1):.1f} µs, spread between pixels {np.nanstd(mean_tot1):.1f} µs")

    #Plots the Iron spectra for all pixels
    #Also plots the iron spectra of a few chosen pixels, with their gaussian and double gaussian fit
    plt.rcParams["font.size"] = 13
    plt.figure("ToT distribution", figsize=(8, 6))
    plt.hist(hits["tot"] * 1e-3, bins=120, range=XRANGE, color="#2a9df4") 
    plt.grid(alpha=0.5)
    plt.xlabel("ToT [µs]")
    plt.ylabel("Entries")
    plt.title(f"{CHIP} iron55, all pixels", fontsize=11)
    plt.tight_layout()

    plt.figure("Overlayed ToT distribution", figsize=(8, 6))
    x = np.linspace(0, XRANGE, 1000)
    for (col, row), color in OVERLAY_PIXELS.items():
        tots = hits[(hits["column"] == col) & (hits["row"] == row)]["tot"].to_numpy() * 1e-3
        amplitude1, mean1, sigma1, amplitude2, mean2, sigma2, amplitude, mean, sigma = fit_peak(tots, col, row)
        plt.hist(tots, bins=120, range=XRANGE, color=color, alpha=0.5, label=f"Pixel ({col}, {row})")
        plt.plot(x, double_gaussian(x, amplitude1, mean1, sigma1, amplitude2, mean2, sigma2), color=color, lw=2, label=f"Fit: {mean1:.1f} ± {sigma1:.1f} µs")
        plt.plot(x, gaussian(x, amplitude, mean, sigma), color=color, lw=2, linestyle='dashed',  label=f"First gauss fit: {mean1:.1f} ± {sigma1:.1f} µs")
    plt.grid(alpha=0.5)
    plt.xlabel("ToT [µs]")
    plt.ylabel("Entries")
    plt.title(f"{CHIP} iron55", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.show()