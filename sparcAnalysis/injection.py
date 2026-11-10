"""
Fits the ToT of every pixel in the sparc6 injection scan with a Gaussian and plots the mean ToT vs Vh.
Also fits the mean ToT vs Vh of every pixel with a straight line, see calibrate().

To use the calibration in another script:
    from injection import calibrate, tot_to_vh
    calibration = calibrate()
    vh = tot_to_vh(tot, col, row, calibration)  # ToT in µs, Vh in mV. Also works with arrays, NaN where the pixel has no fit
"""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from injection_chargeup import otsu_threshold
from sparc_analyzer import SPARCAnalyser

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sparc7/injection"
CHIP = "Sparc 7"

HIGHLIGHTED_PIXELS = {(17, 20): "#e0479e", (19, 19): "#ffc000"}  # (col, row): color
EXAMPLE_VHS = [100, 200, 300, 600, 1000]  # [mV] The Vh points of the example fits


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def linear_fit (x, a, b):
    return a*x + b


def fit_pixel(file, ax=None):
    """Returns the Vh, column, row, and the fitted mean, sigma and fit error on the mean of the ToT [µs] of one raw file"""
    vh, col, row = map(int, re.search(r"_vh(\d+)_c(\d+)_r(\d+)_", file.name).groups())

    # Step 1: load the data
    analyser = SPARCAnalyser(file)
    analyser.do_output = False
    analyser.analyze()
    tots = analyser.hits["tot"].to_numpy() * 1e-3  # ns to µs


    # Step 2: Remove the injection chargeup
    tots = tots[tots > otsu_threshold(tots)]
    if tots.size < 10:
        return vh, col, row, np.nan, np.nan, np.nan


    # Step 3: fit the ToT with a Gaussian. The percentiles keep a few outliers from stretching the histogram
    counts, edges = np.histogram(tots, bins=30)# range=np.percentile(tots, [0.5, 99.5]))
    centers = (edges[:-1] + edges[1:]) / 2
    if ax is not None:
        ax.stairs(counts, edges, fill=True, color="#2a9df4", alpha=0.6)
    try:
        (amplitude, mean, sigma), covariance = curve_fit(gaussian, centers, counts, p0=(counts.max(), np.median(tots), tots.std()))
    except RuntimeError:  # The fit did not converge
        return vh, col, row, np.nan, np.nan, np.nan

    if ax is not None:
        fit_x = np.linspace(edges[0], edges[-1], 200)
        ax.plot(fit_x, gaussian(fit_x, amplitude, mean, sigma), color="black", lw=1.5, label=f"{mean:.2f} ± {abs(sigma):.2f} µs")
        ax.legend(fontsize=9)

    # Reject fits that converged but do not describe the data
    residuals = counts - gaussian(centers, amplitude, mean, sigma)
    r_squared = 1 - np.sum(residuals**2) / np.sum((counts - counts.mean()) ** 2)
    if not (edges[0] < mean < edges[-1] and r_squared > 0.9):
        return vh, col, row, np.nan, np.nan, np.nan
    return vh, col, row, mean, abs(sigma), np.sqrt(covariance[1, 1])



def fit_all_pixels(vhs, means, mean_errors):
    """Returns the slope [µs/mV], intercept [µs] and reduced chi2 of the linear ToT vs Vh fit of every pixel, as arrays indexed [row, col], NaN where there is no fit"""
    slopes = np.full(means.shape[1:], np.nan)
    intercepts = np.full(means.shape[1:], np.nan)
    reduced_chi_squareds = np.full(means.shape[1:], np.nan)

    for row, col in np.ndindex(means.shape[1:]):
        pixel_means = means[:, row, col]
        pixel_mean_errors = mean_errors[:, row, col]

        valid = np.isfinite(pixel_means) & np.isfinite(pixel_mean_errors) & (pixel_mean_errors > 0)
        if valid.sum() < 3:  # Too few points to have any degrees of freedom
            continue
        try:
            popt, _ = curve_fit(linear_fit, vhs[valid], pixel_means[valid], sigma=pixel_mean_errors[valid])
        except RuntimeError:  # The fit did not converge
            continue
        if popt[0] <= 0:  # ToT has to grow with Vh, or the fit can not be inverted
            continue

        slopes[row, col], intercepts[row, col] = popt
        reduced_chi_squareds[row, col] = np.sum(((pixel_means[valid] - linear_fit(vhs[valid], *popt)) / pixel_mean_errors[valid]) ** 2) / (valid.sum() - 2)
    return slopes, intercepts, reduced_chi_squareds


def calibrate():
    """Fits every raw file in DATA_DIR and the ToT vs Vh of every pixel. Returns a dict with
    vhs [mV], the Gaussian means, sigmas and mean_errors [µs] indexed [vh, row, col],
    and the linear fit slopes [µs/mV], intercepts [µs] and reduced_chi_squareds indexed [row, col]"""
    files = sorted(DATA_DIR.glob("*.raw"))
    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(fit_pixel, files, chunksize=16), total=len(files)))
    vh, col, row, mean, sigma, mean_error = np.array(results).T

    vhs = np.unique(vh)
    means = np.full((len(vhs), SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), np.nan)  # [vh, row, col]
    means[np.searchsorted(vhs, vh), row.astype(int), col.astype(int)] = mean

    sigmas = np.full_like(means, np.nan)  # [vh, row, col]
    sigmas[np.searchsorted(vhs, vh), row.astype(int), col.astype(int)] = sigma

    mean_errors = np.full_like(means, np.nan)  # [vh, row, col]
    mean_errors[np.searchsorted(vhs, vh), row.astype(int), col.astype(int)] = mean_error

    # Fit the mean ToT vs Vh of every pixel with a straight line, so that tot_to_vh can use it
    slopes, intercepts, reduced_chi_squareds = fit_all_pixels(vhs, means, mean_errors)
    print(f"Fitted {np.count_nonzero(np.isfinite(slopes))} pixels, median reduced chi2 = {np.nanmedian(reduced_chi_squareds):.2f}")

    return {"vhs": vhs, "means": means, "sigmas": sigmas, "mean_errors": mean_errors,
            "slopes": slopes, "intercepts": intercepts, "reduced_chi_squareds": reduced_chi_squareds}


def tot_to_vh(tot, col, row, calibration):
    """Returns the Vh [mV] that corresponds to a ToT [µs] in a pixel, from inverting ToT = slope * Vh + intercept. calibration is the output of calibrate()"""
    return (tot - calibration["intercepts"][row, col]) / calibration["slopes"][row, col]


if __name__ == "__main__":
    calibration = calibrate()
    vhs, means, sigmas, mean_errors = calibration["vhs"], calibration["means"], calibration["sigmas"], calibration["mean_errors"]


    # Step 4: plot the mean ToT of every pixel vs Vh
    plt.rcParams["font.size"] = 13
    plt.figure(figsize=(8, 6))

    #Plot all pixels spread out a bit in x
    x = np.repeat(vhs, SPARCAnalyser.NROWS * SPARCAnalyser.NCOLS) + np.random.default_rng(0).uniform(-5, 5, means.size)
    plt.plot(x, means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")

    for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
        #Highlighting the mean ToT of some pixels
        plt.plot(vhs, means[:, pixel_row, pixel_col], "o-", color=color, ms=6, label=f"Pixel ({pixel_col}, {pixel_row})")
        
        #Fitting the highlighted pixels using a linear function
        pixel_means = means[:, pixel_row, pixel_col]
        pixel_mean_errors = mean_errors[:, pixel_row, pixel_col]

        valid = np.isfinite(pixel_means)
        popt, pcov = curve_fit(linear_fit, vhs[valid], pixel_means[valid], sigma=pixel_mean_errors[valid])
        plt.plot(vhs, linear_fit(vhs, popt[0], popt[1]), "k--", lw=1.5, label=f"Fit: {popt[0]:.2f} Vh + {popt[1]:.2f}")
        plt.errorbar(vhs, means[:, pixel_row, pixel_col], yerr=sigmas[:, pixel_row, pixel_col], fmt="o-", color=color, ms=6, capsize=3, label=f"Pixel ({pixel_col}, {pixel_row})")

        #Printing chi squared of each fits
        reduced_chi_squared = np.sum(((pixel_means[valid] - linear_fit(vhs[valid], *popt)) / pixel_mean_errors[valid]) ** 2) / (valid.sum() - 2)
        print(f"Pixel ({pixel_col}, {pixel_row}): reduced chi2 = {reduced_chi_squared:.2f}")

    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.xlabel("Vh [mV]")
    plt.ylabel("Mean ToT [µs]")
    plt.title(f"{CHIP} injection scan", fontsize=11)
    plt.legend()
    plt.tight_layout()
    

    # # Example fits: the ToT and Gaussian fit of the highlighted pixels at a few Vh points
    # fig, axes = plt.subplots(len(HIGHLIGHTED_PIXELS), len(EXAMPLE_VHS), figsize=(16, 6), constrained_layout=True)
    # for axes_row, (pixel_col, pixel_row) in zip(axes, HIGHLIGHTED_PIXELS):
    #     for ax, example_vh in zip(axes_row, EXAMPLE_VHS):
    #         file = next(DATA_DIR.glob(f"*_vh{example_vh}_c{pixel_col}_r{pixel_row}_*.raw"))
    #         rejected = np.isnan(fit_pixel(file, ax)[3])
    #         ax.set_title(f"Pixel ({pixel_col}, {pixel_row}), Vh = {example_vh} mV" + (" (rejected)" if rejected else ""), fontsize=10)
    #         ax.grid(alpha=0.5)
    # fig.supxlabel("ToT [µs]")
    # fig.supylabel("Entries")

    plt.show()
