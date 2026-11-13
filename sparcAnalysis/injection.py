"""
Fits the ToT of every active pixel in an injection scan with a Gaussian, and plots the mean ToT vs Vh.
Then fits the mean ToT vs Vh of every pixel with curved_fit (a straight line that bends down near the threshold).
Saves a CSV on the form col,row,slope [e/ns],offset [e].
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
# DATA_DIR = ROOT / "data/sparc7/injection"
CHIP = 'sparc13'
DATA_DIR = ROOT / "../Stephan/data" / CHIP / "injection"

HIGHLIGHTED_PIXELS = {(17, 20): "#e0479e", (0, 25): "#ffc000", (13, 18):"#93C47D"}  # (col, row): color
# SPARCAnalyser.masked_pixels = [(26, 2)]

print(f"Masking pixels {SPARCAnalyser.masked_pixels}")

# ----For showing a few gausuian--------
EXAMPLE_VHS = [60, 70, 80, 90, 100]  # [mV] The Vh points of the example fits
# ----For showing a few gausuian--------


#Fit used for exctracting mean ToT of a pixel
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


#Fit used for the ToT vs Vh injection scan, per pixel
def curved_fit(Vh, a, b, c, threshold):
    return a * Vh + b - (c)/(Vh - threshold + ((c)/(a*threshold + b)))


#Fits the ToT spectra twice with a Gaussian to extract the mean
def fit_pixel(file, ax=None):
    """Returns the Vh, column, row, and the fitted mean, sigma and fit error on the mean of the ToT [µs] of one raw file"""
    vh, col, row = map(int, re.search(r"_vh(\d+)_c(\d+)_r(\d+)_", file.name).groups())

    # Step 1: load the data
    analyser = SPARCAnalyser(file)
    analyser.do_output = False
    analyser.config.add("double_pulsing_limit", 200)
    analyser.config.add("tot_upper_limit", 150000) #Sparc7 (use 50000 for sparc6)
    analyser.analyze()
    tots = analyser.hits["tot"].to_numpy() * 1e-3

    # Step 2: Remove the injection chargeup
    tots = tots[tots > otsu_threshold(tots)]
    if tots.size < 20:
        # if tots.size != 0:
            # print(f"-------------Too few ToTs ({tots.size}): Vh={vh}, pixel ({col}, {row})---------------")
        return vh, col, row, np.nan, np.nan, np.nan

    min_tot, max_tot, mean_tot, sigma_tot = np.min(tots), np.max(tots), np.median(tots), tots.std()
    tot_counts, tot_edges = np.histogram(tots, bins='auto')
    tot_centers = (tot_edges[:-1] + tot_edges[1:]) / 2
    gap = (max(mean_tot - 3 * abs(sigma_tot), 0., min_tot), min(mean_tot + 3 * abs(sigma_tot), max_tot))

    if tot_counts.sum() < 50:
        # print(f"-------------Too few counts to fit first Gaussian: Vh={vh}, pixel={col}, {row}-----------------")
        return vh, col, row, np.nan, np.nan, np.nan
    try:
        (amplitude, mean, sigma), _ = curve_fit(gaussian, tot_centers, tot_counts, p0=(tot_counts.max(), mean_tot, sigma_tot), bounds=([0, gap[0], 0], [np.inf, gap[1], gap[1]-gap[0]]))
    except RuntimeError: 
        print(f"-------------Rejected by the first Gaussian: Vh={vh}, pixel ({col}, {row})---------------------")
        return vh, col, row, np.nan, np.nan, np.nan

    #--Second fit----
    gap = (max(mean - 3 * abs(sigma), 0., min_tot), min(mean + 3 * abs(sigma), max_tot))
    remaining_tot = tots[(tots > gap[0]) & (tots < gap[1])]
    counts, edges = np.histogram(remaining_tot, bins='auto', range=gap)
    centers = (edges[:-1] + edges[1:]) / 2

    if counts.sum() < 50:
        # print(f"-------------Too few counts to fit second Gaussian: Vh={vh}, pixel ({col}, {row})---------------------")
        return vh, col, row, np.nan, np.nan, np.nan
    try:
        (amplitude, mean, sigma), covariance = curve_fit(gaussian, centers, counts, p0=(counts.max(), mean, abs(sigma)), bounds=([0, gap[0], 0], [np.inf, gap[1], gap[1]-gap[0]]))
    except RuntimeError:  
        print(f"-------------Rejected by the second Gaussian: Vh={vh}, pixel ({col}, {row})---------------------")
        return vh, col, row, np.nan, np.nan, np.nan

    if mean-sigma < np.min(edges):
            print(f"Removing fit: Vh={vh}, pixel ({col}, {row}) since mean is too close to edge")
            return vh, col, row, np.nan, np.nan, np.nan

    if ax is not None:
        # Draw the histogram that was fitted, with the fit on top
        ax.stairs(counts, edges, fill=True, color="#2a9df4", alpha=0.6)
        fit_x = np.linspace(edges[0], edges[-1], 200)
        ax.plot(fit_x, gaussian(fit_x, amplitude, mean, sigma), color="black", lw=1.5, label=f"{mean:.2f} ± {abs(sigma):.2f} µs")
        ax.legend(fontsize=9)

    return vh, col, row, mean, abs(sigma), np.sqrt(covariance[1, 1])



#Fits the ToT vs Vh injection scan of one pixel with "curved_fit"
def fit_tot_vs_vh(vhs, tots, tot_errors):
    """Fits curved_fit to the mean ToT [µs] vs Vh [mV] of one pixel. Returns popt = (a, b, c, threshold), or None if the fit fails"""
    try:
        (a0, b0), _ = curve_fit(lambda x, a, b: a*x + b, vhs, tots, sigma=tot_errors)  # Straight line as the starting point
        if a0 <= 0:  # ToT has to grow with Vh, or the fit can not be inverted
            return None
        p0 = (a0, b0, 1.0, vhs.min() * 0.8)
        bounds = ([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, vhs.min()])
        popt, _ = curve_fit(curved_fit, vhs, tots, sigma=tot_errors, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    return popt

#Calls "fit_tot_vs_vh" for all pixels
def fit_all_pixels(vhs, means, mean_errors):
    """Returns the curved_fit parameters a = slope [µs/mV], b = intercept [µs], c [µs*mV], threshold [mV] and the reduced chi2 of the ToT vs Vh fit of every pixel, as arrays indexed [col, row], NaN where there is no fit"""
    slopes = np.full(means.shape[1:], np.nan)
    intercepts = np.full(means.shape[1:], np.nan)
    cs = np.full(means.shape[1:], np.nan)
    thresholds = np.full(means.shape[1:], np.nan)
    reduced_chi_squareds = np.full(means.shape[1:], np.nan)

    for col, row in np.ndindex(means.shape[1:]):
        pixel_means = means[:, col, row]
        pixel_mean_errors = mean_errors[:, col, row]

        valid = np.isfinite(pixel_means) & np.isfinite(pixel_mean_errors) & (pixel_mean_errors > 0)
        if valid.sum() < 5:  # Too few points to have any degrees of freedom
            print("Less than 5 data points for te Vh fit")
            continue
        popt = fit_tot_vs_vh(vhs[valid], pixel_means[valid], pixel_mean_errors[valid])
        if popt is None:
            continue

        slopes[col, row], intercepts[col, row], cs[col, row], thresholds[col, row] = popt
        reduced_chi_squareds[col, row] = np.sum(((pixel_means[valid] - curved_fit(vhs[valid], *popt)) / pixel_mean_errors[valid]) ** 2) / (valid.sum() - len(popt))
    return slopes, intercepts, cs, thresholds, reduced_chi_squareds


#Calls all the above functions, in order to fit the ToT vs Vh
def calibrate():
    """Fits every raw file in DATA_DIR and the ToT vs Vh of every pixel. Returns a dict with
    vhs [mV], the Gaussian means, sigmas and mean_errors [µs] indexed [vh, col, row],
    and the curved_fit slopes [µs/mV], intercepts [µs], cs [µs*mV], thresholds [mV] and reduced_chi_squareds indexed [col, row]"""
    files = sorted(DATA_DIR.glob("*.raw"))
    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(fit_pixel, files, chunksize=16), total=len(files)))
    vh, col, row, mean, sigma, mean_error = np.array(results).T

    vhs = np.unique(vh)
    means = np.full((len(vhs), SPARCAnalyser.NCOLS, SPARCAnalyser.NROWS), np.nan)  # [vh, col, row]
    means[np.searchsorted(vhs, vh), col.astype(int), row.astype(int)] = mean

    sigmas = np.full_like(means, np.nan)  # [vh, col, row]
    sigmas[np.searchsorted(vhs, vh), col.astype(int), row.astype(int)] = sigma

    mean_errors = np.full_like(means, np.nan)  # [vh, col, row]
    mean_errors[np.searchsorted(vhs, vh), col.astype(int), row.astype(int)] = mean_error

    # Fit the mean ToT vs Vh of every pixel with curved_fit, so that tot_to_vh can use it
    slopes, intercepts, cs, thresholds, reduced_chi_squareds = fit_all_pixels(vhs, means, mean_errors)
    # print(f"Fitted {np.count_nonzero(np.isfinite(slopes))} pixels, median reduced chi2 = {np.nanmedian(reduced_chi_squareds):.2f}")
    # print(f"Fitted {np.count_nonzero(np.isfinite(slopes))} pixels, mean reduced chi2 = {np.nanmean(reduced_chi_squareds):.2f}")

    return {"vhs": vhs, "means": means, "sigmas": sigmas, "mean_errors": mean_errors,
            "slopes": slopes, "intercepts": intercepts, "cs": cs, "thresholds": thresholds,
            "reduced_chi_squareds": reduced_chi_squareds}





#Loads a cache so we dont have to recompute every time
CALIBRATION_CACHE = ROOT / "sparcCache" / CHIP / "injection_calibration.npz"
def load_calibration(recompute=False):
    """Returns the same dict as calibrate(), but saves it to CALIBRATION_CACHE the first time and reads it from there after that.
    Use recompute=True (or delete the file) after changing the data or the fits."""
    if CALIBRATION_CACHE.exists() and not recompute:
        print(f"Loading the ToT vs Vh calibration from {CALIBRATION_CACHE}")
        with np.load(CALIBRATION_CACHE) as cache:
            return dict(cache)
    calibration = calibrate()
    CALIBRATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CALIBRATION_CACHE, **calibration)
    print(f"Saved the ToT vs Vh calibration to {CALIBRATION_CACHE}")
    return calibration


#Inverts curved_fit: gives the Vh that a fit with parameters a_bar, b, c_bar, Q_thr (= a, b, c, threshold) maps to a ToT
def tot_to_charge(tot_ns, a, b, c, Vh):
    s = a * Vh + b
    u = s**2 + a * c - s * tot_ns
    charge = Vh + (2 * c * tot_ns) / (np.sqrt(4 * a * s * c * tot_ns + u**2) + u)
    return charge


#Converts a ToT to Vh using the curved_fit of the pixel, or the mean of all pixels if that pixel has no fit
def tot_to_vh(tot, col, row, calibration):
    tot, col, row = np.broadcast_arrays(np.asarray(tot, dtype=float), np.asarray(col, dtype=int), np.asarray(row, dtype=int))
    params = [calibration[key] for key in ("slopes", "intercepts", "cs", "thresholds")]

    vh = np.array(tot_to_charge(tot, *(p[col, row] for p in params)))  # np.array so that a single ToT can also be assigned to

    # Pixels without a fit: the mean Vh of all fitted pixels at the same ToT
    no_fit = np.isnan(params[0][col, row]) & np.isfinite(tot)
    if np.any(no_fit):
        has_fit = np.isfinite(params[0])
        fitted_params = [p[has_fit] for p in params]  # 1D, one entry per fitted pixel
        unique_tots, inverse = np.unique(tot[no_fit], return_inverse=True)  # Each distinct ToT is only computed once
        all_pixel_vhs = tot_to_charge(unique_tots[:, None], *fitted_params)  # [tot, pixel]
        vh[no_fit] = np.nanmean(all_pixel_vhs, axis=1)[inverse]

    return vh[()] if vh.ndim == 0 else vh  # A plain number back for a single ToT



if __name__ == "__main__":
    calibration = load_calibration()
    vhs, means, sigmas, mean_errors = calibration["vhs"], calibration["means"], calibration["sigmas"], calibration["mean_errors"]

    #Plot the ToT for all pixels, spread out slightly in x
    plt.rcParams["font.size"] = 13
    plt.figure(figsize=(8, 6))
    x = np.repeat(vhs, SPARCAnalyser.NROWS * SPARCAnalyser.NCOLS) + np.random.default_rng(0).uniform(-5, 5, means.size)
    plt.plot(x, means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")
    for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
        pixel_means = means[:, pixel_col, pixel_row]
        pixel_mean_errors = mean_errors[:, pixel_col, pixel_row]
        plt.errorbar(vhs, pixel_means, yerr=pixel_mean_errors, fmt="o", color=color, ms=6, capsize=3, label=f"Pixel ({pixel_col}, {pixel_row})")

        popt = (calibration["slopes"][pixel_col, pixel_row], calibration["intercepts"][pixel_col, pixel_row],
                calibration["cs"][pixel_col, pixel_row], calibration["thresholds"][pixel_col, pixel_row])
        if np.isnan(popt[0]):
            print(f"Pixel ({pixel_col}, {pixel_row}): fit failed")
            continue
        valid = np.isfinite(pixel_means) & np.isfinite(pixel_mean_errors) & (pixel_mean_errors > 0)
        fit_vhs = np.linspace(vhs[valid].min(), vhs[valid].max(), 300)
        plt.plot(fit_vhs, curved_fit(fit_vhs, *popt), "--", color=color, lw=1.5)
        print(f"Pixel ({pixel_col}, {pixel_row}): reduced chi2 = {calibration['reduced_chi_squareds'][pixel_col, pixel_row]:.2f}")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.xlabel("Vh [mV]")
    plt.ylabel("Mean ToT [µs]")
    plt.title(f"{CHIP} injection scan", fontsize=11)
    plt.legend()
    plt.tight_layout()

    print("Mean reduced chi squared", np.nanmean(calibration['reduced_chi_squareds']))



    # ----For showing a few gausuian--------
    # Example fits: the ToT and Gaussian fit of the highlighted pixels at a few Vh points
    fig, axes = plt.subplots(len(HIGHLIGHTED_PIXELS), len(EXAMPLE_VHS), figsize=(16, 6), constrained_layout=True)
    for axes_row, (pixel_col, pixel_row) in zip(axes, HIGHLIGHTED_PIXELS):
        for ax, example_vh in zip(axes_row, EXAMPLE_VHS):
            file = next(DATA_DIR.glob(f"*_vh{example_vh}_c{pixel_col}_r{pixel_row}_*.raw"))
            rejected = np.isnan(fit_pixel(file, ax)[3])
            ax.set_title(f"Pixel ({pixel_col}, {pixel_row}), Vh = {example_vh} mV" + (" (rejected)" if rejected else ""), fontsize=10)
            ax.grid(alpha=0.5)
    fig.supxlabel("ToT [µs]")
    fig.supylabel("Entries")
    # ----For showing a few gausuian--------

    plt.show()