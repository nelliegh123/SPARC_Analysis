"""
Fits the ToT of every pixel in the sparc6 injection scan with a Gaussian and plots the mean ToT vs Vh.
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
DATA_DIR = ROOT / "data/sparc6/injection"
PLOT_FILE = ROOT / "sparcPlots/sparc6/ToT_vh/tot_vs_vh_gaussian.png"
HIGHLIGHTED_PIXELS = {(17, 20): "#e0479e", (19, 19): "#ffc000"}  # (col, row): color
EXAMPLE_VHS = [100, 200, 300, 600, 1000]  # [mV] The Vh points of the example fits



def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_pixel(file, ax=None):
    """Returns the Vh, column, row, and the fitted mean and sigma of the ToT [µs] of one raw file. Draws the fit on ax if given"""
    vh, col, row = map(int, re.search(r"_vh(\d+)_c(\d+)_r(\d+)_", file.name).groups())

    # Step 1: load the data
    analyser = SPARCAnalyser(file)
    analyser.do_output = False
    analyser.analyze()
    tots = analyser.hits["tot"].to_numpy() * 1e-3  # ns to µs


    # Step 2: Remove the injection chargeup
    tots = tots[tots > otsu_threshold(tots)]
    if tots.size < 1:
        return vh, col, row, np.nan, np.nan


    # Step 3: fit the ToT with a Gaussian. The percentiles keep a few outliers from stretching the histogram
    counts, edges = np.histogram(tots, bins=30, range=np.percentile(tots, [0.5, 99.5]))
    centers = (edges[:-1] + edges[1:]) / 2
    if ax is not None:
        ax.stairs(counts, edges, fill=True, color="#2a9df4", alpha=0.6)
    try:
        (amplitude, mean, sigma), _ = curve_fit(gaussian, centers, counts, p0=(counts.max(), np.median(tots), tots.std()))
    except RuntimeError:  # The fit did not converge
        return vh, col, row, np.nan, np.nan

    if ax is not None:
        fit_x = np.linspace(edges[0], edges[-1], 200)
        ax.plot(fit_x, gaussian(fit_x, amplitude, mean, sigma), color="black", lw=1.5, label=f"{mean:.2f} ± {abs(sigma):.2f} µs")
        ax.legend(fontsize=9)

    # Reject fits that converged but do not describe the data
    residuals = counts - gaussian(centers, amplitude, mean, sigma)
    r_squared = 1 - np.sum(residuals**2) / np.sum((counts - counts.mean()) ** 2)
    if not (edges[0] < mean < edges[-1] and r_squared > 0.9):
        return vh, col, row, np.nan, np.nan
    return vh, col, row, mean, abs(sigma)


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob("*.raw"))
    with ProcessPoolExecutor() as pool:
        results = list(tqdm(pool.map(fit_pixel, files, chunksize=16), total=len(files)))
    vh, col, row, mean, sigma = np.array(results).T

    # Step 4: plot the mean ToT of every pixel vs Vh
    vhs = np.unique(vh)
    means = np.full((len(vhs), SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), np.nan)  # [vh, row, col]
    means[np.searchsorted(vhs, vh), row.astype(int), col.astype(int)] = mean

    plt.rcParams["font.size"] = 13
    plt.figure(figsize=(8, 6))

    # All pixels, spread out a bit in x so that the points do not lie on top of each other
    x = np.repeat(vhs, SPARCAnalyser.NROWS * SPARCAnalyser.NCOLS) + np.random.default_rng(0).uniform(-5, 5, means.size)
    plt.plot(x, means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")

    plt.plot(vhs, np.nanmedian(means, axis=(1, 2)), "-", color="black", lw=1.5, label="Median of all pixels")

    for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
        plt.plot(vhs, means[:, pixel_row, pixel_col], "o-", color=color, ms=6, label=f"Pixel ({pixel_col}, {pixel_row})")

    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.xlabel("Vh [mV]")
    plt.ylabel("Mean ToT [µs]")
    plt.title("sparc6 injection, Gaussian fit", fontsize=11)
    plt.legend()
    plt.tight_layout()
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(PLOT_FILE, dpi=150)
    # print(f"Saved {PLOT_FILE}")
    print("Finished")

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

    plt.show()
