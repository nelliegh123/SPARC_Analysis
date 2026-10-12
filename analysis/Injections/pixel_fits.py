import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from injection_scan import load_run, fit_gaussian, gaussian, tot_bins, DATA_DIR, PLOT_DIR

"""
Plots the ToT histogram and the Gaussian fit of a few chosen pixels side by side, for one setting.
Uses the same fit (100 bins and the same outlier cut) as fit_scan, so the fitted mean is the one in the ToT maps.
"""

VCASB = 300
VH = 800
PIXELS = [(0, 31), (1, 31), (2, 31)]  # (col, row)

# the default data (pulse duration = 250); for an older data set use e.g. EXTRA_DATASETS["ireset"]["data_dir"] and ["plot_dir"]
DATA = DATA_DIR
LABEL = "Pulse duration = 250"
OUTPUT_DIR = PLOT_DIR / "tot_hitmap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BINS = 100


#-----ToT histogram of one pixel with its Gaussian fit-----
def plot_pixel(ax, col, row):
    hitmap, _, tots = load_run(VCASB, VH, col, row, DATA)
    ax.set_title(f"Pixel ({col}, {row}): {int(hitmap[row, col])} hits")
    ax.set_xlabel("ToT [µs]")
    ax.set_ylabel("Counts")
    if tots.size == 0:
        ax.text(0.5, 0.5, "No hits", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        print(f"px ({col}, {row}): no hits")
        return

    # same cut as in fit_gaussian: hits more than 10 robust standard deviations from the median are left out,
    # otherwise a few wrongly paired hits (hundreds of µs) stretch the bins and the fit doesn't match the histogram
    median = np.median(tots)
    robust_std = 1.4826 * np.median(np.abs(tots - median))
    if robust_std > 0:
        keep = np.abs(tots - median) <= 10 * robust_std
        if (~keep).any():
            print(f"px ({col}, {row}): {np.sum(~keep)} hits more than 10 robust sigma from the median left out")
        tots = tots[keep]
    counts, edges = np.histogram(tots, tot_bins(tots, BINS))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, counts, width=edges[1] - edges[0], alpha=0.5, label="Data")
    amp, mean, sigma, chi2_ndf = fit_gaussian(tots, BINS)
    if np.isnan(mean):
        ax.text(0.5, 0.9, "No Gaussian fit", transform=ax.transAxes, ha="center", fontsize=12)
        print(f"px ({col}, {row}): {tots.size} hits, no Gaussian fit")
    else:
        x_fit = np.linspace(edges[0], edges[-1], 300)
        ax.plot(x_fit, gaussian(x_fit, amp, mean, sigma), color="red",
                label=f"Fit: mean = {mean:.2f} µs\nsigma = {sigma:.2f} µs\n$\\chi^2$/ndf = {chi2_ndf:.2f}")
        print(f"px ({col}, {row}): {tots.size} hits, mean = {mean:.2f} µs, sigma = {sigma:.2f} µs, chi2/ndf = {chi2_ndf:.2f}")
    ax.legend()


#-----------Main function where we call everything-----------
def main():
    fig, axes = plt.subplots(1, len(PIXELS), figsize=(5.5 * len(PIXELS), 4.5), squeeze=False)
    for ax, (col, row) in zip(axes[0], PIXELS):
        plot_pixel(ax, col, row)
    fig.suptitle(f"ToT histograms, {LABEL}, Vcasb = {VCASB} mV, Vh = {VH} mV")
    fig.tight_layout()
    pixel_tag = "_".join(f"{col}_{row}" for col, row in PIXELS)
    output_file = OUTPUT_DIR / f"tot_fits_vcasb_{VCASB}_vh_{VH}_px_{pixel_tag}.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
