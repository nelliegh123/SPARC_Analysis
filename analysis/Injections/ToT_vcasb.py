import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from injection_scan import fit_scan, jitter_scatter, EXTRA_DATASETS, OLD_PLOT_DIR

"""
Plots the ToT vs Vcasb for a fixed Vh.
The ToT is the mean of the Gaussian fit to the ToT histogram for each pixel.
Removes any ToT values more than 5 sigma above the chip mean.
The Vcasb scan only exists in the old data, so this always uses that data.
"""


Vh = 600
Vcasb_VALUES = list(range(220, 370, 10))  # 220 ... 360 mV
SETTINGS = f"Vh = {Vh} mV"


N_SIGMA = 5
DATA = EXTRA_DATASETS["old"]["data_dir"]
OUTPUT_DIR = OLD_PLOT_DIR / "tot_vcasb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#-----Removes pixels more than N_SIGMA above the mean, repeating the cut until nothing more is removed-----
def remove_outliers(tot):
    tot = tot.copy()
    while True:
        cut = np.nanmean(tot) + N_SIGMA * np.nanstd(tot)
        too_high = tot > cut
        if not too_high.any():
            return tot
        tot[too_high] = np.nan


def plot_tot_vcasb(means):
    vcasb = np.array(Vcasb_VALUES)
    flat = means.reshape(len(vcasb), -1)
    cut = np.array([remove_outliers(tot) for tot in flat])
    chip_mean = np.nanmean(cut, axis=1)

    for v, m, n_removed in zip(vcasb, chip_mean, np.isnan(cut).sum(axis=1) - np.isnan(flat).sum(axis=1)):
        print(f"Vcasb = {v} mV: chip mean ToT = {m:.2f} µs" + (f", removed {n_removed} pixels > {N_SIGMA} sigma" if n_removed else ""))

    plt.figure("ToT vs Vcasb", figsize=(8, 6))
    jitter_scatter(vcasb, cut)
    plt.plot(vcasb, chip_mean, "o", color="darkorange", ms=8, label="Chip mean")

    plt.ylim(bottom=0)
    plt.xlabel("Vcasb [mV]")
    plt.ylabel("Mean ToT [µs]")
    plt.title(f"Mean ToT vs Vcasb, all pixels\n{SETTINGS}")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    output_file = OUTPUT_DIR / f"tot_vcasb_vh_{Vh}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")




#-----------Main function where we call everything-----------
def main():
    scan = fit_scan([(vcasb, Vh) for vcasb in Vcasb_VALUES], f"vh_{Vh}_vcasb_scan", data_dir=DATA)
    plot_tot_vcasb(scan["means"])  # indexed [vcasb, row, col]

    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")


if __name__ == "__main__":
    main()
