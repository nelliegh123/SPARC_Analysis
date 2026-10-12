import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from injection_scan import fit_scan, N_INJECTIONS, PLOT_DIR

"""
Compares several (Vcasb, VH) settings in the same plot, one color per setting:
the number of hits per pixel and the mean ToT per pixel.
"""

# (Vcasb, VH) in mV, at most 8 settings so every one gets its own color
SETTINGS = [
    (300, 75),
    (300, 100),
    (300, 200),
    (300, 400),
    (300, 600),
    (300, 800),
    (300, 900),
    (300, 1200),
]

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

COUNTS_DIR = PLOT_DIR / "counts_hitmap"  
TOT_DIR = PLOT_DIR / "tot_hitmap"       
COUNTS_DIR.mkdir(parents=True, exist_ok=True)
TOT_DIR.mkdir(parents=True, exist_ok=True)


def label(vcasb, vh):
    return f"Vcasb = {vcasb} mV, VH = {vh} mV"


#-----Plots one outlined histogram per setting, all with the same bins-----
# Values outside hist_range are left out of the plot and the number left out is printed
def compare_hist(per_setting, xlabel, title, output_file, bins=25, hist_range=None):
    values = [v[~np.isnan(v)] for v in per_setting]
    edges = np.histogram_bin_edges(np.concatenate(values), bins, hist_range)
    for (vcasb, vh), v in zip(SETTINGS, values):
        n_out = np.sum((v < edges[0]) | (v > edges[-1]))
        if n_out:
            print(f"{label(vcasb, vh)}: {n_out} pixels outside the {xlabel} plot range")
    plt.figure(title, figsize=(9, 6))
    for (vcasb, vh), v, color in zip(SETTINGS, values, COLORS):
        plt.hist(v, edges, histtype='stepfilled', facecolor=to_rgba(color, 0.25), edgecolor=color,
                 linewidth=1.5, label=label(vcasb, vh))
    plt.xlabel(xlabel)
    plt.ylabel('Number of pixels')
    # plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")


#-----------Main function where we call everything-----------
def main():
    if len(SETTINGS) > len(COLORS):
        raise ValueError(f"At most {len(COLORS)} settings can be compared in one plot")

    # each setting is fitted (or loaded from cache) under its own name, the same one injection.py uses
    scans = [fit_scan([(vcasb, vh)], f"vcasb_{vcasb}_vh_{vh}") for vcasb, vh in SETTINGS]
    nhits = [scan["nhits"][0] for scan in scans]
    tot_means = [scan["means"][0] for scan in scans]

    compare_hist(nhits, 'Number of hits', 'Number of hits per pixel', COUNTS_DIR / 'hits_hist_compare.png',
                 hist_range=(0, 1.5 * N_INJECTIONS))
    compare_hist(tot_means, 'Mean ToT [µs]', 'Mean ToT per pixel', TOT_DIR / 'tot_mean_hist_compare.png')


if __name__ == "__main__":
    main()
