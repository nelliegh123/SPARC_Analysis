import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

from injection_scan import fit_scan, DATA_DIR, PLOT_DIR, EXTRA_DATASETS, DATA_VCASB, DATA_VH

"""
Hitmaps of several data sets next to each other at one Vh, with the colour scale zoomed in on
HITS_MIN to HITS_MAX hits. Pixels below the range are white, above it green, and without hits gray.
"""

# Vh can be given on the command line: "python hitmap_compare.py 1200"
VH = int(sys.argv[1]) if len(sys.argv) > 1 else 800
HITS_MIN, HITS_MAX = 8000, 10000

# The full scans that are already cached (see bad_pixels.py), so nothing has to be fitted again
# (title, Vcasb, Vh values of the scan, cache name, data folder): the default data and the older data sets,
# with the same Vh lists as everywhere, so the cached fits are shared
DATASETS = [(f"Pulse duration = 250, Vcasb = {DATA_VCASB} mV", DATA_VCASB, DATA_VH, f"vcasb_{DATA_VCASB}_vh_scan", DATA_DIR)] + [
    (f"{d['label']}, Vcasb = {d['vcasb']} mV", d["vcasb"], d["vh_scan"], f"vcasb_{d['vcasb']}_vh_scan", d["data_dir"])
    for d in EXTRA_DATASETS.values()]

OUTPUT_DIR = PLOT_DIR / "counts_hitmap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#-----------Main function where we call everything-----------
def main():
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under("white")
    cmap.set_over("#1baf7a")
    cmap.set_bad("lightgray")

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS) + 1, 5.5), sharey=True)
    for ax, (title, vcasb, vh_scan, name, data_dir) in zip(axes, DATASETS):
        if VH not in vh_scan:
            # e.g. the Ireset scan, which isn't finished yet
            ax.text(0.5, 0.5, f"Vh = {VH} mV\nnot measured yet", transform=ax.transAxes, ha="center", va="center", fontsize=14)
            ax.set_title(title)
            ax.set_axis_off()
            print(f"{title}: no Vh = {VH} mV in this scan")
            continue
        scan = fit_scan([(vcasb, vh) for vh in vh_scan], name, data_dir=data_dir)
        nhits = scan["nhits"][vh_scan.index(VH)]
        n_below = np.sum((nhits > 0) & (nhits < HITS_MIN))
        n_above = np.sum(nhits > HITS_MAX)
        n_none = np.sum(~(nhits > 0))
        print(f"{title}: {n_below} pixels below {HITS_MIN}, {n_above} above {HITS_MAX}, {n_none} without hits")

        im = ax.imshow(np.where(nhits > 0, nhits, np.nan), origin="lower", cmap=cmap, vmin=HITS_MIN, vmax=HITS_MAX)
        ax.set_title(f"{title}\n{n_below} below {HITS_MIN}, {n_above} above {HITS_MAX}")
        ax.set_xlabel("Column")
    axes[0].set_ylabel("Row")
    # one colour bar for all, the triangles show the colours below and above the range
    fig.colorbar(im, ax=axes, label="Number of hits", extend="both", fraction=0.02, pad=0.02)
    fig.suptitle(f"Hitmaps at Vh = {VH} mV (white: below {HITS_MIN} hits, green: above {HITS_MAX}, gray: no hits)")
    output_file = OUTPUT_DIR / f"hitmap_compare_vh_{VH}_{HITS_MIN}_{HITS_MAX}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
