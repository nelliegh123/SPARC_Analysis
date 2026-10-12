import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from injection_scan import load_run, PLOT_DIR

"""
Plots the ToT histograms of a few weak pixels (fewer than 9500 of 10000 hits at VH = 1000 mV)
next to two normal pixels, at several VH values, to see how the weak pixels differ.
"""

VCASB = 300
VH_VALUES = [400, 800, 1200]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# (col, row)
WEAK_PIXELS = [(10, 19), (21, 14), (0, 3), (27, 11), (0, 10), (11, 7)]
NORMAL_PIXELS = [(6, 6), (3, 2)]

TOT_MAX = 20  # µs, hits above this are wrapped timestamps and are left out
BIN_WIDTH = 0.1  # µs

OUTPUT_DIR = PLOT_DIR / "weak_pixels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#-----One panel per pixel, one histogram per VH-----
def plot_weak_pixels():
    pixels = [(p, "weak") for p in WEAK_PIXELS] + [(p, "normal") for p in NORMAL_PIXELS]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for ax, ((col, row), kind) in zip(axes.ravel(), pixels):
        runs = []
        for vh in VH_VALUES:
            hitmap, _, tots = load_run(VCASB, vh, col, row)
            n_wrapped = np.sum(tots > TOT_MAX)
            if n_wrapped:
                print(f"px ({col}, {row}) VH = {vh} mV: {n_wrapped} hits above {TOT_MAX} µs left out")
            runs.append((vh, int(hitmap[row, col]), tots[tots <= TOT_MAX]))

        # same bins for every VH in this panel, covering this pixel's own ToT range
        all_tots = np.concatenate([tots for _, _, tots in runs])
        edges = np.arange(all_tots.min(), all_tots.max() + BIN_WIDTH, BIN_WIDTH)
        for (vh, nhits, tots), color in zip(runs, COLORS):
            ax.hist(tots, edges, histtype='stepfilled', facecolor=to_rgba(color, 0.25),
                    edgecolor=color, linewidth=1.5, label=f"VH = {vh} mV: {nhits} hits")
        ax.set_title(f"Pixel ({col}, {row}), {kind}")
        ax.set_xlabel("ToT [µs]")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)

    fig.suptitle(f"ToT of weak pixels (fewer than 9500 hits at VH = 1000 mV) and two normal pixels, Vcasb = {VCASB} mV")
    fig.tight_layout()
    output_file = OUTPUT_DIR / f"weak_pixels_tot_vcasb_{VCASB}.png"
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    plot_weak_pixels()
