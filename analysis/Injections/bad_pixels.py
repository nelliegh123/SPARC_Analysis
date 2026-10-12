import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle

from injection_scan import fit_scan, PLOT_DIR, NROWS, NCOLS, DATA_VCASB, DATA_VH

"""
Shows where the bad pixels are: pixels that have hits but whose ToT histogram can't be fitted with a Gaussian
and in another color the pixels with fewer than LOW_HITS hits.
A pixel that is both is drawn half in each color. One figure with one heatmap per setting: a row per Vcasb
and a column per Vh in VH_PLOT.
"""

VCASB_VALUES = [DATA_VCASB]  # the default data only has Vcasb = 300 mV
VH_PLOT = [75, 300, 600, 1200]  



VH_SCAN = DATA_VH


LOW_HITS = 50 

BAD_COLOR = "#e34948"
LOW_HITS_COLOR = "#2a78d6"

OUTPUT_DIR = PLOT_DIR / "bad_pixels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#-----Bad pixels: hits but no Gaussian fit, indexed [vh, row, col]-----
def find_bad_pixels(scan):
    return np.isnan(scan["means"]) & (scan["nhits"] > 0)


#-----Colors one pixel, or its left half in color and right half in second_color-----
def draw_pixel(ax, col, row, color, second_color=None):
    if second_color is None:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, color=color, lw=0))
    else:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 0.5, 1, color=color, lw=0))
        ax.add_patch(Rectangle((col, row - 0.5), 0.5, 1, color=second_color, lw=0))


#-----One heatmap with only the bad and low-hit pixels coloured-----
def draw_heatmap(ax, bad_map, low_map, title):
    for row, col in np.argwhere(bad_map | low_map):
        if bad_map[row, col] and low_map[row, col]:
            draw_pixel(ax, col, row, BAD_COLOR, LOW_HITS_COLOR)
        else:
            draw_pixel(ax, col, row, BAD_COLOR if bad_map[row, col] else LOW_HITS_COLOR)
    ax.set_xlim(-0.5, NCOLS - 0.5)
    ax.set_ylim(-0.5, NROWS - 0.5)
    ax.set_aspect("equal")
    # thin lines between the pixels, so the matrix is visible
    ax.set_xticks(np.arange(-0.5, NCOLS), minor=True)
    ax.set_yticks(np.arange(-0.5, NROWS), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"{title}\n{bad_map.sum()} bad fits, {low_map.sum()} with < {LOW_HITS} hits")


def print_pixels(vcasb, vh, bad_map, low_map):
    print(f"Vcasb = {vcasb} mV, Vh = {vh} mV: {bad_map.sum()} bad pixels "
          f"{[(int(c), int(r)) for r, c in np.argwhere(bad_map)]}")
    print(f"    {low_map.sum()} pixels with < {LOW_HITS} hits {[(int(c), int(r)) for r, c in np.argwhere(low_map)]}")


#-----Title, legend and saving, the same for every figure-----
def finish_figure(fig, title, output_file):
    fig.suptitle(title)
    fig.legend(handles=[Patch(color=BAD_COLOR), Patch(color=LOW_HITS_COLOR),
                        (Patch(color=BAD_COLOR), Patch(color=LOW_HITS_COLOR))],
               labels=["Hits, but no Gaussian fit of the ToT", f"Fewer than {LOW_HITS} hits", "Both"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
               loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved {output_file}")


#-----Both Vcasb values: one row per Vcasb, one column per Vh in VH_PLOT-----
def plot_bad_pixels(bad, low):
    fig, axes = plt.subplots(len(VCASB_VALUES), len(VH_PLOT), figsize=(4 * len(VH_PLOT), 4.6 * len(VCASB_VALUES)),
                             sharex=True, sharey=True, squeeze=False)
    for i, vcasb in enumerate(VCASB_VALUES):
        for j, vh in enumerate(VH_PLOT):
            ax = axes[i, j]
            bad_map = bad[vcasb][VH_SCAN.index(vh)]
            low_map = low[vcasb][VH_SCAN.index(vh)]
            draw_heatmap(ax, bad_map, low_map, f"Vh = {vh} mV")
            if i == len(VCASB_VALUES) - 1:
                ax.set_xlabel("Column")
            if j == 0:
                ax.set_ylabel("Row")
                # Vcasb of this row of heatmaps, on the left
                ax.annotate(f"Vcasb = {vcasb} mV", xy=(0, 0.5), xycoords="axes fraction", xytext=(-50, 0),
                            textcoords="offset points", rotation=90, ha="right", va="center",
                            fontsize=14, fontweight="bold")
            print_pixels(vcasb, vh, bad_map, low_map)
    finish_figure(fig, "Bad pixels", OUTPUT_DIR / f"bad_pixels_overview_vcasb_{'_'.join(map(str, VCASB_VALUES))}.png")


#-----------Main function where we call everything-----------
def main():
    bad, low = {}, {}
    for vcasb in VCASB_VALUES:
        scan = fit_scan([(vcasb, vh) for vh in VH_SCAN], f"vcasb_{vcasb}_vh_scan")
        bad[vcasb] = find_bad_pixels(scan)
        low[vcasb] = scan["nhits"] < LOW_HITS
    plot_bad_pixels(bad, low)


if __name__ == "__main__":
    main()
