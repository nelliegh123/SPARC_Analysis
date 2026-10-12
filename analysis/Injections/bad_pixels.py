import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle

from injection_scan import fit_scan, fit_dataset, PLOT_DIR, NROWS, NCOLS, EXTRA_DATASETS, DATA_VCASB, DATA_VH

"""
Shows where the bad pixels are: pixels that have hits but whose ToT histogram can't be fitted with a Gaussian
(the same definition as in injection.py), and in another color the pixels with fewer than LOW_HITS hits.
A pixel that is both is drawn half in each color. One heatmap per setting: one figure with a row per Vcasb and a column per Vh,
and one figure per Vcasb with more Vh values, also for each of the EXTRA_DATASETS in injection_scan.py.
"""

VCASB_VALUES = [DATA_VCASB]  # the default data only has Vcasb = 300 mV
VH_PLOT = [75, 300, 600, 1200]  

VH_SINGLE = [50, 75, 100, 200, 400, 1200]
SINGLE_COLUMNS = 3
MAP_VH = 800  


VH_SCAN = DATA_VH

# For the EXTRA_DATASETS (old data, Ireset = 70 pA, ...): these Vh values plus the highest one that is complete,
# which is also the Vh of the ToT map and hitmap on the right
EXTRA_VH_SINGLE = [200, 300, 400, 500, 600]

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


#-----One Vcasb scan (Vh values vh_scan), one heatmap per Vh in vh_single in rows of SINGLE_COLUMNS,-----
#-----and on the right the mean ToT map (top) and the hitmap (bottom) at map_vh-----
def plot_bad_pixels_single(scan, vcasb, vh_scan, vh_single, map_vh, title, output_file):
    bad = find_bad_pixels(scan)
    low = scan["nhits"] < LOW_HITS
    nrows = max(2, int(np.ceil(len(vh_single) / SINGLE_COLUMNS)))
    fig, axes = plt.subplots(nrows, SINGLE_COLUMNS + 1, figsize=(4 * (SINGLE_COLUMNS + 1) + 1, 4.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for k, ax in enumerate(axes[:, :SINGLE_COLUMNS].ravel()):
        if k >= len(vh_single):
            ax.set_visible(False)
            continue
        vh = vh_single[k]
        bad_map = bad[vh_scan.index(vh)]
        low_map = low[vh_scan.index(vh)]
        draw_heatmap(ax, bad_map, low_map, f"Vh = {vh} mV")
        if k // SINGLE_COLUMNS == nrows - 1:
            ax.set_xlabel("Column")
        if k % SINGLE_COLUMNS == 0:
            ax.set_ylabel("Row")
        print_pixels(vcasb, vh, bad_map, low_map)

    # right column: mean ToT and number of hits of every pixel at map_vh, pixels without a value stay white
    i = vh_scan.index(map_vh)
    maps = [(scan["means"][i], "Mean ToT [µs]", f"Mean ToT, Vh = {map_vh} mV"),
            (np.where(scan["nhits"][i] > 0, scan["nhits"][i], np.nan), "Number of hits", f"Hitmap, Vh = {map_vh} mV")]
    for row, ax in enumerate(axes[:, -1]):
        if row >= len(maps):
            ax.set_visible(False)
            continue
        values, label, map_title = maps[row]
        im = ax.imshow(values, origin="lower", cmap="plasma")
        fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_title(map_title)
        if row == len(maps) - 1:
            ax.set_xlabel("Column")
    finish_figure(fig, title, output_file)


#-----------Main function where we call everything-----------
def main():
    bad, low, scans = {}, {}, {}
    for vcasb in VCASB_VALUES:
        scans[vcasb] = fit_scan([(vcasb, vh) for vh in VH_SCAN], f"vcasb_{vcasb}_vh_scan")
        bad[vcasb] = find_bad_pixels(scans[vcasb])
        low[vcasb] = scans[vcasb]["nhits"] < LOW_HITS
    plot_bad_pixels(bad, low)
    for vcasb in VCASB_VALUES:
        plot_bad_pixels_single(scans[vcasb], vcasb, VH_SCAN, VH_SINGLE, MAP_VH, f"Bad pixels, Vcasb = {vcasb} mV",
                               OUTPUT_DIR / f"bad_pixels_vcasb_{vcasb}.png")

    # the other data sets, each cached under its own name so they don't mix with the standard scans
    for name, d in EXTRA_DATASETS.items():
        d["plot_dir"].mkdir(parents=True, exist_ok=True)
        vh_max = d["vh_scan"][-1]
        plot_bad_pixels_single(fit_dataset(name), d["vcasb"], d["vh_scan"], EXTRA_VH_SINGLE + [vh_max], vh_max,
                               f"Bad pixels, {d['label']}, Vcasb = {d['vcasb']} mV",
                               d["plot_dir"] / f"bad_pixels_{d['tag']}_vcasb_{d['vcasb']}.png")


if __name__ == "__main__":
    main()
