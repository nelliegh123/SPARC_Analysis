import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from injection_scan import fit_scan, fit_dataset, PLOT_DIR, N_INJECTIONS, DATA_VH, EXTRA_DATASETS

"""
Counts hitmap (top row) and ToT hitmap (bottom row) for one Vcasb at a few Vh values.
"""

VCASB = 300
HITS_TOLERANCE = 250  # a pixel with 10000 +- HITS_TOLERANCE hits counts as having all its hits

HIT_CLASSES = ["No hits", f"Fewer than {N_INJECTIONS - HITS_TOLERANCE}",
               f"{N_INJECTIONS} $\\pm$ {HITS_TOLERANCE}", f"More than {N_INJECTIONS + HITS_TOLERANCE}"]
HIT_COLORS = ["white", "#7b2cbf", "#f6d32d", "black"]

TOT_EDGES = [1.0, 2.0, 7.0, 10.0]
TOT_COLORS = ["white", "#3c096c", "#9d4edd", "#f6d32d", "#f77f00", "black"]
TOT_SHORT = [f"< {TOT_EDGES[0]:g}"] + [f"{lo:g}-{hi:g}" for lo, hi in zip(TOT_EDGES, TOT_EDGES[1:])] + [f"> {TOT_EDGES[-1]:g}"]
TOT_CLASSES = ["No fit"] + [f"{band} µs" for band in TOT_SHORT]

VH_VALUES_WANTED = [100, 150, 200, 300]

DATASET = sys.argv[1] if len(sys.argv) > 1 else "standard"
if DATASET == "standard":
    VH_SCAN = DATA_VH
    VH_VALUES = [vh for vh in VH_VALUES_WANTED if vh in VH_SCAN] or VH_SCAN[-4:]
    LABEL = ""
    FILE_TAG = ""
    OUTPUT_DIR = PLOT_DIR / "tot_hitmap"
elif DATASET in EXTRA_DATASETS:
    d = EXTRA_DATASETS[DATASET]
    VCASB = d["vcasb"]
    VH_SCAN = d["vh_scan"]
    VH_VALUES = [vh for vh in VH_VALUES_WANTED if vh in VH_SCAN] or VH_SCAN[-4:]
    LABEL = f"{d['label']}, "
    FILE_TAG = f"{d['tag']}_"
    OUTPUT_DIR = d["plot_dir"]
else:
    raise ValueError(f'Unknown data set "{DATASET}", use "standard" or one of {list(EXTRA_DATASETS)}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#-----------Main function where we call everything-----------
def main():
    scan = fit_scan([(VCASB, vh) for vh in VH_SCAN], f"vcasb_{VCASB}_vh_scan") if DATASET == "standard" else fit_dataset(DATASET)
    nhits = np.array([scan["nhits"][VH_SCAN.index(vh)] for vh in VH_VALUES])
    means = np.array([scan["means"][VH_SCAN.index(vh)] for vh in VH_VALUES])

    hit_class = np.select([~(nhits > 0), nhits < N_INJECTIONS - HITS_TOLERANCE, nhits <= N_INJECTIONS + HITS_TOLERANCE],
                          [0, 1, 2], default=3)
   
    tot_class = np.where(np.isnan(means), 0, np.digitize(means, TOT_EDGES) + 1)
    if len(TOT_COLORS) != len(TOT_CLASSES):
        raise ValueError(f"TOT_COLORS needs {len(TOT_CLASSES)} colours, one per ToT band plus white for no fit")
    rows = [("Hitmap", "Hits", hit_class, HIT_CLASSES, HIT_COLORS, ["fewer", "all", "more"]),
            ("Mean ToT", "Mean ToT", tot_class, TOT_CLASSES, TOT_COLORS, [f"({band})" for band in TOT_SHORT])]

    fig, axes = plt.subplots(2, len(VH_VALUES), figsize=(5 * len(VH_VALUES) + 1, 11), sharex=True, sharey=True)
    for i, (name, legend_title, classes, labels, colors, short) in enumerate(rows):
        for j, vh in enumerate(VH_VALUES):
            n_class = [np.sum(classes[j] == k) for k in range(len(labels))]
            axes[i, j].imshow(classes[j], origin="lower", cmap=ListedColormap(colors), vmin=-0.5, vmax=len(colors) - 0.5)
            counts = [f"{n} {s}" for n, s in zip(n_class[1:], short)]
           
            lines = [", ".join(counts[k:k + 3]) for k in range(0, len(counts), 3)]
            axes[i, j].set_title(f"{name}, Vh = {vh} mV\n" + "\n".join(lines))
            print(f"{name}, Vh = {vh} mV: " + ", ".join(f"{n} {label}" for n, label in zip(n_class, labels)))
        axes[i, 0].set_ylabel("Row")
       
        axes[i, -1].legend(handles=[Patch(facecolor=c, edgecolor="gray", label=label) for c, label in zip(colors, labels)],
                           title=legend_title, loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("Column")
    fig.suptitle(f"{LABEL}Vcasb = {VCASB} mV")
    output_file = OUTPUT_DIR / f"hitmap_tot_{FILE_TAG}vcasb_{VCASB}_vh_{'_'.join(map(str, VH_VALUES))}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
