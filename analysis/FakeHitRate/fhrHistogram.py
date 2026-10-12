import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path
from razorbill.cnstlnmsgpack import UnpackResult
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analyze_data_sl.py is one folder up, in analysis/
from analyze_data_sl import SPARCAnalyser

"""
This code visiualizes noisy pixels. It plots a histogram of fake-hit-rate and counts for all voltages, and
a stacked histogram all together. THe plots are put in plots/FakeHitRate/Histograms. 
"""


MEASUREMENT_TIME = 300.0  # seconds
ROOT = Path(__file__).resolve().parents[2]  # Nellie/, this file is in analysis/FakeHitRate/
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "plots" / "FakeHitRate" / "Histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SAMPLE = "sparc0_12V"
# VOLTAGES = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330]

SAMPLE = "sparc0_12V"
VOLTAGES = [270, 280, 290, 300, 310, 320, 330]

def get_hitmap(raw_file: Path):
    analyzer = SPARCAnalyser(raw_file)
    result = analyzer.unpack_all()
    if result == UnpackResult.FAILED:
        print(f"ERROR: unpack failed for {raw_file}")
        return None
    return analyzer.hitmap




#-----------------Plots 3D lego histogram------------------------
def plot_lego(hitmap: np.ndarray, vcasb: int):
    fhr = hitmap / MEASUREMENT_TIME
    nrows, ncols = fhr.shape

    col, row = np.meshgrid(np.arange(ncols), np.arange(nrows))
    x, y = col.ravel(), row.ravel()
    dx = dy = 0.9
    top = fhr.ravel()
    z = np.zeros_like(top)
    dz = top

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    vmax = max(top.max(), 1e-9)
    colors = plt.cm.viridis(top / vmax)
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

    ax.set_zlim(0, vmax)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel(r"Fake-hit rate (hits s$^{-1}$)")
    ax.set_title(f"{SAMPLE} @ VCASB = {vcasb} mV")

    output_file = OUTPUT_DIR / f"lego_linear_{SAMPLE}_vcasb_{vcasb}.png"
    # fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")




#---------Plots linear histogram-----------
def plot_hist_linear(hitmap, vcasb):
    fig, ax = plt.subplots(figsize=(10, 7))

    for vcasb in VOLTAGES:
        raw_file = DATA_DIR / SAMPLE / f"data_run_vcasb_{vcasb}.raw"
        if not raw_file.exists():
            print(f"WARNING: file not found: {raw_file}")
            continue

        hitmap = get_hitmap(raw_file)
        if hitmap is None:
            continue

        fhr = hitmap.flatten() / MEASUREMENT_TIME
        fhr = fhr[fhr > 0]  # skip pixels with no hits
        ax.hist(fhr, bins=100, alpha=0.5, label=f"VCASB = {vcasb} mV")

    ax.set_xlabel(r"Fake-hit rate (hits s$^{-1}$)")
    ax.set_ylabel("Number of pixels")
    ax.set_title(f"{SAMPLE} - Pixel FHR Histogram (linear scale)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    #output_file = OUTPUT_DIR / f"histogram_linear_{SAMPLE}.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    plt.show()
    plt.close(fig)




#---------Plots log histogram-----------
def plot_hist_log(hitmap, vcasb):
    print("Plotting single vcasb histogram")

    fig, ax = plt.subplots(figsize=(10, 7))
    fhr = hitmap.flatten() / MEASUREMENT_TIME
    ax.hist(fhr, bins=100, alpha=0.5, label=f"VCASB = {vcasb} mV")

    ax.set_yscale("log")
    ax.set_xlabel(r"Fake-hit rate (hits s$^{-1}$)")
    ax.set_ylabel("Number of pixels")
    ax.set_title(f"{SAMPLE} - Pixel FHR Histogram (log scale)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"histogram_log_{SAMPLE}_{vcasb}.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")




#---------Plots stacked log histogram-----------
def plot_stacked_hist_log():
    print("Plotting stacked histogram")
    fhrs, labels = [], []

    for vcasb in VOLTAGES:
        print("The vcasb is: ", vcasb)
        raw_file = DATA_DIR / SAMPLE / f"data_run_vcasb_{vcasb}.raw"
        hitmap = get_hitmap(raw_file)

        fhr = hitmap.flatten() / MEASUREMENT_TIME
        fhr_floor = np.where(fhr == 0, 1 / (MEASUREMENT_TIME * fhr.size), fhr)
        fhrs.append(fhr_floor) 
        labels.append(f"VCASB = {vcasb} mV")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(fhrs, bins=50, histtype="bar", stacked=True, label=labels)

    ax.set_yscale("log")
    ax.set_xlabel(r"Fake-hit rate (hits s$^{-1}$)")
    ax.set_ylabel("Number of pixels")
    ax.set_title(f"{SAMPLE} - Pixel FHR Histogram (log scale)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"stacked_histogram_log_{SAMPLE}_{vcasb}.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")






# --------------In here I call everything i want to do----------------
def main():
    for vcasb in VOLTAGES:
        raw_file = DATA_DIR / SAMPLE / f"data_run_vcasb_{vcasb}.raw"

        print(f"\nProcessing VCASB = {vcasb} mV")
        hitmap = get_hitmap(raw_file)

        plot_hist_log(hitmap, vcasb)
    
    # hitmap = get_hitmap(raw_file)
    plot_stacked_hist_log()


if __name__ == "__main__":
    main()
