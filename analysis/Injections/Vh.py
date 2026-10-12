import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
from razorbill.cnstlnmsgpack import UnpackResult
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analyze_data_sl.py is one folder up, in analysis/
from analyze_data_sl import SPARCAnalyser

ROOT = Path(__file__).resolve().parents[2]  # Nellie/, this file is in analysis/Injections/
DATA_DIR = ROOT / "data" / "Vh"
SAMPLE = "sparc0/ToT"
# VH_VALUES = [90, 100, 110, 120, 130, 140, 150] # mV
VH_VALUES = [95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155] # mV

OUTPUT_DIR = ROOT / "plots" / "Injections" / "s_curve"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_hitmap(raw_file: Path):
    analyzer = SPARCAnalyser(raw_file)
    result = analyzer.unpack_all()
    print(f"Unpack result: {result.name}")
    if result == UnpackResult.FAILED:
        print("ERROR: unpack failed.")
        return None

    n_active_pixels = int(analyzer.active_pixels.sum())
    print(f"Active pixels: {n_active_pixels}")
    print(f"Total hits:    {int(analyzer.hitmap[analyzer.active_pixels].sum())}")
    print(f"Missed hits:   {analyzer.missed_hits}")

    return analyzer.hitmap, analyzer.active_pixels


def main():
    successful_vh = []
    hitmaps = []
    active_pixels = None

    for vh in VH_VALUES:
        raw_file = DATA_DIR / SAMPLE / f"data_run_vh_{vh}.raw"
        if not raw_file.exists():
            print(f"\nWARNING: file not found: {raw_file}")
            continue

        print(f"\n{'-' * 60}\nSample: {SAMPLE}\nVh: {vh} mV")
        result = get_hitmap(raw_file)
        if result is None:
            continue

        hitmap, active = result
        successful_vh.append(vh)
        hitmaps.append(hitmap)
        # only keep pixels that are active in every run
        active_pixels = active if active_pixels is None else active_pixels & active

    if len(successful_vh) == 0:
        print("\nNo valid measurements")
        return

    hitmaps = np.array(hitmaps)  # shape (n_vh, NROWS, NCOLS)
    counts = hitmaps[:, active_pixels]  # shape (n_vh, n_active_pixels), one column per pixel

    fig, ax = plt.subplots(figsize=(10, 7))

    # one line per pixel
    ax.plot(successful_vh, counts, color="tab:blue", alpha=0.1, linewidth=1)
    ax.plot([], [], color="tab:blue", alpha=0.3, linewidth=1, label=f"Single pixel ({counts.shape[1]} pixels)")

    # mean over all pixels
    ax.plot(successful_vh, counts.mean(axis=1), marker="o", color="black", markersize=6, linewidth=2,
            label="Mean over pixels")

    ax.set_xlabel("Vh (mV)", fontsize=13)
    ax.set_ylabel("Counts per pixel", fontsize=13)
    ax.set_title(SAMPLE.capitalize(), fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax.hlines(25, 95, 155, color='black', linestyle='dashed')

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"counts_vs_Vh_{SAMPLE.replace('/', '_')}.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    print(f"Plot saved:\n{output_file.resolve()}")
    plt.show()
    plt.close(fig)

    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")


if __name__ == "__main__":
    main()
