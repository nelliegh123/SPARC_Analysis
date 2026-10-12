import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

import sys
from pathlib import Path
from razorbill.cnstlnmsgpack import UnpackResult
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analyze_data_sl.py is one folder up, in analysis/
from analyze_data_sl import SPARCAnalyser

# MASKED_PIXELS = 4

MEASUREMENT_TIME = 300.0  # seconds
N_PIXELS = 896
LOWER_FHR_LIMIT = 1 / (MEASUREMENT_TIME * N_PIXELS)  # 1 hit/pixel/300 s, floor for measurable FHR
UPPER_FHR_LIMIT = 294  
ROOT = Path(__file__).resolve().parents[2]  # Nellie/, this file is in analysis/FakeHitRate/
DATA_DIR = ROOT / "data"

# #---sparc0----
# SCANS = [
#     {"label": "0.0 V", "color": "tab:blue", "samples": [("sparc0_0V", [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])]},
#     {"label": "-1.2 V", "color": "tab:orange", "samples": [
#         ("sparc0_12V", [220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330]),
#         ("sparc0_12V_extra", [340])]
#     },
#     {"label": "-3.0 V", "color": "tab:green", "samples": [("sparc0_3V", [450, 475, 500, 525, 550, 575])]},
# ]

# #----sparc4---
# SCANS = [
#     {"label": "0.0 V", "color": "tab:blue", "samples": [("sparc4_0V", [0, 40, 80, 120])]},
#     {"label": "-1.2 V", "color": "tab:orange", "samples": [("sparc4_12V", [225, 250, 275, 300, 325, 350])]},
#     # {"label": "-3.0 V", "color": "tab:green", "samples": [("sparc4_3V", [450, 475, 500, 525, 550, 575])]},
# ]

#----sparc6---
SCANS = [
    {"label": "0.0 V", "color": "tab:blue", "samples": [("sparc6_0V", [50, 75, 100, 125])]},
    {"label": "-1.2 V", "color": "tab:orange", "samples": [("sparc6_12V", [225, 250, 275, 300, 325, 350])]},
    # {"label": "-3.0 V", "color": "tab:green", "samples": [("sparc4_3V", [450, 475, 500, 525, 550, 575])]},
]



# # DPTS data
x_0V = [30.286493860845823, 50.750341064120036, 70.39563437926329, 90.8594815825375]
y_0V = [0.00024709112279856045, 0.0022295349551952263, 0.01477892540690517, 0.7346346838552443]
x_12V = [220.19099590723056, 260.300136425648, 300.4092769440655, 339.699863574352, 379.8090040927694]
y_12V = [0.00025746268004948707, 0.00025222359662044737, 0.0007813707376518078, 0.018529220216409483, 2.0117380422807907]
x_3V = [400.27285129604365, 450.2046384720327, 500.1364256480218, 550.068212824011, 600]
y_3V = [0.00024206308919513515, 0.00024206308919513515, 1.9306977288832496, 11.312834366320196, 19.708013774451135]


OUTPUT_DIR = ROOT / "plots" / "FakeHitRate" / "try_masked"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_total_hits(raw_file: Path):
    analyzer = SPARCAnalyser(raw_file)
    result = analyzer.unpack_all()
    print(f"Unpack result: {result.name}")
    if result == UnpackResult.FAILED:
        print("ERROR: unpack failed.")
        return None

    hits_active = analyzer.hitmap.flatten()[analyzer.active_pixels.flatten()]
    n_active_pixels = len(hits_active)
    if n_active_pixels == 0:
        print("ERROR: No active pixels found.")
        return None

    total_hits = int(hits_active.sum())
    print(f"Active pixels: {n_active_pixels}")
    print(f"Total hits:    {total_hits}")
    print(f"Missed hits:   {analyzer.missed_hits}")

    hitmap_active = np.where(analyzer.active_pixels, analyzer.hitmap, 0)  # inactive pixels set to 0

    return total_hits, n_active_pixels, hitmap_active


def clamp_fhr(fhr: float) -> float:
    if fhr < LOWER_FHR_LIMIT:
        print(f"Mean FHR {fhr:.6e} below lower limit, clamping to {LOWER_FHR_LIMIT:.6e} hits/pixel/s")
        return LOWER_FHR_LIMIT
    return fhr


def compute_fhr_curves(samples: list[tuple[str, list[int]]], mask_counts: list[int]):
    """Unpack each raw file once and compute the mean FHR for every number of masked pixels.

    Returns {masked_pixels: (vcasb_list, fhr_list)}.
    """
    curves = {n: ([], []) for n in mask_counts}

    for sample, voltages in samples:
        print()
        print("=" * 70)
        print(f"PROCESSING {sample}")
        print("=" * 70)

        for vcasb in voltages:
            raw_file = DATA_DIR / sample / f"data_run_vcasb_{vcasb}.raw"
            if not raw_file.exists():
                print(f"\nWARNING: file not found: {raw_file}")
                continue

            print(f"\n{'-' * 60}\nSample: {sample}\nVCASB: {vcasb} mV")

            try:
                result = get_total_hits(raw_file)
            except Exception as error:
                print(f"\nERROR while processing:\nSample = {sample}\nVCASB = {vcasb} mV")
                print(f"Error type: {type(error).__name__}")
                print(f"Error message: {error}")
                continue

            if result is None:
                continue

            total_hits, n_active_pixels, hitmap_active = result

            # Sort once (noisiest first); cumulative sum gives the hits removed by masking the N noisiest pixels
            order = np.argsort(hitmap_active, axis=None)[::-1][:max(mask_counts)]
            sorted_hits = hitmap_active.flatten()[order]
            masked_hits_cum = np.concatenate(([0], np.cumsum(sorted_hits)))

            for row, col in zip(*np.unravel_index(order, hitmap_active.shape)):
                print(f"Noisy pixel (row, col) = ({row}, {col}) with {int(hitmap_active[row, col])} hits")

            for n in mask_counts:
                fhr = (total_hits - masked_hits_cum[n]) / ((n_active_pixels - n) * MEASUREMENT_TIME)
                fhr = clamp_fhr(fhr)
                print(f"Mean FHR with {n} masked pixels: {fhr:.4e} hits/pixel/s")
                curves[n][0].append(vcasb)
                curves[n][1].append(fhr)

    if all(len(vcasb) == 0 for vcasb, _ in curves.values()):
        print(f"\nNo valid measurements for {[sample for sample, _ in samples]}")
    return curves


def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    # # DPTS reference curves
    ax.plot(x_0V, y_0V, marker="s", color='grey', linestyle='--', alpha=0.7, markersize=6, linewidth=2)
    ax.plot(x_12V, y_12V, marker="s", color='grey', linestyle='--', alpha=0.7, markersize=6, linewidth=2)
    ax.plot(x_3V, y_3V, marker="s", color='grey', linestyle='--', alpha=0.7, markersize=6, linewidth=2, label="DPTS data")


    mask_counts = [0, 1, 2, 3, 4]
    alphas = np.linspace(0.4, 1.0, len(mask_counts))
    markers = ["o", "^", "s", "*", "D"]  # one marker shape per number of masked pixels
    marker_sizes = {"*": 9}  # the star looks small at the default size
    for scan in SCANS:
        curves = compute_fhr_curves(scan["samples"], mask_counts)
        for alpha, marker, n in zip(alphas, markers, mask_counts):
            vcasb, fhr = curves[n]
            if len(vcasb) == 0:
                continue
            ax.plot(vcasb, fhr, marker=marker, color=scan["color"], alpha=alpha,
                    markersize=marker_sizes.get(marker, 6), linewidth=2)
        # legend entry per scan: colour only, since the marker varies with the number of masked pixels
        ax.plot([], [], color=scan["color"], linewidth=2, label=scan["label"])

    # all marker shapes side by side in a single legend entry
    marker_handles = tuple(ax.plot([], [], marker=marker, color="black", linestyle="none",
                                   markersize=marker_sizes.get(marker, 6))[0] for marker in markers)
    marker_label = f"{', '.join(str(n) for n in mask_counts)} masked pixels"


    # ax.hlines(y=LOWER_FHR_LIMIT, xmin=0, xmax=600, colors="grey", linestyles="dashed", linewidth=1.5, label="Lower limit, 1 hit/pixel/300 s")
    # ax.hlines(y=UPPER_FHR_LIMIT, xmin=0, xmax=600, colors="grey", linestyles="dashed", linewidth=1.5, label="Upper limit, 294 hits/pixel/300 s")
    ax.set_yscale("log")
    ax.set_xlabel("VCASB (mV)", fontsize=13)
    ax.set_ylabel(r"Fake-hit rate (hits pixel$^{-1}$ s$^{-1}$)", fontsize=13)
    ax.set_title("Sparc6", fontsize=15)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [marker_handles], labels + [marker_label], loc="upper left", fontsize=10, handlelength=6,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    ax.set_xlim(-5, 395)



    plt.tight_layout()
    output_file = OUTPUT_DIR / "FHR_vs_VCASB_sparc6_masked.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    print(f"Plot saved:\n{output_file.resolve()}")

    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")






if __name__ == "__main__":
    main()
