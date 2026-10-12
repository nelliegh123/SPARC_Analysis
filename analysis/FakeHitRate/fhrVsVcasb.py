import re
import matplotlib.pyplot as plt

import sys
from pathlib import Path
from razorbill.cnstlnmsgpack import UnpackResult
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analyze_data_sl.py is one folder up, in analysis/
from analyze_data_sl import SPARCAnalyser

MEASUREMENT_TIME = 300.0  # seconds
N_PIXELS = 896
LOWER_FHR_LIMIT = 1 / (MEASUREMENT_TIME * N_PIXELS)  # 1 hit/pixel/300 s, floor for measurable FHR
UPPER_FHR_LIMIT = 294  
ROOT = Path(__file__).resolve().parents[2]  # Nellie/, this file is in analysis/FakeHitRate/
DATA_DIR = ROOT / "data"
SAMPLE = "sparc0"

# # My data
# SCANS = [
#     {"label": "0.0 V", "color": "tab:blue", "samples": [("sparc0_0V", [30, 40, 50, 60, 70, 80, 90, 100, 110])]},
#     {"label": "-1.2 V", "color": "tab:orange", "samples": [
#         ("sparc0_12V", [220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330]),
#         ("sparc0_12V_extra", [340])]
#     },
#     {"label": "-3.0 V", "color": "tab:green", "samples": [("sparc0_3V", [450, 475, 500, 525, 550, 575])]},
# ]

# Each sample is (folder in DATA_DIR, filename prefix), files are <prefix>_<vcasb>.raw
SCANS = [
    {"label": "0.0 V (new resistor)", "color": "tab:blue", "samples": [(f"{SAMPLE}/fhr", "data_run_0V_vcasb")]},
    {"label": "-1.2 V (new resistor)", "color": "tab:orange", "samples": [(f"{SAMPLE}/fhr", "data_run_12V_vcasb")]},
    {"label": "-3.0 V (new resistor)", "color": "tab:green", "samples": [(f"{SAMPLE}/fhr", "data_run_30V_vcasb")]},
    {"label": "0.0 V (old resistor)", "color": "tab:blue", "linestyle": ":", "marker": "^",
     "samples": [(f"vcasb/{SAMPLE}_0V", "data_run_vcasb")]},
    {"label": "-1.2 V (old resistor)", "color": "tab:orange", "linestyle": ":", "marker": "^",
     "samples": [(f"vcasb/{SAMPLE}_12V", "data_run_vcasb"), (f"vcasb/{SAMPLE}_12V_extra", "data_run_vcasb")]},
    {"label": "-3.0 V (old resistor)", "color": "tab:green", "linestyle": ":", "marker": "^",
     "samples": [(f"vcasb/{SAMPLE}_3V", "data_run_vcasb")]},
]


# # DPTS data
x_0V = [30.286493860845823, 50.750341064120036, 70.39563437926329, 90.8594815825375]
y_0V = [0.00024709112279856045, 0.0022295349551952263, 0.01477892540690517, 0.7346346838552443]
x_12V = [220.19099590723056, 260.300136425648, 300.4092769440655, 339.699863574352, 379.8090040927694]
y_12V = [0.00025746268004948707, 0.00025222359662044737, 0.0007813707376518078, 0.018529220216409483, 2.0117380422807907]
x_3V = [400.27285129604365, 450.2046384720327, 500.1364256480218, 550.068212824011, 600]
y_3V = [0.00024206308919513515, 0.00024206308919513515, 1.9306977288832496, 11.312834366320196, 19.708013774451135]
# # x_12V = [101.50068212824011, 140.7912687585266, 180.90040927694406, 220.19099590723056,
# #         260.300136425648, 300.4092769440655, 339.699863574352, 379.8090040927694]
# # y_12V = [0.00025222359662044737, 0.00025746268004948707, 0.00025746268004948707, 0.00025746268004948707,
# #          0.00025222359662044737, 0.0007813707376518078, 0.018529220216409483, 2.0117380422807907]



OUTPUT_DIR = ROOT / "plots" / "FakeHitRate" / "plots_fhr_vs_vcasb"
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

    return total_hits, n_active_pixels


def find_raw_files(sample: str, prefix: str):
    """Return [(vcasb, raw_file)] sorted by VCASB for files named <prefix>_<vcasb>.raw in DATA_DIR/<sample>."""
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.raw")
    files = []
    for raw_file in (DATA_DIR / sample).glob(f"{prefix}_*.raw"):
        match = pattern.fullmatch(raw_file.name)
        if match:
            files.append((int(match.group(1)), raw_file))
    return sorted(files)


def compute_fhr_curve(samples: list[tuple[str, str]]):
    """Return (successful_vcasb, mean_fhr) lists combined across all (sample, prefix) pairs in the scan."""
    successful_vcasb = []
    mean_fhr = []

    for sample, prefix in samples:
        print()
        print("=" * 70)
        print(f"PROCESSING {sample}/{prefix}_*.raw")
        print("=" * 70)

        raw_files = find_raw_files(sample, prefix)
        if not raw_files:
            print(f"\nWARNING: no files found matching {DATA_DIR / sample}/{prefix}_*.raw")

        for vcasb, raw_file in raw_files:
            print(f"\n{'-' * 60}\nFile: {raw_file}\nVCASB: {vcasb} mV")

            try:
                result = get_total_hits(raw_file)
            except Exception as error:
                print(f"\nERROR while processing:\nFile = {raw_file}\nVCASB = {vcasb} mV")
                print(f"Error type: {type(error).__name__}")
                print(f"Error message: {error}")
                continue

            if result is None:
                continue

            total_hits, n_active_pixels = result
            fhr = total_hits / (n_active_pixels * MEASUREMENT_TIME)
            if fhr < LOWER_FHR_LIMIT:
                print(f"Mean FHR {fhr:.6e} below lower limit, clamping to {LOWER_FHR_LIMIT:.6e} hits/pixel/s")
                fhr = LOWER_FHR_LIMIT
            print(f"Mean FHR: {fhr:.6e} hits/pixel/s")
            successful_vcasb.append(vcasb)
            mean_fhr.append(fhr)

    if len(successful_vcasb) == 0:
        print(f"\nNo valid measurements for {[sample for sample, _ in samples]}")
    return successful_vcasb, mean_fhr


def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    # # DPTS reference curves
    for i, (x_dpts, y_dpts) in enumerate([(x_0V, y_0V), (x_12V, y_12V), (x_3V, y_3V)]):
        ax.plot(x_dpts, y_dpts, marker="s", linestyle='--', color="grey", alpha=0.7, markersize=6, linewidth=2,
                label="DPTS" if i == 0 else None)

    # My measured curves
    for scan in SCANS:
        vcasb, fhr = compute_fhr_curve(scan["samples"])
        if len(vcasb) == 0:
            continue
        order = sorted(range(len(vcasb)), key=lambda i: vcasb[i])
        vcasb = [vcasb[i] for i in order]
        fhr = [fhr[i] for i in order]
        ax.plot(vcasb, fhr, marker=scan.get("marker", "o"), linestyle=scan.get("linestyle", "-"),
                color=scan["color"], markersize=6, linewidth=2, label=scan["label"])

    # ax.hlines(y=LOWER_FHR_LIMIT, xmin=0, xmax=600, colors="grey", linestyles="dashed", linewidth=1.5, label="Lower limit, 1 hit/pixel/300 s")
    # ax.hlines(y=UPPER_FHR_LIMIT, xmin=0, xmax=600, colors="grey", linestyles="dashed", linewidth=1.5, label="Upper limit, 294 hits/pixel/300 s")
    ax.set_yscale("log")
    ax.set_xlabel("VCASB (mV)", fontsize=13)
    ax.set_ylabel(r"Fake-hit rate (hits pixel$^{-1}$ s$^{-1}$)", fontsize=13)
    ax.set_title(SAMPLE.capitalize(), fontsize=15)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=10, handlelength=3)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f"FHR_vs_VCASB_{SAMPLE}.png"
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Plot saved:\n{output_file.resolve()}")

    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")






if __name__ == "__main__":
    main()
