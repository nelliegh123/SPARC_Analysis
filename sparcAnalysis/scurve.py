import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from sparc_ana_tools import fit_gaussian
from sparc_cache import cache
from sparc_common import NCOLS, NROWS, ChipParams, get_run_id_inj
from sparc_scripts.double_pulsing_analyzer import DoublePulsingAnalyser

DATA_DIR = Path(sys.argv[1])
N_INJECTIONS = 10000
COLS = list(range(NCOLS))
ROWS = list(range(NROWS))
RUNS: list[ChipParams] = [
    ChipParams(360, 70, 400, 700, 300, vh) for vh in range(30, 150, 10)
]


@cache
def do_ana(
    data_dir: Path, run_idx: int, cp: ChipParams, row: int, col: int
) -> tuple[int, int, int, int]:
    run_id = get_run_id_inj(cp, col, row)
    dpana = DoublePulsingAnalyser(data_dir.joinpath(f"data_{run_id}.raw"))
    dpana.do_output = False
    dpana.analyze()
    return run_idx, row, col, min(N_INJECTIONS, len(dpana.events[(row, col)]))


if __name__ == "__main__":
    vh = [cp.vh for cp in RUNS]
    n_injection_events = np.full(
        (len(RUNS), len(ROWS), len(COLS)), np.nan, dtype=np.float64
    )

    pixels = [
        (DATA_DIR, run_idx, cp, row, col)
        for run_idx, cp in enumerate(RUNS)
        for row in ROWS
        for col in COLS
    ]
    for (
        run_idx,
        row,
        col,
        pixel_n_injections,
    ) in process_map(do_ana, *zip(*pixels), chunksize=1, max_workers=15):
        row_idx = ROWS.index(row)
        col_idx = COLS.index(col)
        n_injection_events[run_idx, row_idx, col_idx] = pixel_n_injections

    # Calculate injection efficiency
    injection_efficiency = n_injection_events / N_INJECTIONS
    mean_injection_efficiency = np.nanmean(injection_efficiency, axis=(1, 2))

    # Interpolate the Vh at which the mean efficiency crosses 50%
    mean_vh_threshold = np.interp(0.5, mean_injection_efficiency, vh)
    print(f"Mean Vh treshold = {mean_vh_threshold:.1f} mV")

    plt.figure("S-Curve")
    plt.xlabel("Vh [mV]")
    plt.ylabel("Injection Efficiency")
    for row_idx in range(len(ROWS)):
        for col_idx in range(len(COLS)):
            plt.plot(
                vh,
                injection_efficiency[:, row_idx, col_idx],
                color="tab:blue",
                linewidth=0.3,
                alpha=0.3,
                marker=None,
            )
    plt.plot(
        vh,
        mean_injection_efficiency,
        color="black",
        linewidth=2.5,
        marker=None,
    )
    plt.axhline(0.5, color="red", linestyle=":", linewidth=1.0)
    plt.axvline(mean_vh_threshold, color="red", linestyle=":", linewidth=1.0)

    # Threshold dispersion: interpolate the 50% crossing for every pixel
    vh_threshold = np.full((len(ROWS), len(COLS)), np.nan, dtype=np.float64)
    for row_idx in range(len(ROWS)):
        for col_idx in range(len(COLS)):
            pixel_injection_efficiency = injection_efficiency[:, row_idx, col_idx]
            vh_threshold[row_idx, col_idx] = np.interp(
                0.5, pixel_injection_efficiency, vh
            )
    vh_threshold_flat = vh_threshold.flatten()
    vh_threshold_mean, vh_threshold_std, vh_threshold_ampl = fit_gaussian(
        vh_threshold_flat
    )
    print(
        f"Threshold dispersion: mean = {vh_threshold_mean:.1f} mV, std = {vh_threshold_std:.1f} mV"
    )

    plt.figure("Threshold Map")
    plt.xlabel("Col")
    plt.ylabel("Row")
    plt.imshow(vh_threshold, interpolation="none", origin="lower")
    plt.colorbar(label="Vh threshold [mV]")

    plt.figure("Threshold Dispersion")
    plt.hist(vh_threshold_flat)
    plt.xlabel("Vh threshold [mV]")
    plt.ylabel("Number of pixels")

    plt.show()
