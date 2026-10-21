import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from double_pulsing_analyzer import DoublePulsingAnalyser

DATA_DIR = Path(sys.argv[1])
VCASB = 300
VH = int(sys.argv[2]) if len(sys.argv) > 2 else 600


def do_ana(data_dir: Path, vcasb: int, vh: int, row: int, col: int) -> tuple[int, int, int, float]:
    print(f'({row:02d}, {col:02d}): starting analysis')
    dpana = DoublePulsingAnalyser(data_dir.joinpath(f'data_run_vcasb_{vcasb}_vh_{vh}_px_{col}_{row}.raw'))
    dpana.config.add("double_pulsing_limit", 750)
    dpana.do_output = False
    dpana.analyze()
    print(f'({row:02d}, {col:02d}): analysis done')
    return row, col, len(dpana.events[(row, col)]), dpana.avg_n_double_pulses[row, col]


if __name__ == '__main__':
    n_events = np.full((DoublePulsingAnalyser.NROWS, DoublePulsingAnalyser.NCOLS), False, dtype=np.uint32)
    avg_n_double_pulses = np.zeros((DoublePulsingAnalyser.NROWS, DoublePulsingAnalyser.NCOLS), dtype=np.float64)

    pixels = [(DATA_DIR, VCASB, VH, row, col) for row in range(DoublePulsingAnalyser.NROWS) for col in range(DoublePulsingAnalyser.NCOLS)]
    with ProcessPoolExecutor(max_workers=15) as executor:
        for row, col, pixel_n_events, pixel_avg_n_double_pulses in executor.map(do_ana, *zip(*pixels)):
            n_events[row, col] = pixel_n_events
            avg_n_double_pulses[row, col] = pixel_avg_n_double_pulses

    plt.figure('Injection Events')
    plt.xlabel('Col')
    plt.ylabel('Row')
    plt.imshow(np.ma.masked_less(n_events, 1), interpolation='none', origin='lower')
    plt.colorbar(label='Event count')

    plt.figure('Number of Double Pulses')
    plt.xlabel('Col')
    plt.ylabel('Row')
    plt.imshow(avg_n_double_pulses, interpolation='none', origin='lower', vmin=0)
    plt.colorbar(label='Average number of double pulses')

    # Column projection
    col_avg_n_double_pulses: list[float] = []
    for col in range(DoublePulsingAnalyser.NCOLS):
        avg_n_double_pulses_sel = avg_n_double_pulses[:, col]
        col_avg_n_double_pulses.append(np.nanmean(avg_n_double_pulses_sel))

    # Arbiter projection
    arbiter_borders = [-0.5, 7.5, 15.5, 21.5, 27.5]
    arbiter_avg_n_double_pulses: list[float] = []
    for start, end in pairwise(arbiter_borders):
        # Take the average of every second (i.e. even) column for each arbiter
        col_avg_n_double_pulses_sel = np.array(col_avg_n_double_pulses)[int(start+0.5):int(end+0.5):2]
        arbiter_avg_n_double_pulses.append(np.nanmean(col_avg_n_double_pulses_sel))

    plt.figure('Double Pulses Column Projection')
    plt.xlabel('Column')
    plt.ylabel('Average number of double pulses')
    plt.bar(range(DoublePulsingAnalyser.NCOLS), col_avg_n_double_pulses)
    for x in arbiter_borders:
        plt.axvline(x, color='purple', linewidth=1, linestyle='--')

    plt.figure('Double Pulses Arbiter Projection')
    plt.xlabel('Arbiter (odd cols only)')
    plt.ylabel('Average number of double pulses')
    plt.bar(['2:1', '4:1', '16:1 + 4:1', '64:1'], arbiter_avg_n_double_pulses)

    # # Row projection
    # arbiters_row_avg_n_double_pulses: list[list[float]] = []
    # for start, end in pairwise(arbiter_borders):
    #     row_avg_n_double_pulses: list[float] = []
    #     for row in range(DoublePulsingAnalyser.NROWS):
    #         row_avg_n_double_pulses_sel = avg_n_double_pulses[row, int(start+0.5):int(end+0.5):2]
    #         row_avg_n_double_pulses.append(np.nanmean(row_avg_n_double_pulses_sel))
    #     arbiters_row_avg_n_double_pulses.append(row_avg_n_double_pulses)

    # for arbiter, row_avg_n_double_pulses in zip(['2:1', '4:1', '16:1 + 4:1', '64:1'], arbiters_row_avg_n_double_pulses):
    #     plt.figure(f'Double Pulses Arbiter {arbiter} Row Projection')
    #     plt.xlabel('Row')
    #     plt.ylabel('Average number of double pulses')
    #     plt.bar(range(DoublePulsingAnalyser.NROWS), row_avg_n_double_pulses)

    plt.show()