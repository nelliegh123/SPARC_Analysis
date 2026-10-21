# SPDX-FileCopyrightText: 2026 The Razorbill Caribou authors
# SPDX-License-Identifier: CC0-1.0

import sys
from itertools import chain
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from razorbill.cnstlnmsgpack import DataRecord
from razorbill.core import Decoder, Frame
from razorbill.devices.sparc import SPARCFrame, SPARCHitFrame

from sparc_analyzer import SPARCAnalyser


class DoublePulsingAnalyser(SPARCAnalyser):
    def __init__(self, file_path: Path) -> None:
        super().__init__(file_path)

        self.event_split = 100e3 # 100us
        self.injection_chargeup_split = 30e3 # 30us

        # One row per frame: row, column, timestamp (ns)
        self._frames: list[tuple[int, int, float]] = []
        self.frames = pd.DataFrame(self._frames, columns=['row', 'column', 'timestamp'])

        self._last_ts = np.full((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), -self.event_split, dtype=np.float64)
        self.ts_diffs: dict[tuple[int, int], list[float]] = {}
        self.events: dict[tuple[int, int], list[list[float]]] = {}
        self.n_double_pulses: dict[tuple[int, int], list[int]] = {}
        self.avg_n_double_pulses = np.zeros((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), dtype=np.float64)
        self.dp_leading: dict[tuple[int, int], list[list[float]]] = {}
        self.dp_falling: dict[tuple[int, int], list[list[float]]] = {}

        for row in range(SPARCAnalyser.NROWS):
            for column in range(SPARCAnalyser.NCOLS):
                self.ts_diffs[row, column] = []
                self.events[row, column] = []
                self.dp_leading[row, column] = []
                self.dp_falling[row, column] = []

    def _on_data(self, sender: str, device: str, data: DataRecord, frame: Frame, decoder: Decoder) -> None:
        super()._on_data(sender, device, data, frame, decoder)

        # Cast to SPARC frame since we know we only have SPARC data
        frame = cast(SPARCFrame, frame)

        if isinstance(frame, SPARCHitFrame):
            row = frame.row
            column = frame.column
            this_ts = self.decoder.combineTimestamps(frame.timestampFPGA, frame.timestampSPARC) * self.timestamp_bin_ns
            self._frames.append((row, column, this_ts))

            last_ts = float(self._last_ts[row, column])
            if last_ts is not None:
                self.ts_diffs[(row, column)].append(this_ts - last_ts)
            self._last_ts[row, column] = this_ts

    def analyze(self) -> None:
        super().analyze()

        tot_mode = self.config.get('tot_mode')

        # Build frame dataframe
        self.frames = pd.DataFrame(self._frames, columns=['row', 'column', 'timestamp']).astype(
            {'row': np.uint8, 'column': np.uint8, 'timestamp': np.float64}
        )

        # Build events per pixel
        for pixel, ts_diffs in self.ts_diffs.items():
            events: list[list[float]] = []
            for ts_diff in ts_diffs:
                # If timestamp difference large than event split time then new event
                if ts_diff >= self.event_split:
                    events.append([0.])
                    continue
                events[-1].append(events[-1][-1] + ts_diff)
            self.events[pixel] = events

        # Injection chargeup filter
        if self.injection_chargeup_split:
            for pixel, events in self.events.items():
                for n, event in enumerate(events):
                    for m, ts_m in enumerate(event):
                        # Discard anything from injection chargeup
                        if ts_m > self.injection_chargeup_split:
                            self.events[pixel][n] = [ts - ts_m for ts in event[m:]]
                            break

        # Count number of double pulses per pixel
        expected_sparc_hit_count = 2 if tot_mode else 1
        for pixel, events in self.events.items():
            n_double_pulses = [len(event) - expected_sparc_hit_count for event in events]
            self.n_double_pulses[pixel] = n_double_pulses
            self.avg_n_double_pulses[pixel] = (
                sum(n_double_pulses) / len(n_double_pulses) if n_double_pulses else np.nan
            )

        # Separate between double pulses on leading and falling edge
        for pixel, events in self.events.items():
            dp_leading: list[list[float]] = []
            if tot_mode:
                dp_falling: list[list[float]] = []
                for event in events:
                    if len(event) > 2:
                        # Find largest gap
                        largest_gap = 0.
                        largest_gap_n = 0
                        for n in range(len(event) - 1):
                            gap = event[n+1] - event[n]
                            if gap > largest_gap:
                                largest_gap = gap
                                largest_gap_n = n+1
                        # Split pulses
                        dp_event_leading = event[1:largest_gap_n]
                        if dp_event_leading:
                            dp_leading.append(dp_event_leading)
                        dp_event_falling = [ts - event[largest_gap_n] for ts in event[largest_gap_n+1:]]
                        if dp_event_falling:
                            dp_falling.append(dp_event_falling)
                self.dp_falling[pixel] = dp_falling
            else:
                for event in events:
                    if len(event) > 1:
                        dp_leading.append(event[1:])
            self.dp_leading[pixel] = dp_leading

        if self.do_output:
            merged_ts_diffs = list(chain.from_iterable(self.ts_diffs.values()))
            merged_events = list(chain.from_iterable(self.events.values()))
            total_events = len(merged_events)
            merged_n_double_pulses = list(chain.from_iterable(self.n_double_pulses.values()))
            total_avg_n_double_pulses = np.mean(merged_n_double_pulses) if merged_n_double_pulses else np.nan
            merged_dp_leading = list(chain.from_iterable(chain.from_iterable(self.dp_leading.values())))
            merged_dp_falling = list(chain.from_iterable(chain.from_iterable(self.dp_falling.values())))

            print(f'{total_events} events over with {total_avg_n_double_pulses:.2g} double pulses on average')

            plt.figure('Timestamp Differences [0ns - 512ns]')
            plt.xlabel('Timestamp Difference [ns]')
            plt.ylabel('Entries')
            plt.yscale('log')
            plt.hist([diff for diff in merged_ts_diffs if diff < 512], bins=128, range=(0, 512))

            plt.figure('Timestamp Differences [100ns - 100us]')
            plt.xlabel('Timestamp Difference [us]')
            plt.ylabel('Entries')
            plt.yscale('log')
            plt.hist([diff * 1e-3 for diff in merged_ts_diffs if diff > 100 and diff < 100e3], bins=1000, range=(0, 100))

            plt.figure('Double Pulsing Histogram')
            plt.xlabel('Number of double pulses')
            plt.ylabel('Entries')
            plt.yscale('log')
            plt.hist(merged_n_double_pulses, bins=50, range=(0, 50))

            plt.figure('Double Pulsing Hitmap')
            plt.xlabel('Col')
            plt.ylabel('Row')
            plt.imshow(self.avg_n_double_pulses, interpolation='none', origin='lower', vmin=0)
            plt.colorbar(label='Average number of double pulses')

            if merged_dp_leading:
                max_ts = min(max(merged_dp_leading), 1000)
                plt.figure('Time Distribution Leading Edge Double Pulses')
                plt.xlabel('Time of double pulse after leading edge [ns]')
                plt.ylabel('Entries')
                plt.yscale('log')
                plt.hist(merged_dp_leading, bins=int(max_ts/self.timestamp_bin_ns), range=(0, max_ts))

            if merged_dp_falling:
                max_ts = min(max(merged_dp_falling), 1000)
                plt.figure('Time Distribution Falling Edge Double Pulses')
                plt.xlabel('Time of double pulse after falling edge [ns]')
                plt.ylabel('Entries')
                plt.yscale('log')
                plt.hist(merged_dp_falling, bins=int(max_ts/self.timestamp_bin_ns), range=(0, max_ts))


if __name__ == '__main__':
    sparc_analyzer = DoublePulsingAnalyser(Path(sys.argv[1]))
    sparc_analyzer.analyze()
    plt.show()
