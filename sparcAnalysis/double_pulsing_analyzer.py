# SPDX-FileCopyrightText: 2026 The Razorbill Caribou authors
# SPDX-License-Identifier: CC0-1.0

import argparse
import os
import sys
from datetime import datetime
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from razorbill.cnstlnmsgpack import DataRecord, UnpackResult
from razorbill.core import Configuration, Decoder, Frame
from razorbill.devices.sparc import SPARCFrame, SPARCHitFrame
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from sparc_analyzer import SPARCAnalyser 

"""
Like sparc_analyzer, plus the double pulses of every event in every pixel.

The decoded data of a raw file is cached in sparcCache/decoded, so only the first run has to read the raw file.
If the run is still being written, the cached part is used until you pass --refresh.
"""

CACHE_DIR = Path(__file__).resolve().parent.parent / 'sparcCache' / 'decoded'


class DoublePulsingAnalyser(SPARCAnalyser):
    def __init__(self, file_path: Path, use_cache: bool = False, refresh_cache: bool = False) -> None:
        super().__init__(file_path)

        self.file_path = Path(file_path)
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache

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
        self.avg_n_double_pulses_err = np.zeros((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), dtype=np.float64)
        self.dp_rising: dict[tuple[int, int], list[list[float]]] = {}
        self.dp_falling: dict[tuple[int, int], list[list[float]]] = {}

        for row in range(SPARCAnalyser.NROWS):
            for column in range(SPARCAnalyser.NCOLS):
                self.ts_diffs[row, column] = []
                self.events[row, column] = []
                self.dp_rising[row, column] = []
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

    def unpack_all(self) -> UnpackResult:
        if self.use_cache and not self.refresh_cache:
            cached_result = self._load_cache()
            if cached_result is not None:
                return cached_result

        # Taken before reading, so a file that grows during reading no longer matches the cache
        stat = self.file_path.stat()

        # The unpacker doesn't expose its file position, so this counts entries rather than showing a percentage
        unpack_result = UnpackResult.UNPACKED
        with tqdm(desc='Reading data', unit=' entries', disable=not self.do_output) as progress:
            while unpack_result == UnpackResult.UNPACKED:
                unpack_result = self._unpacker.unpack()
                progress.update()

        if self.use_cache and unpack_result != UnpackResult.FAILED:
            self._save_cache(unpack_result, stat.st_size, stat.st_mtime_ns)
        return unpack_result

    def _cache_path(self) -> Path:
        return CACHE_DIR / self.file_path.parent.name / f'{self.file_path.stem}.npz'

    def _save_cache(self, unpack_result: UnpackResult, size: int, mtime_ns: int) -> None:
        # Columns instead of one tuple per hit: pd.DataFrame takes both, and this frees a lot of memory
        n_hits, n_frames = len(self._hits), len(self._frames)
        hits = cast(list, self._hits)
        frames = cast(list, self._frames)
        self._hits = {
            'row': np.fromiter(map(itemgetter(0), hits), np.uint8, n_hits),
            'column': np.fromiter(map(itemgetter(1), hits), np.uint8, n_hits),
            'timestamp': np.fromiter(map(itemgetter(2), hits), np.float64, n_hits),
            'tot': np.fromiter(map(itemgetter(3), hits), np.float64, n_hits),
        }
        self._frames = {
            'row': np.fromiter(map(itemgetter(0), frames), np.uint8, n_frames),
            'column': np.fromiter(map(itemgetter(1), frames), np.uint8, n_frames),
            'timestamp': np.fromiter(map(itemgetter(2), frames), np.float64, n_frames),
        }

        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first, so an interrupted run can't leave a broken cache behind
        temp_path = path.with_suffix('.tmp.npz')
        np.savez(
            temp_path,
            unpack_result=unpack_result.name,
            file_size=size,
            file_mtime_ns=mtime_ns,
            hit_row=self._hits['row'],
            hit_column=self._hits['column'],
            hit_timestamp=self._hits['timestamp'],
            hit_tot=self._hits['tot'],
            frame_row=self._frames['row'],
            frame_column=self._frames['column'],
            frame_timestamp=self._frames['timestamp'],
            active_pixels=self.active_pixels,
            n_missed_sparc_hits=self.n_missed_sparc_hits,
            time_start=self.time_start.isoformat(),
            time_end=self.time_end.isoformat(),
            tot_mode=bool(self.config.get('tot_mode')),
            timestamp_bin_ns=self.timestamp_bin_ns,
        )
        os.replace(temp_path, path)
        if self.do_output:
            print(f'Saved cache {path}')

    def _load_cache(self) -> UnpackResult | None:
        """Fills the analyser from the cache. Returns None if the raw file has to be read instead."""
        path = self._cache_path()
        if not path.exists():
            return None

        with np.load(path) as cache:
            unpack_result = UnpackResult.__members__[str(cache['unpack_result'])]
            cached_size = int(cache['file_size'])
            stat = self.file_path.stat()
            unchanged = cached_size == stat.st_size and int(cache['file_mtime_ns']) == stat.st_mtime_ns
            # The raw file only grows while the run is being written, anything else means different data
            run_still_growing = unpack_result == UnpackResult.WAITING_FOR_DATA and stat.st_size > cached_size
            if not unchanged and not run_still_growing:
                return None

            self._hits = {
                'row': cache['hit_row'],
                'column': cache['hit_column'],
                'timestamp': cache['hit_timestamp'],
                'tot': cache['hit_tot'],
            }
            self._frames = {
                'row': cache['frame_row'],
                'column': cache['frame_column'],
                'timestamp': cache['frame_timestamp'],
            }
            self.active_pixels = cache['active_pixels']
            self.n_missed_sparc_hits = int(cache['n_missed_sparc_hits'])
            self.time_start = datetime.fromisoformat(str(cache['time_start']))
            self.time_end = datetime.fromisoformat(str(cache['time_end']))
            self.config = Configuration()
            self.config.add('tot_mode', bool(cache['tot_mode']))
            self.timestamp_bin_ns = float(cache['timestamp_bin_ns'])

        self._rebuild_ts_diffs()

        if self.do_output:
            print(f'Using cache {path}')
            if not unchanged:
                print(
                    f'WARNING: the raw file has grown from {cached_size / 1e9:.2f} GB to {stat.st_size / 1e9:.2f} GB '
                    'since this cache was made, so the newest data is NOT included. Use --refresh to read it.'
                )
        return unpack_result

    def _rebuild_ts_diffs(self) -> None:
        """Gives the same ts_diffs as _on_data, which isn't called when the data comes from the cache"""
        pixels = self._frames['row'].astype(np.int64) * SPARCAnalyser.NCOLS + self._frames['column']
        order = np.argsort(pixels, kind='stable')
        pixels, timestamps = pixels[order], self._frames['timestamp'][order]
        starts = np.flatnonzero(np.diff(pixels, prepend=-1))
        ts_diffs = np.diff(timestamps, prepend=0.)
        # _last_ts starts at -event_split, so the first frame of a pixel always starts a new event
        ts_diffs[starts] = timestamps[starts] + self.event_split
        for start, end in zip(starts, np.append(starts[1:], pixels.size)):
            row, column = divmod(int(pixels[start]), SPARCAnalyser.NCOLS)
            self.ts_diffs[row, column] = ts_diffs[start:end].tolist()

    def _pixel_progress(self, pixel_data: dict, description: str) -> tqdm:
        return tqdm(pixel_data.items(), desc=description, unit=' pixels', disable=not self.do_output)

    def analyze(self) -> None:
        super().analyze()

        tot_mode = self.config.get('tot_mode')

        # Build frame dataframe
        self.frames = pd.DataFrame(self._frames, columns=['row', 'column', 'timestamp']).astype(
            {'row': np.uint8, 'column': np.uint8, 'timestamp': np.float64}
        )

        # Masking pixels
        for row, col in self.masked_pixels:
            self.frames = self.frames[~((self.frames['row'] == row) & (self.frames['column'] == col))]
            self.ts_diffs[row, col] = []



        # Build events per pixel
        for pixel, ts_diffs in self._pixel_progress(self.ts_diffs, 'Building events'):
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
            for pixel, events in self._pixel_progress(self.events, 'Filtering chargeup'):
                for n, event in enumerate(events):
                    for m, ts_m in enumerate(event):
                        # Discard anything from injection chargeup
                        if ts_m > self.injection_chargeup_split:
                            self.events[pixel][n] = [ts - ts_m for ts in event[m:]]
                            break

        # Count number of double pulses per pixel
        expected_sparc_hit_count = 2 if tot_mode else 1
        for pixel, events in self._pixel_progress(self.events, 'Counting double pulses'):
            n_double_pulses = [max(len(event) - expected_sparc_hit_count, 0) for event in events]
            self.n_double_pulses[pixel] = n_double_pulses
            self.avg_n_double_pulses[pixel] = np.mean(n_double_pulses) if n_double_pulses else np.nan
            self.avg_n_double_pulses_err[pixel] = np.mean(np.sqrt(n_double_pulses)) if n_double_pulses else np.nan

        # Separate between double pulses on rising and falling edge
        for pixel, events in self._pixel_progress(self.events, 'Separating double pulses'):
            dp_rising: list[list[float]] = []
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
                        dp_event_rising = event[1:largest_gap_n]
                        if dp_event_rising:
                            dp_rising.append(dp_event_rising)
                        dp_event_falling = [ts - event[largest_gap_n] for ts in event[largest_gap_n+1:]]
                        if dp_event_falling:
                            dp_falling.append(dp_event_falling)
                self.dp_falling[pixel] = dp_falling
            else:
                for event in events:
                    if len(event) > 1:
                        dp_rising.append(event[1:])
            self.dp_rising[pixel] = dp_rising

        if self.do_output:
            merged_ts_diffs = list(chain.from_iterable(self.ts_diffs.values()))
            merged_events = list(chain.from_iterable(self.events.values()))
            total_events = len(merged_events)
            merged_n_double_pulses = list(chain.from_iterable(self.n_double_pulses.values()))
            total_avg_n_double_pulses = np.nanmean(self.avg_n_double_pulses.flatten())
            total_avg_n_double_pulses_err = np.sqrt(np.nansum(self.avg_n_double_pulses_err.flatten()**2) / np.count_nonzero(~np.isnan(self.avg_n_double_pulses_err.flatten()))) if np.count_nonzero(~np.isnan(self.avg_n_double_pulses_err.flatten())) > 0 else np.nan
            merged_dp_rising = list(chain.from_iterable(chain.from_iterable(self.dp_rising.values())))
            merged_dp_falling = list(chain.from_iterable(chain.from_iterable(self.dp_falling.values())))

            print(f'{total_events} events over with {total_avg_n_double_pulses:.2f} ± {total_avg_n_double_pulses_err:.2f} double pulses on average')

            # plt.figure('Timestamp Differences [0ns - 512ns]')
            # plt.xlabel('Timestamp Difference [ns]')
            # plt.ylabel('Entries')
            # plt.yscale('log')
            # plt.hist([diff for diff in merged_ts_diffs if diff < 512], bins=128, range=(0, 512))

            # plt.figure('Timestamp Differences [100ns - 100us]')
            # plt.xlabel('Timestamp Difference [us]')
            # plt.ylabel('Entries')
            # plt.yscale('log')
            # plt.hist([diff * 1e-3 for diff in merged_ts_diffs if diff > 100 and diff < 100e3], bins=1000, range=(0, 100))

            # plt.figure('Double Pulsing Histogram')
            # plt.xlabel('Number of double pulses')
            # plt.ylabel('Entries')
            # plt.xticks([5 * n for n in range(11)])
            # plt.yscale('log')
            # plt.hist(merged_n_double_pulses, bins=51, range=(-0.5, 50.5))

            # plt.figure('Double Pulsing Hitmap')
            # plt.xlabel('Col')
            # plt.ylabel('Row')
            # plt.imshow(self.avg_n_double_pulses, interpolation='none', origin='lower', vmin=0)
            # plt.colorbar(label='Average number of double pulses')

            # if merged_dp_rising:
            #     max_ts = min(max(merged_dp_rising), 1000)
            #     plt.figure('Time Distribution rising Edge Double Pulses')
            #     plt.xlabel('Time of double pulse after rising edge [ns]')
            #     plt.ylabel('Entries')
            #     plt.yscale('log')
            #     plt.hist(merged_dp_rising, bins=int(max_ts/self.timestamp_bin_ns)+1, range=(-0.5*self.timestamp_bin_ns, max_ts+0.5*self.timestamp_bin_ns))

            # if merged_dp_falling:
            #     max_ts = min(max(merged_dp_falling), 1000)
            #     plt.figure('Time Distribution Falling Edge Double Pulses')
            #     plt.xlabel('Time of double pulse after falling edge [ns]')
            #     plt.ylabel('Entries')
            #     plt.yscale('log')
            #     plt.hist(merged_dp_falling, bins=int(max_ts/self.timestamp_bin_ns)+1, range=(-0.5*self.timestamp_bin_ns, max_ts+0.5*self.timestamp_bin_ns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double pulsing analysis of one raw file.')
    parser.add_argument('file', type=Path, help='the raw file')
    parser.add_argument('--refresh', action='store_true', help='read the raw file again and update the cache')
    args = parser.parse_args()

    sparc_analyzer = DoublePulsingAnalyser(args.file, use_cache=True, refresh_cache=args.refresh)
    # Off for source data like iron-55, which has no injection chargeup. Use 30e3 for injection data.
    sparc_analyzer.injection_chargeup_split = 0
    sparc_analyzer.analyze()
    plt.show()
