# SPDX-FileCopyrightText: 2026 The Razorbill Caribou authors
# SPDX-License-Identifier: CC0-1.0

import argparse
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from razorbill.cnstlnmsgpack import BOR, EOR, DataRecord, FileUnpacker, UnpackResult
from razorbill.core import Configuration, Decoder, Frame
from razorbill.devices.sparc import SPARCBusyFrame, SPARCDecoder, SPARCFrame

"""
Takes data from one pixel.
Gives active pixels, hits, missed hits, run duration, hit rate, histogram of hits in the pixel and the ToT distribution.

The hits of a raw file are cached in sparcCache/decoded, so only the first run has to read the raw file.
If the run is still being written, the cached part is used until you pass --refresh.
"""

CACHE_DIR = Path(__file__).resolve().parent.parent / 'sparcCache' / 'decoded'


class SPARCAnalyser:
    NROWS = 32
    NCOLS = 28


    masked_pixels = []
    # print(f"Pixels {masked_pixels} are masked")
    

    def __init__(self, file_path: Path, use_cache: bool = False, refresh_cache: bool = False) -> None:
        self._unpacker = FileUnpacker(file_path, self._on_data, self._on_bor, self._on_eor)

        self.file_path = Path(file_path)
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache

        self.do_output = True

        # One row per hit: row, column, timestamp (ns), ToT (ns)
        self._hits: list[tuple[int, int, float, float]] = []
        self.hits = pd.DataFrame(self._hits, columns=['row', 'column', 'timestamp', 'tot'])

        self.active_pixels = np.full((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), False, dtype=bool)
        self.hitmap = np.zeros((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), dtype=np.uint64)
        self.n_missed_sparc_hits = 0
        self.time_start = datetime.now(tz=UTC)
        self.time_end = self.time_start
        self.config = Configuration()

        self.masked_pixels = SPARCAnalyser.masked_pixels 

        # self.masked_pixels = SPARCAnalyser.masked_pixels

    def _on_data(self, sender: str, device: str, data: DataRecord, frame: Frame, decoder: Decoder) -> None:
        """Callback for processing data records"""
        # Cast to SPARC frame since we know we only have SPARC data
        frame = cast(SPARCFrame, frame)

        # Count missed hits
        if isinstance(frame, SPARCBusyFrame):
            self.n_missed_sparc_hits += cast(SPARCBusyFrame, frame).busyCount

        # Process frame to get hits
        hits = decoder.process(frame)
        for hit in hits:
            # hit.timestamp is already in ns (leading edge), hit.raw (ToT) is in timestamp bins
            self._hits.append((hit.row, hit.column, hit.timestamp, hit.raw * self.timestamp_bin_ns))

    def _on_bor(self, sender: str, bor: BOR, device_map: Mapping[str, tuple[Decoder, Configuration]]) -> None:
        """Callback for processing Begin-Of-Run (BOR) message"""
        # Extract active pixels from registers
        self._read_active_pixels(bor.userTags['sparc']['registers'])

        # Configure decoder
        decoder, config = device_map['sparc']
        config.update(self.config)
        decoder.configure(config)

        # Store decoder and config
        self.decoder = cast(SPARCDecoder, decoder)
        self.config = config

        # Extract timestamp frequency
        self.timestamp_bin_ns = 1e3 / cast(float, self.config.get('timestamp_frequency'))

    def _on_eor(self, sender: str, eor: EOR) -> None:
        """Callback for processing End-Of-Run (EOR) message"""
        # Extract run time
        self.time_start = eor.runMetadata['time_start']
        self.time_end = eor.runMetadata['time_end']

    def _read_active_pixels(self, registers: dict[str, int]):
        for col in range(SPARCAnalyser.NCOLS):
            col_register = f'enable_col_{(col + 1) // 2}'
            col_even_odd = 'even' if col % 2 == 0 else 'odd'
            for row in range(SPARCAnalyser.NROWS):
                row_register = f'enable_row_{row}_{col_even_odd}'
                if registers[row_register] == 1 and registers[col_register] == 1:
                    self.active_pixels[row, col] = True

    def unpack_all(self) -> UnpackResult:
        if self.use_cache and not self.refresh_cache:
            cached_result = self._load_cache()
            if cached_result is not None:
                return cached_result

        # Taken before reading, so a file that grows during reading no longer matches the cache
        stat = self.file_path.stat()

        unpack_result = UnpackResult.UNPACKED
        while unpack_result == UnpackResult.UNPACKED:
            unpack_result = self._unpacker.unpack()

        if self.use_cache and unpack_result != UnpackResult.FAILED:
            self._save_cache(unpack_result, stat.st_size, stat.st_mtime_ns)
        return unpack_result

    def _cache_path(self) -> Path:
        return CACHE_DIR / self.file_path.parent.name / f'{self.file_path.stem}_hits.npz'

    def _save_cache(self, unpack_result: UnpackResult, size: int, mtime_ns: int) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first, so an interrupted run can't leave a broken cache behind
        temp_path = path.with_suffix('.tmp.npz')
        np.savez(
            temp_path,
            unpack_result=unpack_result.name,
            file_size=size,
            file_mtime_ns=mtime_ns,
            hits=np.array(self._hits, dtype=np.float64).reshape(-1, 4),
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

            self._hits = cache['hits']
            self.active_pixels = cache['active_pixels']
            self.n_missed_sparc_hits = int(cache['n_missed_sparc_hits'])
            self.time_start = datetime.fromisoformat(str(cache['time_start']))
            self.time_end = datetime.fromisoformat(str(cache['time_end']))
            self.config = Configuration()
            self.config.add('tot_mode', bool(cache['tot_mode']))
            self.timestamp_bin_ns = float(cache['timestamp_bin_ns'])

        if self.do_output:
            print(f'Using cache {path}')
            if not unchanged:
                print(
                    f'WARNING: the raw file has grown from {cached_size / 1e9:.2f} GB to {stat.st_size / 1e9:.2f} GB '
                    'since this cache was made, so the newest data is NOT included. Use --refresh to read it.'
                )
        return unpack_result

    def analyze(self) -> None:
        unpack_result = self.unpack_all()
        if self.do_output:
            print(unpack_result.name)

        if unpack_result == UnpackResult.FAILED:
            return

        # Build hit dataframe
        self.hits = pd.DataFrame(self._hits, columns=['row', 'column', 'timestamp', 'tot']).astype(
            {'row': np.uint8, 'column': np.uint8, 'timestamp': np.float64, 'tot': np.float64}
        )


        #Masking pixels
        # if len(self.masked_pixels) != 0:
        #     print('Masking pixels', self.masked_pixels)

        for i in range(len(self.masked_pixels)):
            col, row = self.masked_pixels[i]
            self.hits = self.hits[~((self.hits['row'] == row) & (self.hits['column'] == col ))]
            self.active_pixels[row, col] = False


     

        # Run stats
        self.n_active_pixels = self.active_pixels.sum()
        np.add.at(self.hitmap, (self.hits['row'].to_numpy(), self.hits['column'].to_numpy()), 1)
        self.hitmap = np.ma.masked_where(~self.active_pixels, self.hitmap)
        self.n_hits = self.hitmap.sum()
        self.run_duration = self.time_end - self.time_start

        if self.do_output:
            
            print(f'Active pixels: {self.n_active_pixels} ({self.active_pixels.size - self.n_active_pixels} masked)')
            print(f'Hits: {self.n_hits}')
            print(f'Missed SPARC hits: {self.n_missed_sparc_hits}')
            if self.run_duration:
                print(f'Run duration: {self.run_duration}')
                print(f'Hit rate: {self.n_hits / (self.run_duration.total_seconds() * self.n_active_pixels) :.3g} hits/pixel/second')

            plt.figure('Hitmap')
            plt.xlabel('Col')
            plt.ylabel('Row')
            plt.imshow(np.ma.masked_less(self.hitmap, 1), interpolation='none', origin='lower', vmin=0)
            plt.colorbar(label='Hits')

            if self.config.get("tot_mode"):
                plt.figure('ToT distribution')
                plt.xlabel('ToT [us]')
                plt.ylabel('Entries')
                plt.hist(self.hits['tot'] * 1e-3, bins=50, range=(0, 60))  

                # #-----Plots only a single pixel------------
                # pixel_hits = self.hits[(self.hits['column'] == 6) & (self.hits['row'] == 6)]
                # plt.hist(pixel_hits['tot'] * 1e-3, bins=75, range=(0, 52))     
                # #-----Plots only a single pixel------------

                plots = [(0, 0), (3, 3), (6, 6)]
                plt.figure('Overlayed ToT distribution')
                plt.xlabel('ToT [us]')
                plt.ylabel('Entries')
                for i in range(len(plots)):
                    col, row = plots[i]
                    pixel_hits = self.hits[(self.hits['column'] == col) & (self.hits['row'] == row)]
                    plt.hist(pixel_hits['tot'] * 1e-3, bins=75, range=(0, 15), alpha=0.7, label=f'Pixel ({col}, {row})')  
                plt.legend()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis of one raw file.')
    parser.add_argument('file', type=Path, help='the raw file')
    parser.add_argument('--refresh', action='store_true', help='read the raw file again and update the cache')
    args = parser.parse_args()

    sparc_analyzer = SPARCAnalyser(args.file, use_cache=True, refresh_cache=args.refresh)
    sparc_analyzer.analyze()
    plt.show()
