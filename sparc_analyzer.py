# SPDX-FileCopyrightText: 2026 The Razorbill Caribou authors
# SPDX-License-Identifier: CC0-1.0

import sys
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


class SPARCAnalyser:
    NROWS = 32
    NCOLS = 28

    def __init__(self, file_path: Path) -> None:
        self._unpacker = FileUnpacker(file_path, self._on_data, self._on_bor, self._on_eor)

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
            self._hits.append((hit.row, hit.column, hit.timestamp * self.timestamp_bin_ns, hit.raw * self.timestamp_bin_ns))

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
        unpack_result = UnpackResult.UNPACKED
        while unpack_result == UnpackResult.UNPACKED:
            unpack_result = self._unpacker.unpack()
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
                plt.hist(self.hits['tot'] * 1e-3, bins=50)


if __name__ == '__main__':
    sparc_analyzer = SPARCAnalyser(Path(sys.argv[1]))
    sparc_analyzer.analyze()
    plt.show()
