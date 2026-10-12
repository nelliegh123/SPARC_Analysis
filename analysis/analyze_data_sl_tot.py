# SPDX-FileCopyrightText: 2026 The Razorbill Caribou authors
# SPDX-License-Identifier: CC0-1.0

import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from razorbill.cnstlnmsgpack import BOR, EOR, DataRecord, FileUnpacker, UnpackResult
from razorbill.core import Configuration, Decoder, Frame
from razorbill.devices.sparc import SPARCBusyFrame, SPARCFrame


class SPARCAnalyser:
    NROWS = 32
    NCOLS = 28

    def __init__(self, file_path: Path) -> None:
        self._unpacker = FileUnpacker(file_path, self._on_data, self._on_bor, self._on_eor)
        # Reads the file, initialized beginning and end of run

        self.active_pixels = np.full((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), False, dtype=bool)
        # Creates a empty (alse) grid of the sparc pixels
        self.hitmap = np.zeros((SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS), dtype=np.uint64)
        # Creates a grid of zeros to count the hits for each pixel
        self.missed_hits = 0
        # Initializes the counter for missed hits
        self.time_start = datetime.now(tz=UTC)
        # Initializes the start time of the run to the current time
        self.time_end = self.time_start
        # Initializes the end time of the run to the current time, this will be updated when the EOR message is processed

        self.tots = []


    def _on_data(self, sender: str, device: str, data: DataRecord, frame: Frame, decoder: Decoder) -> None:
        """Callback for processing data records"""
        # Cast to SPARC frame since we know we only have SPARC data
        frame = cast(SPARCFrame, frame)

        # Count missed hits
        if isinstance(frame, SPARCBusyFrame):
            self.missed_hits += cast(SPARCBusyFrame, frame).busyCount

        # Process frame to get hits
        hits = decoder.process(frame)
        for hit in hits:
            if hit.row < SPARCAnalyser.NROWS and hit.column < SPARCAnalyser.NCOLS:
                self.hitmap[hit.row, hit.column] += 1
                self.tots.append(hit.raw)
            else:
                print(f"Ignoring hit outside matrix {hit.row}, {hit.column}")

    def _on_bor(self, sender: str, bor: BOR, device_map: Mapping[str, tuple[Decoder, Configuration]]) -> None:
        """Callback for processing Begin-Of-Run (BOR) message"""
        # Extract active (as in not masked) pixels from registers
        self._read_active_pixels(bor.userTags['sparc']['registers'])

        # Configure decoder (settings can be overridden here)
        decoder, config = device_map['sparc']
        # config.set("tot_lower_limit", 40)
        # config.set("tot_upper_limit", 8192)
        decoder.configure(config)

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
        # Based on the register values, this function sets the corresponding pixels in the active_pixels array to True if they are enabled (active) and False otherwise.

    def unpack_all(self) -> UnpackResult:
        unpack_result = UnpackResult.UNPACKED
        while unpack_result == UnpackResult.UNPACKED:
            unpack_result = self._unpacker.unpack()
        return unpack_result

    def analyze(self) -> None:
        unpack_result = self.unpack_all()

        print(unpack_result.name)
        if unpack_result == UnpackResult.FAILED:
            return


        # Count active pixels
        n_active_pixels = self.active_pixels.sum()
        print(f'Active pixels: {n_active_pixels} ({self.active_pixels.size - n_active_pixels} masked)')

        # Apply mask
        self.hitmap = np.ma.masked_where(~self.active_pixels, self.hitmap)

        # Print total number of hits
        n_hits = self.hitmap.sum()
        # print(f'Type hitmap: {type(self.hitmap)}')
        # print(f'Hitmap: {(self.hitmap)}')
        print(f'Hits:          {n_hits}')
        print(f'Missed hits:   {self.missed_hits}')

        # Calculate the hit rate
        run_duration = self.time_end - self.time_start
        if run_duration:
            print(f'Run duration:  {run_duration}')
            hit_rate = n_hits / (run_duration.total_seconds() * n_active_pixels)
            print(f'Hit rate:      {hit_rate :.3g} hits/pixel/second')
            print("-"*40)

        # Hitmap
        plt.figure('Hitmap')
        plt.imshow(np.ma.masked_less(self.hitmap, 1), interpolation='none', origin='lower', vmin=0)
        plt.colorbar(label='Hits')
        plt.xlabel('Col')
        plt.ylabel('Row')
        plt.grid(False)
        plt.savefig('hitmap.png', dpi=150)




        # ToT
        plt.figure('ToT')
        plt.hist(np.array(self.tots)*4, 100)
        plt.xlabel('ToT [ns]')
        plt.ylabel('Hits')
        plt.savefig('tot_hist.png', dpi=150)





if __name__ == '__main__':
    sparc_analyzer = SPARCAnalyser(Path(sys.argv[1]))
    sparc_analyzer.analyze()
    plt.show()
