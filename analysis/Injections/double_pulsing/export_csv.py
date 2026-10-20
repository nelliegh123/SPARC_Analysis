"""Writes every hit word of one run to a CSV file, one row per word."""
import csv
from pathlib import Path

from razorbill.cnstlnmsgpack import FileUnpacker, UnpackResult
from razorbill.devices.sparc import SPARCHitFrame

ROOT = Path(__file__).resolve().parents[3]
RAW_FILE = ROOT / "data" / "sparc7" / "double_pulsing" / "pd_750" / "data_run_vcasb_300_vh_300_px_6_6.raw"
CSV_FILE = ROOT / "data" / "sparc7" / "double_pulsing" / "csv" / "pd_750.csv"

rows = []


def on_bor(sender, bor, device_map):
    decoder, config = device_map["sparc"]
    decoder.configure(config)

def on_data(sender, device, data, frame, decoder):
    if not isinstance(frame, SPARCHitFrame):
        return  # skip trigger and busy words
    fpga = frame.timestampFPGA
    sparc = frame.timestampSPARC
    combined = decoder.combineTimestamps(fpga, sparc)
    rows.append([fpga, sparc, combined, frame.row, frame.column])

def on_eor(sender, eor):
    pass


unpacker = FileUnpacker(RAW_FILE, on_data, on_bor, on_eor)
while unpacker.unpack() == UnpackResult.UNPACKED:
    pass

CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fpga_timestamp", "sparc_timestamp", "combined_timestamp", "row", "column"])
    writer.writerows(rows)

print(f"Saved {len(rows)} words to {CSV_FILE}")
