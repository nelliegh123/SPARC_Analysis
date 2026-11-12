"""
Plots the ToT vs Vh of every pixel. 

Usage:
    python ToT_vh.py DATA_DIR {sparc_analyzer,double_pulsing_analyzer} [--pixel COL ROW] [--vcasb VCASB] [--min-hits N]

The two analyzers give the ToT of one injection in different ways:
    sparc_analyzer:          the ToT of the decoder hits
    double_pulsing_analyzer: the time from the first leading edge frame to the first falling edge frame of the event,
                             so double pulses on either edge do not change the ToT
At high Vh every injection also gives a short pulse ~32 µs before the real one (the injection chargeup).
Both analyzers drop it in the same way as DoublePulsingAnalyser does.
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from double_pulsing_analyzer import DoublePulsingAnalyser
from sparc_analyzer import SPARCAnalyser
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
FILE_NAME = re.compile(r"data_vcasb(\d+)_ireset\d+_vcasn\d+_ibias\d+_idb\d+_vh(\d+)_c(\d+)_r(\d+)_run_0\.raw")
COLORS = ["#e0479e", "#ffc000", "#2ca02c", "#9467bd"]
EVENT_SPLIT_NS = 100e3  # A new injection starts after a gap this long
CHARGEUP_SPLIT_NS = 30e3 


#-------The ToTs (µs) of the injections of one pixel run, without the injection chargeup------
def sparc_analyzer_tots(file, col, row):
    ana = SPARCAnalyser(file)
    ana.do_output = False
    ana.analyze()
    check_tot_mode(ana, file)
    hits = ana.hits[(ana.hits["row"] == row) & (ana.hits["column"] == col)].sort_values("timestamp")
    timestamps, tots = hits["timestamp"].to_numpy(), hits["tot"].to_numpy()

    # same event split and chargeup filter as DoublePulsingAnalyser, but on the leading edges of the decoder hits
    keep = np.ones(tots.size, dtype=bool)
    starts = np.flatnonzero(np.diff(timestamps, prepend=-np.inf) >= EVENT_SPLIT_NS)
    for start, end in zip(starts, np.append(starts[1:], tots.size)):
        after = np.flatnonzero(timestamps[start:end] - timestamps[start] > CHARGEUP_SPLIT_NS)
        if after.size:
            keep[start:start + after[0]] = False
    return tots[keep] * 1e-3


def double_pulsing_analyzer_tots(file, col, row):
    ana = DoublePulsingAnalyser(file)
    ana.do_output = False
    ana.analyze()
    check_tot_mode(ana, file)
    # events start at 0 (the first leading edge frame); the largest gap is the one before the first falling edge frame
    return np.array([event[np.argmax(np.diff(event)) + 1] for event in ana.events[row, col] if len(event) >= 2]) * 1e-3


def check_tot_mode(ana, file):
    if not ana.config.get("tot_mode"):
        raise SystemExit(f"{file.name} was not taken in ToT mode")


ANALYZERS = {"sparc_analyzer": sparc_analyzer_tots, "double_pulsing_analyzer": double_pulsing_analyzer_tots}


#-------Median ToT (µs) and number of ToT values of one pixel run------
def median_tot(job):
    analyzer, file, col, row = job
    tots = ANALYZERS[analyzer](file, col, row)
    return (np.median(tots) if tots.size else np.nan), tots.size


#-------The runs in DATA_DIR as {(vh, col, row): file}------
def find_runs(data_dir, vcasb):
    runs, vcasbs = {}, set()
    for file in data_dir.glob("*.raw"):
        match = FILE_NAME.fullmatch(file.name)
        if not match:
            print(f"Skipping {file.name}: not data_vcasb<vcasb>_ireset<ireset>_vcasn<vcasn>_ibias<ibias>_idb<idb>_vh<vh>_c<col>_r<row>_run_0.raw")
            continue
        vcasbs.add(int(match[1]))
        if vcasb is None or int(match[1]) == vcasb:
            runs[int(match[2]), int(match[3]), int(match[4])] = file
    if vcasb is None and len(vcasbs) > 1:
        raise SystemExit(f"{data_dir} has several Vcasb values {sorted(vcasbs)}, choose one with --vcasb")
    if not runs:
        raise SystemExit(f"No raw files found in {data_dir}")
    return runs


#-------Cache of analysed raw files as {file name: (size, mtime_ns, median ToT, number of ToT values)}------
def cache_path(data_dir, analyzer):
    return ROOT / "sparcCache" / data_dir.parent.name / f"{data_dir.name}_{analyzer}.npz"


def load_cache(path):
    if not path.exists():
        return {}
    with np.load(path) as cache:
        return {str(name): (int(size), int(mtime), float(median), int(n)) for name, size, mtime, median, n
                in zip(cache["name"], cache["size"], cache["mtime"], cache["median"], cache["n"])}


def save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(cache)
    size, mtime, median, n = (np.array(column) for column in zip(*(cache[name] for name in names)))
    np.savez(path, name=np.array(names), size=size, mtime=mtime, median=median, n=n)


def file_stamp(file):
    stat = file.stat()
    return stat.st_size, stat.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description="Plots the median ToT vs Vh of every pixel of an injection scan.")
    parser.add_argument("data_dir", type=Path, help="folder with the raw files of the scan")
    parser.add_argument("--analyzer", default="double_pulsing_analyzer",  choices=ANALYZERS, help="If double pulses should be removed or not")
    parser.add_argument("--pixel", nargs=2, type=int, action="append", default=[(17, 20), (19, 19)], metavar=("COL", "ROW"),
                        help="Show fit of this/these pixels")
    parser.add_argument("--vcasb", type=int, help="Vcasb [mV] to use, if the folder has several")
    parser.add_argument("--min-hits", type=int, default=10,
                        help="leave out pixel runs with fewer ToT values than this (default 100)")
    parser.add_argument("--recompute", action="store_true", help="ignore the cache and analyse every raw file again")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    runs = find_runs(data_dir, args.vcasb)
    vhs = sorted({vh for vh, _, _ in runs})
    nrows, ncols = SPARCAnalyser.NROWS, SPARCAnalyser.NCOLS
    for vh in vhs:
        n = sum(key[0] == vh for key in runs)
        if n < nrows * ncols:
            print(f"WARNING: only {n} of {nrows * ncols} pixel files found for Vh = {vh} mV")

    print(f"Analysing data at {len(vhs)} different vh values using {args.analyzer}")

    cache_file = cache_path(data_dir, args.analyzer)
    cache = {} if args.recompute else load_cache(cache_file)
    stamps = {file: file_stamp(file) for file in runs.values()}
    todo = [file for file in runs.values() if cache.get(file.name, (None, None))[:2] != stamps[file]]
    print(f"{len(runs) - len(todo)} pixel runs from cache {cache_file}, analysing {len(todo)}")
    if todo:
        with ProcessPoolExecutor() as pool:
            jobs = [(args.analyzer, file, *map(int, FILE_NAME.fullmatch(file.name).group(3, 4))) for file in todo]
            for file, (median, n) in zip(todo, tqdm(pool.map(median_tot, jobs, chunksize=16), total=len(jobs), desc="Analyzing data")):
                cache[file.name] = (*stamps[file], float(median), int(n))
        save_cache(cache_file, cache)

    medians = np.full((len(vhs), nrows, ncols), np.nan)  # [vh, row, col]
    for (vh, col, row), file in runs.items():
        _, _, median, n = cache[file.name]
        if n >= args.min_hits:
            medians[vhs.index(vh), row, col] = median
    for vh, values in zip(vhs, medians):
        print(f"Vh = {vh:6d} mV: {np.isfinite(values).sum():3d} pixels, median ToT {np.nanmedian(values) if np.isfinite(values).any() else np.nan:.3f} µs")

    plt.rcParams.update({"font.size": 13})
    plt.figure(figsize=(8, 6))
    rng = np.random.default_rng(0)
    half_width = 0.2 * np.min(np.diff(vhs)) if len(vhs) > 1 else 0
    for i, vh in enumerate(vhs):
        y = medians[i].ravel()
        plt.plot(vh + rng.uniform(-half_width, half_width, y.size), y, ".", color="#2a9df4", alpha=0.5, ms=8, mew=0,
                 label="All pixels" if i == 0 else None)
    for (col, row), color in zip(args.pixel, COLORS):
        plt.plot(vhs, medians[:, row, col], "o-", color=color, ms=6, label=f"Pixel ({col}, {row})")

    plt.ylim(bottom=0)
    plt.grid(alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.xlabel("Vh [mV]")
    plt.ylabel("Median ToT [µs]")
    plt.title(f"{data_dir.name}, {args.analyzer}", fontsize=11)
    for handle in plt.legend(framealpha=0.95, edgecolor="dimgray").legend_handles:
        handle.set_alpha(1)

    out_dir = ROOT / "sparcPlots" / data_dir.parent.name / "ToT_vh"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tot_vs_vh_{data_dir.name}_{args.analyzer}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
