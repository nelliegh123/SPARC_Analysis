"""Shared code for the injection scans: where the files are, how to read one pixel's run,
the Gaussian ToT fit, and fitting a whole scan with caching. The scripts (injection.py, ToT.py,
ToT_vcasb.py, s_curve.py) import from here and only contain their settings and plots."""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  
from analyze_data_sl_tot import SPARCAnalyser

ROOT = Path(__file__).resolve().parents[2] 
# Default data: the Vh scan with pulse duration = 250 (SPARC.toml, run with injection_scan_sl.py), only Vcasb = 300 mV, complete up to DATA_VH[-1]
DATA_DIR = ROOT / "data" / "sparc0" / "injection_pd_250"
PLOT_DIR = ROOT / "plots" / "Injections"
OLD_PLOT_DIR = ROOT / "plots" / "WrongInjections"  # plots of the older data sets in EXTRA_DATASETS
CACHE_DIR = ROOT / "cache"  # one subfolder per data folder, so fits of different data sets never mix

NROWS = SPARCAnalyser.NROWS
NCOLS = SPARCAnalyser.NCOLS
N_INJECTIONS = 10000
TOT_US_PER_COUNT = 0.004  # one ToT count is 4 ns; all ToT values in this code are in µs

SCAN_VH = [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
DATA_VCASB = 300
DATA_VH = SCAN_VH  # the Vh values that are complete in DATA_DIR; add new ones here

# Older injection data sets, each a Vh scan at one Vcasb in its own folder, selectable in the scripts by name.
# "vh_scan" lists the Vh values that are complete (see fit_dataset).
EXTRA_DATASETS = {
    "old": {"label": "old data", "data_dir": DATA_DIR.parent / "injection", "vcasb": 300,
            "vh_scan": SCAN_VH, "tag": "old", "plot_dir": OLD_PLOT_DIR / "old"},
    "ireset": {"label": "Ireset = 70 pA", "data_dir": DATA_DIR.parent / "injection_ireset_70pA", "vcasb": 300,
               "vh_scan": SCAN_VH, "tag": "ireset_70pA", "plot_dir": OLD_PLOT_DIR / "Ireset70pa"},
}



#-------Reads the raw file and loads hitmap etc------
# data_dir is DATA_DIR unless another injection data folder is given, e.g. EXTRA_DATASETS["ireset"]["data_dir"]
def raw_file(vcasb, vh, col, row, data_dir=DATA_DIR):
    return Path(data_dir) / f"data_run_vcasb_{vcasb}_vh_{vh}_px_{col}_{row}.raw"

def load_run(vcasb, vh, col, row, data_dir=DATA_DIR):
    analyzer = SPARCAnalyser(raw_file(vcasb, vh, col, row, data_dir))
    analyzer.unpack_all()
    return analyzer.hitmap, analyzer.active_pixels, np.array(analyzer.tots) * TOT_US_PER_COUNT


def tot_bins(values, bins=100):
    counts = np.round(np.asarray(values) / TOT_US_PER_COUNT)
    width = max(1, int(np.ceil((counts.max() - counts.min() + 1) / bins)))
    n_bins = int(np.ceil((counts.max() - counts.min() + 1) / width))
    return (counts.min() - 0.5 + width * np.arange(n_bins + 1)) * TOT_US_PER_COUNT


#-----------Gaussian fit-------------
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_gaussian(values, bins=100):
    failed = (np.nan, np.nan, np.nan, np.nan)
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return failed

    # Hits more than 10 standard deviations from the median are left out of the fit.
    median = np.median(values)
    robust_std = 1.4826 * np.median(np.abs(values - median))
    if robust_std > 0:
        values = values[np.abs(values - median) <= 10 * robust_std]

    counts, edges = np.histogram(values, tot_bins(values, bins))
    centers = 0.5 * (edges[:-1] + edges[1:])

    sel = counts > 0  
    if sel.sum() < 4:
        return failed

    shift, scale = values.mean(), values.std()
    if scale == 0:
        return failed
    x = (centers[sel] - shift) / scale
    try:
        with warnings.catch_warnings():
            # the covariance isn't used; it can't be estimated for fits that are rejected below anyway
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(gaussian, x, counts[sel], p0=[counts.max(), 0, 1], sigma=np.sqrt(counts[sel]), absolute_sigma=True)
    except RuntimeError:
        return failed
    amp, mean, sigma = popt[0], shift + scale * popt[1], scale * abs(popt[2])
    popt = (amp, mean, sigma)

    if not (edges[0] <= mean <= edges[-1]) or sigma > edges[-1] - edges[0]:
        return failed

    chi2 = np.sum(((counts[sel] - gaussian(centers[sel], *popt)) / np.sqrt(counts[sel])) ** 2)
    ndf = sel.sum() - 3
    return amp, mean, sigma, chi2 / ndf


#-------Fits one pixel, returns mean, sigma, chi2/ndf and the hits in the injected pixel------
def _fit_run(args):
    vcasb, vh, col, row, data_dir = args
    if not raw_file(vcasb, vh, col, row, data_dir).exists():
        return None
    hitmap, _, tots = load_run(vcasb, vh, col, row, data_dir)
    _, mean, sigma, chi2_ndf = fit_gaussian(tots, 100)
    return mean, sigma, chi2_ndf, hitmap[row, col]


#-------Fits every pixel for a list of (vcasb, vh) settings, in parallel------
# Returns a dict of arrays indexed [setting, row, col]: "means" and "sigmas" (µs), "chi2" (chi2/ndf),
# "nhits" (hits in the injected pixel), plus "nfiles" [setting] = how many pixel files were found.
# Results are cached in CACHE_DIR/<name>.npz and reused as long as the settings are the same, so use a
# different name for data from another data_dir.
def fit_scan(settings, name, data_dir=DATA_DIR):
    settings = [tuple(s) for s in settings]
    cache_file = CACHE_DIR / Path(data_dir).name / f"{name}.npz"
    if cache_file.exists():
        cache = np.load(cache_file)
        if [tuple(s) for s in cache["settings"].tolist()] == settings:
            print(f"Loaded cached fits from {cache_file}")
            return {key: cache[key] for key in ("means", "sigmas", "chi2", "nhits", "nfiles")}
        print(f"Cache is out of date, refitting {name}")

    shape = (len(settings), NROWS, NCOLS)
    scan = {key: np.full(shape, np.nan) for key in ("means", "sigmas", "chi2", "nhits")}
    scan["nfiles"] = np.zeros(len(settings), dtype=int)

    jobs = [(i, vcasb, vh, col, row) for i, (vcasb, vh) in enumerate(settings)
            for col in range(NCOLS) for row in range(NROWS)]
    print(f"Fitting {len(jobs)} pixel runs for {name} on {os.cpu_count()} cores")
    with ProcessPoolExecutor() as pool:
        results = pool.map(_fit_run, [job[1:] + (data_dir,) for job in jobs], chunksize=64)
        for (i, _, _, col, row), result in zip(jobs, results):
            if result is None:
                continue
            scan["nfiles"][i] += 1
            for key, value in zip(("means", "sigmas", "chi2", "nhits"), result):
                scan[key][i, row, col] = value

    complete = True
    for (vcasb, vh), n in zip(settings, scan["nfiles"]):
        if n < NROWS * NCOLS:
            complete = False
            print(f"WARNING: only {n} of {NROWS * NCOLS} pixel files found for VCASB = {vcasb} mV, VH = {vh} mV")

    # an incomplete scan is not cached, otherwise the missing pixels would stay missing after the data arrives
    if not complete:
        print(f"Not caching {name}, because pixel files are missing")
        return scan
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, settings=np.array(settings), **scan)
    print(f"Saved fits to {cache_file}")
    return scan


#-------Fits (or loads from cache) the whole Vh scan of one of the EXTRA_DATASETS------
def fit_dataset(name):
    d = EXTRA_DATASETS[name]
    return fit_scan([(d["vcasb"], vh) for vh in d["vh_scan"]], f"vcasb_{d['vcasb']}_vh_scan", data_dir=d["data_dir"])


#-------Plots one dot per pixel at each x value, spread sideways so they don't pile up------
def jitter_scatter(x_values, values, label="All pixels", ms=8):
    x_values = np.asarray(x_values)
    if len(x_values) > 1:
        gaps = np.diff(x_values)
        nearest_gap = np.minimum(np.append(gaps, gaps[-1]), np.insert(gaps, 0, gaps[0]))
        half_width = 0.15 * np.minimum(nearest_gap, np.median(gaps))
    else:
        half_width = [0]
    rng = np.random.default_rng(0)  
    for i, x0 in enumerate(x_values):
        y = np.ravel(values[i])
        x = x0 + rng.uniform(-half_width[i], half_width[i], y.size)
        plt.plot(x, y, ".", color="#2a9df4", alpha=0.5, ms=ms, mew=0, label=label if i == 0 else None)
