import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.optimize import curve_fit
from injection_scan import (fit_scan, jitter_scatter, load_run, DATA_DIR, PLOT_DIR, NROWS, NCOLS, N_INJECTIONS, DATA_VH,
                            EXTRA_DATASETS, fit_dataset)

"""
Plots the ToT vs Vh for a fixed Vcasb.
The ToT is the mean of the Gaussian fit to the ToT histogram for each pixel.
Full scans are available at Vcasb = 250 and 300 mV, with VH from 25 to 1200 mV,
and the other data sets in EXTRA_DATASETS in injection_scan.py (old data, Ireset = 70 pA).
"""

# On the command line: a Vcasb for the standard data, "python ToT_vh.py 250" (300 if nothing is given),
# or the name of one of the EXTRA_DATASETS, e.g. "python ToT_vh.py ireset" or "python ToT_vh.py pd250".
ARG = sys.argv[1] if len(sys.argv) > 1 else "300"
if ARG in EXTRA_DATASETS:
    DATASET = EXTRA_DATASETS[ARG]
    VCASB = DATASET["vcasb"]
    VH_VALUES = DATASET["vh_scan"]
    DATA = DATASET["data_dir"]
    SETTINGS = f"{DATASET['label']}, VCASB = {VCASB} mV"
    FILE_TAG = f"{DATASET['tag']}_vcasb_{VCASB}"
    OUTPUT_DIR = DATASET["plot_dir"]
else:
    DATASET = None
    VCASB = int(ARG)
    VH_VALUES = DATA_VH
    DATA = DATA_DIR
    SETTINGS = f"VCASB = {VCASB} mV"
    FILE_TAG = f"vcasb_{VCASB}"
    OUTPUT_DIR = PLOT_DIR / "tot_vh"

# (6, 6) is near the top of the ToT range of the chip, (10, 19) near the bottom (about 0.7 x the ToT of (6, 6))
HIGHLIGHT = [(6, 6), (10, 19)]
DOT_SIZE = 14  # marker size of the "All pixels" dots (one per pixel)
HIGHLIGHT_COLORS = ["#f5a623", "#d55e00", "#8c3a00"]  # light, dark and darker orange, one per HIGHLIGHT pixel
FIT_VH_MAX = 1200 #600

# Noisy pixels are found only at VH <= NOISE_VH_MAX: the injection is too small to fire any pixel there,
# so every hit is noise. At higher VH the cut would instead pick out pixels with a low threshold.
NOISE_VH_MAX = 50

# If True, a pixel's ToT at a given VH is only plotted if it had N_INJECTIONS hits there, within HITS_TOLERANCE
FULL_HITS_ONLY = False
HITS_TOLERANCE = 0.05  # 5 %, so 9500 to 10500 hits
TAG = "_full_hits" if FULL_HITS_ONLY else ""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



#-----Finds noisy pixels: hit count more than 5 standard deviations above the chip mean at any VH <= NOISE_VH_MAX-----
def find_noisy_pixels(nhits):
    low_vh = np.array(VH_VALUES) <= NOISE_VH_MAX
    if not low_vh.any():
        print(f"No VH <= {NOISE_VH_MAX} mV in this run, skipping the noisy-pixel cut")
        return np.zeros((NROWS, NCOLS), dtype=bool)
    n = nhits[low_vh]
    mean = np.nanmean(n, axis=(1, 2), keepdims=True)
    std = np.nanstd(n, axis=(1, 2), keepdims=True)
    noisy = (n > mean + 5 * std).any(axis=0)
    print(f"Removing {noisy.sum()} noisy pixels (hits > mean + 5 sigma at VH <= {NOISE_VH_MAX} mV): "
          f"{[(int(c), int(r)) for r, c in np.argwhere(noisy)]}")
    return noisy


#-----Keeps a pixel's ToT at a VH only if its hit count there is within HITS_TOLERANCE of N_INJECTIONS-----
def keep_full_hits(means, nhits):
    full = np.abs(nhits - N_INJECTIONS) <= HITS_TOLERANCE * N_INJECTIONS
    for vh, n_full, n_fit in zip(VH_VALUES, full.sum(axis=(1, 2)), (~np.isnan(means)).sum(axis=(1, 2))):
        print(f"VH = {vh} mV: {n_full} pixels with {N_INJECTIONS} +- {HITS_TOLERANCE:.0%} hits ({n_fit} with a ToT fit)")
    means = means.copy()
    means[~full] = np.nan
    return means


#-----ToT vs VH surrogate function, Eq. (1) with VH in place of the injected charge-----
def surrogate(vh, a, b, c, d):
    return a * vh + b - c / (vh - d)


# Fits the surrogate to one pixel's mean ToT [µs] for VH <= FIT_VH_MAX. Returns the parameters, or None.
def fit_surrogate(vh, tot):
    sel = ~np.isnan(tot) & (vh <= FIT_VH_MAX)
    if sel.sum() < 5:
        return None
    x, y = vh[sel], tot[sel]
    # d acts as the threshold, so it has to stay below the lowest VH where the pixel fires
    p0 = [(y[-1] - y[0]) / (x[-1] - x[0]), 0, 10, x[0] - 20]
    lower, upper = [0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, x[0] - 1e-3]
    try:
        popt, _ = curve_fit(surrogate, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
    except RuntimeError:
        return None
    names = ["a", "b", "c", "d"]
    for name, value, lo, hi in zip(names, popt, lower, upper):
        if np.isclose(value, lo, atol=1e-6) or np.isclose(value, hi, atol=1e-3):
            print(f"WARNING: fit parameter {name} = {value:.4g} is at its limit, the fit does not describe the data well")
    return popt


#-----All pixels on one plot, with the HIGHLIGHT pixels and their surrogate fits on top-----
# All pixels are one point each (the mean of their Gaussian fit). If all_hits is True, the HIGHLIGHT pixels are shown
# with every single hit instead of their mean. The fits always use the Gaussian means.
def plot_all_pixels(means, all_hits=False):
    vh = np.array(VH_VALUES)
    plt.figure("ToT vs VH all pixels", figsize=(8, 6))
    jitter_scatter(vh, means, ms=DOT_SIZE)
    # the same sideways spread as jitter_scatter, for the highlighted pixels' hits
    gaps = np.diff(vh)
    half_width = 0.15 * np.minimum(np.minimum(np.append(gaps, gaps[-1]), np.insert(gaps, 0, gaps[0])), np.median(gaps))
    rng = np.random.default_rng(1)

    for (col, row), color, ls in zip(HIGHLIGHT, HIGHLIGHT_COLORS, ["-", "--", ":"]):
        tot = means[:, row, col]
        if not all_hits:
            plt.plot(vh, tot, "o", color=color, ms=6, label=f"Pixel ({col}, {row})")
        else:
            for i, (x0, w) in enumerate(zip(vh, half_width)):
                _, _, hits = load_run(VCASB, x0, col, row, DATA)
                plt.plot(x0 + rng.uniform(-w, w, hits.size), hits, ".", color=color, ms=1.5, mew=0, alpha=0.3,
                         rasterized=True)
            # a legend entry with a visible marker, the hits themselves are tiny and transparent
            plt.plot([], [], "o", color=color, ms=6, label=f"Pixel ({col}, {row}), all hits")

        popt = fit_surrogate(vh.astype(float), tot)
        if popt is None:
            print(f"WARNING: surrogate fit failed for px ({col}, {row})")
            continue
        a, b, c, d = popt
        print(f"px ({col}, {row}) fit: a = {a:.4f} µs/mV, b = {b:.2f} µs, c = {c:.1f} µs*mV, d = {d:.1f} mV")
        fit_end = min(FIT_VH_MAX, vh[-1])  # the scan can end before FIT_VH_MAX
        x_fit = np.linspace(max(d + 1, vh[~np.isnan(tot)].min() - 20), fit_end, 200)
        plt.plot(x_fit, surrogate(x_fit, *popt), ls, color="black", label=f"Pixel ({col}, {row}) fit")
        if vh[-1] > FIT_VH_MAX:
            x_ext = np.linspace(FIT_VH_MAX, vh[-1], 100)  # extrapolation beyond the fit range
            plt.plot(x_ext, surrogate(x_ext, *popt), ls, color="black", alpha=0.3)

    plt.ylim(bottom=0)
    plt.xlabel("Vh [mV]")
    plt.ylabel("ToT [µs]" if all_hits else "Mean ToT [µs]")
    title = f"Mean ToT vs VH, pixels with {N_INJECTIONS} +- {HITS_TOLERANCE:.0%} hits" if FULL_HITS_ONLY else "Mean ToT vs VH, all pixels"
    if all_hits:
        title = "ToT vs VH: mean ToT of all pixels, every hit of the highlighted pixels"
    plt.title(f"{title}\n{SETTINGS}")
    # the pixels first, then the fits
    handles, labels = plt.gca().get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: labels[k].endswith(" fit"))
    plt.legend([handles[k] for k in order], [labels[k] for k in order])
    plt.grid(alpha=0.3)
    output_file = OUTPUT_DIR / f"tot_vs_vh_{FILE_TAG}_all_pixels{TAG}{'_hits' if all_hits else ''}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")




#-----------Main function where we call everything-----------
def main():
    scan = fit_dataset(ARG) if DATASET else fit_scan([(VCASB, vh) for vh in VH_VALUES], f"vcasb_{VCASB}_vh_scan")
    means, nhits = scan["means"].copy(), scan["nhits"]  # indexed [vh, row, col]

    noisy = find_noisy_pixels(nhits)
    means[:, noisy] = np.nan
    if FULL_HITS_ONLY:
        means = keep_full_hits(means, nhits)

    plot_all_pixels(means)

    # the same plot with every single hit of the highlighted pixels
    plot_all_pixels(means, all_hits=True)

    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")


if __name__ == "__main__":
    main()
