import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis" / "Injections"))
from double_pulsing_analyzer import DoublePulsingAnalyser
from injection_scan import CACHE_DIR, DATA_VH, NCOLS, NROWS, ROOT, cache_folder, fit_gaussian, gaussian, jitter_scatter, raw_file, tot_bins

"""
Same plot as analysis/Injections/ToT_vh.py, but with double pulses masked before the Gaussian fit.
The ToT is taken from the injection events of DoublePulsingAnalyser instead of the decoder hits:
ToT = largest time difference within the event, i.e. from the last leading edge frame to the falling edge.
Extra frames after the falling edge and double pulses before the leading edge are ignored.
"""

VCASB = 300
DATA = ROOT / "data" / "sparc7" / "injection_pd_8k_ireset_10pA"
PLOT_DIR = ROOT / "plots" / "sparc7" / "TEST"
OUT = PLOT_DIR / "tot_vh" / DATA.name
NAME = DATA.name
OUT.mkdir(parents=True, exist_ok=True)
CACHE = CACHE_DIR / cache_folder(DATA) / f"vcasb_{VCASB}_vh_scan_dp_masked.npz"

N_SIGMA_OUTLIER = 5  # hits further than this many (robust) sigma from the median ToT are removed as outliers
MAX_REL_SIGMA = 0.3  # Gaussian fits with sigma/mean above this are not Gaussian and are thrown away
PIXELS = [(7, 5), (26, 30)]  # (col, row)
COLORS = ["#e0479e", "#ffc000"]
VH = np.array(DATA_VH, dtype=float)
plt.rcParams.update({"font.size": 13})


def func(vh, a, b, c, d):
    return a * vh + b - c / (vh - d)


def fit(tot):
    ok = ~np.isnan(tot)
    x, y = VH[ok], tot[ok]

    p0 = [(y[-1] - y[0]) / (x[-1] - x[0]), 0, 10, x[0] - 20]

    popt, _ = curve_fit(func, x, y, p0=p0, bounds=([0, -np.inf, 0, 0], [np.inf, np.inf, np.inf, x[0] - 1e-3]), maxfev=20000)
    return popt, x[0]


#-------The masked ToTs (µs) of one pixel run without outliers, the number of injection events and of removed outliers------
def masked_tots(vh, col, row):
    ana = DoublePulsingAnalyser(raw_file(VCASB, vh, col, row, DATA))
    ana.do_output = False
    ana.analyze()
    events = ana.events[row, col]
    tots = np.array([np.max(np.diff(event)) * 1e-3 for event in events if len(event) >= 2])
    if tots.size == 0:
        return tots, len(events), 0
    median = np.median(tots)
    robust_std = 1.4826 * np.median(np.abs(tots - median))
    if robust_std == 0:
        return tots, len(events), 0
    ok = np.abs(tots - median) <= N_SIGMA_OUTLIER * robust_std
    return tots[ok], len(events), np.sum(~ok)


def rejected(mean, sigma):
    return np.isnan(mean) or sigma / mean > MAX_REL_SIGMA


#-------Fits the masked ToTs of one pixel, returns the mean (NaN if rejected), the fraction of injection events kept and the number of outliers------
def fit_pixel(args):
    vh, col, row = args
    if not raw_file(VCASB, vh, col, row, DATA).exists():
        return np.nan, np.nan, 0
    tots, n_events, n_outliers = masked_tots(vh, col, row)
    kept = tots.size / n_events if n_events else np.nan
    _, mean, sigma, _ = fit_gaussian(tots, 100)
    return np.nan if rejected(mean, sigma) else mean, kept, n_outliers


if (CACHE.exists() and np.array_equal(np.load(CACHE)["vh"], VH)
        and np.load(CACHE).get("max_rel_sigma") == MAX_REL_SIGMA
        and np.load(CACHE).get("n_sigma_outlier") == N_SIGMA_OUTLIER):
    means, kept, outliers = np.load(CACHE)["means"], np.load(CACHE)["kept"], np.load(CACHE)["outliers"]
    print(f"Loaded cached fits from {CACHE}")
    print()
else:
    jobs = [(vh, col, row) for vh in DATA_VH for row in range(NROWS) for col in range(NCOLS)]
    print(f"Fitting {len(jobs)} pixel runs")
    with ProcessPoolExecutor() as pool:
        results = np.array(list(pool.map(fit_pixel, jobs, chunksize=16)))
    means, kept, outliers = results.T.reshape(3, len(DATA_VH), NROWS, NCOLS)  # [vh, row, col]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, vh=VH, means=means, kept=kept, outliers=outliers,
             max_rel_sigma=MAX_REL_SIGMA, n_sigma_outlier=N_SIGMA_OUTLIER)
    print(f"Saved fits to {CACHE}")

print(f"Removed {int(outliers.sum())} outlier hits (> {N_SIGMA_OUTLIER} sigma) in {int((outliers > 0).sum())} pixel runs")
for col, row in PIXELS:
    per_vh = ", ".join(f"{vh}: {int(n)}" for vh, n in zip(DATA_VH, outliers[:, row, col]) if n > 0)
    print(f"  px ({col}, {row}): {int(outliers[:, row, col].sum())} removed" + (f" (Vh mV: n = {per_vh})" if per_vh else ""))
print()

#-------Fits ToT vs Vh of the highlighted pixels once, used by both plots------
FITS = {}
for col, row in PIXELS:
    if np.isnan(means[:, row, col]).sum() > len(VH) - 5:
        print(f"Too few points to fit px ({col}, {row})")
        continue
    FITS[col, row] = fit(means[:, row, col])
    print(f"px ({col}, {row}) fit: a, b, c, d = {np.round(FITS[col, row][0], 4)}")
print()


def plot(means, all_hits):
    plt.figure(figsize=(8, 6))
    jitter_scatter(VH, means, ms=14)
    rng = np.random.default_rng(1)

    for (col, row), color in zip(PIXELS, COLORS):
        if all_hits:
            gaps = np.diff(VH)
            half_width = 0.15 * np.minimum(np.minimum(np.append(gaps, gaps[-1]), np.insert(gaps, 0, gaps[0])), np.median(gaps))
            for vh, w, mean in zip(DATA_VH, half_width, means[:, row, col]):
                # only the hits of Vh points whose Gaussian fit is used
                if np.isnan(mean):
                    continue
                tot = masked_tots(vh, col, row)[0]
                plt.plot(vh + rng.uniform(-w, w, tot.size), tot, ".", color=color, ms=5, mew=0, alpha=0.3,
                         rasterized=True)
            plt.plot([], [], ".", color=color, ms=14, mew=0, label=f"Pixel ({col}, {row})")
        else:
            plt.plot(VH, means[:, row, col], ".", color=color, ms=14, mew=0, label=f"Pixel ({col}, {row})")

    for (col, row), ls in zip(PIXELS, ["-", "--"]):
        if (col, row) not in FITS:
            continue
        popt, first_vh = FITS[col, row]
        x = np.linspace(max(popt[3] + 1, first_vh - 20), VH[-1], 200)
        plt.plot(x, func(x, *popt), ls, color="black", label=f"Pixel ({col}, {row}) fit")

    plt.ylim(bottom=0)
    plt.gca().set_axisbelow(True)
    plt.grid(alpha=0.5)
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel("Vh [mV]")
    plt.ylabel("ToT [µs]")
    for handle in plt.legend(framealpha=0.95, edgecolor="dimgray").legend_handles:
        handle.set_alpha(1)
    out = OUT / f"tot_vs_vh_{NAME}_all_pixels{'_hits' if all_hits else ''}_dp_masked.png"
    plt.savefig(out, dpi=150)
    plt.show()
    plt.close()
    print(f"Saved {out}")


#-------Shows the Gaussian fits of the masked ToTs of the highlighted pixels, one window per pixel------
def plot_gaussians():
    n_cols = int(np.ceil(np.sqrt(len(DATA_VH))))
    n_rows = int(np.ceil(len(DATA_VH) / n_cols))
    for (col, row), color in zip(PIXELS, COLORS):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows))
        for ax, vh in zip(axes.flat, DATA_VH):
            ax.set_title(f"Vh = {vh} mV", fontsize=10)
            if not raw_file(VCASB, vh, col, row, DATA).exists():
                continue
            tots = masked_tots(vh, col, row)[0]
            if tots.size == 0:
                continue
            ax.hist(tots, tot_bins(tots, 100), color=color, alpha=0.7)
            amp, mean, sigma, chi2_ndf = fit_gaussian(tots, 100)
            if np.isnan(mean):
                continue
            x = np.linspace(tots.min(), tots.max(), 300)
            bad = rejected(mean, sigma)
            ax.plot(x, gaussian(x, amp, mean, sigma), "--" if bad else "-", color="gray" if bad else "black")
            ax.text(0.97, 0.95, f"µ = {mean:.2f} µs\nσ = {sigma:.2f} µs\nχ²/ndf = {chi2_ndf:.1f}" + ("\nrejected" if bad else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.tick_params(labelsize=8)
        for ax in axes.flat[len(DATA_VH):]:
            ax.set_visible(False)
        fig.supxlabel("ToT [µs]")
        fig.supylabel("Counts")
        fig.suptitle(f"Pixel ({col}, {row})")
        fig.tight_layout()
    plt.show()


#-------Fits ToT vs Vh of every pixel and maps the fit parameters over the chip; prints even vs odd columns------
def plot_fit_maps():
    params = np.full((4, NROWS, NCOLS), np.nan)  # a, b, c, d [row, col]
    for row in range(NROWS):
        for col in range(NCOLS):
            if np.isnan(means[:, row, col]).sum() > len(VH) - 5:
                continue
            try:
                params[:, row, col] = fit(means[:, row, col])[0]
            except RuntimeError:
                pass
    print(f"ToT vs Vh fits failed or skipped for {np.isnan(params[0]).sum()} pixels")

    names = ["a [µs/mV]", "b [µs]", "c [µs·mV]", "d [mV]"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, values, name in zip(axes, params, names):
        even, odd = values[:, 0::2], values[:, 1::2]
        print(f"{name:10s} even columns {np.nanmean(even):.4g} ± {np.nanstd(even):.3g}, "
              f"odd columns {np.nanmean(odd):.4g} ± {np.nanstd(odd):.3g}")
        # colour scale from the 1st to 99th percentile, so single outlier pixels don't wash out the map
        vmin, vmax = np.nanpercentile(values, [1, 99])
        im = ax.imshow(values, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=name, shrink=0.8)
        ax.imshow(np.where(np.isnan(values), 1.0, np.nan), origin="lower", cmap=ListedColormap(["black"]))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
    fig.suptitle("ToT vs Vh fit: a·Vh + b − c / (Vh − d)")
    fig.tight_layout()
    out = OUT / f"tot_vh_fit_maps_{NAME}_dp_masked.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


plot(means, all_hits=False)
plot(means, all_hits=True)
plot_fit_maps()
plot_gaussians()


