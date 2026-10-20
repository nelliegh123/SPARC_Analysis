import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from injection_scan import fit_scan, jitter_scatter, load_run, raw_file, DATA_DIR, PLOT_DIR, DATA_VH

"""
Plots the mean ToT vs Vh of every pixel, with two highlighted pixels and a fit to them.
"""

VCASB = 300
DATA = DATA_DIR
OUT = PLOT_DIR / "tot_vh" / DATA.name
NAME = DATA.name  
OUT.mkdir(parents=True, exist_ok=True)

PIXELS = [(6, 6), (19, 19)]  # (col, row)
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


def plot(means, all_hits):
    plt.figure(figsize=(8, 6))
    jitter_scatter(VH, means, ms=14)
    rng = np.random.default_rng(1)

    for (col, row), color in zip(PIXELS, COLORS):
        if all_hits: 
            gaps = np.diff(VH)
            half_width = 0.15 * np.minimum(np.minimum(np.append(gaps, gaps[-1]), np.insert(gaps, 0, gaps[0])), np.median(gaps))
            for vh, w in zip(DATA_VH, half_width):
                if not raw_file(VCASB, vh, col, row, DATA).exists():
                    continue
                tot = load_run(VCASB, vh, col, row, DATA)[2]
                plt.plot(vh + rng.uniform(-w, w, tot.size), tot, ".", color=color, ms=1.5, mew=0, alpha=0.3,
                         rasterized=True)
            plt.plot([], [], "o", color=color, ms=6, label=f"Pixel ({col}, {row})")
        else:
            plt.plot(VH, means[:, row, col], "o", color=color, ms=6, label=f"Pixel ({col}, {row})")

    for (col, row), ls in zip(PIXELS, ["-", "--"]):
        if np.isnan(means[:, row, col]).sum() > len(VH) - 5:
            print(f"Too few points to fit px ({col}, {row})")
            continue
        popt, first_vh = fit(means[:, row, col])
        print(f"px ({col}, {row}) fit: a, b, c, d = {np.round(popt, 4)}")
        x = np.linspace(max(popt[3] + 1, first_vh - 20), VH[-1], 200)
        plt.plot(x, func(x, *popt), ls, color="black", label=f"Pixel ({col}, {row}) fit")

    plt.ylim(bottom=0)
    plt.xlabel("Vh [mV]")
    plt.ylabel("ToT [µs]")

    for handle in plt.legend(framealpha=0.95, edgecolor="dimgray").legend_handles:
        handle.set_alpha(1)
    out = OUT / f"tot_vs_vh_{NAME}_all_pixels{'_hits' if all_hits else ''}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


scan = fit_scan([(VCASB, vh) for vh in DATA_VH], f"vcasb_{VCASB}_vh_scan", data_dir=DATA)
means, nhits = scan["means"].copy(), scan["nhits"]  # [vh, row, col]

# # noisy pixels: more hits than mean + 5 sigma at Vh <= 50 mV, where the injection is too small to fire anything
# low = nhits[VH <= 50]
# means[:, (low > np.nanmean(low, axis=(1, 2), keepdims=True) + 5 * np.nanstd(low, axis=(1, 2), keepdims=True)).any(0)] = np.nan

plot(means, all_hits=False)
plot(means, all_hits=True)
