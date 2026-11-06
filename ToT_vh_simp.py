"""
Plots the median ToT of every pixel of the finished sparc6 injection scan:
    1. ToT vs Vh, with all pixels, the median of all pixels and a few highlighted pixels
    2. A map of the ToT of every pixel at one Vh

The ToT values were already calculated with DoublePulsingAnalyser and saved in CACHE_FILE,
so this script only reads that file and never analyses the raw data.

"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "sparcCache/sparc6/injection_double_pulsing_analyzer.npz"
PLOT_DIR = ROOT / "sparcPlots/sparc6/ToT_vh"
PLOT_FILE = PLOT_DIR / "tot_vs_vh_injection_simp.png"
MAP_PLOT_FILE = PLOT_DIR / "tot_map_injection_simp.png"

NROWS, NCOLS = 32, 28
MIN_HITS = 0  # Leave out pixel runs with fewer ToT values than this
MAP_VH = 1000  # The Vh [mV] to show in the pixel map
HIGHLIGHTED_PIXELS = {(17, 20): "#e0479e", (19, 19): "#ffc000"}  # (col, row): color


#-------Read the cached ToT values------
cache = np.load(CACHE_FILE)
names, cached_medians, n_tots = cache["name"], cache["median"], cache["n"]

# The Vh, column and row of each entry are in the file name
settings = [re.search(r"_vh(\d+)_c(\d+)_r(\d+)_", str(name)).groups() for name in names]
vh, col, row = np.array(settings, dtype=int).T
vhs = np.unique(vh)

# Put the median ToTs in an array with the shape [vh, row, col]
medians = np.full((len(vhs), NROWS, NCOLS), np.nan)
enough_hits = n_tots >= MIN_HITS
medians[np.searchsorted(vhs, vh[enough_hits]), row[enough_hits], col[enough_hits]] = cached_medians[enough_hits]
medians[:, 17, 7] = np.nan  # Row 17, column 7 is masked in the analyzer, but was cached before the mask was added


#-------Print the median ToT of all pixels at each Vh------
median_per_vh = np.nanmedian(medians, axis=(1, 2))
for vh_value, values, median in zip(vhs, medians, median_per_vh):
    print(f"Vh = {vh_value:4d} mV: {np.isfinite(values).sum():3d} pixels, median ToT {median:6.3f} µs")


#-------Plot 1: ToT vs Vh------
plt.rcParams["font.size"] = 13
plt.figure(figsize=(8, 6))

# All pixels, spread out a bit in x so that the points do not lie on top of each other
x = np.repeat(vhs, NROWS * NCOLS) + np.random.default_rng(0).uniform(-5, 5, medians.size)
plt.plot(x, medians.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")

plt.plot(vhs, median_per_vh, "-", color="black", lw=1.5, label="Median of all pixels")

for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
    plt.plot(vhs, medians[:, pixel_row, pixel_col], "o-", color=color, ms=6, label=f"Pixel ({pixel_col}, {pixel_row})")

plt.ylim(bottom=0)
plt.grid(alpha=0.5)
plt.xlabel("Vh [mV]")
plt.ylabel("Median ToT [µs]")
plt.title("sparc6 injection, double_pulsing_analyzer", fontsize=11)
plt.legend()
plt.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=150)
print(f"Saved {PLOT_FILE}")


#-------Plot 2: map of the ToT of every pixel at MAP_VH------
plt.figure(figsize=(8, 7))
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("lightgray")  # Pixels without a ToT value
plt.imshow(medians[np.searchsorted(vhs, MAP_VH)], cmap=cmap, interpolation="none", origin="lower")
plt.colorbar(label="Median ToT [µs]")
plt.xlabel("Column")
plt.ylabel("Row")
plt.title(f"sparc6 injection, double_pulsing_analyzer, Vh = {MAP_VH} mV", fontsize=11)
plt.tight_layout()

plt.savefig(MAP_PLOT_FILE, dpi=150)
print(f"Saved {MAP_PLOT_FILE}")

plt.show()
