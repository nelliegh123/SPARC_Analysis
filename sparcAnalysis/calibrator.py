#THis code will combine the results from injection.py and iron55.py.
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

from injection import HIGHLIGHTED_PIXELS, calibrate, fit, tot_to_vh
from iron55 import OVERLAY_PIXELS, TOT_BINS, fit_all_pixels as fit_fe_peaks, fit_peak, gaussian, load_hits
from sparc_analyzer import SPARCAnalyser

Q_FE = 1600 * e  # [C]


def vh_to_electrons(vh, capacitance):
    """Returns the charge [electrons] from Q=C*Vh, with Vh in mV and the capacitance in fF"""
    return capacitance * 1e-15 * vh * 1e-3 / e


#Step1: Use the per pixel fit of ToT vs Vh to calibrate the Iron55 from being in units of ToT, to being in units of Vh.
calibration = calibrate()
hits = load_hits()
mean_tot_fe, _ = fit_fe_peaks(hits)  # [µs] The Fe peak is fitted in ToT by iron55.py, indexed [row, col]
rows, cols = np.indices(mean_tot_fe.shape)
vh_fe = tot_to_vh(mean_tot_fe, cols, rows, calibration)  # [mV], indexed [row, col]

hits["vh"] = tot_to_vh(hits["tot"].to_numpy() * 1e-3, hits["column"].to_numpy(), hits["row"].to_numpy(), calibration)  # ns to µs
hits = hits.dropna(subset=["vh"])  # Pixels without a ToT vs Vh fit

#Step 2: calculate the capacitance of each pixel.
# This we can do by using the relation C=Q/Vh. So we use the chare Q_Fe=1600 electrons, and the Vh values that we have extracted from fitting the iron 55 spectra of each pixel, after changign the x axis to Vh instead of ToT.
capacitance = Q_FE / (vh_fe * 1e-3) * 1e15  # [fF], indexed [row, col]

print(f"Fitted {np.count_nonzero(np.isfinite(capacitance))} pixels")
print(f"Vh of the Fe peak: median {np.nanmedian(vh_fe):.0f} mV, spread between pixels {np.nanstd(vh_fe):.0f} mV")
print(f"Capacitance: median {np.nanmedian(capacitance):.3f} fF, spread between pixels {np.nanstd(capacitance):.3f} fF")

#Step 3: plot the mean ToT of every pixel vs the injected charge, using Q=C_pixel*Vh
vhs, means, sigmas = calibration["vhs"], calibration["means"], calibration["sigmas"]  # means and sigmas are indexed [vh, row, col]
slopes, intercepts = calibration["slopes"], calibration["intercepts"]
electrons = vh_to_electrons(vhs[:, None, None], capacitance)  # [vh, row, col]

plt.rcParams["font.size"] = 13

plt.figure("ToT vs charge", figsize=(8, 6))
plt.plot(electrons.ravel(), means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")
for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
    pixel_electrons = electrons[:, pixel_row, pixel_col]
    plt.errorbar(pixel_electrons, means[:, pixel_row, pixel_col], yerr=sigmas[:, pixel_row, pixel_col], fmt="o", color=color, ms=6, capsize=3, label=f"Pixel ({pixel_col}, {pixel_row})")
    plt.plot(pixel_electrons, fit(vhs, slopes[pixel_row, pixel_col], intercepts[pixel_row, pixel_col]), "k--", lw=1.5)
plt.ylim(bottom=0)
plt.grid(alpha=0.5)
plt.xlabel("Injected charge [electrons]")
plt.ylabel("Mean ToT [µs]")
plt.title("sparc7 injection", fontsize=11)
plt.legend()
plt.tight_layout()

#Step 4: plot the iron55 spectra in electrons, using Q=C_pixel*Vh
hits["electrons"] = vh_to_electrons(hits["vh"].to_numpy(), capacitance[hits["row"].to_numpy(), hits["column"].to_numpy()])

plt.figure("Charge distribution", figsize=(8, 6))
plt.hist(hits["electrons"], bins=120, range=(0, 2500), color="#2a9df4")
plt.grid(alpha=0.5)
plt.xlabel("Charge [electrons]")
plt.ylabel("Entries")
plt.title("sparc7 iron55, all pixels", fontsize=11)
plt.tight_layout()

plt.figure("Overlayed charge distribution", figsize=(8, 6))
x = np.linspace(TOT_BINS[0], TOT_BINS[-1], 1000)  # [µs]
for (pixel_col, pixel_row), color in OVERLAY_PIXELS.items():
    pixel_tots = hits[(hits["column"] == pixel_col) & (hits["row"] == pixel_row)]["tot"].to_numpy() * 1e-3  # ns to µs
    pixel_capacitance = capacitance[pixel_row, pixel_col]
    electrons_per_us = vh_to_electrons(1 / calibration["slopes"][pixel_row, pixel_col], pixel_capacitance)

    def tot_to_electrons(tot):
        return vh_to_electrons(tot_to_vh(tot, pixel_col, pixel_row, calibration), pixel_capacitance)

    # The peak is fitted in ToT by iron55.py, and the histogram and the fit are then converted to electrons
    amplitude, mean, sigma = fit_peak(pixel_tots)
    counts, _ = np.histogram(pixel_tots, bins=TOT_BINS)
    plt.stairs(counts, tot_to_electrons(TOT_BINS), fill=True, color=color, alpha=0.5, label=f"Pixel ({pixel_col}, {pixel_row})")
    plt.plot(tot_to_electrons(x), gaussian(x, amplitude, mean, sigma), color=color, lw=2, label=f"Fit: {tot_to_electrons(mean):.0f} ± {sigma * electrons_per_us:.0f} electrons")
plt.grid(alpha=0.5)
plt.xlabel("Charge [electrons]")
plt.ylabel("Entries")
plt.title("sparc7 iron55", fontsize=11)
plt.legend()
plt.tight_layout()

#Step 5: overlay the uncalibrated and calibrated iron55 spectra. The uncalibrated ToT has no charge scale, so all its hits use the same conversion: the median fit and the median capacitance
median_capacitance = np.nanmedian(capacitance)
uncalibrated_vh = (hits["tot"].to_numpy() * 1e-3 - np.nanmedian(intercepts)) / np.nanmedian(slopes)  # ns to µs
electrons_per_mv = vh_to_electrons(1, median_capacitance)

fig, ax = plt.subplots(num="Uncalibrated vs calibrated", figsize=(8, 6))
ax.hist(uncalibrated_vh * electrons_per_mv, bins=120, range=(0, 2500), color="#e0479e", alpha=0.5, label="Uncalibrated (same conversion for all pixels)")
ax.hist(hits["electrons"], bins=120, range=(0, 2500), histtype="step", color="#2a9df4", lw=2, label="Calibrated (per pixel)")
ax.grid(alpha=0.5)
ax.set_xlabel("Charge [electrons]")
ax.set_ylabel("Entries")
ax.set_title("sparc7 iron55, all pixels", fontsize=11)
ax.legend()

# The top axis uses the median capacitance, because every calibrated pixel has its own conversion between electrons and Vh
top_axis = ax.secondary_xaxis("top", functions=(lambda electrons: electrons / electrons_per_mv, lambda vh: vh * electrons_per_mv))
top_axis.set_xlabel("Vh [mV]")
fig.tight_layout()

plt.show()
