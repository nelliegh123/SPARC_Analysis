#THis code will combine the results from injection.py and iron55.py.
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

from injection import DATA_DIR, HIGHLIGHTED_PIXELS, curved_fit, load_calibration, tot_to_vh
from iron55 import OVERLAY_PIXELS, TOT_BINS, fit_peak, gaussian, load_fe_peaks, load_hits
from sparc_analyzer import SPARCAnalyser


Q_FE = 1610 * e  # [C]
Q_FE_2 = 1780 * e
CSV_PATH = DATA_DIR.parent / "calibration.csv"  # col,row,slope [e/ns],offset [e]

def tot_to_charge(tot_ns):
    s = a_bar * Q_thr + b
    u = s**2 + a_bar * c_bar - s * tot_ns
    charge = Q_thr + (2 * c_bar * tot_ns) / (sqrt(4 * a_bar * s * c_bar * tot_ns + u**2) + u)
    return charge

def vh_to_electrons(vh, capacitance):
    """Returns the charge [electrons] from Q=C*Vh, with Vh in mV and the capacitance in fF"""
    return capacitance * 1e-15 * vh * 1e-3 / e


#Step1: Use the per pixel fit of ToT vs Vh to calibrate the Iron55 from being in units of ToT, to being in units of Vh.
calibration = load_calibration()
hits = load_hits()
mean_tot_fe, _, _, _ = load_fe_peaks(hits)  # The Fe peak is fitted in ToT by iron55.py, indexed [col, row]
cols, rows = np.indices(mean_tot_fe.shape)
vh_fe = tot_to_vh(mean_tot_fe, cols, rows, calibration)

hits["vh"] = tot_to_vh(hits["tot"].to_numpy() * 1e-3, hits["column"].to_numpy(), hits["row"].to_numpy(), calibration)  # ns to µs
hits = hits.dropna(subset=["vh"])  # Pixels without a ToT vs Vh fit


#Step 2: calculate the capacitance of each pixel.
capacitance = Q_FE / (vh_fe * 1e-3) * 1e15  # [fF]

print(f"Fitted {np.count_nonzero(np.isfinite(capacitance))} pixels")
print(f"Vh of the Fe peak: median {np.nanmedian(vh_fe):.0f} mV, spread between pixels {np.nanstd(vh_fe):.0f} mV")
print(f"Capacitance: median {np.nanmedian(capacitance):.3f} fF, spread between pixels {np.nanstd(capacitance):.3f} fF")


#Step 3: convert the injected charge from vh to electrons using Q=C_pixel*Vh
vhs, means, sigmas = calibration["vhs"], calibration["means"], calibration["sigmas"]
slopes, intercepts = calibration["slopes"], calibration["intercepts"]
electrons = vh_to_electrons(vhs[:, None, None], capacitance)

pixel_electrons_per_mv = vh_to_electrons(1, capacitance)  # [e/mV], indexed [col, row]
slopes_e = pixel_electrons_per_mv / slopes * 1e-3  # [e/ns], µs to ns
offsets_e = -pixel_electrons_per_mv * intercepts / slopes  # [e]
valid = np.isfinite(slopes_e) & np.isfinite(offsets_e)
average_slope_e, average_offset_e = slopes_e[valid].mean(), offsets_e[valid].mean()  # Best effort for pixels without a calibration

with open(CSV_PATH, "w") as csv_file:
    csv_file.write("col,row,slope [e/ns],offset [e]\n")
    csv_file.write(f"-1,-1,{average_slope_e:.6g},{average_offset_e:.6g}\n")
    for col, row in zip(*np.nonzero(valid)):
        csv_file.write(f"{col},{row},{slopes_e[col, row]:.6g},{offsets_e[col, row]:.6g}\n")
print(f"Wrote the calibration of {valid.sum()} pixels to {CSV_PATH}, average {average_slope_e:.3g} e/ns, {average_offset_e:.3g} e")


#Step 4: convert the iron55 spectra from ToT to electrons, using Q=C_pixel*Vh
hits["electrons"] = vh_to_electrons(hits["vh"].to_numpy(), capacitance[hits["column"].to_numpy(), hits["row"].to_numpy()])



#---Plottign------
#Plotting the mean ToT/pixel vs the injected chage
plt.rcParams["font.size"] = 13
plt.figure("ToT vs charge", figsize=(8, 6))
plt.plot(electrons.ravel(), means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")
for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
    pixel_electrons = electrons[:, pixel_col, pixel_row]
    plt.errorbar(pixel_electrons, means[:, pixel_col, pixel_row], yerr=sigmas[:, pixel_col, pixel_row], fmt="o", color=color, ms=6, capsize=3, label=f"Pixel ({pixel_col}, {pixel_row})")
    plt.plot(pixel_electrons, curved_fit(vhs, slopes[pixel_col, pixel_row], intercepts[pixel_col, pixel_row], calibration["cs"][pixel_col, pixel_row], calibration["thresholds"][pixel_col, pixel_row]), "k--", lw=1.5)
plt.ylim(bottom=0)
plt.grid(alpha=0.5)
plt.xlabel("Injected charge [electrons]")
plt.ylabel("Mean ToT [µs]")
plt.title("sparc7 injection", fontsize=11)
plt.legend()
plt.tight_layout()



#Plotting the iron55 spectra in electrons
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
    pixel_capacitance = capacitance[pixel_col, pixel_row]
    electrons_per_us = vh_to_electrons(1 / calibration["slopes"][pixel_col, pixel_row], pixel_capacitance)

    def tot_to_electrons(tot):
        return vh_to_electrons(tot_to_vh(tot, pixel_col, pixel_row, calibration), pixel_capacitance)

    # The peak is fitted in ToT by iron55.py, and the histogram and the fit are then converted to electrons
    amplitude1, mean1, sigma1, *_ = fit_peak(pixel_tots, pixel_col, pixel_row)
    counts, _ = np.histogram(pixel_tots, bins=TOT_BINS)
    plt.stairs(counts, tot_to_electrons(TOT_BINS), fill=True, color=color, alpha=0.5, label=f"Pixel ({pixel_col}, {pixel_row})")
    plt.plot(tot_to_electrons(x), gaussian(x, amplitude1, mean1, sigma1), color=color, lw=2, label=f"Fit: {tot_to_electrons(mean1):.0f} ± {sigma1 * electrons_per_us:.0f} electrons")
plt.grid(alpha=0.5)
plt.xlabel("Charge [electrons]")
plt.ylabel("Entries")
plt.title("sparc7 iron55", fontsize=11)
plt.legend()
plt.tight_layout()


#Step 5: overlay the uncalibrated and calibrated iron55 spectra.
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

top_axis = ax.secondary_xaxis("top", functions=(lambda electrons: electrons / electrons_per_mv, lambda vh: vh * electrons_per_mv))
top_axis.set_xlabel("Vh [mV]")
fig.tight_layout()

plt.show()