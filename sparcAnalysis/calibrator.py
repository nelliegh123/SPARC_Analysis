import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e
from scipy.optimize import OptimizeWarning, curve_fit

from injection import DATA_DIR, HIGHLIGHTED_PIXELS, curved_fit, load_calibration, tot_to_vh
from iron55 import OVERLAY_PIXELS, TOT_BINS, fit_peak, gaussian, load_fe_peaks, load_hits


Q_FE = 1610 * e  # [C]
Q_FE_2 = 1780 * e  # [C]

CSV_PATH = DATA_DIR.parent / "calibration.csv"  # One row per pixel: the curved_fit parameters and the capacitance, see Step 3

def linear_fit(x, slope):
    return slope * x

calibration = load_calibration()
hits = load_hits()
hits["tot_us"] = hits["tot"] * 1e-3  # ns to µs
mean1, sigma1, mean2, sigma2 = load_fe_peaks(hits)


#Step 1: convert iron ToT to Vh using tot_to_vh
vh1_iron, vh2_iron = np.full(mean1.shape, np.nan), np.full(mean2.shape, np.nan)  # [mV], indexed [col, row]
for col, row in np.ndindex(mean1.shape):
    tot1, tot2 = mean1[col][row], mean2[col][row]
    vh1_iron[col][row] = tot_to_vh(tot1, col, row, calibration)
    vh2_iron[col][row] = tot_to_vh(tot2, col, row, calibration)


#Step 2: for each pixel fit Q vs ToT. Vh = slope * Q, a line through zero
slopes_q = np.full(mean1.shape, np.nan)  # [mV/e], indexed [col, row]
for col, row in np.ndindex(mean1.shape):
    x = (Q_FE / e, Q_FE_2 / e)  # [electrons]
    y = (vh1_iron[col][row], vh2_iron[col][row])
    if not np.all(np.isfinite(y)):  # No Fe peaks in this pixel
        continue
    (slope,), _ = curve_fit(linear_fit, x, y)
    slopes_q[col][row] = slope
C = 1 / slopes_q  # [e/mV]

# Safety check against a bad Fe peak fit: keep pixels within 50 % of the median C
outlier = np.abs(C / np.nanmedian(C) - 1) > 0.5
print()
print(f"Removing {np.count_nonzero(outlier)} pixels with C more than 50 % from the median")
C[outlier] = np.nan
print(f"C: median {np.nanmedian(C):.3f} e/mV, spread between pixels {np.nanstd(C):.3f} e/mV, from {np.count_nonzero(np.isfinite(C))} pixels")
print()


#Step 3: Use the ToT to charge conversion as

#convert a => ā = a / C_inj, b => b̄, c => c̄ = c * C_inj, V_h,thr => Q_thr = V_h,thr * C_inj
# The line from Step 2 passes through zero, so b̄ = b.
# The µs from the injection fit are converted to ns, so the parameters match tot_ns.
a_bar = calibration["slopes"] / C * 1e3  # [ns/e]
b_bar = calibration["intercepts"] * 1e3 #+ 0 * C  # [ns], + 0 * C so that pixels without a C are NaN here too
c_bar = calibration["cs"] * C * 1e3  # [ns*e]
Q_thr = calibration["thresholds"] * C  # [e]

def tot_to_charge(tot_ns, a_bar, b, c_bar, Q_thr):
    s = a_bar * Q_thr + b
    u = s**2 + a_bar * c_bar - s * tot_ns
    charge = Q_thr + (2 * c_bar * tot_ns) / (np.sqrt(4 * a_bar * s * c_bar * tot_ns + u**2) + u)
    return charge



#Step 4: save the data in a csv file
pixel_params = np.array([a_bar, b_bar, c_bar, Q_thr])  
valid = np.all(np.isfinite(pixel_params), axis=0)  
average_params = np.mean(pixel_params[:, valid], axis=1) 
# average_params = np.median(pixel_params[:, valid], axis=1) 

with open(CSV_PATH, "w") as csv_file:
    csv_file.write("col,row,slope [ns/e],offset [ns],bend [ns*e],threshold [e]\n")
    csv_file.write("-1,-1," + ",".join(f"{p:.6g}" for p in average_params) + "\n")
    for col, row in zip(*np.nonzero(valid)):
        csv_file.write(f"{col},{row}," + ",".join(f"{p:.6g}" for p in pixel_params[:, col, row]) + "\n")
print(f"Wrote the calibration of {valid.sum()} pixels to {CSV_PATH}, the other {np.count_nonzero(~valid)} pixels use the average")
filled_params = np.where(valid, pixel_params, average_params[:, None, None])  


#Step5: USe the ToT_to_charge to plot th calibrated iron spectra (in units of e) and the calibrated ToT vs Vh. scan (in units of e)
hit_cols, hit_rows = hits["column"].to_numpy(), hits["row"].to_numpy()
hits["electrons"] = tot_to_charge(hits["tot"].to_numpy(), *filled_params[:, hit_cols, hit_rows])

plt.rcParams["font.size"] = 13
plt.figure("Charge distribution", figsize=(8, 6))
plt.hist(hits["electrons"], bins=120, range=(0, 2500), color="#2a9df4")
plt.grid(alpha=0.5)
plt.xlabel("Charge [electrons]")
plt.ylabel("Entries")
plt.title("sparc7 iron55, all pixels", fontsize=11)
plt.tight_layout()

# The injected charge of every Vh point: Q = Vh * C, with each pixel's own C from Step 2
vhs, means, mean_errors = calibration["vhs"], calibration["means"], calibration["mean_errors"]
injected_electrons = vhs[:, None, None] * C  # [electrons], indexed [vh, col, row]

plt.figure("ToT vs injected charge", figsize=(8, 6))
plt.plot(injected_electrons.ravel(), means.ravel(), ".", color="#2a9df4", alpha=0.5, ms=8, mew=0, label="All pixels")
for (pixel_col, pixel_row), color in HIGHLIGHTED_PIXELS.items():
    if not valid[pixel_col, pixel_row]:
        print(f"Pixel ({pixel_col}, {pixel_row}): no calibration, not highlighted")
        continue
    pixel_electrons = injected_electrons[:, pixel_col, pixel_row]
    plt.errorbar(pixel_electrons, means[:, pixel_col, pixel_row], yerr=mean_errors[:, pixel_col, pixel_row], fmt="o", color=color, ms=6, capsize=3, label=f"Pixel ({pixel_col}, {pixel_row})")
    fit_electrons = np.linspace(Q_thr[pixel_col, pixel_row], np.nanmax(pixel_electrons), 300)  # From the threshold (ToT = 0), below it the curve goes through its pole
    plt.plot(fit_electrons, curved_fit(fit_electrons, *pixel_params[:, pixel_col, pixel_row]) * 1e-3, "--", color=color, lw=1.5)  # ns to µs
plt.ylim(bottom=0)
plt.grid(alpha=0.5)
plt.xlabel("Injected charge [electrons]")
plt.ylabel("Mean ToT [µs]")
plt.title("sparc7 injection", fontsize=11)
plt.legend()
plt.tight_layout()

plt.show()
