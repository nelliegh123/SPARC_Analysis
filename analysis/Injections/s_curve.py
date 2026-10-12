import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from injection_scan import fit_scan, complete_vh, ROOT, NROWS, NCOLS

"""
S-curves: the number of hits of every pixel vs the injection voltage Vh, around the threshold.
Each pixel's S-curve is fitted with an error function; its 50 % point is the pixel's threshold and its width
the noise (both in mV of Vh). The Vcasb and the Vh values are read from the file names in DATA; only Vh values
whose pixel files are all there are used, so this also works while a scan is still running.
"""

#-----------Settings-----------
DATA = ROOT / "data" / "sparc7" / "s_curve_full"     
OUTPUT_DIR = ROOT / "plots" / "sparc7" / "s_curve"  
N_INJECTIONS = 25  
VH_MAX = 300      
HIGHLIGHT = [(6, 6), (10, 19)]  # (col, row), the same pixels as in ToT_vh.py
HIGHLIGHT_COLORS = ["#f5a623", "#d55e00"]
#------------------------------


#-----The Vcasb of the scan-----
def find_vcasb():
    vcasbs = {int(f.name.split("_vcasb_")[1].split("_")[0]) for f in DATA.glob("data_run_vcasb_*_vh_*_px_*.raw")}
    if len(vcasbs) != 1:
        raise SystemExit(f"Expected the files of one Vcasb in {DATA}, found {sorted(vcasbs) or 'none'}")
    return vcasbs.pop()


#-----S-curve: 0 hits far below the threshold, N_INJECTIONS far above it-----
def s_curve(vh, threshold, noise):
    return 0.5 * N_INJECTIONS * (1 + erf((vh - threshold) / (np.sqrt(2) * noise)))


# Fits one pixel's hits vs Vh. Returns (threshold, noise) in mV, or (nan, nan) if the pixel never crosses 50 %.
def fit_s_curve(vh, hits):
    ok = ~np.isnan(hits)
    x = vh[ok]
    # extra hits from double pulsing (a few % above N_INJECTIONS) would pull the fit, so cap at N_INJECTIONS
    y = np.minimum(hits[ok], N_INJECTIONS)
    if y.max() < 0.5 * N_INJECTIONS or y.min() > 0.5 * N_INJECTIONS:
        return np.nan, np.nan
    first_above = x[np.argmax(y >= 0.5 * N_INJECTIONS)]
    try:
        popt, _ = curve_fit(s_curve, x, y, p0=[first_above, 10], bounds=([0, 0.1], [x.max(), x.max()]))
    except RuntimeError:
        return np.nan, np.nan
    return popt


#-----------All S-curves on top of each other, the HIGHLIGHT pixels with their fit------------
def plot_s_curves(vh, hits, thresholds, noises, title, output_file):
    plt.figure('S-curves', figsize=(9, 6))
    for row in range(NROWS):
        for col in range(NCOLS):
            plt.plot(vh, hits[:, row, col], color='#2a9df4', linewidth=0.5, alpha=0.3,
                     label='All pixels' if (row, col) == (0, 0) else None)
    x_fit = np.linspace(0, vh.max(), 500)
    for (col, row), color in zip(HIGHLIGHT, HIGHLIGHT_COLORS):
        plt.plot(vh, hits[:, row, col], 'o', color=color, ms=6, label=f'Pixel ({col}, {row})')
        thr, noise = thresholds[row, col], noises[row, col]
        if not np.isnan(thr):
            plt.plot(x_fit, s_curve(x_fit, thr, noise), '-', color=color, linewidth=1.5,
                     label=f'Pixel ({col}, {row}) fit: threshold {thr:.0f} mV, noise {noise:.1f} mV')
    plt.axhline(N_INJECTIONS, color='black', linestyle='--', linewidth=1, label=f'{N_INJECTIONS} injections')
    plt.axhline(0.5 * N_INJECTIONS, color='gray', linestyle=':', linewidth=1, label='50 % (threshold)')
    plt.xlim(0, vh.max())
    plt.xlabel('Vh [mV]')
    plt.ylabel('Number of hits')
    plt.title(title)
    plt.legend(loc='center right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")


#-----------Main function where we call everything-----------
def main():
    vcasb = find_vcasb()
    vh_values = [vh for vh in complete_vh(DATA, vcasb) if vh <= VH_MAX]
    if not vh_values:
        raise SystemExit(f"No complete Vh values (up to {VH_MAX} mV) in {DATA} yet")
    print(f"{DATA}: Vcasb = {vcasb} mV, complete Vh values {vh_values}")

    scan = fit_scan([(vcasb, vh) for vh in vh_values], f"vcasb_{vcasb}_vh_scan", data_dir=DATA)
    hits = scan["nhits"]  # indexed [vh, row, col]

    vh = np.array(vh_values, dtype=float)
    thresholds = np.full((NROWS, NCOLS), np.nan)
    noises = np.full((NROWS, NCOLS), np.nan)
    for row in range(NROWS):
        for col in range(NCOLS):
            thresholds[row, col], noises[row, col] = fit_s_curve(vh, hits[:, row, col])

    ok = ~np.isnan(thresholds)
    if not ok.any():
        print(f"S-curve fits: none, no pixel reaches 50 % of its {N_INJECTIONS} injections")
    else:
        print(f"S-curve fits: {ok.sum()} of {ok.size} pixels; threshold median {np.median(thresholds[ok]):.1f} mV "
              f"(16-84 %: {np.percentile(thresholds[ok], 16):.1f}-{np.percentile(thresholds[ok], 84):.1f} mV), "
              f"noise median {np.median(noises[ok]):.1f} mV")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title = f"S-curves, all pixels\n{DATA.parent.name}/{DATA.name}, Vcasb = {vcasb} mV, {N_INJECTIONS} injections per pixel"
    plot_s_curves(vh, hits, thresholds, noises, title, OUTPUT_DIR / f"s_curves_{DATA.name}_vcasb_{vcasb}.png")


if __name__ == "__main__":
    main()
