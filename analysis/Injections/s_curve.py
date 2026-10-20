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
the noise. 
"""

#-----------Settings-----------
DATA = ROOT / "data" / "sparc7" / "s_curve_fine"     
OUTPUT_DIR = ROOT / "plots" / "sparc7" / "s_curve"  
N_INJECTIONS = 1000  
VH_MAX = 300      

HIGHLIGHT = [("lowest threshold", "#e0479e", "s", "--"),                      # pink squares, dashed
             ("average threshold", "#e41a1c", "^", ":"),                      # red triangles, dotted
             ("highest threshold", "#1a9e3f", "o", (0, (6, 2, 1, 2, 1, 2))),  # green circles, dash-dot-dot
             ("highest noise", "#ffc000", "D", "-.")]                         # yellow diamonds, dash-dot
FIT_LINEWIDTH = 2    # line width of the fits of the highlighted pixels
FONT_SIZE = 13       
XLIM = (0, 150)     
#------------------------------


plt.rcParams.update({"font.size": FONT_SIZE})


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
    y = hits[ok]
    if y.max() < 0.5 * N_INJECTIONS or y.min() > 0.5 * N_INJECTIONS:
        return np.nan, np.nan
    first_above = x[np.argmax(y >= 0.5 * N_INJECTIONS)]
    try:
        popt, _ = curve_fit(s_curve, x, y, p0=[first_above, 10], bounds=([0, 0.1], [x.max(), x.max()]))
    except RuntimeError:
        return np.nan, np.nan
    return popt


#-----The (col, row) of each HIGHLIGHT pixel; a pixel is only picked once, otherwise the next one is taken-----
def pick_pixels(thresholds, noises):
    ok = ~np.isnan(thresholds)
    mean_threshold = np.mean(thresholds[ok])
    order = {"lowest threshold": thresholds,
             "average threshold": np.abs(thresholds - mean_threshold),
             "highest threshold": -thresholds,
             "highest noise": -noises}
    picked = []
    for which, *_ in HIGHLIGHT:
        values = np.where(ok, order[which], np.inf)
        for flat in np.argsort(values, axis=None):
            row, col = np.unravel_index(flat, values.shape)
            if (col, row) not in picked:
                picked.append((int(col), int(row)))
                break
    return picked


#-----------All S-curves on top of each other, the HIGHLIGHT pixels with their fit------------
def plot_s_curves(vh, hits, thresholds, noises, title, output_file):
    hits = hits / N_INJECTIONS  # plotted as the fraction of the injections that gave a hit, the fits scaled the same way
    plt.figure('S-curves', figsize=(9, 6))
    for row in range(NROWS):
        for col in range(NCOLS):
            plt.plot(vh, hits[:, row, col], color='#2a9df4', linewidth=0.5, alpha=0.3,
                     label='All pixels' if (row, col) == (0, 0) else None)
    x_lo, x_hi = XLIM if XLIM else (vh.min(), vh.max())
    x_fit = np.linspace(x_lo, x_hi, 500)  # the fits are drawn over the whole x range
    for (which, color, marker, linestyle), (col, row) in zip(HIGHLIGHT, pick_pixels(thresholds, noises)):
        thr, noise = thresholds[row, col], noises[row, col]
        plt.plot(vh, hits[:, row, col], marker, color=color, ms=7,
                 label=f'Pixel ({col}, {row}), {which}')
        plt.plot(x_fit, s_curve(x_fit, thr, noise) / N_INJECTIONS, linestyle=linestyle, color=color, linewidth=FIT_LINEWIDTH,
                 label=f'Pixel ({col}, {row}) fit: $\\mu$ = {thr:.1f} mV, $\\sigma$ = {noise:.1f} mV')
        print(f"{which}: pixel ({col}, {row}), threshold {thr:.1f} mV, noise {noise:.1f} mV")
    plt.axhline(0.5, color='gray', linestyle=':', linewidth=1)    # 50 %, the threshold
    plt.xlim(x_lo, x_hi)
    plt.xlabel('Vh [mV]')
    plt.ylabel('Normalized counts')
    plt.ylim(0, 1.05)

    # the legend outside the plot, to the right of it and centred vertically
    legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0, framealpha=0.95, edgecolor='dimgray')
    for handle in legend.legend_handles:  # full colour in the legend, even though the S-curves are see-through
        handle.set_alpha(1)
    plt.grid(alpha=0.3)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')  # 'tight' so the legend outside isn't cut off
    plt.close()
    print(f"Saved {output_file}")


#-----------Map of the chip (column vs row) with one value of every pixel (threshold or noise) as colour------------
def plot_map(values, label, output_file):
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("lightgray")  # pixels without an S-curve fit
    plt.figure("Map", figsize=(7, 7))
    im = plt.imshow(values, origin="lower", cmap=cmap)
    plt.colorbar(im, label=label, fraction=0.046, pad=0.04)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_file}")


#-----------Histograms of the threshold and the noise of all pixels, next to each other------------
def plot_histograms(thresholds, noises, output_file):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (axis, values, x label, bin width in mV)
    for ax, values, label, width in ((axes[0], thresholds, r"Threshold $\mu$ [mV]", 1.5),
                                     (axes[1], noises, r"Noise $\sigma$ [mV]", 0.15)):
        values = values[~np.isnan(values)]
        edges = np.arange(np.floor(values.min() / width) * width, values.max() + width, width)
        # filled histogram without gaps between the bins, with a darker outline
        ax.hist(values, edges, histtype="stepfilled", color="#2a9df4", alpha=0.35)
        ax.hist(values, edges, histtype="step", color="#2a78d6", linewidth=1.5)
        ax.text(0.97, 0.95, f"mean {np.mean(values):.1f} mV", transform=ax.transAxes,
                ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="dimgray", alpha=0.95))
        ax.set_xlabel(label)
        ax.set_ylabel("Counts")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)  # grid behind the histogram
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved {output_file}")


#-----------Noise histograms of the even and the odd columns on top of each other, to compare them------------
def plot_noise_even_odd(noises, output_file, width=0.15):
    all_values = noises[~np.isnan(noises)]
    edges = np.arange(np.floor(all_values.min() / width) * width, all_values.max() + width, width)  # same bins for both
    fig, ax = plt.subplots(figsize=(7, 5))
    lines = []
    # (first column, name, fill colour, outline colour)
    for start, name, fill, outline in ((0, "Even columns", "#2a9df4", "#2a78d6"),
                                       (1, "Odd columns", "#f49a2a", "#d6702a")):
        values = noises[:, start::2]
        values = values[~np.isnan(values)]
        ax.hist(values/1000, edges, histtype="stepfilled", color=fill, alpha=0.3)
        ax.hist(values/1000, edges, histtype="step", color=outline, linewidth=1.5, label=f"{name} ({values.size} pixels)")
        lines.append(f"{name.split()[0]}: mean {np.mean(values):.2f} mV, std {np.std(values):.2f} mV")
        print(f"Noise, {name.lower()}: {values.size} pixels, mean {np.mean(values):.2f} mV, "
              f"median {np.median(values):.2f} mV, std {np.std(values):.2f} mV")
    ax.text(0.97, 0.80, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="dimgray", alpha=0.95))
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="dimgray")
    ax.set_xlabel(r"Noise $\sigma$ [mV]")
    ax.set_ylabel("Normalized counts")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
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
    tag = f"{DATA.name}_vcasb_{vcasb}"
    plot_map(thresholds, r"Threshold $\mu$ [mV]", OUTPUT_DIR / f"threshold_map_{tag}.png")
    plot_map(noises, r"Noise $\sigma$ [mV]", OUTPUT_DIR / f"noise_map_{tag}.png")
    plot_histograms(thresholds, noises, OUTPUT_DIR / f"threshold_noise_hist_{tag}.png")
    plot_noise_even_odd(noises, OUTPUT_DIR / f"noise_hist_even_odd_{tag}.png")


    ##Ploease plot the noise hist of the even and odd columns separately, to see if there is a difference

if __name__ == "__main__":
    main()
