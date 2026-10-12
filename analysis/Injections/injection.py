import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from injection_scan import fit_scan, load_run, fit_gaussian, gaussian, tot_bins, PLOT_DIR

Vcasb = 300
VH = 1200
SETTINGS = f"Vcasb = {Vcasb} mV, VH = {VH} mV"
TAG = f"vcasb_{Vcasb}_vh_{VH}"

CHECK_PIXEL = (6, 6)  #(col, row) of a good pixel
ONE_BAD_FIT = False    #True: plot the histogram of only the first bad pixel. False: of all of them

COUNTS_DIR = PLOT_DIR / "counts_hitmap"  
TOT_DIR = PLOT_DIR / "tot_hitmap"       
COUNTS_DIR.mkdir(parents=True, exist_ok=True)
(TOT_DIR / "bad_fits").mkdir(parents=True, exist_ok=True)
(TOT_DIR / "good_fits").mkdir(parents=True, exist_ok=True)


#-----Plots the hitmap, so just the number of hits in each pixel-----
def plot_hitmap(nhits):
    plt.figure('Hitmap')
    plt.imshow(nhits, origin='lower', cmap='plasma')
    plt.colorbar(label='Number of hits')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title(f'Hitmap\n{SETTINGS}')
    output_file = COUNTS_DIR / f'hitmap_{TAG}.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")


#----Plots the bad ToT histogram for one pixel, used for the bad fits----
def bad_hist(row, col, values, bins=50):
    counts, edges = np.histogram(values, tot_bins(values, bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure('ToT histogram')
    plt.bar(centers, counts, width=edges[1]-edges[0], alpha=0.5)
    plt.xlabel('ToT [µs]')
    plt.ylabel('Counts')
    plt.title(f'Pixel ({col}, {row}) ToT histogram\n{SETTINGS}')
    plt.savefig(TOT_DIR / 'bad_fits' / f'tot_histogram_{TAG}_px_{col}_{row}.png', dpi=150)
    plt.close()


#----Plots the good ToT histogram for one pixel and the gaussian fit, used for the good histograms----
def good_hist(row, col, values, bins=50):
    counts, edges = np.histogram(values, tot_bins(values, bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure('ToT histogram')
    plt.bar(centers, counts, width=edges[1]-edges[0], alpha=0.5)
    plt.xlabel('ToT [µs]')
    plt.ylabel('Counts')
    # plt.xlim(0, 10)
    fit_amp, fit_mean, fit_sigma, chi2_ndf = fit_gaussian(values, bins)
    if not np.isnan(fit_mean):
        x_fit = np.linspace(edges[0], edges[-1], 200)
        y_fit = gaussian(x_fit, fit_amp, fit_mean, fit_sigma) 
        plt.plot(x_fit, y_fit, color='red', label=f'Fit: mean={fit_mean:.2f} µs, sigma={fit_sigma:.2f} µs, $\\chi^2$/ndf={chi2_ndf:.2f}')
        plt.legend()
    plt.title(f'Pixel ({col}, {row}) ToT histogram\n{SETTINGS}')
    plt.savefig(TOT_DIR / 'good_fits' / f'tot_histogram_{TAG}_px_{col}_{row}.png', dpi=150)
    plt.close()


#-----------------Plots 3D ToT lego histogram------------------------
def plot_tot_lego(tot_means):
    nrows, ncols = tot_means.shape
    col, row = np.meshgrid(np.arange(ncols), np.arange(nrows))
    x, y = col.ravel(), row.ravel()
    top = tot_means.ravel()
    good = ~np.isnan(top)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    if good.any():
        vmin, vmax = np.nanmin(top), np.nanmax(top)
        colors = plt.cm.plasma((top[good] - vmin) / max(vmax - vmin, 1e-9))
        ax.bar3d(x[good], y[good], 0, 0.9, 0.9, top[good], color=colors, shade=True)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Mean ToT [µs]")
    ax.set_title(f"Mean ToT\n{SETTINGS}")
    output_file = TOT_DIR / f"tot_mean_lego_{TAG}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")


#---plots a 2d histogram of the hits in all pixels --
def plot_hits(nhits):
    plt.figure('Hits histogram')
    plt.hist(nhits[~np.isnan(nhits)], bins=50)
    plt.xlabel('Number of hits')
    plt.ylabel('Count')
    plt.title(f'Number of hits per pixel\n{SETTINGS}')
    output_file = COUNTS_DIR / f'hits_hist_{TAG}.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")


#-----Plots a histogram of the mean ToT for all pixels-----
def ToT_hist(tot_means, bins=25):
    values = tot_means[~np.isnan(tot_means)]  
    counts, edges = np.histogram(values, bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure('Mean ToT histogram')
    plt.bar(centers, counts, width=edges[1]-edges[0], alpha=0.5)
    plt.xlabel('Mean ToT [µs]')
    plt.ylabel('Number of pixels')
    plt.xlim(left=0)
    plt.title(f'Mean ToT of all pixels ({values.size} pixels)\n{SETTINGS}')
    output_file = TOT_DIR / f'tot_mean_hist_{TAG}.png'
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved {output_file}")




#-----------Main function where we call everything-----------
def main():
    scan = fit_scan([(Vcasb, VH)], f"vcasb_{Vcasb}_vh_{VH}")
    tot_means = scan["means"][0]  # indexed [row, col]

    #-Pixels with hits but no Gaussian fit are shown in green-
    bad_fit = np.isnan(tot_means) & (scan["nhits"][0] > 0)
    check_col, check_row = CHECK_PIXEL
    if bad_fit[check_row, check_col]:
        print(f"WARNING: CHECK_PIXEL ({check_col}, {check_row}) has a bad fit, choose a good pixel to check the fit")
    n_plotted = 0
    for row, col in np.argwhere(bad_fit):
        print(f"WARNING: bad ToT fit for px ({col}, {row})")
        if (ONE_BAD_FIT and n_plotted == 1) or (col, row) == (check_col, check_row):
            continue
        _, _, tots = load_run(Vcasb, VH, col, row)
        #-Plots the bad ToT histogram for this pixel-
        bad_hist(row, col, tots, 50)
        n_plotted += 1

    #-Plots one histogram with its Gaussian fit, to check the fit-
    _, _, check_tots = load_run(Vcasb, VH, check_col, check_row)
    good_hist(check_row, check_col, check_tots, 50)

    #-Plots the mean ToT of all pixels as a histogram-
    ToT_hist(tot_means)

    # #-Plots the hitmap of each pixel-
    # plot_hitmap(scan["nhits"][0])

    # #-Plots the hit of each chip as a histogram-
    # plot_hits(scan["nhits"][0])

    # #-Plots the mean ToT of all pixels as a 3D lego plot-
    # plot_tot_lego(tot_means)



    # plt.figure('ToT mean map')
    # plt.imshow(tot_means, origin='lower', cmap='plasma')
    # plt.colorbar(label='Mean ToT [µs]')
    # plt.imshow(np.where(bad_fit, 1.0, np.nan), origin='lower', cmap=ListedColormap(['black']))
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    # plt.title(f'Mean ToT\n{SETTINGS}')
    # plt.savefig(TOT_DIR / f'tot_mean_map_{TAG}.png', dpi=150)
    


    print(f"\n{'=' * 70}\nANALYSIS FINISHED\n{'=' * 70}")


if __name__ == "__main__":
    main()
