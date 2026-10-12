# SPARC injection and fake-hit-rate analysis

This repo contains scripts for data taking and analysis for the SPARC chip (28 columns × 32 rows). 

## Layout

```
Nellie/
├── SPARC.toml                  config file for the chip
├── injection_scan_sl.py        injection scan: injects pixel at a time. 10000 times per pixel and Vh/Vcasb
├── fhr_scan.py                 fake-hit-rate scan (only measuring noise)
├── sparc_masking_generator.py  register values to mask / unmask pixels
├── sparc_ireset_calculator.py  converts between Ireset and Vreset
├── analysis/
│   ├── analyze_data_sl.py      SPARCAnalyser: reads a raw file (hitmap, active pixels)
│   ├── analyze_data_sl_tot.py  Also returns ToT data
│   ├── Injections/             injection analysis (see below)
│   └── FakeHitRate/            fake-hit-rate analysis
└── plots/                      output of the analysis scripts
```

Not in the repository (see `.gitignore`): `data/` (the raw files), `cache/` (fitted results) and `venv/`.

## Requirements

Python 3.11 with `numpy`, `scipy`, `matplotlib`, and `razorbill` (to decode the raw files).
Data taking also needs `ConstellationDAQ`.

## Injection analysis (`analysis/Injections/`)

`injection_scan.py` holds the shared code: where the data is, reading a pixel's raw file, the Gaussian fit
of the ToT histogram, and fitting a whole scan in parallel. The other scripts only contain their settings
and plots. Run them from anywhere, e.g. `python analysis/Injections/ToT_vh.py`.

| Script | Plots |
|---|---|
| `injection.py` | one Vh and Vcasb. Plots ToT histogram, Gaussian fit of a check pixel, bad fits, hitmap, mean ToT map |
| `injection_compare.py` | histograms of hits and mean ToT per pixel for several Vh values in one plot |
| `ToT_vh.py` | Plots ToT vs Vh for all pixels, two highlighted pixels with every hit and a fit |
| `s_curve.py` | S-curves (hits vs Vh) with an error-function fit: threshold and noise of every pixel |
| `hitmap_tot_vh.py` | Counts- and ToT hitmaps at four Vh values |
| `bad_pixels.py` | where the pixels without a Gaussian fit or with very few hits are |
| `hitmap_compare.py` | hitmaps of several data sets next to each other at one Vh |
| `pixel_fits.py` | ToT histogram and fit of a few chosen pixels |
| `ToT_vcasb.py` | mean ToT vs Vcasb (old data only) |
| `Vh.py` | hits vs Vh from the earlier Vh measurements |

### Data sets

The default data is the Vh scan in `data/sparc0/injection_pd_250`, set as `DATA_DIR` in
`injection_scan.py`: pulse duration 250 (see `SPARC.toml`), Vcasb = 300 mV, Vh from 25 to 1200 mV.
Its plots go to `plots/Injections/<plot type>/`.

Older data sets are listed in `EXTRA_DATASETS` in `injection_scan.py` and can be chosen by name on the
command line, e.g. `python analysis/Injections/ToT_vh.py ireset`:

| Name | Data | Plots |
|---|---|---|
| `old` | `data/sparc0/injection` | `plots/WrongInjections/old/` |
| `ireset` | `data/sparc0/injection_ireset_70pA` (reset current 70 pA) | `plots/WrongInjections/Ireset70pa/` |

### Cached fits

Fitting a whole scan (about 16000 pixel files) takes about half a minute. The results are saved in
`cache/<data folder>/` and reused as long as the list of settings is the same. Delete a data set's cache
folder to refit it, for example after changing the decoder settings in `analyze_data_sl_tot.py`.
When new Vh values have been measured, add them to `DATA_VH` (or the data set's `vh_scan`) in
`injection_scan.py`; the scripts then refit once on their own. Scans with missing pixel files are not cached.

### Things to know

- The decoder drops hits with a ToT below 40 or above 8192 counts (4 ns each), set in
  `analyze_data_sl_tot.py`. The upper limit cuts the ToT at 32.77 µs, which with pulse duration 250 is
  reached above about Vh = 1000 mV.
- Some pixels show double pulsing (extra short hits after a pulse), so a pixel can have slightly more
  hits than the 10000 injections.
- In the plots, pixels are given as (column, row).

## Fake-hit-rate analysis (`analysis/FakeHitRate/`)

`fhrVsVcasb.py`, `fhrVsVcasbMasked.py` and `fhrHistogram.py` plot the fake-hit rate vs Vcasb (with and
without masked pixels) and histograms of the noisy pixels. Their plots go to `plots/FakeHitRate/`.
