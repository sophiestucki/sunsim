# SunSim

SunSim simulates the photometric and spectroscopic variability of the Sun and other stars caused by active regions (spots and faculae) and transiting planets. It computes time series of:

- **photometry**: the light curve in a given band;
- **spectroscopy**: the disc-integrated spectrum at every epoch;
- **CCFs**: the cross-correlation function with a line mask, and its radial velocity (RV), FWHM, contrast and bisector span (BIS).

The best place to start is the tutorial, [`tutorial/tutorial.ipynb`](tutorial/tutorial.ipynb), which explains the method and goes through every mode with examples.

## Installation

From the folder of the repository:

```bash
pip install -e .            # editable install: changes to the code are used immediately
pip install -e ".[all]"     # with the optional packages (pandas: fast reading of text maps; cmasher: colormaps;
                            # sunpy, drms: maps of spots and faculae from SDO/HMI images)
```

or `pip install .` for a normal install. Then, in any folder:

```python
import sunsim
from sunsim import StarSim, flux_grid, CCF_grid, active_region, planet
```

Requirements: Python ≥ 3.9, `numpy`, `numba`, `scipy`, `astropy`, `matplotlib` (installed with the package); optional: `pandas`, `cmasher`.

### Paths of the data on your computer

The spectra, masks, maps and models are not part of the repository, and no notebook, test or benchmark contains a path of one computer. Every path is read by `sunsim.local_path(key)`:

1. from the environment variable `SUNSIM_<KEY>` (e.g. `SUNSIM_PHOENIX_MODELS=/data/phoenix`), or
2. from the section `[paths]` of a file `local_paths.ini` in the folder of the repository (or the current folder, or `~/.config/sunsim/`; or the file named by `SUNSIM_PATHS_FILE`).

Copy [`local_paths.example.ini`](local_paths.example.ini) to `local_paths.ini` (not tracked by git) and fill in the folders you have. A missing path gives an error that says which key to set; the tests and benchmarks that need data you do not have are skipped. The keys are listed in `sunsim/local_paths.py` (`KEYS`): `phoenix_models`, `muram_spectra`, `muram_lowres_spectra`, `muram_spot_spectra`, `ccf_masks`, `sdo_data`, `sdo_maps`, `harpsn_sun`, `harpsn_interventions`, `sorce_tsi`, `tsis_tsi`, `old_starsim`, `old_python`, `starsim4`, `starsim_david`, ...

## Quick start

```python
import numpy as np
from sunsim import StarSim, SOLAR_DIFFERENTIAL_ROTATION, flux_grid, active_region

# specific-intensity spectra {mu: {'wav': ..., 'intensity': ...}} of the quiet photosphere, spots and faculae
grids = {}
for name, spectra in (('qp', flux_quiet), ('sp', flux_spot), ('fc', flux_facula)):
    g = flux_grid(60, spectra, name, None, 3800, 8000, mode='photometry')
    g.built_grid()
    grids['flux_grid_' + name] = g.grid

time = np.linspace(0, 25.4, 100)                                    # [days]
ss = StarSim(time, N_rings=60, inclination=0, rotation_period=25.4,
             differential_rotation=SOLAR_DIFFERENTIAL_ROTATION,
             active_regions_map=[active_region('sp', 10, 90, 180),   # spot: radius 10 deg, colatitude 90, longitude 180
                                 active_region('fc', 15, 90, 180)],  # facula around it
             mode=['photometry'], **grids)
ss.generate_timeseries()
ss.flux_var, ss.ff_sp, ss.ff_fc                                     # light curve and filling factors

from sunsim import plot_utils
plot_utils.plot_photometry(ss, show_disc=True)                      # filling factors + light curve + the disc
```

Photometry, spectroscopy and CCFs can be computed in the same simulation: `mode=['photometry', 'spectroscopy', 'ccf']` with the photometry grids in `flux_grid_xx`, the spectroscopy grids in `spec_grid_xx` and the CCF grids in `ccf_grid_xx`.

See the tutorial for spectroscopy, CCFs, instruments (CCFs order by order), planets, evolving regions, 2D maps and the tools of section 10.
Users of the old StarSim can run SunSim from an old configuration file: `sunsim.simulate_old_config('my_old.conf', times=t, observables=['lc', 'spec', 'rv', 'fwhm', 'bis'], spot_map='spotmap.dat')`, or read the old spot map into regions for the new interface: `sunsim.read_old_spotmap('spotmap.dat', config='my_old.conf')` (tutorial, section 9).

## How it works

1. The visible stellar disc is divided into concentric rings, and each ring into cells of similar area. All the cells of a ring share the same μ (cosine of the angle between the surface normal and the line of sight).
2. For each kind of surface (quiet photosphere, spot, facula), a *grid* gives the signal of every cell as if it were entirely of that kind. Grids are built from specific-intensity spectra given at several values of μ.
3. At every epoch the positions of the active regions and planets give the *filling factors* of every cell, i.e. the fraction of the cell covered by quiet photosphere, spots, faculae and planets. The signal of the star is then

   `signal(t) = Σ_cells [ ff_quiet · grid_quiet + ff_spot · grid_spot + ff_facula · grid_facula ]`

   Planets are dark discs and contribute no flux. The fraction of a cell covered by a planet is the exact projected area of the planet disc inside the cell, so planets of any size (even much smaller than a cell) have the right transit depth.
4. In CCF mode, every cell's CCF is Doppler-shifted by its rotation velocity before the sum, and the RV, FWHM, contrast and BIS are measured on the result.

The geometry of all epochs is computed first with NumPy. All epochs are then processed in a single parallel Numba call.

## Features

- **Any spectral library.** The spectra only need the format `{mu: {'wav': wavelengths [Å], 'intensity': intensity}}`. The tutorial uses MURaM spectra. Helpers interpolate Phoenix models in temperature, log g and metallicity: SPECINT models (several μ, low resolution) and HiRes models (disc-integrated, high resolution).
- **Spots and faculae as independent regions.** They can have any position and size. Where they overlap, spots lie on top of faculae.
- **Evolving regions.** Regions have an appearance time and lifetime, and the radius can follow any evolution law (for example linear growth and decay).
- **2D maps of the disc.** Spots and faculae can be given as images, such as identification masks from SDO/HMI, with one spot map and one facula map per epoch. The pixels are projected onto the stellar grid; `max_rings(resolution)` gives the largest number of rings a map resolution supports.
- **Maps from SDO/HMI images.** `sunsim.sdo` classifies every pixel of HMI continuum images and magnetograms as quiet Sun, spot, plage or network (Yeo et al. 2013; Sen et al. 2023) and writes the spot and facula maps of every day: `sdo.make_sdo_maps(...)` in a notebook, or the terminal command `sunsim-sdo-maps --data hmi/ --out maps/ --start 2017-07-01 --end 2017-07-11` (with `--download` from JSOC, and `--conf` to simulate the maps right away). Tutorial: [`tutorial/tutorial_sdo.ipynb`](tutorial/tutorial_sdo.ipynb).
- **Long SDO runs.** For thousands of days at full resolution, `make_sdo_maps(..., n_rings=N, fmt='npz', workers=k)` projects every day on the cells of the SunSim grid right after the identification (`load_filling_factors` then replaces the maps), saves the maps as compressed bit masks, runs days in parallel and resumes a run that stopped (~1.7 s and 1.4 GB per day at 4096 px with one worker). Several values of the thresholds at once (`make_sdo_sweep`, or several values of the options of `sunsim-sdo-maps`): every day is read once for all the sets, and `simulate_sdo_sets` computes the grids once for all of them and keeps them in a file for the next runs. `simulate_in_chunks(time, chunk_size, ...)` simulates chunks of epochs and joins them, and, with circular regions, the CCFs and the rotating spectra of the cells are computed only for the cells touched by a region when storing them would take more than `ccf_cells_max_memory` / `spec_cells_max_memory` (bit-identical results; `ccf_on_the_fly=False` / `spec_on_the_fly=False` always keep them). `StarSim.results()` and `save_results(path)` return / save all the time series, the CCFs included.
- **Comparison with the observed Sun.** `sunsim.observations` loads the TSI of SORCE/TIM and TSIS-1 and the HARPS-N solar RV, FWHM, BIS and contrast (daily averages), matches them with a simulation by day, gives the MAD and RMS of the residuals (RV: one offset per HARPS-N instrumental period, best day shift), and plots the comparison (`plot_comparison`, `plot_correlation`).
- **Rotation.** The rotation can be rigid or follow any differential rotation law: coefficients of a polynomial in sin²(lat), or any function `law(latitude) → Ω(lat) − Ω_eq` [deg/day]; `solar_differential_rotation(B)` gives the solar law of StarSim. Differential rotation affects both the motion of the regions and the Doppler velocity of the cells.
- **Spectra from a limb-darkening law.** `limb_darkened_spectra(wav, I_mu1, law, coefficients)` builds the spectra at many μ from one spectrum at μ = 1 and a law (linear, quadratic, square-root, logarithmic, power-2, Claret's nonlinear, or your own function), with coefficients that can depend on the wavelength.
- **Continuum flattening.** `flatten_spectra(spectra)` divides the spectra of every μ by the continuum of the disc-centre spectrum, which keeps the limb darkening.
- **Transiting planets.** Several planets can be simulated, on circular or eccentric orbits, each with its own projected spin–orbit angle. The Keplerian RV of the star is included, and the Rossiter–McLaughlin effect follows from the simulation.
- **Facular limb brightening** for spectral libraries without facular models (such as Phoenix), with a configurable temperature-contrast law.
- **Air or vacuum.** The CCFs are computed in air by default: `mask_weights(path, mask_medium=..., medium='air')` reads a mask in air or vacuum (three-column HARPS masks: air, two columns: vacuum, unless you say otherwise), and `CCF_grid(..., medium='air', spectrum_medium='vacuum', mask_medium='air', orders_medium='air')` converts the spectra (Phoenix and MURaM are in vacuum), the mask and the orders of the instrument to the medium of the CCF (`vacuum2air` / `air2vacuum`). `medium='vacuum'` computes everything in vacuum, as the old StarSim.
- **CCF options.** Convective blueshift can be added with a bisector model: CIFIST (μ-dependent, `cifist_bisector`), Dumusque 2014 (`dumusque_bisector`), each with an optional linear μ dependence of its amplitude (`mu_scaling=(a, b)`: × (a + b μ)), or any function of μ. The spectra can be degraded to the resolution of an instrument (HARPS, HARPS-N, NEID, EXPRESS or any R), and the CCFs computed order by order with the blaze and efficiency of the instrument; the files of a HARPS-N example are installed with the package (`sunsim.INSTRUMENTS_DIR`).
- **Figures.** `sunsim.plot_utils` plots the filling factors, the light curve, RV, FWHM, BIS and contrast of one or several simulations, and the disc with the regions and the planet paths (matplotlib or cmasher colormaps).
- **Checked arguments.** Wrong inputs (an inclination in degrees, a latitude instead of a colatitude, grids of another number of rings, ...) give a clear error when the objects are created.

## Repository layout

| Path | Content |
|---|---|
| `sunsim/` | the package |
| `sunsim/simulation.py` | `StarSim`: the simulation (geometry, time series, CCF parameters) |
| `sunsim/flux_grid.py` | `flux_grid`: flux (photometry) or spectrum (spectroscopy) of the cells for one kind of surface: one per ring, or one per cell for spectroscopy with rotation |
| `sunsim/ccf_grid.py` | `CCF_grid`: CCF of every ring for one kind of surface (whole spectrum or order by order), and the bisector treatment |
| `sunsim/active_region.py` | `active_region`: a circular spot or facula, with its lifetime and evolution law |
| `sunsim/active_region_mask.py` | `active_region_mask`: spots and faculae from 2D maps of the disc; `max_rings`, `pixels_per_cell` |
| `sunsim/planet.py` | `planet`: a transiting planet (orbit, position on the disc, Keplerian RV, obliquity) |
| `sunsim/spectrum_utils.py` | Phoenix interpolation, facular limb brightening, limb-darkening laws (`limb_darkened_spectra`), continuum flattening (`flatten_spectra`), instrumental resolution, CCF masks, black body |
| `sunsim/rotation.py` | differential rotation laws: `solar_differential_rotation`, `polynomial_differential_rotation`, any function of the latitude |
| `sunsim/observations.py` | comparison with the observations: SORCE/TSIS-1 TSI, HARPS-N (daily), MAD / RMS of the residuals, figures |
| `sunsim/sdo.py` | SDO/HMI images → spot and facula maps (identification, `make_sdo_maps`, `download_hmi`, figures, terminal command `sunsim-sdo-maps`) |
| `sunsim/local_paths.py` | `local_path`: the folders of the data and of the other codes on this computer (`local_paths.ini`) |
| `sunsim/plot_utils.py` | figures of the outputs |
| `sunsim/old_config.py` | `simulate_old_config`: SunSim from a configuration file of the old StarSim; `read_old_config`, `read_old_spotmap`: its parameters and its spot map |
| `sunsim/nbspectra.py` | Numba kernels: grid of the disc, filling factors, planet coverage, Doppler shifts, cross-correlation, resampling, time series, CCF fits |
| `sunsim/_checks.py` | checks of the arguments |
| `sunsim/data/instruments/` | files of the HARPS-N example: wavelengths and blaze of the orders (HARPS files), total efficiency of HARPS-N |
| `tutorial/tutorial.ipynb` | tutorial with examples of every mode |
| `tutorial/tutorial_sdo.ipynb` | tutorial of the maps from SDO/HMI images (`sunsim.sdo`), and their simulation |
| `tests/test_api_errors.py` | wrong usages of the API and the error each one must give, and correct usages (`python tests/test_api_errors.py`) |
| `tests/old_version/` | automatic comparison with the old StarSim on 60 cases (`python tests/old_version/test_old_version.py`) |
| `tests/test_disc_spectrum_vs_ccf.ipynb` | check: CCF of the disc-integrated spectrum (spectroscopy mode) vs CCF mode, with and without rotation |
| `benchmarks/benchmark_old_vs_optimized.ipynb` | old StarSim vs SunSim: time, memory, outputs, parameter sweeps, tests, 89 real SDO epochs |
| `benchmarks/benchmark_starsim4.ipynb` | StarSim v4 vs SunSim |
| `benchmarks/benchmark_starsim_david.ipynb` | original StarSim (D. Baroch) vs SunSim: outputs (reference and special cases, `fast` mode), time and RAM |
| `benchmarks/benchmark_v4_vs_david_fast.ipynb` | StarSim v4 vs the `fast` mode of StarSim (D. Baroch): time and RAM (reads the runners of the two notebooks above, in the same folder) |
| `benchmarks/sdo_benchmark.py` | benchmark of the 2D maps on 89 real SDO epochs (old StarSim vs SunSim) |
| `benchmarks/trappist1_phoenix_mu/` | old StarSim vs SunSim for the spectra (`'spec'`): TRAPPIST-1, Phoenix SPECINT, one spot (configuration file and spot map of the old run, notebook, script running the old code) |
| `analysis/` | working notebooks that use SunSim but are not part of it (solar line masks, an early timing test) |
| `local_paths.example.ini` | template of `local_paths.ini`, the paths of the data on your computer |

## Tests

```bash
python tests/test_api_errors.py                      # ~80 usages (wrong ones and the errors they give, correct ones) (a few seconds)
python tests/old_version/test_old_version.py         # SunSim against the old StarSim on 60 cases
python tests/old_version/test_old_version.py -k sdo  # only the cases whose name contains 'sdo'
```

The comparison with the old StarSim needs the old code: the paths `old_starsim` (folder of the old package) and `old_python` (Python of its environment) of `local_paths.ini`; without them it is skipped. It runs the old code once per case and keeps its results in `tests/old_version/reference/`; it then only runs SunSim (~1 min). The cases on the real SDO maps also need the path `sdo_maps`. Changes made on purpose are listed as `KNOWN` differences, with the reason.

## Conventions

- **Units.** Times are in days, wavelengths in Å, velocities in m/s and angles in radians, except that active-region positions and sizes are in degrees.
- **Region position.** `active_region(typ, size, colatitude, longitude)`: the **colatitude** (0 = north pole, 90 = equator, 180 = south pole); `region.latitude` gives the latitude (90 − colatitude).
- **Inclination.** `inclination` is the angle between the rotation axis and the plane of the sky: 0 = equator-on, π/2 = pole-on.
- **Rotation period.** `rotation_period` is the period at the equator.

## Credits

The original version of this code is based on the work of **David Baroch**. Several developments have been added since, and their integration into this version is ongoing.

**Sophie Stucki**
- Separation of faculae from spots
- Treatment of spot bisector
- Projection of 2D pixel images onto the stellar grid
- Integration of new spectral libraries
- Pipeline producing identification masks from SDO/HMI images
- Architecture, optimization and flexibility improvements
- Transmission spectroscopy at high resolution
- Tests and benchmarks

**Jordi Blanco Pozo**
- Planetary transits within the 2D-image projection framework
- Performance and speed improvements
- Tests and benchmarks

**Òscar Porqueras**
- Time series of the disc-integrated flux at low resolution
- Transmission spectroscopy at low resolution
- Filters
- Inverse mode (not available in the new version)
- Performance and speed improvements
- Tests and benchmarks

**Hugo Tabernero**
- Instrumental effects
