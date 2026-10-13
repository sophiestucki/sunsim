# SunSim

SunSim simulates the photometric and spectroscopic variability of the Sun and other stars caused by active regions (spots and faculae) and transiting planets. It computes time series of:

- **photometry**: the light curve in a given band;
- **spectroscopy**: the disc-integrated spectrum at every epoch;
- **CCFs**: the cross-correlation function with a line mask, and its radial velocity (RV), FWHM, contrast and bisector span (BIS).

The best place to start is the tutorial, [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb), which explains the method and goes through every mode with examples.

## Installation

From the folder of the repository:

```bash
pip install -e .            # editable install: changes to the code are used immediately
pip install -e ".[all]"     # with the optional packages (pandas: fast reading of text maps; cmasher: colormaps)
```

or `pip install .` for a normal install. Then, in any folder:

```python
import sunsim
from sunsim import StarSim, flux_grid, CCF_grid, active_region, planet
```

Requirements: Python ≥ 3.9, `numpy`, `numba`, `scipy`, `astropy`, `matplotlib` (installed with the package); optional: `pandas`, `cmasher`.

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

See the tutorial for spectroscopy, CCFs, instruments (CCFs order by order), planets, evolving regions and 2D maps.
Users of the old StarSim can run SunSim from an old configuration file: `sunsim.simulate_old_config('my_old.conf', times=t, observables=['lc', 'rv', 'fwhm', 'bis'])`.

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
- **Rotation.** The rotation can be rigid or follow any differential rotation law (solar law provided). Differential rotation affects both the motion of the regions and the Doppler velocity of the cells.
- **Transiting planets.** Several planets can be simulated, on circular or eccentric orbits, each with its own projected spin–orbit angle. The Keplerian RV of the star is included, and the Rossiter–McLaughlin effect follows from the simulation.
- **Facular limb brightening** for spectral libraries without facular models (such as Phoenix), with a configurable temperature-contrast law.
- **CCF options.** Convective blueshift can be added with a bisector model (for example Dumusque 2014). The spectra can be degraded to the resolution of an instrument (HARPS, HARPS-N, NEID, EXPRESS or any R), and the CCFs computed order by order with the blaze and efficiency of the instrument; the files of a HARPS-N example are installed with the package (`sunsim.INSTRUMENTS_DIR`).
- **Figures.** `sunsim.plot_utils` plots the filling factors, the light curve, RV, FWHM, BIS and contrast of one or several simulations, and the disc with the regions and the planet paths (matplotlib or cmasher colormaps).
- **Checked arguments.** Wrong inputs (an inclination in degrees, a latitude instead of a colatitude, grids of another number of rings, ...) give a clear error when the objects are created.

## Repository layout

| Path | Content |
|---|---|
| `sunsim/` | the package |
| `sunsim/simulation.py` | `StarSim`: the simulation (geometry, time series, CCF parameters) |
| `sunsim/flux_grid.py` | `flux_grid`: flux (photometry) or spectrum (spectroscopy) of every cell for one kind of surface |
| `sunsim/ccf_grid.py` | `CCF_grid`: CCF of every ring for one kind of surface (whole spectrum or order by order), and the bisector treatment |
| `sunsim/active_region.py` | `active_region`: a circular spot or facula, with its lifetime and evolution law |
| `sunsim/active_region_mask.py` | `active_region_mask`: spots and faculae from 2D maps of the disc; `max_rings`, `pixels_per_cell` |
| `sunsim/planet.py` | `planet`: a transiting planet (orbit, position on the disc, Keplerian RV, obliquity) |
| `sunsim/spectrum_utils.py` | Phoenix interpolation, facular limb brightening, instrumental resolution, CCF masks, black body |
| `sunsim/plot_utils.py` | figures of the outputs |
| `sunsim/old_config.py` | `simulate_old_config`: SunSim from a configuration file of the old StarSim |
| `sunsim/nbspectra.py` | Numba kernels: grid of the disc, filling factors, planet coverage, Doppler shifts, cross-correlation, resampling, time series, CCF fits |
| `sunsim/_checks.py` | checks of the arguments |
| `sunsim/data/instruments/` | files of the HARPS-N example: wavelengths and blaze of the orders (HARPS files), total efficiency of HARPS-N |
| `notebooks/tutorial.ipynb` | tutorial with examples of every mode |
| `notebooks/benchmark_old_vs_optimized.ipynb` | old StarSim vs SunSim: time, memory, outputs, parameter sweeps, tests, 89 real SDO epochs |
| `notebooks/benchmark_starsim4.ipynb` | StarSim v4 vs SunSim |
| `notebooks/test_disc_spectrum_vs_ccf.ipynb` | check: CCF of the disc-integrated spectrum (spectroscopy mode) vs CCF mode, with and without rotation |
| `notebooks/` (other) | working notebooks (masks, timing tests, earlier tutorials) |
| `tests/test_api_errors.py` | wrong usages of the API and the error each one must give (`python tests/test_api_errors.py`) |
| `tests/old_version/` | automatic comparison with the old StarSim on 60 cases (`python tests/old_version/test_old_version.py`) |
| `benchmarks/sdo_benchmark.py` | benchmark of the 2D maps on 89 real SDO epochs (old StarSim vs SunSim) |
| `legacy/` | code of earlier versions that SunSim does not use |

## Tests

```bash
python tests/test_api_errors.py                      # ~60 wrong usages and the errors they give (a few seconds)
python tests/old_version/test_old_version.py         # SunSim against the old StarSim on 60 cases
python tests/old_version/test_old_version.py -k sdo  # only the cases whose name contains 'sdo'
```

The comparison with the old StarSim runs the old code (environment `env_sunsim`, folder of the old package: `OLD_STARSIM_PYTHON`, `OLD_STARSIM_PACKAGE`) once per case and keeps its results in `tests/old_version/reference/`; it then only runs SunSim (~1 min). Changes made on purpose are listed as `KNOWN` differences, with the reason.

## Conventions

- **Units.** Times are in days, wavelengths in Å, velocities in m/s and angles in radians, except that active-region positions and sizes are in degrees.
- **Region position.** `active_region(typ, size, latitude, longitude)` takes the **colatitude** (90 = equator) under the name `latitude`.
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
- Architecture, optimization and flexibility improvments
- Transmission spectroscopy at high resolution

**Jordi Blanco Pozo**
- Planetary transits within the 2D-image projection framework
- Performance and speed improvements

**Òscar Porqueras**
- Time series of the disc-integrated flux at low resolution
- Transmission spectroscopy at low resolution
- Inverse mode (not available)
