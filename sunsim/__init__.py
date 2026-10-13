"""
SunSim: photometric and spectroscopic variability of the Sun and other stars caused by active regions (spots and
faculae) and transiting planets (light curves, spectra, CCFs and their RV, FWHM, contrast and BIS).

    from sunsim import StarSim, flux_grid, CCF_grid, active_region, planet

Modules
    simulation           StarSim: the simulation (geometry, time series, CCF parameters)
    flux_grid            flux_grid: flux (photometry) or spectrum (spectroscopy) of every cell for one kind of surface
    ccf_grid             CCF_grid: CCF of every ring for one kind of surface, and the bisector treatment
    active_region        active_region: a circular spot or facula
    active_region_mask   active_region_mask: spots and faculae from 2D maps of the disc; max_rings, pixels_per_cell
    planet               planet: a transiting planet
    spectrum_utils       Phoenix interpolation, limb brightening of the faculae, instrumental resolution, masks, ...
    plot_utils           figures of the outputs (import sunsim.plot_utils: it imports matplotlib)
    old_config           simulate_old_config: run SunSim from a configuration file of the old StarSim
    nbspectra            Numba kernels
See the tutorial (notebooks/tutorial.ipynb) and the README.
"""
from pathlib import Path

__version__ = '1.0.0'

from .simulation import StarSim, SOLAR_DIFFERENTIAL_ROTATION
from .flux_grid import flux_grid
from .ccf_grid import CCF_grid, bisector_fit, dumusque_bisector
from .active_region import active_region
from .active_region_mask import active_region_mask, max_rings, pixels_per_cell
from .planet import planet
from .spectrum_utils import (black_body, vacuum2air, air2vacuum, facula_dT, limb_brightening_factor,
                             add_limb_brightening, interpolate_phoenix_mu, interpolate_phoenix, phoenix_continuum,
                             phoenix_grid, phoenix_weights, clear_phoenix_cache, add_resol, mask_weights,
                             FACULA_DT_LAWS, INSTRUMENT_RESOLUTION, PHOENIX_URL, PHOENIX_SPECINT_URL)
from .old_config import simulate_old_config, read_old_config
from . import nbspectra                 # the Numba kernels (also exported by `from sunsim import *`)

#data installed with the package: files of the HARPS-N example (CCFs order by order)
DATA_DIR = Path(__file__).resolve().parent / 'data'
INSTRUMENTS_DIR = DATA_DIR / 'instruments'

__all__ = ['StarSim', 'SOLAR_DIFFERENTIAL_ROTATION', 'flux_grid', 'CCF_grid', 'bisector_fit', 'dumusque_bisector',
           'active_region', 'active_region_mask', 'max_rings', 'pixels_per_cell', 'planet',
           'black_body', 'vacuum2air', 'air2vacuum', 'facula_dT', 'limb_brightening_factor', 'add_limb_brightening',
           'interpolate_phoenix_mu', 'interpolate_phoenix', 'phoenix_continuum', 'phoenix_grid', 'phoenix_weights',
           'clear_phoenix_cache', 'add_resol', 'mask_weights', 'FACULA_DT_LAWS', 'INSTRUMENT_RESOLUTION',
           'PHOENIX_URL', 'PHOENIX_SPECINT_URL', 'simulate_old_config', 'read_old_config', 'DATA_DIR',
           'INSTRUMENTS_DIR', 'nbspectra', '__version__']
