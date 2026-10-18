"""
Automatic tests of SunSim against the OLD version of StarSim / SunSim, through old_config.simulate_old_config.

    python tests/old_version/test_old_version.py              # all the cases
    python tests/old_version/test_old_version.py -k phx         # the cases whose name contains 'phx'
    python tests/old_version/test_old_version.py --list         # the cases and what they test
    python tests/old_version/test_old_version.py --refresh      # run the old code again for the selected cases
    python tests/old_version/test_old_version.py --jobs 6       # old runs in parallel (default 4)

(run with the Python of SunSim; the old code runs in its own environment.)

Paths (sunsim.local_path: environment variables SUNSIM_<KEY> or the file local_paths.ini, see local_paths.example.ini):
  old_starsim      folder of the old StarSim package (starsim.py, ...); without it the tests are skipped. The tests run
                   a work copy of it (reference/_old_package): its code, and links to the data it reads from its own
                   folder (models/, masks/, MPS-ATLAS_spectra/, SDO_input/), from the old package when it has them,
                   else from 'old_starsim_data', 'phoenix_models', 'muram_spectra', 'ccf_masks'
  old_starsim_data an old package with the data (optional; e.g. a previous version with SDO_input/ and the masks)
  old_python       Python of the environment of the old code (default: the Python that runs the tests)
  sdo_maps         real SDO maps for the cases 'sdo_real_*' (skipped without it)
  old_harps_files  folder where the old code reads its HARPS-N files (cases with instrument = HARPS-N)

How it works
------------
Every case is a configuration of the old code (BASE, modified by the case), regions (rows of the old spot map, or 2D
maps), epochs and observables. For every case:
1. the configuration file is written in reference/ (plot_grid_map = 0: the old code writes no plot);
2. the OLD code is run once in its environment (run_old_starsim.py, a new process, caches in reference/_cache: the old
   package is not modified) and its results are kept in reference/<case>_<hash>.npz. The hash covers the
   configuration, the regions, the epochs, the observables and the source of the old code: the old code runs again
   only when one of them changes (or with --refresh). The first run takes ~20-30 min (the old MURaM high resolution
   mode is slow), the next ones ~1-2 min;
3. SunSim computes the same case (simulate_old_config, reproduce_old_bugs=True unless the case says otherwise);
4. the outputs are compared: light curve, filling factors, RV, FWHM, BIS, contrast, with the tolerances of the case.

Tolerances
----------
EXACT: the two codes must give the same numbers (differences of rounding, and of the CCF fits: scipy in the old code,
Numba in SunSim, ~1e-5 m/s). Some cases have KNOWN differences, from changes made on purpose in SunSim; their
tolerance is a bound a bit above the difference measured when the change was made, so that a regression in these cases
is still detected, and the reason is printed:
  overlap      the visible part of a facula covered by spots: in every cell the old code subtracted from every facula
               the fraction of the LAST VISIBLE spot of its list (spots first), whatever spot covers the facula; SunSim
               subtracts the spots that cover it, deciding once per epoch how every spot and facula are placed
               (generate_ff_core). Same results for one spot inside its facula; different when a facula is covered in
               part by a spot, by several spots, or when another spot is visible anywhere on the disc.
  fixed bugs   the cases run with reproduce_old_bugs=False show the size of the old bugs that SunSim does not reproduce
               (old_config docstring): Phoenix angles, constant limb extrapolation with Phoenix, 2D-map projection,
               coverage of the cells by the planets.
When a change in SunSim changes the results on purpose, update the tolerance of the cases concerned and write why in
KNOWN below.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))              # the repository (if sunsim is not installed)

from sunsim.local_paths import local_path

REFERENCE = HERE / 'reference'
OLD_CODE = local_path('old_starsim', required=False)
OLD_DATA = local_path('old_starsim_data', required=False) or OLD_CODE
#the spectra of the photosphere read by the old code, by version: every name is given the file of MURAM_PHOTOSPHERE, so
#that the old code and SunSim use the same spectra whatever the name in the version of the old code
MURAM_ALIASES = {'av_solar_ssd_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy':
                 'av_solar_ssd_200G_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy',
                 'av_solar_ssd_10_mu_angles_PHOENIX_high_res_grid_cleaned.npy':
                 'av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid_cleaned.npy'}


def prepare_old_package():
    """Work copy of the old package in reference/_old_package/starsim: the code (*.py and the small files) of the path
    'old_starsim', and links to the data it reads from its own folder, taken from the old package itself when it has
    them, else from 'old_starsim_data' (an old package with the data), 'phoenix_models', 'muram_spectra' and
    'ccf_masks'. The folders of the paths are not modified. Returns the folder of the copy, or None."""
    import shutil
    if OLD_CODE is None or not (OLD_CODE / 'starsim.py').is_file():
        return None
    copy = REFERENCE / '_old_package' / 'starsim'
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for f in OLD_CODE.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            shutil.copy2(f, copy / f.name)

    def source(name, fallback=None):
        for base in (OLD_CODE, OLD_DATA):
            if base is not None and (base / name).exists():
                return base / name
        return fallback

    phoenix = local_path('phoenix_models', required=False)
    models = copy / 'models'
    models.mkdir()
    for sub in ('filters', 'Phoenix_mu', 'Phoenix'):
        src = source(Path('models') / sub, None if phoenix is None or sub == 'filters' else phoenix / sub)
        if src is not None and src.exists():
            (models / sub).symlink_to(src)
    masks = copy / 'masks'                     # the masks of the data package, and those of the old package itself
    masks.mkdir()
    for base in (local_path('ccf_masks', required=False), OLD_DATA and OLD_DATA / 'masks', OLD_CODE / 'masks'):
        if base is not None and base.is_dir():
            for f in base.iterdir():
                if f.is_file():
                    if (masks / f.name).is_symlink():
                        (masks / f.name).unlink()
                    (masks / f.name).symlink_to(f)
    for name, key in (('SDO_input', None), ('orders_CRX', None)):
        src = source(name, local_path(key, required=False) if key else None)
        if src is not None and src.exists():
            (copy / name).symlink_to(src)
    spectra = copy / 'MPS-ATLAS_spectra'
    spectra.mkdir()
    muram = source('MPS-ATLAS_spectra', local_path('muram_spectra', required=False))
    if muram is not None:
        for f in muram.iterdir():
            if f.suffix == '.npy':
                (spectra / f.name).symlink_to(f)
        for alias, real in MURAM_ALIASES.items():
            if (muram / real).exists():
                if (spectra / alias).is_symlink() or (spectra / alias).exists():
                    (spectra / alias).unlink()
                (spectra / alias).symlink_to(muram / real)
    return copy


OLD_PACKAGE = prepare_old_package() or Path('<old_starsim not set>')
OLD_PYTHON = str(local_path('old_python', required=False) or sys.executable)
OLD_HARPS = local_path('old_harps_files', required=False)
#real SDO maps (spot and facula maps of the Sun, 944 x 944 pixels, one per day): the cases 'sdo_real_*' use the epochs
#SDO_REAL_EPOCHS of this folder (linked, not copied, in reference/maps_sdo_real); they are skipped if it is not there
SDO_REAL = local_path('sdo_maps', required=False) or Path('<sdo_maps not set>')
SDO_REAL_EPOCHS = [float(t) for t in range(0, 89, 8)]          # 12 epochs, every 8 days


class SkipCase(Exception):
    """A case that cannot run on this computer (missing input files)."""
OBS = ['lc', 'rv', 'fwhm', 'bis', 'contrast']

# Configuration of the reference case: a Sun-like star on grid points of the Phoenix models (5800 K, spot 5300 K,
# facula 5900 K, logg 4.5), every key of the last old version.
BASE = {
    'files': {'filter_name': 'TJO_R.dat', 'orders_CRX_filename': 'orders_wv_CARMENES.dat'},
    'general': {'simulation_mode': 'grid', 'n_grid_rings': 15, 'wavelength_lower_limit': 3801.0,
                'wavelength_upper_limit': 7999.0, 'spectral_library': 'phoenix',
                'mps_spectra_folder': str(OLD_PACKAGE / 'MPS-ATLAS_spectra') + '/', 'stagger_spectra_folder': '-',
                'mps_mu_grid': 0, 'ccf_norm': 1, 'flux_flat': 1, 'highres_mu': 1, 'smoothing_kernel': 0,
                'remove_bis_spot': 1, 'add_median_diff': 0.0},
    'rv': {'ccf_template': 'mask', 'ccf_mask': str(OLD_PACKAGE / 'masks' / 'HARPS_G20_mps_friendly.mas'),
           'ccf_weight_lines': 0, 'path_weight_lines': 'flux_ratios.dat', 'ccf_rv_range': 18.0, 'ccf_rv_step': 0.25,
           'instrument': 'None'},
    'star': {'temperature_photosphere': 5800.0, 'spot_T_contrast': 500.0, 'facula_T_contrast': 100.0,
             'convective_shift': 0.0, 'rotation_period': 25.0, 'radius': 1.0, 'mass': 1.0, 'logg': 4.5,
             'inclination': 90.0, 'differential_rotation': 0.0},
    'LD': {'use_phoenix_limb_darkening': 1, 'limb_darkening_law': 'quadratic', 'limb_darkening_q1': 0.43,
           'limb_darkening_q2': 0.122, 'limb_extrapolation': 'linear', 'limb_lim': 0.1, 'meunier': 0},
    'spots': {'plot_grid_map': 0, 'spots_evo_law': 'constant', 'reference_time': 0.0, 'sdo_input': 0,
              'SDO_input_path': str(OLD_PACKAGE / 'SDO_input') + '/'},
    'planet': {'simulate_planet': 0, 'planet_period': 3.0, 'planet_transit_t0': 5.0, 'planet_radius': 0.1,
               'planet_impact_param': 0.3, 'planet_esinw': 0.0, 'planet_ecosw': 0.0, 'planet_spin_orbit_angle': 0.0,
               'planet_semi_amplitude': 50.0},
}

# tolerances (max |old - new|) of the EXACT cases
# lc, filling factors, BIS: the same numbers (rounding). RV, FWHM, contrast come from a gaussian fit (scipy in the old
# code, Numba in SunSim): they agree to ~1e-4 of the signals (up to a few mm/s for RV signals of 100 m/s).
EXACT = {'lc': 1e-9, 'ff_sp': 1e-9, 'ff_fc': 1e-9, 'ff_pl': 1e-9, 'rv': 1e-2, 'fwhm': 5e-2, 'bis': 1e-3,
         'contrast': 1e-6}

KNOWN = {
    'overlap': 'faculae and spots: the old code subtracted from every facula the fraction of the LAST VISIBLE spot of '
               'its list (whatever spot covers the facula), SunSim the spots that cover it (generate_ff_core); '
               'differs when a facula is covered in part by a spot, by several spots, or when another spot is visible',
    'constant limb': 'old bug not reproduced (reproduce_old_bugs=False): limb_extrapolation constant with Phoenix '
                     'took the disc-centre intensity',
    'projection': "old ring limits of the 2D maps not reproduced (map_projection='new')",
    'sdo overlap': 'old bug not reproduced (reproduce_old_bugs=False): the old code averaged the spot and facula maps '
                   'independently, a pixel in both maps (the SDO facula maps include most spot pixels) was spot and '
                   'facula; SunSim puts the spot on top of the facula in every pixel (spots_over_faculae)',
    'muram limb': 'old bug not reproduced (reproduce_old_bugs=False): with MURaM, below limb_lim the old code used the '
                  'intensity at mu = 1 (the first angle of the files, not sorted) instead of the smallest mu',
    'harpsn efficiency': 'old bug not reproduced: with numpy 1.x the old code evaluated the HARPS-N efficiency '
                         'polynomial in float32 (meaningless order weights, BIS NaN); SunSim in float64, as the old '
                         'code with numpy >= 2',
    'planet coverage': "old bug not reproduced (reproduce_old_bugs=False): the old code covered the cells with a linear "
                       "ramp, too deep for planets smaller than a cell; SunSim the exact area (planet_coverage='area')",
}


def spot(colat, lon, r, t0=0.0, life=1e4, c2=0.0, c3=0.0):
    """Row of the old spot map: a spot."""
    return [0, t0, life, colat, lon, r, c2, c3]


def facula(colat, lon, r, t0=0.0, life=1e4, c2=0.0, c3=0.0):
    """Row of the old spot map: a facula."""
    return [1, t0, life, colat, lon, r, c2, c3]


SF = [spot(90, 180, 8), facula(90, 180, 14)]          # concentric spot and facula on the equator


def case(name, what, conf=None, regions=SF, times=(0.0, 25.0, 40), observables=OBS, tol=None, known=None,
         bugs=True, maps=None, old_patch=None):
    """A test case: conf modifies BASE ({section: {key: value}}), times = (start, end, number), tol = tolerances that
    replace those of EXACT, known = keys of KNOWN, bugs = reproduce_old_bugs, maps = 'synthetic' 2D maps (see _maps) or
    'sdo_real' (the real SDO maps of SDO_REAL, see _sdo_real_maps), old_patch = run the old code with a runtime patch
    ('efficiency64': the instrument efficiency in float64, as with numpy >= 2; see run_old_starsim.py)."""
    return dict(name=name, what=what, conf=conf or {}, regions=regions, times=times, observables=list(observables),
                tol=dict(EXACT, **(tol or {})), known=list(known or []), bugs=bugs, maps=maps, old_patch=old_patch)


rng = np.random.default_rng(7)
MANY = []
for k in range(8):                                   # 8 regions far from each other (no overlap)
    colat, lon = 30 + 15 * k, (45 * k) % 360
    MANY.append(spot(colat, lon, rng.uniform(2, 5)) if k % 2 == 0 else facula(colat, lon, rng.uniform(3, 8)))

CASES = [
    # ------------------------------------------------------------------ Phoenix: star and geometry
    case('phx_ref', 'Phoenix, spot + concentric facula on the equator, 15 rings'),
    case('phx_spot_only', 'one spot alone', regions=[spot(90, 180, 8)]),
    case('phx_facula_only', 'one facula alone', regions=[facula(90, 180, 12)]),
    case('phx_incl_60', 'inclination 60 deg, regions at colatitude 60',
         conf={'star': {'inclination': 60.0}}, regions=[spot(60, 180, 8), facula(60, 180, 14)]),
    case('phx_incl_120', 'inclination 120 deg (south pole visible), regions at colatitude 130',
         conf={'star': {'inclination': 120.0}}, regions=[spot(130, 100, 8), facula(130, 100, 14)]),
    case('phx_pole_on', 'pole-on star (inclination 0, vsini = 0), circumpolar regions',
         conf={'star': {'inclination': 0.0}}, regions=[spot(30, 90, 8), facula(30, 90, 14)]),
    case('phx_fast_rotator', 'rotation period 8 d (vsini 6 km/s), two rotations', conf={'star': {'rotation_period': 8.0}},
         times=(0.0, 16.0, 40)),
    case('phx_big_star', 'radius 1.6 Rsun, period 30 d', conf={'star': {'radius': 1.6, 'rotation_period': 30.0}},
         times=(0.0, 30.0, 40)),
    case('phx_diffrot', 'solar differential rotation (B = -2.66), regions at 3 latitudes, two rotations (one spot: '
         'several visible spots change the faculae, see overlap_other_spot)',
         conf={'star': {'differential_rotation': -2.66}},
         regions=[spot(45, 0, 6), facula(45, 0, 10), facula(90, 120, 8), facula(135, 240, 10)],
         times=(0.0, 50.0, 50)),
    case('phx_reftime', 'reference time of the longitudes = 10 d', conf={'spots': {'reference_time': 10.0}},
         regions=[spot(70, 40, 7), facula(70, 40, 12)]),
    case('phx_many_regions', '8 spots and faculae at several latitudes and longitudes', regions=MANY),
    case('phx_rings_30', '30 rings', conf={'general': {'n_grid_rings': 30}}),
    case('phx_rings_8', '8 rings', conf={'general': {'n_grid_rings': 8}}),
    # ------------------------------------------------------------------ evolution and lifetime of the regions
    case('phx_lifetime', 'spot appearing at 5 d and disappearing at 15 d, facula from 8 to 20 d',
         regions=[spot(90, 200, 8, t0=5.0, life=10.0), facula(90, 200, 14, t0=8.0, life=12.0)]),
    case('phx_evo_linear', "spots_evo_law linear: spot radius 2 -> 10 deg in 20 d",
         conf={'spots': {'spots_evo_law': 'linear'}},
         regions=[spot(90, 180, 2, t0=2.0, life=20.0, c2=10.0), facula(90, 180, 4, t0=2.0, life=20.0, c2=16.0)]),
    case('phx_evo_quadratic', "spots_evo_law quadratic: radius up to 10 deg at mid-life",
         conf={'spots': {'spots_evo_law': 'quadratic'}},
         regions=[spot(90, 180, 10, t0=2.0, life=20.0), facula(90, 180, 16, t0=2.0, life=20.0)]),
    # ------------------------------------------------------------------ temperatures, spectra, CCF options
    case('phx_offgrid_T', 'Teff 5778 K, spot 5378 K (between Phoenix models with different mu, old behaviour)',
         conf={'star': {'temperature_photosphere': 5778.0, 'spot_T_contrast': 400.0, 'facula_T_contrast': 30.0}}),
    case('phx_logg_4p4', 'logg 4.4 (interpolation between logg 4.0 and 4.5)', conf={'star': {'logg': 4.4}}),
    case('phx_spot_5350', 'spot at 5350 K (interpolation between the 5300 and 5400 K models)',
         conf={'star': {'spot_T_contrast': 450.0}}),
    case('phx_cs_1', 'convective_shift 1: CIFIST and Dumusque bisectors', conf={'star': {'convective_shift': 1.0}}),
    case('phx_cs_neg', 'convective_shift -0.5', conf={'star': {'convective_shift': -0.5}}),
    case('phx_ccf_norm0', 'ccf_norm 0 (CCF = 1 - ccf/min) and convective_shift 1 (other Dumusque coefficients)',
         conf={'general': {'ccf_norm': 0}, 'star': {'convective_shift': 1.0}}),
    case('phx_flux_flat0', 'flux_flat 0 (spectra divided by 5e14 instead of their continuum)',
         conf={'general': {'flux_flat': 0}}),
    case('phx_weight_lines', 'mask lines weighted by order (ccf_weight_lines 1)', conf={'rv': {'ccf_weight_lines': 1}}),
    case('phx_rv_grid', 'CCF velocity range 25 km/s, step 0.5 km/s', conf={'rv': {'ccf_rv_range': 25.0, 'ccf_rv_step': 0.5}}),
    case('phx_no_filter', 'filter file missing: box filter between the wavelength limits',
         conf={'files': {'filter_name': 'no_such_filter.dat'}}, observables=['lc']),
    # ------------------------------------------------------------------ limb darkening and brightening
    case('phx_limb_constant', 'limb_extrapolation constant below limb_lim (old behaviour)',
         conf={'LD': {'limb_extrapolation': 'constant'}}),
    case('phx_limb_lim_02', 'limb_lim 0.2, linear', conf={'LD': {'limb_lim': 0.2}}),
    case('phx_ld_quadratic', 'limb darkening law quadratic instead of the intensity models',
         conf={'LD': {'use_phoenix_limb_darkening': 0}}),
    case('phx_ld_sqrt', 'limb darkening law sqrt', conf={'LD': {'use_phoenix_limb_darkening': 0, 'limb_darkening_law': 'sqrt'}}),
    case('phx_meunier', 'limb brightening of the faculae: Meunier et al. (2010) law', conf={'LD': {'meunier': 1}}),
    case('phx_mu_grid', 'mps_mu_grid 1: Phoenix intensities on mu = 0.1 ... 1 (old behaviour in RV)',
         conf={'general': {'mps_mu_grid': 1}}),
    # ------------------------------------------------------------------ planets
    case('phx_planet', 'transiting planet (P = 3 d, b = 0.3) and a spot',
         conf={'planet': {'simulate_planet': 1}}, regions=[spot(90, 150, 8)], times=(0.0, 10.0, 80)),
    case('phx_small_planet', 'small planet (0.02 Rstar, smaller than a cell) and a spot',
         conf={'planet': {'simulate_planet': 1, 'planet_radius': 0.02}}, regions=[spot(90, 150, 8)],
         times=(4.8, 5.2, 80)),
    case('phx_planet_ecc', 'eccentric planet (e sin w = 0.1, e cos w = 0.05), b = 0.5, spin-orbit 30 deg',
         conf={'planet': {'simulate_planet': 1, 'planet_esinw': 0.1, 'planet_ecosw': 0.05, 'planet_impact_param': 0.5,
                          'planet_spin_orbit_angle': 30.0}}, regions=[spot(90, 150, 8)], times=(0.0, 10.0, 80)),
    # ------------------------------------------------------------------ MURaM (old RV mode slow: fewer epochs)
    case('mps_ref', 'MURaM low resolution, spot + concentric facula', conf={'general': {'spectral_library': 'mps'}},
         times=(0.0, 25.0, 25)),
    case('mps_remove_bis0', 'MURaM, remove_bis_spot 0', conf={'general': {'spectral_library': 'mps', 'remove_bis_spot': 0}},
         times=(0.0, 25.0, 25)),
    case('mps_median_diff', 'MURaM, add_median_diff 20 m/s', conf={'general': {'spectral_library': 'mps', 'add_median_diff': 20.0}},
         times=(0.0, 25.0, 25)),
    case('mps_cs_1', 'MURaM, convective_shift 1 (Dumusque bisector on the spots)',
         conf={'general': {'spectral_library': 'mps'}, 'star': {'convective_shift': 1.0}}, times=(0.0, 25.0, 25)),
    case('mps_incl_diffrot', 'MURaM, inclination 70, differential rotation, 3 latitudes',
         conf={'general': {'spectral_library': 'mps'}, 'star': {'inclination': 70.0, 'differential_rotation': -2.66}},
         regions=[spot(50, 0, 6), facula(50, 0, 10), facula(130, 240, 10), facula(90, 120, 8)], times=(0.0, 50.0, 25)),
    case('mps_ld_law', 'MURaM with a limb darkening law (old behaviour)',
         conf={'general': {'spectral_library': 'mps'}, 'LD': {'use_phoenix_limb_darkening': 0}}, observables=['lc']),
    case('hr_ref', 'MURaM high resolution (spectrum at the mu of every ring), 6 rings, remove_bis_spot, convective_shift -1',
         conf={'general': {'spectral_library': 'mps_highres', 'n_grid_rings': 6, 'ccf_norm': 0, 'flux_flat': 0},
               'star': {'convective_shift': -1.0}}, times=(0.0, 25.0, 20)),
    # ------------------------------------------------------------------ instrument (CCFs order by order)
    case('hr_harpsn', 'MURaM high resolution with the HARPS-N instrument: R = 118000, 72 HARPS orders, blaze x efficiency '
         '(old code with the float64 efficiency, as with numpy >= 2)',
         conf={'general': {'spectral_library': 'mps_highres', 'n_grid_rings': 6, 'ccf_norm': 0, 'flux_flat': 0},
               'rv': {'instrument': 'HARPS-N'}, 'star': {'convective_shift': -1.0}}, times=(0.0, 25.0, 20),
         old_patch='efficiency64'),
    case('sdo_real_harpsn', 'real SDO maps, MURaM high resolution with the HARPS-N instrument, 6 rings',
         conf={'general': {'spectral_library': 'mps_highres', 'n_grid_rings': 6, 'ccf_norm': 0, 'flux_flat': 0},
               'rv': {'instrument': 'HARPS-N'}, 'spots': {'sdo_input': 1}}, regions=None, times=None, maps='sdo_real',
         old_patch='efficiency64'),
    # ------------------------------------------------------------------ 2D maps
    case('sdo_spots', '2D maps (SDO_input of the old package), MURaM, 15 rings',
         conf={'general': {'spectral_library': 'mps'}, 'spots': {'sdo_input': 1}}, regions=None, times=None),
    case('sdo_real_mps', 'real SDO maps (12 epochs of 2015-2016, 944 x 944), MURaM, 15 rings (old faculae)',
         conf={'general': {'spectral_library': 'mps'}, 'spots': {'sdo_input': 1}}, regions=None, times=None,
         maps='sdo_real'),
    case('sdo_real_phoenix', 'real SDO maps, Phoenix, 15 rings (old faculae)',
         conf={'spots': {'sdo_input': 1}}, regions=None, times=None, maps='sdo_real'),
    case('sdo_real_rings30', 'real SDO maps, MURaM, 30 rings',
         conf={'general': {'spectral_library': 'mps', 'n_grid_rings': 30}, 'spots': {'sdo_input': 1}}, regions=None,
         times=None, maps='sdo_real'),
    case('sdo_real_hr', 'real SDO maps, MURaM high resolution, 6 rings',
         conf={'general': {'spectral_library': 'mps_highres', 'n_grid_rings': 6, 'ccf_norm': 0, 'flux_flat': 0},
               'spots': {'sdo_input': 1}}, regions=None, times=None, maps='sdo_real'),
    case('sdo_faculae', 'synthetic 2D maps with spots and faculae, Phoenix (old behaviour of the faculae)',
         conf={'spots': {'sdo_input': 1}}, regions=None, times=None, maps='synthetic'),
    # ------------------------------------------------------------------ known differences
    case('overlap_decentred', 'facula decentred from its spot (partial overlap)',
         regions=[spot(90, 180, 8), facula(90, 192, 14)], known=['overlap'],
         tol={'ff_fc': 0.3, 'lc': 5e-5, 'rv': 0.3, 'fwhm': 0.5, 'bis': 0.2, 'contrast': 1e-4}),
    case('overlap_two_spots', 'two spots on one facula',
         regions=[spot(90, 175, 5), spot(92, 188, 4), facula(90, 182, 14)], known=['overlap'],
         tol={'ff_fc': 0.3, 'lc': 5e-5, 'rv': 0.3, 'fwhm': 0.5, 'bis': 0.2, 'contrast': 1e-4}),
    case('overlap_other_spot', 'spot inside its facula and another spot visible elsewhere on the disc',
         regions=[spot(90, 180, 8), facula(90, 180, 14), spot(90, 100, 5)], known=['overlap'],
         tol={'ff_fc': 0.5, 'lc': 2e-4, 'rv': 1.0, 'fwhm': 1.0, 'bis': 0.5, 'contrast': 2e-4}),
    case('phx_offgrid_T_default', 'Teff 5778 K, spot 5378 K with reproduce_old_bugs=False (the Phoenix angles are '
         'always those of the old code)',
         conf={'star': {'temperature_photosphere': 5778.0, 'spot_T_contrast': 400.0, 'facula_T_contrast': 30.0}},
         bugs=False),
    case('fixed_muram_limb', 'MURaM below limb_lim with the smallest mu (reproduce_old_bugs=False)',
         conf={'general': {'spectral_library': 'mps'}}, bugs=False, known=['muram limb'],
         tol={'lc': 1e-4, 'rv': 0.4, 'fwhm': 3.0, 'bis': 0.3, 'contrast': 1.5e-4}),
    case('fixed_limb_constant', 'limb_extrapolation constant with the SunSim limb (reproduce_old_bugs=False)',
         conf={'LD': {'limb_extrapolation': 'constant'}}, bugs=False, known=['constant limb'],
         tol={'lc': 5e-4, 'rv': 3.0, 'fwhm': 12.0, 'bis': 1.0, 'contrast': 1e-3}),
    case('fixed_projection', '2D maps with the SunSim projection (reproduce_old_bugs=False)',
         conf={'general': {'spectral_library': 'mps'}, 'spots': {'sdo_input': 1}}, regions=None, times=None,
         bugs=False, known=['projection', 'muram limb'],
         tol={'ff_sp': 0.05, 'lc': 5e-5, 'rv': 0.2, 'fwhm': 3.0, 'bis': 1.6, 'contrast': 1e-4}),
    case('fixed_sdo_real', 'real SDO maps with the faculae and the projection of SunSim (reproduce_old_bugs=False)',
         conf={'general': {'spectral_library': 'mps'}, 'spots': {'sdo_input': 1}}, regions=None, times=None,
         maps='sdo_real', bugs=False, known=['sdo overlap', 'projection', 'muram limb'],
         tol={'ff_sp': 1e-3, 'ff_fc': 0.2, 'lc': 0.04, 'rv': 10.0, 'fwhm': 9.0, 'bis': 4.5, 'contrast': 3.5e-3}),
    case('fixed_harpsn', 'hr_harpsn against the old code with numpy 1.x (efficiency in float32: meaningless)',
         conf={'general': {'spectral_library': 'mps_highres', 'n_grid_rings': 6, 'ccf_norm': 0, 'flux_flat': 0},
               'rv': {'instrument': 'HARPS-N'}, 'star': {'convective_shift': -1.0}}, times=(0.0, 25.0, 20),
         bugs=False, known=['harpsn efficiency'],
         tol={'rv': 30.0, 'fwhm': 1300.0, 'bis': np.inf, 'contrast': 0.015}),     # bis: NaN in the old code
    case('fixed_planet_coverage', 'transiting planet of phx_planet with the exact coverage (reproduce_old_bugs=False)',
         conf={'planet': {'simulate_planet': 1}}, regions=[spot(90, 150, 8)], times=(0.0, 10.0, 80),
         bugs=False, known=['planet coverage'],
         tol={'ff_pl': 0.06, 'lc': 7e-4, 'rv': 0.7, 'fwhm': 0.2, 'bis': 0.1, 'contrast': 7e-4}),
    case('fixed_small_planet', 'phx_small_planet with the exact coverage (reproduce_old_bugs=False)',
         conf={'planet': {'simulate_planet': 1, 'planet_radius': 0.02}}, regions=[spot(90, 150, 8)],
         times=(4.8, 5.2, 80), bugs=False, known=['planet coverage'],
         tol={'ff_pl': 0.2, 'lc': 2.5e-3, 'rv': 2.0, 'fwhm': 1.2, 'bis': 0.2, 'contrast': 2.5e-3}),
]


# =============================================================================================
# files of a case
# =============================================================================================

def _maps(folder):
    """Synthetic 2D maps (200 x 200, 8 epochs): a spot and a larger facula around it, moving across the disc, and a
    second facula alone (pixel values = covered fraction)."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 200
    y, x = np.mgrid[0:n, 0:n]
    x, y = (2 * x + 1) / n - 1, (2 * y + 1) / n - 1
    for k, t in enumerate(np.linspace(0.0, 7.0, 8)):
        cx, cy = -0.7 + 0.2 * k, 0.1
        d = np.hypot(x - cx, y - cy)
        sp = (d < 0.12).astype(float)
        fc = ((d < 0.25) & (d >= 0.12)).astype(float) + (np.hypot(x + 0.2, y + 0.5) < 0.15)
        np.savetxt(folder / ('spot_map_%s.txt' % float(t)), sp, fmt='%.1f')
        np.savetxt(folder / ('faculae_map_%s.txt' % float(t)), np.clip(fc, 0, 1), fmt='%.1f')


def _sdo_real_maps(folder):
    """Links to the maps of SDO_REAL at the epochs SDO_REAL_EPOCHS (the old code takes every spot_map_*.txt of the
    folder as an epoch)."""
    if not SDO_REAL.is_dir():
        raise SkipCase("the real SDO maps are not on this computer (%s; set the path 'sdo_maps')" % SDO_REAL)
    folder.mkdir(parents=True, exist_ok=True)
    for t in SDO_REAL_EPOCHS:
        for kind in ('spot_map', 'faculae_map'):
            src = SDO_REAL / ('%s_%s.txt' % (kind, t))
            if not src.exists():
                raise SkipCase('%s is missing' % src)
            dst = folder / src.name
            if not dst.is_symlink():
                dst.symlink_to(src)


def write_conf(c):
    """Configuration file of a case (in reference/); returns its path."""
    conf = ConfigParser()
    conf.optionxform = str                                  # keep the case of the keys, as in the old files
    for section, kv in BASE.items():
        conf[section] = {k: str(v) for k, v in kv.items()}
    for section, kv in c['conf'].items():
        for k, v in kv.items():
            conf[section][k] = str(v)
    if c['maps'] == 'synthetic':
        folder = REFERENCE / ('maps_' + c['name'])
        _maps(folder)
        conf['spots']['SDO_input_path'] = str(folder) + '/'
    elif c['maps'] == 'sdo_real':
        folder = REFERENCE / 'maps_sdo_real'
        _sdo_real_maps(folder)
        conf['spots']['SDO_input_path'] = str(folder) + '/'
    conf['spots']['plot_grid_map'] = '0'                    # the old code writes no plot file
    path = REFERENCE / ('%s.conf' % c['name'])
    REFERENCE.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        conf.write(f)
    return path


def epochs(c):
    if c['times'] is None:
        return None
    t0, t1, n = c['times']
    return np.linspace(t0, t1, n)


def case_hash(c, conf_path):
    """Hash of what determines the old results: configuration, regions, epochs, observables, maps, old source code."""
    h = hashlib.sha1()
    conf = Path(conf_path).read_bytes().replace(('maps_' + c['name']).encode(), b'maps')
    h.update(conf.replace(str(REFERENCE).encode(), b'REFERENCE'))      # independent of the folder of the tests
    h.update(json.dumps([c['regions'], c['times'], c['observables']]).encode())
    if c['old_patch']:
        h.update(c['old_patch'].encode())
    for name in ('starsim.py', 'spectra.py', 'nbspectra.py'):
        h.update((OLD_PACKAGE / name).read_bytes())
    if c['maps'] == 'synthetic':
        for f in sorted((REFERENCE / ('maps_' + c['name'])).glob('*.txt')):
            h.update(f.read_bytes())
    elif c['maps'] == 'sdo_real':                           # large files: names, sizes and dates
        for f in sorted((REFERENCE / 'maps_sdo_real').glob('*.txt')):
            st = f.stat()
            h.update(('%s %d %d' % (f.name, st.st_size, st.st_mtime_ns)).encode())
    if c['conf'].get('rv', {}).get('instrument') == 'HARPS-N':     # the files read by the old code
        for f in ('HARPS.2025-12-16T23-21-06.037_blaze_A.fits', 'HARPS.2024-10-17T21-57-53.302_wave_A.fits',
                  'plot-data-3.csv'):
            p = None if OLD_HARPS is None else OLD_HARPS / f
            h.update(p.read_bytes() if p is not None and p.exists() else b'missing')
    return h.hexdigest()[:12]


# =============================================================================================
# old and new runs
# =============================================================================================

def old_path(c, conf_path):
    """File of the old results: named by the hash only, so that cases with the same old setup share it (e.g. a case and
    its reproduce_old_bugs=False twin)."""
    return REFERENCE / ('old_%s.npz' % case_hash(c, conf_path))


def old_results(c, refresh=False):
    """Results of the old code for a case (from reference/, or computed by the old code)."""
    conf_path = write_conf(c)
    out = old_path(c, conf_path)
    if out.exists() and not refresh:
        return dict(np.load(out)), 0.0
    job = {'old_package': str(OLD_PACKAGE), 'conf': str(conf_path), 'out': str(out),
           'times': None if c['times'] is None else epochs(c).tolist(), 'observables': c['observables'],
           'regions': c['regions'], 'patch': c['old_patch']}
    job_path = REFERENCE / ('%s.job.json' % c['name'])
    job_path.write_text(json.dumps(job))
    env = dict(os.environ, PYTHONPYCACHEPREFIX=str(REFERENCE / '_cache' / 'pycache'),
               NUMBA_CACHE_DIR=str(REFERENCE / '_cache' / 'numba'), MPLBACKEND='Agg')
    t = time.perf_counter()
    p = subprocess.run([OLD_PYTHON, str(HERE / 'run_old_starsim.py'), str(job_path)], capture_output=True, text=True,
                       env=env, cwd=str(REFERENCE))
    if p.returncode != 0 or not out.exists():
        raise RuntimeError('old code failed for %s:\n%s' % (c['name'], p.stderr[-2000:]))
    return dict(np.load(out)), time.perf_counter() - t


def new_results(c):
    """Results of SunSim for a case (old_config.simulate_old_config)."""
    from sunsim import simulate_old_config
    conf_path = REFERENCE / ('%s.conf' % c['name'])
    return simulate_old_config(conf_path, times=epochs(c), observables=c['observables'], spot_map=c['regions'],
                               old_package=OLD_PACKAGE, reproduce_old_bugs=c['bugs'], verbose=False)


def compare(c, old, new):
    """{quantity: (max |old - new|, tolerance, amplitude of the old signal)} for the outputs of the case."""
    out = {}
    for k in ('lc', 'ff_sp', 'ff_fc', 'ff_pl', 'rv', 'fwhm', 'bis', 'contrast'):
        if k not in old or k not in new:
            continue
        a, b = np.asarray(old[k], dtype=float), np.asarray(new[k], dtype=float)
        both = np.isnan(a) & np.isnan(b)                       # NaN in both codes: the same
        if np.any(np.isnan(a) != np.isnan(b)):
            d = np.inf                                          # NaN in only one of them: a failure
        else:
            d = float(np.max(np.abs(a - b)[~both])) if np.any(~both) else 0.0
        amp = float(np.nanmax(a) - np.nanmin(a)) if np.any(~np.isnan(a)) else 0.0
        out[k] = (d, c['tol'][k], amp)
    return out


def run_case(c, refresh=False):
    """Run one case; returns a dict with its status ('PASS', 'KNOWN' (pass, with a known difference), 'FAIL',
    'ERROR'), the comparison and the outputs of both codes."""
    try:
        old, t_old = old_results(c, refresh)
        t = time.perf_counter()
        new = new_results(c)
        t_new = time.perf_counter() - t
    except SkipCase as e:
        return dict(case=c, status='SKIP', error=str(e), diffs={})
    except Exception as e:                                       # noqa: BLE001  (reported, not raised)
        return dict(case=c, status='ERROR', error='%s: %s' % (type(e).__name__, e), diffs={})
    diffs = compare(c, old, new)
    failed = [k for k, (d, tol, _) in diffs.items() if not d <= tol]
    status = 'FAIL' if failed else ('KNOWN' if c['known'] else 'PASS')
    return dict(case=c, status=status, failed=failed, diffs=diffs, old=old, new=new, t_old=t_old, t_new=t_new)


def run_cases(pattern=None, refresh=False, jobs=4, verbose=True):
    """Run the cases whose name contains `pattern` (all by default). The old code of the cases without results runs
    first, `jobs` cases at a time. Returns the list of results (run_case)."""
    cases = [c for c in CASES if pattern is None or pattern in c['name']]
    todo, seen = [], set()
    for c in cases:
        try:
            path = old_path(c, write_conf(c))
        except SkipCase:
            continue                                            # reported by run_case
        if (refresh or not path.exists()) and path not in seen:   # one old run for the cases that share a setup
            todo.append(c)
            seen.add(path)
    if todo and verbose:
        print('old code: %d case(s) to run (%d at a time): %s' % (len(todo), jobs, ', '.join(c['name'] for c in todo)),
              flush=True)

    def old_only(c):
        try:
            _, t = old_results(c, refresh)
            if verbose:
                print('  old %-22s %6.1f s' % (c['name'], t), flush=True)
        except Exception as e:                                   # noqa: BLE001  (shown again by run_case)
            if verbose:
                print('  old %-22s FAILED: %s' % (c['name'], str(e).splitlines()[0]), flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        list(pool.map(old_only, todo))
    results = [run_case(c) for c in cases]         # the old results exist now (or the error is reported)
    if verbose:
        print_results(results)
    return results


def print_results(results):
    """Table of the results: status, largest difference relative to its tolerance, details of the failures."""
    print('\n%-22s %-6s %-60s' % ('case', 'status', 'largest difference / tolerance (quantity)'))
    for r in results:
        c = r['case']
        if r['status'] in ('ERROR', 'SKIP'):
            print('%-22s %-6s %s' % (c['name'], r['status'], r['error'].splitlines()[0][:150]))
            continue
        k, (d, tol, amp) = max(r['diffs'].items(), key=lambda kv: kv[1][0] / kv[1][1])
        print('%-22s %-6s %8.2g / %-8.2g (%s, signal %.3g)%s' % (c['name'], r['status'], d, tol, k, amp,
              '   known: ' + ', '.join(c['known']) if c['known'] else ''))
        for q in r.get('failed', []):
            d, tol, amp = r['diffs'][q]
            print('%-22s        FAILED %s: max |old - new| = %.3g > %.3g (signal %.3g)' % ('', q, d, tol, amp))
    counts = {s: sum(r['status'] == s for r in results) for s in ('PASS', 'KNOWN', 'FAIL', 'ERROR', 'SKIP')}
    print('\n%d cases: %d pass, %d pass with a known difference, %d fail, %d error%s'
          % (len(results), counts['PASS'], counts['KNOWN'], counts['FAIL'], counts['ERROR'],
             ', %d skipped (missing input files)' % counts['SKIP'] if counts['SKIP'] else ''))
    if counts['KNOWN']:
        print('known differences:')
        for key in sorted({k for r in results for k in r['case']['known']}):
            print('  %-15s %s' % (key, KNOWN[key]))


def main():
    parser = argparse.ArgumentParser(description='Tests of SunSim against the old version (see the module docstring).')
    parser.add_argument('-k', dest='pattern', help='only the cases whose name contains this text')
    parser.add_argument('--refresh', action='store_true', help='run the old code again for the selected cases')
    parser.add_argument('--jobs', type=int, default=4, help='old runs at the same time (default 4)')
    parser.add_argument('--list', action='store_true', help='list the cases')
    args = parser.parse_args()
    if args.list:
        for c in CASES:
            print('%-22s %s%s' % (c['name'], c['what'], ('  [known: %s]' % ', '.join(c['known'])) if c['known'] else ''))
        return 0
    if not (OLD_PACKAGE / 'starsim.py').is_file():
        print("The old StarSim is not on this computer: tests skipped. Set the path 'old_starsim' (environment variable "
              "SUNSIM_OLD_STARSIM or local_paths.ini, see local_paths.example.ini) to the folder of the old package, and "
              "'old_python' to the Python of its environment.")
        return 0
    results = run_cases(args.pattern, args.refresh, args.jobs)
    return 1 if any(r['status'] in ('FAIL', 'ERROR') for r in results) else 0


if __name__ == '__main__':
    sys.exit(main())
