"""
old_config: run SunSim from a configuration file (.conf) of the old version of StarSim / SunSim, without learning the new
interface. The old parameters are read, the spectra and the grids are built as the old code builds them for its spectral
library (Phoenix, MURaM low resolution 'mps', MURaM high resolution 'mps_highres'), and the time series is computed by
the new StarSim (main.StarSim), for circular active regions (spot map) or for 2D maps of the disc (sdo_input = 1).

    from sunsim import simulate_old_config
    res = simulate_old_config('starsim_sun_sdo_mps_lowres.conf', times=np.linspace(0, 25, 100), observables=['lc', 'rv'])
    res['lc'], res['rv'], res['fwhm'], res['bis'], res['contrast'], res['ff_sp'], ...   (the keys of old StarSim.results)
    res['report']                                     (what was read, the defaults used, the keys ignored, the paths found)
    res['starsim']                                    (the new StarSim object, to go further with the new interface)

The configuration file
----------------------
Old files differ: some lack keys that the last old version reads, some have keys it no longer uses, some have paths
of another computer. read_old_config():
* reads the file tolerantly (also a section header written [[files]]);
* fills the missing OPTION keys with the defaults of OPTION_DEFAULTS (what the old code did before the key existed, or
  its usual value) and lists them in the report; the missing PHYSICAL keys (temperature, rotation, radius, ...) are an
  error, since no default is sensible;
* lists the keys that are ignored (unknown, or not used by the old code: bb_ratio, facular_area_ratio, force_cifist,
  smoothing_kernel, orders_CRX_filename, plot_grid_map, the [optimization] section);
* finds the files: a path is used if it exists, otherwise a file of the same name is searched in the folders of the old
  package (masks/, models/filters/, MPS-ATLAS_spectra/, SDO_input/); the paths can also be given (paths={...}).

The old package folder (old_package) is the folder of the old starsim code, with models/Phoenix, models/Phoenix_mu,
models/filters, masks and spotmap.dat. By default: the parent of the folder of the configuration file if it is called
'conf', or the folder of the file itself.

What is done for every spectral library (as in the old code)
-------------------------------------------------------------
Photometry ('lc'), brightness of one cell of every ring = sum over the wavelengths of I(mu) * projected area / 4 pi *
filter (the old sum, not an integral):
  phoenix     : Phoenix SPECINT at T_photosphere and T_spot (logg); the faculae have the spectrum of the photosphere
                times the bolometric limb brightening ((T + dT(mu)) / T)^4, dT(mu) = 155 - 290 mu + 128 mu^2 K
                (meunier = 0) or 250.9 - 407.7 mu + 190.9 mu^2 K (meunier = 1). mps_mu_grid = 1: the intensities are
                first interpolated at mu = 0.1, 0.2, ..., 1.0.
  mps         : MURaM low resolution intensities (av_solar_ssd_200G / av_solar_faculae_200G ..._low_res_specint_grid_
  mps_highres   cleaned.npy) for the photosphere and the faculae (no limb brightening: it is in the MURaM faculae); the
                spot is the photosphere times the black body ratio B(T_spot) / B(T_photosphere).
CCF ('rv', 'bis', 'fwhm', 'contrast'), mask of the configuration (air -> vacuum for 3-column masks, lines weighted by
order with ccf_weight_lines, weights normalised to 1), velocity grid -range..range by step:
  phoenix     : CCF of the Phoenix HiRes spectra at T_photosphere, T_spot and T_facula (flattened with flux_flat). The
                bisector of each of these CCFs is removed, and a bisector is added in every ring, times convective_shift:
                CIFIST bisector of the mu of the ring for the photosphere, Dumusque (2014) bisector for the spots and the
                faculae. Every ring is weighted by its brightness (SPECINT, with the limb brightening for the faculae).
  mps         : CCF of the MURaM high resolution spectra at the disc centre (spot: black body ratio). The photosphere and
                the faculae keep their own bisector (MURaM has the convective blueshift). The spot: its bisector is
                removed if remove_bis_spot = 1 (plus median and add_median_diff if add_median_diff > 0), and the Dumusque
                bisector times convective_shift is added. Rings weighted by their MURaM brightness.
  mps_highres : the MURaM high resolution spectrum is interpolated at the mu of every ring and its CCF is computed in
                every ring (highres_mu = 1); the spot as in mps. The CCF amplitude of every ring is its brightness.
The CCF of every cell is the CCF of its ring shifted by the rotation velocity of the cell, as in the old code (and with
the old rigid rotation velocity, even with differential_rotation; the regions move with the differential rotation).
Below limb_lim: limb_extrapolation = 'linear': I(mu) = mu / limb_lim * I(limb_lim); 'constant': I(mu) = I at the
smallest tabulated mu.

Regions, planets, outputs
-------------------------
* circular regions: the spot map of the old code (spotmap.dat of the old package, or spot_map=path or array): one row per
  region: type (0 spot, 1 facula), appearance time, lifetime, colatitude, longitude at reference_time, coefficients of
  spots_evo_law ('constant': radius; 'linear': radius from c1 to c2 over the lifetime; 'quadratic': c1 * 4 dt (T - dt)
  / T^2). Differential rotation B [deg/day] -> SunSim coefficients (1.698 B / 2.66, 2.346 B / 2.66).
* 2D maps (sdo_input = 1): spot_map_<t>.txt / faculae_map_<t>.txt in SDO_input_path, as in the old code; the epochs are
  those of the maps (or `times`, which must be among them). The maps are read by active_region_mask with the orientation
  of the old projection (projection_pxl_to_ss_grid). The pixels are given to the cells with the angular edges of the
  rings (the old code: halfway between the projected radii of the rings; map_projection='old' reproduces it exactly).
* planet (simulate_planet = 1): semi-major axis 4.2097 P^(2/3) M^(1/3) / R (old code), spin-orbit angle in degrees.
* outputs: res['lc'] = flux / flux of the quiet star, res['rv'] = RV - RV of the quiet star (+ Keplerian RV),
  res['contrast'] = contrast / contrast of the quiet star, res['fwhm'], res['bis'], res['CCF'] (velocities, CCFs),
  res['ccf_quiet'], res['ff_ph'], res['ff_sp'], res['ff_fc'], res['ff_pl'] [%], res['time']. As in the old code, the
  CCFs of res['CCF'] are shifted to a minimum of 1e-6 (the old compute_ccf_params modifies them in place); the CCFs
  without this shift are in res['CCF_unshifted'].

Differences with the old code (bugs not reproduced by default; everything else as above)
-----------------------------------------------------------------------------------------
simulate_old_config(..., reproduce_old_bugs=True) reproduces all of them, and then gives the outputs of the old code
(checked by tests/old_version/test_old_version.py on ~50 cases). One difference remains in both cases, because it is in
the simulation itself (generate_ff_core): the faculae covered by spots. In every cell the old code subtracted from every
facula the fraction of the last visible spot of its list, whatever spot covers the facula; SunSim subtracts the spots
that cover it. Same results for one spot inside its facula; different when a facula is covered in part by a spot, by
several spots, or when another spot is visible elsewhere on the disc.
* Phoenix SPECINT models with different mu angles (temperatures between 5700 and 5800 K, 5300 and 5400 K, ...): the old
  code added the rows of the models of the interpolation without interpolating them in mu, and used the angles of one
  model (the photosphere in photometry, the spot in RV) for every surface; here every model is interpolated in mu on the
  angles of its lowest-T, lowest-logg corner first (~0.4 % of the signals in the tests; exact with the old addition).
* limb_extrapolation = 'constant' with Phoenix: the old code took the intensity at mu = 1 for the rings below limb_lim
  (its first row); here: at the smallest tabulated mu, as intended (and as the old code does for MURaM since 2026-10-09).
* limb darkening law (use_phoenix_limb_darkening = 0) with MURaM: the old code multiplied the law by the spectrum at
  mu = 0.1; here: by the spectrum at mu = 1, as with Phoenix.
* mps_mu_grid = 1 in RV: the old code interpolated with the angle of the original grid instead of the target angle.
* 2D maps: the old code computed the faculae as spots in photometry and as quiet photosphere in the CCFs (phoenix, mps);
  here the faculae have their own spectra, as with circular regions.
* planets: the old code covered a cell with a linear ramp in the distance between the centres of the cell and of the
  planet, so a cell whose centre is under the planet was about half covered whatever its radius: planets smaller than
  a cell were much too deep (x8 for 0.01 Rstar with 15 rings, x2 for 0.03 Rstar, +9 % for 0.1 Rstar); here the exact
  area of the planet in every cell (StarSim planet_coverage='area'; 'ramp' with reproduce_old_bugs).
* instrument = HARPS-N: the old code evaluated its efficiency polynomial (degree 10 in the wavelength in A) on the
  float32 wavelengths of the orders. With numpy 1.x the arithmetic stays in float32 and the powers of ~5000 A lose all
  their precision: the efficiency went from -1800 to +800 instead of 14-90 % (negative on a third of the pixels), so
  the order weights, the RV, the FWHM and the BIS (NaN) of the old code were wrong, and so sensitive to the rounding
  that they cannot be reproduced. With numpy >= 2 the same line gives float64 (correct) values. SunSim always uses
  the float64 efficiency, i.e. the results of the old code with numpy >= 2 (also with reproduce_old_bugs).
* the CCF parameters (RV, FWHM, BIS, contrast) are measured by the new code (Numba fit, same model).
Not supported (as in the old compute_forward, or hard-coded in it): simulation_mode 'fast' (grid is used),
ccf_template = 'model', spectral_library = 'stagger', the 'spec' observable, mps_highres with highres_mu = 0 (smoothed
ratio): a clear error or warning is given. instrument = HARPS-N with mps_highres (CCFs order by order) is converted,
with the files of the instruments/ folder of SunSim (or the old hard-coded paths); see _instrument_ccf.
"""
import os
import re
import warnings
import configparser
from pathlib import Path

import numpy as np

from . import nbspectra
from .simulation import StarSim
from .active_region import active_region
from .planet import planet
from .ccf_grid import CCF_grid, bisector_fit, dumusque_bisector
from .spectrum_utils import interpolate_phoenix_mu, interpolate_phoenix, black_body, air2vacuum

C_CGS = 2.99792458e10
# warning of np.polyfit for ill-conditioned fits: numpy >= 2 (np.exceptions) or older (np.RankWarning)
_RANK_WARNING = getattr(getattr(np, 'exceptions', None), 'RankWarning', None) or getattr(np, 'RankWarning', Warning)

#files of the instrument option of the old code (rv: instrument = HARPS-N, with mps_highres). The old code read them
#at fixed paths (OLD_INSTRUMENT_PATHS); they are looked for in the instruments/ folder of SunSim first.
INSTRUMENT_FILES = {'blaze_file': 'HARPS.2025-12-16T23-21-06.037_blaze_A.fits',
                    'wave_file': 'HARPS.2024-10-17T21-57-53.302_wave_A.fits',
                    'efficiency_file': 'HARPS-N_total_efficiency.csv'}
OLD_INSTRUMENT_PATHS = {'blaze_file': '/Users/sophiestucki/Downloads/HARPS.2025-12-16T23-21-06.037_blaze_A.fits',
                        'wave_file': '/Users/sophiestucki/Downloads/HARPS.2024-10-17T21-57-53.302_wave_A.fits',
                        'efficiency_file': '/Users/sophiestucki/Downloads/plot-data-3.csv'}
INSTRUMENT_FOLDER = Path(__file__).resolve().parent / 'data' / 'instruments'

MURAM_FILES = {
    ('low', 'ph'): 'av_solar_ssd_200G_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy',
    ('low', 'fc'): 'av_solar_faculae_200G_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy',
    ('high', 'ph'): 'av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid_cleaned.npy',
    ('high', 'fc'): 'av_solar_faculae_200G_10_mu_angles_PHOENIX_high_res_grid_cleaned.npy',
}

# physical keys: no default (an error if missing). (section, key, type)
REQUIRED = [
    ('general', 'n_grid_rings', int), ('general', 'wavelength_lower_limit', float),
    ('general', 'wavelength_upper_limit', float),
    ('star', 'temperature_photosphere', float), ('star', 'spot_T_contrast', float),
    ('star', 'facula_T_contrast', float), ('star', 'rotation_period', float), ('star', 'radius', float),
    ('star', 'inclination', float),
]
# needed only for some options: (section, key, type, needed when)
CONDITIONAL = [
    ('star', 'logg', float, 'spectral_library = phoenix'),
    ('star', 'mass', float, 'simulate_planet = 1'),
    ('rv', 'ccf_mask', str, 'RV observables'), ('rv', 'ccf_rv_range', float, 'RV observables'),
    ('rv', 'ccf_rv_step', float, 'RV observables'),
    ('LD', 'limb_darkening_q1', float, 'use_phoenix_limb_darkening = 0'),
    ('LD', 'limb_darkening_q2', float, 'use_phoenix_limb_darkening = 0'),
    ('planet', 'planet_period', float, 'simulate_planet = 1'), ('planet', 'planet_transit_t0', float, 'simulate_planet = 1'),
    ('planet', 'planet_radius', float, 'simulate_planet = 1'), ('planet', 'planet_impact_param', float, 'simulate_planet = 1'),
    ('planet', 'planet_esinw', float, 'simulate_planet = 1'), ('planet', 'planet_ecosw', float, 'simulate_planet = 1'),
    ('planet', 'planet_spin_orbit_angle', float, 'simulate_planet = 1'),
    ('planet', 'planet_semi_amplitude', float, 'simulate_planet = 1'),
]
# options: (section, key, type, default, what the default means)
OPTION_DEFAULTS = [
    ('files', 'filter_name', str, '', 'no filter: box between the wavelength limits'),
    ('general', 'simulation_mode', str, 'grid', 'grid (the only mode of the old compute_forward)'),
    ('general', 'spectral_library', str, 'phoenix', 'Phoenix (the only library of the first versions)'),
    ('general', 'mps_spectra_folder', str, 'MPS-ATLAS_spectra', 'MPS-ATLAS_spectra of the old package'),
    ('general', 'mps_mu_grid', int, 0, 'Phoenix intensities at their own angles'),
    ('general', 'ccf_norm', int, 1, 'CCF normalised to [0, 1], as in the first versions'),
    ('general', 'flux_flat', int, 1, 'high resolution spectra divided by their continuum, as in the first versions'),
    ('general', 'highres_mu', int, 1, 'mps_highres: spectrum interpolated at the mu of every ring'),
    ('general', 'remove_bis_spot', int, 1, 'MURaM: bisector of the spot CCF removed'),
    ('general', 'add_median_diff', float, 0.0, 'no shift of the spot CCF'),
    ('star', 'convective_shift', float, 0.0, 'no bisector added'),
    ('star', 'differential_rotation', float, 0.0, 'rigid rotation'),
    ('rv', 'ccf_template', str, 'mask', 'mask'),
    ('rv', 'ccf_weight_lines', int, 0, 'lines not weighted by order'),
    ('rv', 'path_weight_lines', str, '', '-'),
    ('rv', 'instrument', str, 'None', 'no instrument'),
    ('LD', 'use_phoenix_limb_darkening', int, 1, 'limb darkening of the intensity models'),
    ('LD', 'limb_darkening_law', str, 'quadratic', 'quadratic'),
    ('LD', 'limb_extrapolation', str, 'linear', 'linear: I(mu) = mu / limb_lim * I(limb_lim) below limb_lim'),
    ('LD', 'limb_lim', float, None, 'the smallest tabulated mu'),
    ('LD', 'meunier', int, 0, 'StarSim limb brightening law of the faculae (155 - 290 mu + 128 mu^2 K)'),
    ('spots', 'spots_evo_law', str, 'constant', 'constant radius'),
    ('spots', 'reference_time', float, 0.0, 'longitudes defined at t = 0'),
    ('spots', 'sdo_input', int, 0, 'circular regions (spot map)'),
    ('spots', 'sdo_input_path', str, 'SDO_input', 'SDO_input of the old package'),
    ('planet', 'simulate_planet', int, 0, 'no planet'),
]
# read by the old code but without effect on a forward simulation, or not read at all
IGNORED = {('files', 'orders_crx_filename'), ('general', 'stagger_spectra_folder'), ('general', 'smoothing_kernel'),
           ('general', 'bb_ratio'), ('general', 'phoenix_spectra'), ('star', 'facular_area_ratio'),
           ('star', 'force_cifist'), ('spots', 'plot_grid_map'), ('rv', 'ccf_interpolation_spectra')}


# =============================================================================================
# configuration
# =============================================================================================

def _parse(conf_file):
    """ConfigParser of the file, tolerant: inline comments, repeated keys, section headers such as [[files]]."""
    text = Path(conf_file).read_text()
    text = re.sub(r'(?m)^\s*\[+\s*([A-Za-z_]+)\s*\]+', r'[\1]', text)
    parser = configparser.ConfigParser(inline_comment_prefixes='#', strict=False)
    parser.read_string(text)
    return parser


def _get(parser, section, key, typ):
    raw = parser.get(section, key).strip()
    if typ is int:
        return int(float(raw))
    return typ(raw)


def read_old_config(conf_file, old_package=None, observables=('lc',), paths=None):
    """Settings of an old configuration file, with the defaults of the missing option keys.
    Returns (cfg, report): cfg {key: value} (key names of the old code, lower case), report {'missing defaults',
    'ignored', 'paths', 'notes'}. Raises an error that lists the missing physical keys."""
    conf_file = Path(conf_file)
    parser = _parse(conf_file)
    if old_package is None:
        old_package = conf_file.parent.parent if conf_file.parent.name == 'conf' else conf_file.parent
    old_package = Path(old_package)
    report = {'file': str(conf_file), 'old package': str(old_package), 'missing defaults': [], 'ignored': [],
              'paths': {}, 'notes': []}
    if not (old_package / 'models').is_dir() and not (old_package / 'masks').is_dir():
        report['notes'].append('%s has no models/ or masks/ folder: give the folder of the old starsim code with '
                               'old_package=... if files are not found' % old_package)
    cfg = {'old_package': old_package, '_old_bugs': False}

    known = set()
    for section, key, typ, default, meaning in OPTION_DEFAULTS:
        known.add((section, key))
        if parser.has_option(section, key):
            cfg[key] = _get(parser, section, key, typ)
        else:
            cfg[key] = default
            report['missing defaults'].append('[%s] %s -> %r (%s)' % (section, key, default, meaning))
    missing = []
    for section, key, typ in REQUIRED:
        known.add((section, key))
        if parser.has_option(section, key):
            cfg[key] = _get(parser, section, key, typ)
        else:
            missing.append('[%s] %s' % (section, key))
    rv_wanted = any(o in observables for o in ('rv', 'bis', 'fwhm', 'contrast'))
    for section, key, typ, when in CONDITIONAL:
        known.add((section, key))
        if parser.has_option(section, key):
            cfg[key] = _get(parser, section, key, typ)
        elif key not in cfg:
            cfg[key] = None
    needed = {'RV observables': rv_wanted, 'simulate_planet = 1': bool(cfg['simulate_planet']),
              'use_phoenix_limb_darkening = 0': not cfg['use_phoenix_limb_darkening'],
              'spectral_library = phoenix': cfg['spectral_library'] == 'phoenix'}
    for section, key, typ, when in CONDITIONAL:
        if cfg.get(key) is None and needed[when]:
            missing.append('[%s] %s (needed for %s)' % (section, key, when))
    if missing:
        raise KeyError('%s lacks keys that have no sensible default:\n  %s' % (conf_file, '\n  '.join(missing)))

    for section in parser.sections():
        if section == 'optimization':
            report['ignored'].append('[optimization] (inversion options)')
            continue
        for key in parser[section]:
            if (section, key) not in {(a, b.lower()) for a, b in known}:
                why = 'not used by the old code for a forward simulation' if (section, key) in IGNORED else 'unknown key'
                report['ignored'].append('[%s] %s (%s)' % (section, key, why))

    # files
    def find(value, folders, what, is_dir=False):
        overrides = paths or {}
        if what in overrides:
            report['paths'][what] = '%s (given)' % overrides[what]
            return Path(overrides[what])
        if not value:
            return None
        p = Path(value)
        base = re.split(r'[\\/]+', str(value).rstrip('\\/'))[-1]       # also for Windows paths (C:\\...\\name)
        candidates = [p] if p.is_absolute() else [old_package / f / p for f in folders] + [conf_file.parent / p]
        candidates += [old_package / f / base for f in folders]
        for c in candidates:
            if (c.is_dir() if is_dir else c.is_file()):
                if c != p:
                    report['paths'][what] = '%s (the file gives %s)' % (c, value)
                else:
                    report['paths'][what] = str(c)
                return c
        report['paths'][what] = 'NOT FOUND: %s' % value
        return None

    cfg['filter_path'] = find(cfg['filter_name'], ['models/filters', 'filters'], 'filter') if cfg['filter_name'] else None
    if cfg['filter_name'] and cfg['filter_path'] is None:
        report['notes'].append('filter %s not found: box filter between the wavelength limits (as the old code)'
                               % cfg['filter_name'])
    cfg['mask_path'] = find(cfg['ccf_mask'], ['masks'], 'ccf_mask') if cfg.get('ccf_mask') else None
    cfg['weight_path'] = (find(cfg['path_weight_lines'], ['masks'], 'path_weight_lines')
                          if cfg['ccf_weight_lines'] else None)
    if cfg['spectral_library'] in ('mps', 'mps_highres'):
        # only a folder of the same name is accepted in place of a missing one (other MURaM spectra are not substituted)
        cfg['mps_folder'] = find(cfg['mps_spectra_folder'].rstrip('/'), ['.'], 'mps_spectra_folder', is_dir=True)
    if cfg['sdo_input']:
        cfg['sdo_folder'] = find(cfg['sdo_input_path'].rstrip('/'), ['.'], 'SDO_input_path', is_dir=True)
    cfg['phoenix_mu_folder'] = old_package / 'models' / 'Phoenix_mu'
    cfg['phoenix_folder'] = old_package / 'models' / 'Phoenix'
    if paths:
        for k in ('phoenix_mu_folder', 'phoenix_folder', 'mps_folder', 'sdo_folder'):
            if k in paths:
                cfg[k] = Path(paths[k])
                report['paths'][k] = '%s (given)' % paths[k]
    if cfg['spectral_library'] == 'mps_highres' and cfg['instrument'] == 'HARPS-N':
        for k, name in INSTRUMENT_FILES.items():
            for where, path in (('given', (paths or {}).get(k)), ('sunsim/data/instruments', INSTRUMENT_FOLDER / name),
                                ('path of the old code', OLD_INSTRUMENT_PATHS[k])):
                if path is not None and Path(path).exists():
                    cfg[k] = Path(path)
                    report['paths'][k] = '%s (%s)' % (path, where)
                    break
            else:
                cfg[k] = None
                report['paths'][k] = 'NOT FOUND (%s in %s, or paths={%r: ...})' % (name, INSTRUMENT_FOLDER, k)
    return cfg, report


# =============================================================================================
# spectra, as the old code reads them
# =============================================================================================

def _old_continuum(wv, flux, lower, upper):
    """Continuum of the old code: maximum of the flux in 19 bins between lower-1 and upper+1 A, 6th degree polyfit."""
    bins = np.linspace(lower - 1, upper + 1, 20)
    x, y = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        idx = (wv > a) & (wv < b)
        if not idx.any():
            raise ValueError('the wavelength range %g-%g A goes beyond the spectra (%.0f-%.0f A): no point to fit the '
                             'continuum between %.0f and %.0f A (the old code fails there too)'
                             % (lower, upper, wv.min(), wv.max(), a, b))
        k = np.argmax(flux[idx])
        x.append(wv[idx][k])
        y.append(flux[idx][k])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', _RANK_WARNING)
        coeff = np.polyfit(x, y, 6)
    return np.poly1d(coeff)(wv)


def _flatten(cfg, wv, flux):
    """High resolution spectrum as the old code uses it: divided by its continuum (flux_flat) or by 5e14."""
    if cfg['flux_flat']:
        return flux / _old_continuum(wv, flux, cfg['wavelength_lower_limit'], cfg['wavelength_upper_limit'])
    return flux / 5e14


def _muram(cfg, kind, surface):
    """MURaM intensities {mu: (wav, intensity)} (erg/s/cm^2/cm/sr), mu ascending; kind 'low' or 'high'."""
    data = np.load(Path(cfg['mps_folder']) / MURAM_FILES[(kind, surface)], allow_pickle=True).item()
    out = {}
    for key in sorted(data, key=float):
        wav = np.asarray(data[key]['wav'], dtype=np.float64)
        out[float(key)] = (wav, np.asarray(data[key]['intensity'], dtype=np.float64) / (1e-8 * wav)**2 * C_CGS)
        if np.isnan(out[float(key)][1]).any():
            cfg.setdefault('_notes', set()).add('%s contains NaN (mu = %s): the outputs that use it are NaN, as with '
                                                'the old code' % (MURAM_FILES[(kind, surface)], key))
    return out


def _lowres(cfg, surface, context='lc'):
    """Low resolution intensities of a surface ('ph', 'sp', 'fc') in the band: acd (ascending), wav, rows (n_mu, n_wav).
    context: 'lc' (photometry) or 'rv' (brightness of the rings for the CCFs), for the old treatment of the angles."""
    lo, hi = cfg['wavelength_lower_limit'], cfg['wavelength_upper_limit']
    T_ph = cfg['temperature_photosphere']
    T_sp = T_ph - cfg['spot_T_contrast']
    if cfg['spectral_library'] == 'phoenix':
        T = T_sp if surface == 'sp' else T_ph           # Phoenix faculae: photosphere x limb brightening (bolometric)
        if cfg['_old_bugs']:
            # old: the rows of the models are added without interpolation in mu, and every surface takes the angles of
            # the photosphere models in photometry, of the spot models in RV (the last ones read)
            _, wav, rows = _phoenix_specint_old(cfg, T, lo, hi)
            acd = _phoenix_specint_old(cfg, T_ph if context == 'lc' else T_sp, lo, hi)[0]
        else:
            acd, wav, rows = _phoenix_specint(cfg, T, lo, hi)
        # the old code adds a row of zeros at mu = 0
        acd = np.concatenate(([0.0], acd))
        rows = np.vstack([np.zeros(len(wav)), rows])
        if cfg['mps_mu_grid']:
            grid = np.linspace(0.1, 1.0, 10)
            if cfg['_old_bugs'] and context == 'rv':
                # old RV: interpolation weight with the angle acd[i] of the original grid (old order: descending, 0
                # last) instead of the target angle
                acd_old = acd[::-1]
                new = []
                for i, m in enumerate(grid):
                    j = min(max(int(np.searchsorted(acd, m, side='left')), 1), len(acd) - 1)
                    new.append(rows[j - 1] + (rows[j] - rows[j - 1]) * (acd_old[i] - acd[j - 1]) / (acd[j] - acd[j - 1]))
                rows = np.array(new)
            else:
                rows = np.array([_interp_mu(acd, rows, m) for m in grid])
            acd = grid
        return acd, wav, rows
    data = _muram(cfg, 'low', 'fc' if surface == 'fc' else 'ph')
    acd = np.array(list(data))
    wav = data[acd[0]][0]
    keep = (wav > lo) & (wav < hi)
    rows = np.array([data[m][1][keep] for m in acd])
    wav = wav[keep]
    if surface == 'sp':
        rows = rows * black_body(wav, T_sp) / black_body(wav, T_ph)
    return acd, wav, rows


def _phoenix_specint(cfg, T, lo, hi):
    """Phoenix SPECINT intensities interpolated in T and logg (lo < wavelength < hi), mu ascending. The models of the
    interpolation can tabulate different mu (e.g. 5700 and 5800 K): every model is first interpolated in mu on the angles
    of the model with the lowest T and logg (the angles used by the old code, which added the rows of the models
    without this interpolation)."""
    from . import spectrum_utils as su
    weights = su.phoenix_weights(Path(cfg['phoenix_mu_folder']), T, cfg['logg'], 0.0, 'SPECINT')
    models = sorted(weights, key=lambda f: f.name)              # lowest T, then lowest logg first
    data = {f: su._read_specint(str(f), os.stat(f).st_mtime_ns, float(lo), float(hi)) for f in models}
    mu0, wav, _ = data[models[0]]
    order = np.argsort(mu0)
    mu0 = np.asarray(mu0)[order]
    total = None
    resampled = False
    for f in models:
        mu, _, inten = data[f]
        o = np.argsort(mu)
        mu, inten = np.asarray(mu)[o], np.asarray(inten)[o]
        if not np.array_equal(mu, mu0):
            resampled = True
            j = np.clip(np.searchsorted(mu, mu0), 1, len(mu) - 1)
            w = ((mu0 - mu[j - 1]) / (mu[j] - mu[j - 1]))[:, None]
            inten = inten[j - 1] + w * (inten[j] - inten[j - 1])
        total = weights[f] * inten if total is None else total + weights[f] * inten
    if resampled:
        cfg.setdefault('_notes', set()).add('Phoenix SPECINT models with different mu angles (T = %g K): interpolated '
                                            'in mu on the angles of %s before the interpolation in T and logg'
                                            % (T, models[0].name))
    return mu0, np.asarray(wav), total


def _phoenix_specint_old(cfg, T, lo, hi):
    """Phoenix SPECINT as the old code: the rows of the models of the interpolation added without interpolation in mu
    (every model in its own order, mu ascending), with the angles of the lowest-T, lowest-logg model."""
    from . import spectrum_utils as su
    weights = su.phoenix_weights(Path(cfg['phoenix_mu_folder']), T, cfg['logg'], 0.0, 'SPECINT')
    models = sorted(weights, key=lambda f: f.name)
    data = {f: su._read_specint(str(f), os.stat(f).st_mtime_ns, float(lo), float(hi)) for f in models}
    rows = None
    for f in models:
        o = np.argsort(data[f][0])
        r = weights[f] * np.asarray(data[f][2])[o]
        rows = r if rows is None else rows + r
    mu0 = np.sort(np.asarray(data[models[0]][0]))
    return mu0, np.asarray(data[models[0]][1]), rows


def _interp_mu(acd, rows, mu):
    """Linear interpolation of rows (tabulated at the ascending angles acd) at mu, as the old code (acd_low < mu <= acd_upp)."""
    j = int(np.searchsorted(acd, mu, side='left'))
    j = min(max(j, 1), len(acd) - 1)
    return rows[j - 1] + (rows[j] - rows[j - 1]) * (mu - acd[j - 1]) / (acd[j] - acd[j - 1])


def _ring_intensity(cfg, acd, rows, mu, limb_lim):
    """Intensity of a ring at mu with the old limb rules (limb_lim, limb_extrapolation, limb darkening law)."""
    phoenix = cfg['spectral_library'] == 'phoenix'
    if not cfg['use_phoenix_limb_darkening']:
        if cfg['_old_bugs'] and not phoenix:
            return rows[0] * _ld_law(cfg, mu)             # old (MURaM): law x spectrum at the smallest mu
        return rows[-1] * _ld_law(cfg, mu)                # law x disc-centre spectrum
    if mu > limb_lim:
        return _interp_mu(acd, rows, mu)
    if cfg['limb_extrapolation'] == 'constant':
        if cfg['_old_bugs'] and phoenix:
            return rows[-1]                               # old (Phoenix): the disc-centre intensity
        positive = np.flatnonzero(acd > 0)[0]
        return rows[positive]                             # the smallest tabulated mu
    if limb_lim == acd.min():
        return mu / limb_lim * rows[0]
    return mu / limb_lim * _interp_mu(acd, rows, limb_lim)


def _ld_law(cfg, mu):
    q1, q2, law = cfg['limb_darkening_q1'], cfg['limb_darkening_q2'], cfg['limb_darkening_law']
    if law == 'linear':
        return 1 - q1 * (1 - mu)
    if law == 'quadratic':
        a, b = 2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)
        return 1 - a * (1 - mu) - b * (1 - mu)**2
    if law == 'sqrt':
        a, b = np.sqrt(q1) * (1 - 2 * q2), 2 * np.sqrt(q1) * q2
        return 1 - a * (1 - mu) - b * (1 - np.sqrt(mu))
    if law == 'log':
        a, b = q2 * q1**2 + 1, q1**2 - 1
        return 1 - a * (1 - mu) - b * mu * (1 - np.log(mu))
    raise ValueError("limb_darkening_law must be linear, quadratic, sqrt or log, got %r" % law)


def _limb_brightening(cfg, mu):
    """Bolometric limb brightening of the Phoenix faculae: ((T + dT(mu)) / T)^4."""
    T = cfg['temperature_photosphere']
    if cfg['meunier']:
        return ((T + (250.9 - 407.7 * mu + 190.9 * mu**2)) / T)**4
    return ((T + (155 - 290 * mu + 128 * mu**2)) / T)**4


def _cifist(mu):
    """CIFIST bisector (km/s as a function of the CCF height) at mu, as cifist_coeff_interpolate of the old code."""
    amv = np.arange(1, 0.0, -0.1)
    cxm = np.array([[-3.51974861, 11.1702017, -13.22368296, 6.67694456, -0.63201573, -0.44695616, -0.36838495],
                    [-4.05903967, 13.21901003, -16.47215949, 9.51023171, -2.13104764, -0.05153799, -0.36973749],
                    [-3.92153131, 12.76694663, -15.96958217, 9.39599116, -2.34394028, 0.12546611, -0.42092905],
                    [-3.81892968, 12.62209118, -16.06973368, 9.71487198, -2.61439945, 0.25356088, -0.43310756],
                    [-5.37213406, 17.6604689, -22.52477323, 13.91461247, -4.13186181, 0.60271171, -0.46427559],
                    [-6.35351933, 20.92046705, -26.83933359, 16.86220487, -5.28285592, 0.90643187, -0.47696283],
                    [-7.67270144, 25.60866105, -33.4381214, 21.58855269, -7.1527039, 1.35990694, -0.48001707],
                    [-9.24152009, 31.09337903, -41.07410957, 27.04196984, -9.32910982, 1.89291407, -0.455407],
                    [-11.62006536, 39.30962189, -52.38161244, 34.98243089, -12.40650704, 2.57940618, -0.37337442],
                    [-14.14768805, 47.9566719, -64.20294114, 43.23156971, -15.57423374, 3.13318175, -0.14451226]])
    if mu <= 0.1:
        c = cxm[9] + (cxm[8] - cxm[9]) * (mu - amv[9]) / (amv[8] - amv[9])
    else:
        low = np.max(amv[amv < mu])
        i_low, i_upp = np.where(amv == low)[0][0], np.where(amv == np.min(amv[amv >= mu]))[0][0]
        c = cxm[i_low] + (cxm[i_upp] - cxm[i_low]) * (mu - amv[i_low]) / (amv[i_upp] - amv[i_low])
    return np.poly1d(c)


# =============================================================================================
# grids
# =============================================================================================

def _filter(cfg, wav):
    if cfg['filter_path'] is None:
        return np.where((wav >= cfg['wavelength_lower_limit']) & (wav <= cfg['wavelength_upper_limit']), 1.0, 0.0)
    w, f = np.loadtxt(cfg['filter_path'], unpack=True)
    order = np.argsort(w)
    return np.interp(wav, w[order], f[order], left=0.0, right=0.0)


def _limb_lim(cfg, acd):
    lim = cfg['limb_lim']
    if lim is None or lim < acd.min():
        return acd.min()
    return lim


def _photometric_grids(cfg, geometry, has_faculae):
    """Brightness of one cell of every ring for the 3 surfaces (old compute_immaculate_lc / _facula_lc)."""
    amu, pare = geometry['amu'], geometry['pare']
    grids = {}
    for surface in ('ph', 'sp', 'fc'):
        if surface == 'fc' and not has_faculae:
            grids['fc'] = grids['ph'].copy()
            continue
        src = 'ph' if (surface == 'fc' and cfg['spectral_library'] == 'phoenix') else surface
        acd, wav, rows = _lowres(cfg, src)
        lim = _limb_lim(cfg, acd)
        filt = _filter(cfg, wav)
        b = np.array([np.sum(_ring_intensity(cfg, acd, rows, mu, lim) * pare[i] / (4 * np.pi) * filt)
                      for i, mu in enumerate(amu)])
        if surface == 'fc' and cfg['spectral_library'] == 'phoenix':
            b = b * np.array([_limb_brightening(cfg, mu) for mu in amu])
        grids[surface] = b
    if cfg['_old_bugs'] and cfg['sdo_input']:
        grids['fc'] = grids['sp'].copy()          # old, 2D maps: the faculae have the brightness of the spots
    return grids


class RingCCF(object):
    """CCF of every ring for one kind of surface, in the form StarSim uses (as a CCF_grid after built_grid()):
    rvs (velocity grid), ccf_rings (N_rings, n_rv: StarSim multiplies them by the projected area of a cell / 4 pi),
    rvs_ring (N_rings, n_rv: velocity axis of the CCF of every ring, with its bisector)."""
    def __init__(self, N_rings, rvs, ccf_rings, rvs_ring, ccf_centre=None):
        self.N_rings = N_rings
        self.rvs = np.asarray(rvs, dtype=np.float64)
        self.ccf_rings = np.asarray(ccf_rings, dtype=np.float64)
        self.rvs_ring = np.asarray(rvs_ring, dtype=np.float64)
        self.ccf_centre = ccf_centre


def _old_mask(cfg):
    """Lines and weights of the mask as the old code reads them (air -> vacuum for 3 columns, order weights, sum 1)."""
    d = np.loadtxt(cfg['mask_path'], unpack=True)
    if len(d) == 2:
        wvm, fm = np.atleast_1d(d[0]).astype(float), np.atleast_1d(d[1]).astype(float)
    elif len(d) == 3:
        wvm, fm = np.atleast_1d(air2vacuum((d[0] + d[1]) / 2)), np.atleast_1d(d[2]).astype(float)
    else:
        raise ValueError('mask %s: two (wavelength, weight) or three (wv1, wv2, weight) columns' % cfg['mask_path'])
    fm = fm.copy()
    if cfg['ccf_weight_lines']:
        if cfg['weight_path'] is None:
            raise FileNotFoundError('ccf_weight_lines = 1 but the file of the order weights was not found')
        order, weight, wvi, wvf = np.loadtxt(cfg['weight_path'], unpack=True)
        j = 0                                   # weight_mask of the old code, with its behaviour
        for i in range(len(wvm)):
            if wvm[i] < wvi[j]:
                fm[i] = 0.0
            elif wvi[j] <= wvm[i] <= wvf[j]:
                fm[i] = fm[i] * weight[j]
            elif wvm[i] > wvf[j]:
                j += 1
                if j >= len(wvi):
                    fm[i] = 0.0
                    break
    return np.asarray(wvm, dtype=np.float64), np.asarray(fm / np.sum(fm), dtype=np.float64)


def _spot_rv_axis(cfg, rv, ccf, mu):
    """Velocity axis of a spot CCF (old: remove_bis_spot, add_median_diff, Dumusque bisector x convective_shift)."""
    dum = dumusque_bisector(bool(cfg['ccf_norm']))(mu)
    add = dum(ccf) * 1000 * cfg['convective_shift']
    if cfg['spectral_library'] == 'phoenix':
        return rv - bisector_fit(rv, ccf, 'cubic')(ccf) + add
    if cfg['remove_bis_spot']:
        bis = bisector_fit(rv, ccf, 'cubic')(ccf)
        if cfg['add_median_diff'] > 0:
            # as the old code: + median of the bisector for mps, + median of minus the bisector for mps_highres
            median = np.median(bis) if cfg['spectral_library'] == 'mps' else np.median(-bis)
            return rv - bis + median + cfg['add_median_diff'] + add
        return rv - bis + add
    return rv + cfg['add_median_diff'] + add


def _instrument_ccf(cfg, wv, flux, rv, wvm, fm, N):
    """CCF_grid that computes the CCFs order by order, as the old instrument = HARPS-N option: spectrum degraded to
    R = 118000, resampled on the 72 orders of the HARPS wavelength file, times the blaze of every order times the
    efficiency (degree-10 polynomial fitted to the efficiency table, nm -> A, as the old code), CCF of every order with
    its mask lines, sum of the orders."""
    absent = [k for k in INSTRUMENT_FILES if cfg.get(k) is None]
    if absent:
        raise FileNotFoundError('instrument = HARPS-N: %s not found; put %s in %s or give paths={...}'
                                % (', '.join(absent), ', '.join(INSTRUMENT_FILES[k] for k in absent), INSTRUMENT_FOLDER))
    from astropy.io import fits
    wvb = fits.getdata(cfg['wave_file'])                    # float32 in the file (CCF_grid computes the edges in float64)
    blaze = fits.getdata(cfg['blaze_file'])
    table = np.genfromtxt(cfg['efficiency_file'], delimiter=',', names=True)
    x, y = table[table.dtype.names[0]] * 10, table[table.dtype.names[1]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', _RANK_WARNING)      # the old polynomial fit is badly conditioned
        efficiency = np.poly1d(np.polyfit(x, y, 10))
    # the efficiency in float64 (see the differences with the old code), blaze * efficiency stored in float32 as the
    # old in-place product blaze_orders[i] *= eff on the float32 blaze of the file
    eff = efficiency(np.asarray(wvb, dtype=np.float64))
    blaze = (np.asarray(blaze, dtype=np.float32) * eff).astype(np.float32)
    return CCF_grid(N, {1.0: {'wav': wv, 'intensity': flux}}, rv, wvm, fm, normalization=bool(cfg['ccf_norm']),
                    instrument='HARPS-N', blaze_function=(wvb, blaze), instrumental_efficiency=None,
                    mu_ratio=np.ones(N))


def _ccf_grids(cfg, geometry, has_faculae, report):
    """CCF of every ring for the 3 surfaces (old compute_immaculate_*_rv), as RingCCF objects."""
    N, amu, pare = geometry['N'], geometry['amu'], geometry['pare']
    lib = cfg['spectral_library']
    lo, hi = cfg['wavelength_lower_limit'], cfg['wavelength_upper_limit']
    step, rng = 1000 * cfg['ccf_rv_step'], 1000 * cfg['ccf_rv_range']
    rv = np.arange(-rng, rng + step, step)
    wvm, fm = _old_mask(cfg)
    norm = bool(cfg['ccf_norm'])
    T_ph = cfg['temperature_photosphere']
    T = {'ph': T_ph, 'sp': T_ph - cfg['spot_T_contrast'], 'fc': T_ph + cfg['facula_T_contrast']}
    cs = cfg['convective_shift']

    def ccf_of(wv, flux, phoenix_grid):
        if wv.max() < wvm.max() + 1 or wv.min() > wvm.min() - 1:
            raise ValueError('the wavelength range must cover the mask (+-1 A): %.1f-%.1f A, mask %.1f-%.1f A'
                             % (wv.min(), wv.max(), wvm.min(), wvm.max()))
        return nbspectra.cross_correlation_mask(rv, np.ascontiguousarray(wv, dtype=np.float64),
                                                np.ascontiguousarray(flux, dtype=np.float64), wvm, fm, phoenix_grid, norm)

    out = {}
    if lib == 'mps_highres':
        by_order = cfg['instrument'] == 'HARPS-N'
        if by_order:
            report['notes'].append('instrument HARPS-N: CCFs order by order (resolution 118000, HARPS orders, blaze x '
                                   'efficiency), as the old code')
        elif cfg['instrument'] not in ('None', '', 'none'):
            report['notes'].append("instrument %s: the old code only used its blaze files for HARPS-N; CCF of the "
                                   "whole spectrum, as the old code" % cfg['instrument'])
        if not cfg['highres_mu']:
            report['notes'].append('highres_mu = 0 (smoothed ratio of the old code) is not supported: the spectrum is '
                                   'interpolated at the mu of every ring (highres_mu = 1)')
        for surface in ('ph', 'sp', 'fc'):
            if surface == 'fc' and not has_faculae:
                out['fc'] = out['ph']
                continue
            data = _muram(cfg, 'high', 'fc' if surface == 'fc' else 'ph')
            acd = np.array(list(data))
            wav = data[1.0][0]
            keep = (wav > lo - 1) & (wav < hi + 1)
            wv = wav[keep]
            rows = []
            for m in acd:
                f = data[m][1][keep]
                if surface == 'sp':
                    f = f * black_body(wv, T['sp']) / black_body(wv, T_ph)
                rows.append(_old_continuum_div(cfg, wv, f) if cfg['flux_flat'] else f)
            rows = np.array(rows)
            lim = _limb_lim(cfg, acd)
            ccfs, axes = [], []
            orders = _instrument_ccf(cfg, wv, rows[-1], rv, wvm, fm, N) if by_order else None
            for i, mu in enumerate(amu):
                f_ring = _ring_intensity(cfg, acd, rows, mu, lim)
                c = orders._ccf_of_spectrum(f_ring) if by_order else ccf_of(wv, f_ring, False)
                ccfs.append(c)
                axes.append(_spot_rv_axis(cfg, rv, c, mu) if surface == 'sp' else rv)
            # old: ccf_ring = ccf * pare / 4 pi (ccf * pare by order); StarSim multiplies ccf_rings by pare / 4 pi
            scale = 4 * np.pi if by_order else 1.0
            out[surface] = RingCCF(N, rv, scale * np.array(ccfs), np.array(axes))
        return rv, out

    # phoenix and mps: the CCF of the disc-centre spectrum, weighted by the brightness of every ring
    spectra = {}
    if lib == 'phoenix':
        for surface in ('ph', 'sp', 'fc'):
            if surface == 'fc' and not has_faculae:
                continue
            s = interpolate_phoenix(cfg['phoenix_folder'], T[surface], cfg['logg'], wavelength_lower_limit=lo,
                                    wavelength_upper_limit=hi, overhead=1.0)[1.0]
            spectra[surface] = (s['wav'], _flatten(cfg, s['wav'], s['intensity']))
    else:
        for surface in ('ph', 'fc'):
            if surface == 'fc' and not has_faculae:
                continue
            wav, f = _muram(cfg, 'high', surface)[1.0]
            keep = (wav > lo - 1) & (wav < hi + 1)
            spectra[surface] = (wav[keep], _flatten(cfg, wav[keep], f[keep]))
        wv, fph = spectra['ph']
        spectra['sp'] = (wv, fph * black_body(wv, T['sp']) / black_body(wv, T_ph))
    ccf = {s: ccf_of(wv, f, lib == 'phoenix') for s, (wv, f) in spectra.items()}

    # brightness of one cell of every ring (old sccf / Ngrid / flxph), low resolution intensities without filter
    def brightness(surface):
        src = 'ph' if (surface == 'fc' and lib == 'phoenix') else surface
        acd, wav, rows = _lowres(cfg, src, 'rv')
        lim = _limb_lim(cfg, acd)
        b = np.array([np.sum(_ring_intensity(cfg, acd, rows, mu, lim)) for mu in amu])
        if surface == 'fc' and lib == 'phoenix':
            b = b * np.array([_limb_brightening(cfg, mu) for mu in amu])
        return b
    b_ph = brightness('ph')
    total = np.sum(geometry['Ngrid_in_ring'] * b_ph * pare / (4 * np.pi))
    for surface in ('ph', 'sp', 'fc'):
        if surface == 'fc' and not has_faculae:
            out['fc'] = out['ph']
            continue
        b = b_ph if surface == 'ph' else brightness(surface)
        c = ccf[surface]
        if lib == 'phoenix':
            centre_axis = rv - bisector_fit(rv, c, 'cubic')(c)          # bisector of the reference CCF removed
            if surface == 'ph':
                axes = [centre_axis + _cifist(mu)(c) * 1000 * cs for mu in amu]
            elif surface == 'sp':
                axes = [_spot_rv_axis(cfg, rv, c, mu) for mu in amu]
            else:
                axes = [centre_axis + dumusque_bisector(norm)(mu)(c) * 1000 * cs for mu in amu]
        else:                                                           # mps: MURaM keeps its bisector
            if surface == 'sp':
                axes = [_spot_rv_axis(cfg, rv, c, mu) for mu in amu]
            else:
                axes = [rv for mu in amu]
        # old: ccf_ring = ccf * (brightness of one cell) / (brightness of the star); StarSim multiplies by pare / 4 pi
        ccf_rings = np.array([c * b[i] / total for i in range(N)])
        out[surface] = RingCCF(N, rv, ccf_rings, np.array(axes), ccf_centre=c)
    if cfg['_old_bugs'] and cfg['sdo_input']:
        out['fc'] = out['ph']                     # old, 2D maps (phoenix, mps): the faculae have the CCF of the photosphere
    return rv, out


def _old_continuum_div(cfg, wv, f):
    return f / _old_continuum(wv, f, cfg['wavelength_lower_limit'], cfg['wavelength_upper_limit'])


# =============================================================================================
# regions, planet
# =============================================================================================

def _regions(cfg, spot_map):
    """active_region objects from the rows of the old spot map."""
    rows = np.atleast_2d(np.loadtxt(spot_map) if isinstance(spot_map, (str, Path)) else np.asarray(spot_map, dtype=float))
    law = cfg['spots_evo_law']
    B = cfg['differential_rotation']
    regions = []
    for r in rows:
        typ = 'sp' if r[0] == 0 else 'fc'
        t0, life, colat, lon = r[1], r[2], r[3], r[4]
        c = list(r[5:8]) + [0.0] * max(0, 3 - len(r[5:8]))
        if law == 'constant':
            evolution = None
        elif law == 'linear':
            evolution = (lambda dt, size, c1=c[0], c2=c[1], T=life: c1 + dt * (c2 - c1) / T)
        elif law == 'quadratic':
            evolution = (lambda dt, size, c1=c[0], T=life: -4 * c1 * dt * (dt - T) / T**2)
        else:
            raise ValueError("spots_evo_law %r is not implemented in the old code either (constant, linear, quadratic)" % law)
        regions.append(active_region(typ, c[0], colat, lon, appearance_time=t0, lifetime=life,
                                     reference_time=cfg['reference_time'], evolution=evolution))
    return regions


def _old_projection(n, N, Ngrid_in_ring):
    """Cell of every pixel of an n x n map as projection_pxl_to_ss_grid of the old code (flattened, -1 outside the
    disc): pixel coordinates linspace(-1, 1, n), ring limits halfway between the projected radii of the rings (the
    new projection uses the angular edges of the rings), cell = nearest position angle, image flipped horizontally."""
    _, _, _, _, rs, _, _, _, _, _, _ = nbspectra.generate_grid_coordinates_nb(N)
    r_unique = np.unique(np.round(np.sort(np.asarray(rs)), 6))
    lim_r = np.append((r_unique[1:N] + r_unique[:N - 1]) / 2.0, 1.0)
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)                         # X[i, j] = x[j], Y[i, j] = x[i]
    R = np.sqrt(X**2 + Y**2)
    inside = R <= 1.0
    ring = np.minimum(np.searchsorted(lim_r, R, side='left'), N - 1)
    angle = np.arctan2(X, Y) + np.pi / 2.0
    angle = np.where(angle >= 0, angle, angle + 2 * np.pi)
    nin = np.asarray(Ngrid_in_ring, dtype=np.int64)
    offset = np.concatenate(([0], np.cumsum(nin)[:-1]))
    n_cells = nin[ring]
    idx = np.round(((angle + np.pi / 2.0) % (2 * np.pi)) / (2 * np.pi / n_cells)).astype(np.int64) % n_cells
    cells = np.where(inside, offset[ring] + idx, -1)[:, ::-1]
    return np.ascontiguousarray(cells, dtype=np.int32).ravel()


def _old_projection_mask(folder):
    """active_region_mask that projects the maps on the grid as the old code (map_projection='old')."""
    from .active_region_mask import active_region_mask

    class OldProjectionMask(active_region_mask):
        def projection(self, shape, N_rings, Ngrid_in_ring):
            key = (shape[0], shape[1], N_rings)
            with self._lock:
                if key not in self._projections:
                    if shape[0] != shape[1]:
                        raise ValueError('the old projection needs square maps, got %d x %d' % shape)
                    cells = _old_projection(shape[0], N_rings, Ngrid_in_ring)
                    npix = np.bincount(cells[cells >= 0], minlength=int(np.sum(Ngrid_in_ring))).astype(np.float64)
                    self._projections[key] = (cells, npix)
                return self._projections[key]
    return OldProjectionMask(folder, origin='lower', flip_horizontal=False)


def _planet(cfg):
    a = 4.2097 * cfg['planet_period']**(2 / 3) * cfg['mass']**(1 / 3) / cfg['radius']      # old semi-major axis
    return planet(planet_period=cfg['planet_period'], planet_transit_t0=cfg['planet_transit_t0'],
                  planet_radius=cfg['planet_radius'], planet_semi_major_axis=a,
                  planet_impact_param=cfg['planet_impact_param'], planet_esinw=cfg['planet_esinw'],
                  planet_ecosw=cfg['planet_ecosw'], planet_spin_orbit_angle=np.deg2rad(cfg['planet_spin_orbit_angle']),
                  planet_semi_amplitude=cfg['planet_semi_amplitude'])


# =============================================================================================
# main function
# =============================================================================================

def simulate_old_config(conf_file, times=None, observables=('lc',), spot_map=None, old_package=None, paths=None,
                        map_projection=None, reproduce_old_bugs=False, verbose=True):
    """Run SunSim with the parameters of an old configuration file (see the module docstring).

    conf_file   : path of the old .conf file
    times       : epochs [days]; with 2D maps (sdo_input = 1) the epochs of the maps by default
    observables : as the old compute_forward: 'lc' and / or 'rv', 'bis', 'fwhm', 'contrast'
    spot_map    : circular regions: path of a spot map or array of its rows (default: spotmap.dat of the old package)
    old_package : folder of the old starsim code (models, masks, spotmap.dat); see the module docstring
    paths       : paths that replace those of the file: {'ccf_mask', 'filter', 'path_weight_lines', 'mps_spectra_folder'
                  (or 'mps_folder'), 'SDO_input_path' (or 'sdo_folder'), 'phoenix_folder', 'phoenix_mu_folder'}
    reproduce_old_bugs : True -> also reproduce the behaviours of the old code listed in "Differences with the old
                  code" (Phoenix angles, constant limb extrapolation with Phoenix, mps_mu_grid in RV, limb darkening law
                  with MURaM, faculae of the 2D maps, old projection of the maps): the outputs are those of the old code
                  (used by the tests against the old version, test_old_version.py). The rule for the faculae covered in
                  part by spots (generate_ff_core) is always the new one.
    map_projection : 2D maps only. None: 'old' with reproduce_old_bugs, 'new' otherwise. 'new': the pixels are given to
                  the cells with the angular edges of the rings (active_region_mask); 'old': as the old code (ring limits halfway between the projected radii of
                  the rings), which reproduces its results exactly. ~9 % of the pixels, near the ring limits, change of
                  ring; the filling factors and the signals differ by <1 % of their amplitude.
    verbose     : print the report (defaults used, keys ignored, paths, differences)
    Returns a dict with the keys of the old StarSim.results ('time', 'lc', 'flux', 'ff_ph', 'ff_sp', 'ff_fc', 'ff_pl',
    'rv', 'contrast', 'fwhm', 'bis', 'CCF', 'ccf_quiet', 'raw_xbis', 'raw_ybis'), 'report' and 'starsim' (the new
    StarSim object)."""
    observables = [observables] if isinstance(observables, str) else list(observables)
    if 'spec' in observables:
        raise NotImplementedError("'spec' is not converted: use flux_grid(..., mode='spectroscopy') and "
                                  "StarSim(mode=['spectroscopy']) of the new version")
    unknown = [o for o in observables if o not in ('lc', 'rv', 'bis', 'fwhm', 'contrast')]
    if unknown:
        raise ValueError('unknown observables %s (old compute_forward: lc, rv, bis, fwhm, contrast)' % unknown)
    do_lc = 'lc' in observables
    do_rv = any(o in observables for o in ('rv', 'bis', 'fwhm', 'contrast'))

    if paths is not None:      # accept the names of the configuration keys
        paths = dict(paths)
        for a, b in (('mps_spectra_folder', 'mps_folder'), ('SDO_input_path', 'sdo_folder'), ('filter_name', 'filter')):
            if a in paths:
                paths[b] = paths.pop(a)
    cfg, report = read_old_config(conf_file, old_package, observables, paths)
    cfg['_old_bugs'] = bool(reproduce_old_bugs)
    if map_projection is None:
        map_projection = 'old' if reproduce_old_bugs else 'new'
    lib = cfg['spectral_library']
    if lib == 'stagger':
        raise NotImplementedError("spectral_library = 'stagger' is not handled by the old compute_forward either")
    if lib not in ('phoenix', 'mps', 'mps_highres'):
        raise ValueError('unknown spectral_library %r' % lib)
    if do_rv and cfg['ccf_template'] != 'mask':
        raise NotImplementedError("ccf_template = %r: the old compute_forward only computes CCFs with a mask"
                                  % cfg['ccf_template'])
    if cfg['simulation_mode'] != 'grid':
        report['notes'].append("simulation_mode = %r: 'grid' is used (the only mode of the old compute_forward)"
                               % cfg['simulation_mode'])
    for key, folder in (('mps_folder', lib in ('mps', 'mps_highres')), ('mask_path', do_rv),
                        ('sdo_folder', bool(cfg['sdo_input']))):
        if folder and cfg.get(key) is None:
            name = {'mps_folder': 'mps_spectra_folder', 'mask_path': 'ccf_mask', 'sdo_folder': 'SDO_input_path'}[key]
            raise FileNotFoundError('%s not found: %s. Give it with paths={%r: ...}'
                                    % (name, report['paths'].get(name, report['paths']), name))
    if lib in ('mps', 'mps_highres'):
        needed = [MURAM_FILES[('low', 'ph')], MURAM_FILES[('low', 'fc')]]
        if do_rv:
            needed += [MURAM_FILES[('high', 'ph')], MURAM_FILES[('high', 'fc')]]
        absent = [f for f in needed if not (Path(cfg['mps_folder']) / f).exists()]
        if absent:
            raise FileNotFoundError('the MURaM folder %s lacks %s (the file names used by the old code)'
                                    % (cfg['mps_folder'], ', '.join(absent)))
    if reproduce_old_bugs:
        report['notes'].append('reproduce_old_bugs: the old behaviours are reproduced (except the rule of the faculae '
                               'covered in part by spots)')
    else:
        report['notes'].append('differences with the old code: see the module docstring (bugs not reproduced)')

    N = cfg['n_grid_rings']
    Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, pare = nbspectra.generate_grid_coordinates_nb(N)
    geometry = {'N': N, 'Ngrid_in_ring': np.asarray(Ngrid_in_ring, dtype=np.int64), 'amu': np.asarray(amu),
                'pare': np.asarray(pare)}
    inclination = np.deg2rad(90 - cfg['inclination'])
    B = cfg['differential_rotation']
    diff_rot = (1.698 * B / 2.66, 2.346 * B / 2.66)

    # regions
    mask = None
    regions = None
    if cfg['sdo_input']:
        from .active_region_mask import active_region_mask
        if map_projection == 'old':
            mask = _old_projection_mask(str(cfg['sdo_folder']))
        elif map_projection == 'new':
            # orientation of the old projection (projection_pxl_to_ss_grid): row 0 of the maps is the bottom of the disc
            mask = active_region_mask(str(cfg['sdo_folder']), origin='lower', flip_horizontal=False)
            report['notes'].append("2D maps: pixels given to the cells with the angular edges of the rings; "
                                   "map_projection='old' reproduces the old ring limits (differences < 1 %)")
        else:
            raise ValueError("map_projection must be 'new' or 'old', got %r" % (map_projection,))
        if times is None:
            times = mask.times
        has_faculae = True
    else:
        if times is None:
            raise ValueError('give the epochs: simulate_old_config(conf, times=...)')
        if spot_map is None:
            spot_map = cfg['old_package'] / 'spotmap.dat'
            report['paths']['spot map'] = str(spot_map)
        regions = _regions(cfg, spot_map)
        has_faculae = any(r.type == 'fc' for r in regions)
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))

    # grids
    grids, modes = {}, []
    if do_lc:
        g = _photometric_grids(cfg, geometry, has_faculae)
        grids.update(flux_grid_qp=g['ph'], flux_grid_sp=g['sp'], flux_grid_fc=g['fc'])
        modes.append('photometry')
    if do_rv:
        rv, c = _ccf_grids(cfg, geometry, has_faculae, report)
        grids.update(ccf_grid_qp=c['ph'], ccf_grid_sp=c['sp'], ccf_grid_fc=c['fc'])
        modes.append('ccf')

    planets = _planet(cfg) if cfg['simulate_planet'] else None
    ss = StarSim(times, N, inclination=inclination, rotation_period=cfg['rotation_period'], differential_rotation=diff_rot,
                 stellar_radius=cfg['radius'], active_regions_map=regions, active_regions_mask=mask, planet=planets,
                 mode=modes, planet_coverage='ramp' if cfg['_old_bugs'] else 'area', **grids)
    if do_rv:
        # the old code Doppler-shifts the cells with the rigid rotation velocity (also with differential rotation)
        theta = np.arccos(zs * np.cos(-inclination) - xs * np.sin(-inclination))
        phi = np.arctan2(ys, xs * np.cos(-inclination) + zs * np.sin(-inclination))
        ss.rvel = np.ascontiguousarray(ss.vsini * np.sin(theta) * np.sin(phi), dtype=np.float64)
    ss.generate_timeseries(verbose=False)

    res = {'time': times, 'ff_ph': ss.ff_quiet, 'ff_sp': ss.ff_sp, 'ff_fc': ss.ff_fc, 'ff_pl': ss.ff_pl,
           'report': report, 'starsim': ss}
    if do_lc:
        quiet = np.dot(geometry['Ngrid_in_ring'], g['ph'])
        res['flux'] = ss.flux_var
        res['lc'] = ss.flux_var / quiet
    if do_rv:
        q = StarSim(times[:1], N, inclination=inclination, rotation_period=cfg['rotation_period'],
                    stellar_radius=cfg['radius'], mode=['ccf'], ccf_grid_qp=c['ph'], ccf_grid_sp=c['sp'],
                    ccf_grid_fc=c['fc'])
        q.rvel = ss.rvel
        q.generate_timeseries(ccf_params=False, verbose=False)
        quiet_ccf = q.ccf_var[0]
        rv0, c0, _, _, _, _ = ss.compute_ccf_params(rv=ss.ccf_rv, ccf=[quiet_ccf])
        res['rv'] = ss.rv_var - rv0[0]                  # rv_var already has the Keplerian RV of the planets
        res['contrast'] = ss.contrast_var / c0[0]
        res['fwhm'] = ss.fwhm_var
        res['bis'] = ss.bis_var
        # the old compute_ccf_params shifts the CCFs in place (ccf - min + 1e-6) and results['CCF'] keeps them shifted
        shifted = ss.ccf_var - ss.ccf_var.min(axis=1, keepdims=True) + 0.000001
        res['CCF'] = np.vstack((ss.ccf_rv, shifted))
        res['CCF_unshifted'] = np.vstack((ss.ccf_rv, ss.ccf_var))
        res['ccf_quiet'] = quiet_ccf
        res['raw_xbis'], res['raw_ybis'] = ss.xbis_var, ss.ybis_var
        for k in ('ff_ph', 'ff_sp', 'ff_fc'):
            res[k + '_rv'] = res[k]
    report['notes'].extend(sorted(cfg.get('_notes', [])))
    if verbose:
        print_report(report)
    return res


def print_report(report):
    """Print what was read from an old configuration file (defaults used, keys ignored, paths, notes)."""
    print('Old configuration %s (old package %s)' % (report['file'], report['old package']))
    for title, key in (('missing keys, default used', 'missing defaults'), ('keys ignored', 'ignored')):
        if report[key]:
            print('  %s:' % title)
            for line in report[key]:
                print('    ' + line)
    if report['paths']:
        print('  files:')
        for k, v in report['paths'].items():
            print('    %s: %s' % (k, v))
    for note in report['notes']:
        print('  note: ' + note)
