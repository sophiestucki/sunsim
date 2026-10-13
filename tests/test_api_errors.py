"""
Wrong usages of the SunSim API and the error (or warning) each one must give, plus a correct usage that must still run.
Synthetic spectra only (no Phoenix file is needed), runs in a few seconds:

    python tests/test_api_errors.py            # every case
    python tests/test_api_errors.py -v         # print the message of every error

Add a case here when a check is added to _checks.py / the classes.
"""
import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository (if not installed)
from sunsim import (StarSim, flux_grid, CCF_grid, active_region, planet, mask_weights, interpolate_phoenix,  # noqa: E402
                    interpolate_phoenix_mu, add_resol, max_rings)

N = 5
WV = np.linspace(5000.0, 5100.0, 20001)           # uniform grid, 0.005 A
LINES = np.array([5020.0, 5050.0, 5080.0])


def spectrum(scale=1.0):
    """Low resolution specific intensities at 4 mu (limb darkened), the format of interpolate_phoenix_mu."""
    return {mu: {'wav': WV, 'intensity': scale * (0.4 + 0.6 * mu) * np.ones_like(WV)} for mu in (0.1, 0.4, 0.7, 1.0)}


def lines(depth=0.5):
    """High resolution spectrum with three gaussian lines, at mu = 1 only."""
    f = np.ones_like(WV)
    for c in LINES:
        f -= depth * np.exp(-0.5 * ((WV - c) / 0.05) ** 2)
    return {1.0: {'wav': WV, 'intensity': f}}


def photometry_grids():
    out = []
    for s in (1.0, 0.5, 1.1):
        g = flux_grid(N, spectrum(s), 'x', None, 5000.0, 5100.0)
        g.built_grid()
        out.append(g)
    return out


RV = 1000.0 * np.arange(-15, 15.5, 0.5)
REGIONS = [active_region('sp', 5.0, 90.0, 0.0)]
GRIDS = photometry_grids()


def starsim(**kw):
    args = dict(time=np.linspace(0, 25, 10), N_rings=N, inclination=0.0, rotation_period=25.0,
                active_regions_map=REGIONS, flux_grid_qp=GRIDS[0], flux_grid_sp=GRIDS[1], flux_grid_fc=GRIDS[2])
    args.update(kw)
    s = StarSim(**args)
    s.generate_timeseries(verbose=False)
    return s


def ccf(**kw):
    args = dict(N_rings=N, high_res_flux=lines(), RVs=RV, wvm=LINES, fm=np.ones(3), mu_ratio=np.ones(N))
    args.update(kw)
    return CCF_grid(**args)


def bad_mask_file(tmp):
    p = os.path.join(tmp, 'four_columns.mas')
    np.savetxt(p, np.ones((5, 4)))
    return p


TMP = tempfile.mkdtemp(prefix='sunsim_api_test_')

#(name, function, expected exception or warning class)
CASES = [
    # active_region
    ('region: unknown type', lambda: active_region('spot', 5, 90, 0), ValueError),
    ('region: negative size', lambda: active_region('sp', -5, 90, 0), ValueError),
    ('region: size > 180 deg', lambda: active_region('sp', 200, 90, 0), ValueError),
    ('region: latitude instead of colatitude', lambda: active_region('sp', 5, -30, 0), ValueError),
    ('region: NaN longitude', lambda: active_region('sp', 5, 90, np.nan), ValueError),
    ('region: size as a string', lambda: active_region('sp', '5', 90, 0), TypeError),
    ('region: appears at +inf', lambda: active_region('sp', 5, 90, 0, appearance_time=np.inf), ValueError),
    # planet
    ('planet: missing radius', lambda: planet(planet_period=3, planet_semi_major_axis=10), ValueError),
    ('planet: negative period', lambda: planet(planet_period=-3, planet_radius=0.1, planet_semi_major_axis=10), ValueError),
    ('planet: eccentricity >= 1', lambda: planet(planet_period=3, planet_radius=0.1, planet_semi_major_axis=10,
                                                 planet_esinw=0.8, planet_ecosw=0.8), ValueError),
    ('planet: spin-orbit angle in degrees', lambda: planet(planet_period=3, planet_radius=0.1, planet_semi_major_axis=10,
                                                           planet_spin_orbit_angle=45), ValueError),
    ('planet: orbit inside the star', lambda: planet(planet_period=3, planet_radius=0.1, planet_semi_major_axis=0.5),
     UserWarning),
    # flux_grid
    ('flux_grid: N_rings = 0', lambda: flux_grid(0, spectrum(), 'x', None, 5000, 5100), ValueError),
    ('flux_grid: N_rings not an integer', lambda: flux_grid(5.5, spectrum(), 'x', None, 5000, 5100), TypeError),
    ('flux_grid: one mu only', lambda: flux_grid(N, lines(), 'x', None, 5000, 5100), ValueError),
    ('flux_grid: spectra not a dict', lambda: flux_grid(N, WV, 'x', None, 5000, 5100), TypeError),
    ('flux_grid: mu > 1', lambda: flux_grid(N, {**spectrum(), 2.0: spectrum()[1.0]}, 'x', None, 5000, 5100), ValueError),
    ('flux_grid: wavelength limits swapped', lambda: flux_grid(N, spectrum(), 'x', None, 5100, 5000), ValueError),
    ('flux_grid: inclination in degrees', lambda: flux_grid(N, spectrum(), 'x', None, 5000, 5100, inclination=90), ValueError),
    ('flux_grid: unknown mode', lambda: flux_grid(N, spectrum(), 'x', None, 5000, 5100, mode='rv'), ValueError),
    ('flux_grid: missing filter file', lambda: flux_grid(N, spectrum(), 'x', '/no/such/filter.dat', 5000, 5100).built_grid(),
     UserWarning),
    # CCF_grid
    ('CCF_grid: velocities decreasing', lambda: ccf(RVs=RV[::-1]), ValueError),
    ('CCF_grid: mask and weights of different lengths', lambda: ccf(fm=np.ones(2)), ValueError),
    ('CCF_grid: mu_ratio of the wrong length', lambda: ccf(mu_ratio=np.ones(N + 1)), ValueError),
    ('CCF_grid: negative mu_ratio', lambda: ccf(mu_ratio=-np.ones(N)), ValueError),
    ('CCF_grid: limb_lim > 1', lambda: ccf(limb_lim=2), ValueError),
    ('CCF_grid: unknown limb extrapolation', lambda: ccf(limb_extrapolation='cubic'), ValueError),
    ('CCF_grid: mask outside the spectrum', lambda: ccf(wvm=np.array([4000.0, 5050.0, 5080.0])), ValueError),
    ('CCF_grid: phoenix_resolution on a non-Phoenix grid', lambda: ccf(phoenix_resolution=True), ValueError),
    # StarSim
    ('StarSim: inclination in degrees', lambda: starsim(inclination=90), ValueError),
    ('StarSim: negative rotation period', lambda: starsim(rotation_period=-25), ValueError),
    ('StarSim: NaN in the times', lambda: starsim(time=np.array([0.0, np.nan])), ValueError),
    ('StarSim: N_rings different from the grids', lambda: starsim(N_rings=N + 1), ValueError),
    ('StarSim: a single region not in a list (accepted)', lambda: starsim(active_regions_map=REGIONS[0]), None),
    ('StarSim: regions given as an array', lambda: starsim(active_regions_map=np.array([[1, 5, 90, 0]])), TypeError),
    ('StarSim: unknown mode', lambda: starsim(mode=['lc']), ValueError),
    ('StarSim: mode not a list (accepted)', lambda: starsim(mode='photometry'), None),
    ('StarSim: flux grid missing', lambda: starsim(flux_grid_fc=None), ValueError),
    ('StarSim: flux grid not built', lambda: starsim(flux_grid_qp=flux_grid(N, spectrum(), 'x', None, 5000, 5100)),
     RuntimeError),
    ('StarSim: ccf mode without CCF grids', lambda: starsim(mode=['ccf']), ValueError),
    ('StarSim: planet not a planet object', lambda: starsim(planet={'period': 3}), TypeError),
    ('StarSim: flux grid with NaN', lambda: starsim(flux_grid_qp=np.full_like(GRIDS[0].grid, np.nan)), UserWarning),
    # spectrum_utils
    ('mask_weights: missing file', lambda: mask_weights('/no/such/mask.mas'), FileNotFoundError),
    ('mask_weights: four columns', lambda: mask_weights(bad_mask_file(TMP)), ValueError),
    ('interpolate_phoenix: wavelength limits swapped',
     lambda: interpolate_phoenix('.', 5800, 4.5, wavelength_lower_limit=6000, wavelength_upper_limit=5000), ValueError),
    ('interpolate_phoenix_mu: NaN temperature', lambda: interpolate_phoenix_mu('.', np.nan, 4.5), ValueError),
    ('add_resol: unknown instrument', lambda: add_resol(WV, np.ones_like(WV), 'ESPRESSO'), ValueError),
    ('add_resol: negative resolving power', lambda: add_resol(WV, np.ones_like(WV), -5), ValueError),
    ('CCF_grid: unknown instrument', lambda: ccf(instrument='ESPRESSO'), ValueError),
    ('CCF_grid: blaze without instrument', lambda: ccf(blaze_function=([WV], [np.ones(len(WV))])), ValueError),
    ('CCF_grid: efficiency without blaze', lambda: ccf(instrument='HARPS-N', instrumental_efficiency=(WV, np.ones(len(WV)))), ValueError),
    # active_region_mask.max_rings
    ('max_rings: size 0', lambda: max_rings(0), ValueError),
    ('max_rings: 3D shape', lambda: max_rings((64, 64, 3)), ValueError),
    ('max_rings: min_pixels = 0', lambda: max_rings(64, min_pixels=0), ValueError),
    # correct usages that must run without error
    ('correct: photometry', lambda: starsim(), None),
    ('correct: CCF grid', lambda: ccf().built_grid(bisector=None), None),
    ('correct: CCF grid with the resolution of an instrument only', lambda: ccf(instrument='HARPS-N').built_grid(bisector=None), None),
    ('correct: add_resol with R as a number', lambda: add_resol(WV, lines()[1.0]['intensity'], 50000), None),
    ('correct: CCF grid by order', lambda: ccf(instrument=60000, blaze_function=([WV[100:-100:2]], [np.ones(len(WV[100:-100:2]))])).built_grid(bisector=None), None),
    ('correct: max_rings(64) = 10', lambda: None if max_rings(64) == 10 else 1 / 0, None),
]


def run(verbose=False):
    n_fail = 0
    for name, func, expected in CASES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                func()
                err = None
            except Exception as e:      # noqa: BLE001  (the type is checked below)
                err = e
        warned = [w for w in caught if issubclass(w.category, UserWarning) and not issubclass(w.category, DeprecationWarning)]
        if expected is None:
            ok, got = err is None, 'no error' if err is None else '%s: %s' % (type(err).__name__, err)
        elif issubclass(expected, Warning):
            ok = err is None and any(issubclass(w.category, expected) for w in warned)
            got = ('%s: %s' % (type(err).__name__, err) if err is not None else
                   ('warning: %s' % warned[0].message if warned else 'no warning'))
        else:
            ok = isinstance(err, expected)
            got = 'no error' if err is None else '%s: %s' % (type(err).__name__, err)
        n_fail += not ok
        if verbose or not ok:
            print('%-4s %-52s %s' % ('ok' if ok else 'FAIL', name, got))
        else:
            print('%-4s %s' % ('ok', name))
    print('\n%d cases, %d failed' % (len(CASES), n_fail))
    return n_fail


if __name__ == '__main__':
    sys.exit(1 if run(verbose='-v' in sys.argv) else 0)
