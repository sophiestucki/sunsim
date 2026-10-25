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
                    interpolate_phoenix_mu, add_resol, max_rings, limb_darkened_spectra, flatten_spectra,
                    solar_differential_rotation, cifist_bisector, dumusque_bisector, local_path, sdo,
                    observations)

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


def spectroscopy_grids(rotation=False):
    out = []
    for sc in (1.0, 0.5, 1.1):
        g = flux_grid(N, spectrum(sc), 'x', None, 5000.0, 5100.0, mode='spectroscopy', stellar_rotation=rotation)
        g.built_grid()
        out.append(g)
    return out


def spectroscopy_rings_vs_cells():
    """The spectra with the grids per ring (no rotation) and the same grids repeated for every cell must be equal."""
    g = spectroscopy_grids()
    assert g[0].grid.shape[0] == N and g[0].per_ring
    kw = dict(mode=['spectroscopy'], active_regions_map=[active_region('sp', 8, 90, 0), active_region('fc', 15, 90, 0)])
    rings = starsim(flux_grid_qp=g[0].grid, flux_grid_sp=g[1].grid, flux_grid_fc=g[2].grid, **kw).spec_var
    cells = starsim(flux_grid_qp=g[0].cell_grid, flux_grid_sp=g[1].cell_grid, flux_grid_fc=g[2].cell_grid, **kw).spec_var
    mixed = starsim(flux_grid_qp=g[0].grid, flux_grid_sp=g[1].cell_grid, flux_grid_fc=g[2].grid, **kw).spec_var
    assert np.allclose(rings, cells, rtol=1e-12, atol=0) and np.allclose(rings, mixed, rtol=1e-12, atol=0)
    assert np.allclose(g[0].disc_spectrum, g[0].cell_grid.sum(axis=0), rtol=1e-12, atol=0)


def spectroscopy_on_the_fly():
    """Rotating spectroscopy, inclined star with differential rotation: the spectra of the touched cells computed on the
    fly (spec_cells_max_memory=0, flux_grid objects) must be bit-identical to the stored grids of the cells."""
    kw = dict(stellar_rotation=True, inclination=0.6, rotation_period=10.0, differential_rotation=solar_differential_rotation(),
              mode='spectroscopy')
    g = []
    for sc in (1.0, 0.5, 1.1):
        x = {mu: {'wav': WV, 'intensity': v['intensity'] * lines()[1.0]['intensity']} for mu, v in spectrum(sc).items()}
        x = flux_grid(N, x, 'x', None, 5000.0, 5100.0, **kw)
        x.built_grid()
        g.append(x)
    assert g[0]._grid is None                                   # not built by built_grid
    disc = g[0].disc_spectrum.copy()
    run = dict(inclination=0.6, rotation_period=10.0, differential_rotation=solar_differential_rotation(),
               mode=['spectroscopy'], active_regions_map=[active_region('sp', 8, 70, 30), active_region('fc', 15, 110, 200)])
    lazy = starsim(spec_grid_qp=g[0], spec_grid_sp=g[1], spec_grid_fc=g[2], spec_cells_max_memory=0, **run).spec_var
    assert g[0]._grid is None                                   # still not built
    full = starsim(spec_grid_qp=g[0].grid, spec_grid_sp=g[1].grid, spec_grid_fc=g[2].grid, **run).spec_var
    assert np.array_equal(lazy, full) and np.array_equal(disc, g[0].grid.sum(axis=0))


def _observations_check():
    """sunsim.observations on synthetic data: statistics, matching by day, the offset per instrumental period."""
    obs = observations
    assert obs.mad([1, 2, 3, 4, 100]) == 1.0 and abs(obs.rms([1, -1, 1, -1]) - 1.0) < 1e-15
    assert obs.jd_of_date('2015-07-18') == 2457221.5
    t_sim = np.arange(0.0, 200.0)
    rv_sim = 3 * np.sin(2 * np.pi * t_sim / 27)
    t_obs = np.arange(0.0, 200.0) + 0.5
    rv_obs = rv_sim + np.where(t_obs > 100, 5.0, 0.0)            # an offset after an intervention at day 100
    harps = {'t': t_obs, 'rv': rv_obs, 'rv_err': np.full(200, 0.1)}
    io, i_s = obs.match_days(t_obs, t_sim, 0)
    assert np.array_equal(io, i_s) and len(io) == 200
    r = obs.compare_rv(harps, t_sim, rv_sim, interventions=[100.0])
    assert r['rms'] < 1e-12 and len(r['periods']) == 2 and all(p['shift'] == 0 for p in r['periods'])
    assert obs.compare_rv(harps, t_sim, rv_sim)['rms'] > 1.0     # without the intervention, the offset stays
    ph = obs.compare_photometry({'name': 'x', 't': t_obs, 'tsi': 1361 * (1 + 1e-4 * rv_sim)}, t_sim, 1 + 1e-4 * rv_sim)
    assert ph['n'] == 200 and ph['rms'] < 1e-3


def ccf(**kw):
    args = dict(N_rings=N, high_res_flux=lines(), RVs=RV, wvm=LINES, fm=np.ones(3), mu_ratio=np.ones(N))
    args.update(kw)
    return CCF_grid(**args)


def _ccf_on_the_fly_check():
    """ccf_on_the_fly True / False / 'auto' give the same CCFs; save_results writes and reads back the time series."""
    g = []
    for sc in (1.0, 0.5, 1.1):
        x = ccf(high_res_flux={1.0: {'wav': WV, 'intensity': sc * lines()[1.0]['intensity']}})
        x.built_grid(bisector=None)
        g.append(x)
    kw = dict(mode=['photometry', 'ccf'], ccf_grid_qp=g[0], ccf_grid_sp=g[1], ccf_grid_fc=g[2], inclination=0.3,
              active_regions_map=[active_region('sp', 8, 80, 0), active_region('fc', 15, 80, 10)])
    out = {v: starsim(ccf_on_the_fly=v, **kw) for v in (True, False, 'auto')}
    assert out[True].ccf_cells_on_the_fly() and not out[False].ccf_cells_on_the_fly()
    assert np.array_equal(out[True].ccf_var, out[False].ccf_var) and np.array_equal(out[True].rv_var, out['auto'].rv_var)
    ss = out[True]
    d = tempfile.mkdtemp()
    path = ss.save_results(os.path.join(d, 'run'))
    back = np.load(path)
    assert np.array_equal(back['ccf'], ss.ccf_var) and np.array_equal(back['rv'], ss.rv_var)
    ss.save_results(os.path.join(d, 'txt'), fmt='txt')
    ts = np.loadtxt(os.path.join(d, 'txt', 'timeseries.txt'))
    assert ts.shape == (len(ss.obs_times), 11) and np.allclose(np.loadtxt(os.path.join(d, 'txt', 'ccf.txt')), ss.ccf_var)
    assert 'ccf' not in np.load(ss.save_results(os.path.join(d, 'no_ccf'), ccf=False))


def _quiet_velocity_run(v, ff_sp=None, **kw):
    """CCF mode with cell filling factors (map mode) and a quiet_velocity v (None, an array or a field)."""
    from sunsim import precomputed_filling_factors
    g = []
    for sc in (1.0, 0.5, 1.1):
        x = ccf(high_res_flux={1.0: {'wav': WV, 'intensity': sc * lines()[1.0]['intensity']}})
        x.built_grid(bisector=None)
        g.append(x)
    t = np.arange(4.0)
    n_cells = len(StarSim(t, N, 0.0, 25.0).vec_grid)
    sp = np.zeros((4, n_cells)) if ff_sp is None else ff_sp
    mask = precomputed_filling_factors(t, sp, np.zeros((4, n_cells)), N)
    args = dict(time=t, N_rings=N, inclination=0.0, rotation_period=25.0, active_regions_mask=mask, mode=['ccf'],
                ccf_grid_qp=g[0], ccf_grid_sp=g[1], ccf_grid_fc=g[2], quiet_velocity=v)
    args.update(kw)
    s = StarSim(**args)
    s.generate_timeseries(verbose=False)
    return s, n_cells


def _quiet_velocity_check():
    """quiet_velocity: 0 changes nothing; the same velocity in every quiet cell shifts the RV by it; a fully spotted
    cell is not shifted; the field object."""
    from sunsim import supergranulation as sg
    ref, n_cells = _quiet_velocity_run(None)
    zero, _ = _quiet_velocity_run(np.zeros((4, n_cells)))
    assert np.array_equal(ref.ccf_var, zero.ccf_var)
    shifts = np.array([0.0, 500.0, -1000.0, 1500.0])           # multiples of the step of RV: exact translations
    moved, _ = _quiet_velocity_run(sg.quiet_velocity_field(np.arange(4.0), shifts[:, None] + np.zeros((4, n_cells)), N))
    inner = np.abs(ref.ccf_rv) < ref.ccf_rv.max() - 4000                # the ends are extrapolated cell by cell
    for k, s in enumerate(shifts):
        expected = np.interp(ref.ccf_rv - s, ref.ccf_rv, ref.ccf_var[k])
        assert np.max(np.abs(moved.ccf_var[k] - expected)[inner]) < 1e-9 * np.ptp(ref.ccf_var[k])
    sp = np.zeros((4, n_cells))
    sp[:, 0] = 1.0                                               # the central cell is a spot: not moved
    one = np.zeros((4, n_cells))
    one[:, 0] = 500.0
    spotted, _ = _quiet_velocity_run(one, ff_sp=sp)
    nov, _ = _quiet_velocity_run(None, ff_sp=sp)
    assert np.array_equal(spotted.ccf_var, nov.ccf_var)
    idx, dt = sg.closest([1.0, 2.6, 9.0], [0.0, 2.0, 3.0])
    assert list(idx) == [0, 2, 2] and np.allclose(dt, np.array([1.0, 0.4, 6.0]) * 86400)


def _haywood_check():
    """Dopplergram corrections of Haywood et al. (2016): the geometry of the lines of sight, eq. 10 and 13 against the
    vectors computed directly, the average over the frames of a window and the cells without data."""
    from sunsim import supergranulation as sg
    R, D = 6.96e8, 1.5e11
    rho_sun = np.arcsin(R / D)
    rng = np.random.default_rng(1)
    r, a = rho_sun * np.sqrt(rng.uniform(0, 0.999, 500)), rng.uniform(0, 2 * np.pi, 500)
    tx, ty = np.tan(r) * np.cos(a), np.tan(r) * np.sin(a)
    u, p, mu = sg.los_geometry(np.r_[0.0, tx, 2 * rho_sun], np.r_[0.0, ty, 0.0], D, R)
    u, p = np.array(u), np.array(p)
    assert np.allclose(u[:, 0], [0, 0, -1]) and np.allclose(p[:, 0], [0, 0, R]) and np.isclose(mu[0], 1.0)
    assert np.isnan(mu[-1])                                                      # off the disc
    u, p, mu = u[:, 1:-1], p[:, 1:-1], mu[1:-1]
    o = np.array([0.0, 0.0, D])[:, None]
    los = (p - o) / np.linalg.norm(p - o, axis=0)
    assert np.allclose(np.linalg.norm(p, axis=0), R, rtol=1e-9) and np.allclose(u, los, atol=1e-12)
    assert np.allclose(mu, np.sum(p / R * (o - p), axis=0) / np.linalg.norm(o - p, axis=0), atol=1e-9)
    vsc = np.array([-120.0, 25.0, -2900.0])                                       # OBS_VW, OBS_VN, OBS_VR
    assert np.allclose(sg.spacecraft_velocity(u, *vsc), -np.sum(los * vsc[:, None], axis=0))
    b0 = np.deg2rad(7.0)
    axis = np.array([0.0, np.cos(b0), np.sin(b0)])
    sinphi = axis @ p / R
    omega = np.polyval(np.array(sg.SNODGRASS_ULRICH)[::-1], sinphi ** 2) * np.pi / 180 / 86400
    v_surface = omega * np.cross(axis, p.T).T
    v_rot = sg.rotation_velocity(u, p, b0, R)
    assert np.allclose(v_rot, np.sum(los * v_surface, axis=0), atol=1e-9)
    g = sg.rotation_basis(u, p, b0, R)
    assert np.allclose(sum(a * gk for a, gk in zip(sg.SNODGRASS_ULRICH, g)), v_rot)
    # fit of a1, a2, a3 on the cells of several frames, with any offset per ring and per frame
    ring = sg.ring_of_cells(N)
    n_f, n_c = 6, len(ring)
    G = rng.normal(0, 30, (n_f, 3, n_c))
    G[:, 1] = G[:, 0] * rng.uniform(0, 0.8, n_c)
    G[:, 2] = G[:, 1] * rng.uniform(0, 0.8, n_c)
    truth = np.array([14.2, -1.8, -2.4])
    sw = rng.uniform(1, 3, (n_f, n_c))
    sw[:, -2:] = 0                                                               # cells without quiet pixels
    v = np.einsum('k,fkc->fc', truth, G) + rng.normal(0, 200, (n_f, N))[:, ring] + rng.normal(0, 0.5, (n_f, n_c))
    alphas, err, rms = sg.fit_rotation(v * sw, sw, G * sw[:, None], N)
    assert np.allclose(alphas, truth, atol=0.05) and rms < 1 and np.all(err > 0)
    assert np.allclose(sg.remove_rotation(v * sw, G * sw[:, None], truth), (v - np.einsum('k,fkc->fc', truth, G)) * sw)
    west = p[0] > 0.9 * R                                                        # the west limb recedes
    assert west.any() and np.all(v_rot[west] > 1500) and np.all(v_rot[p[0] < -0.9 * R] < -1500)
    # window average: the frames within window / 2 of the exposure, as the mean of the frames
    jd = 2459340.0 + np.arange(40) * 45.0 / 86400
    v = rng.normal(0, 300, (40, 6))
    sw = np.full((40, 6), 2.0)
    mean, _, n, ok = sg.window_average(jd, v * sw, sw, [jd[20], jd[0]], window=700.0)
    sel = np.abs(jd - jd[20]) * 86400 <= 350
    assert n[0] == sel.sum() == 15 and ok[0] and not ok[1] and np.allclose(mean[0], v[sel].mean(axis=0))
    ring = sg.ring_of_cells(N)
    vc = np.where(ring == N - 1, np.nan, 10.0 * ring)[None, :]
    filled, empty = sg.fill_empty_cells(vc, np.ones_like(vc), N)
    assert np.all(empty[0] == (ring == N - 1)) and np.allclose(filled[0], 10.0 * np.minimum(ring, N - 2))


def _overlap_check():
    """Relative difference between the planet filling factor of two identical (overlapping) planets and of one."""
    pl = lambda: planet(planet_period=4, planet_transit_t0=0.0, planet_radius=0.1, planet_semi_major_axis=10)
    t = np.linspace(-0.05, 0.05, 11)
    one = starsim(time=t, planet=pl(), active_regions_map=[])
    two = starsim(time=t, planet=[pl(), pl()], active_regions_map=[])
    return np.max(np.abs(two.ff_pl - one.ff_pl)) / one.ff_pl.max()


def _synthetic_sun(n=101):
    """Synthetic intensity, |B| and mu of a disc with one spot (dark, strong field) and one plage (bright, weaker field)."""
    y, x = np.indices((n, n)) - (n - 1) / 2
    mu = sdo.mu_from_rho(np.hypot(x, y), (n - 1) / 2)
    intensity = np.where(np.isfinite(mu), 1.0, np.nan)
    blos = np.where(np.isfinite(mu), 2.0, np.nan)                      # noise
    spot = np.hypot(x - 10, y) < 4
    plage = (np.hypot(x + 20, y - 10) < 8)
    intensity[spot] = 0.5
    blos[spot] = 2000.0
    intensity[plage] = 1.02
    blos[plage] = 200.0
    blos[50, 80] = 100.0                                               # one isolated bright pixel: network
    return intensity, blos, mu, spot, plage


def _sdo_check():
    intensity, blos, mu, spot, plage = _synthetic_sun()
    c = sdo.classify_pixels(intensity, blos, mu, noise_th=8.0)
    pl, nt, n_pl = sdo.split_plage_network(c['bright'], 20)
    n_sp = sdo.count_spots(c['spot'])[0]
    ok = (np.array_equal(c['spot'], spot) and np.array_equal(pl, plage) and nt[50, 80] and n_pl == 1 and n_sp == 1
          and abs(c['I_quiet'] - 1.0) < 1e-12)
    return None if ok else 1 / 0


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
    ('correct: spectroscopy, grids per ring = grids per cell', spectroscopy_rings_vs_cells, None),
    ('StarSim: spectroscopy grids with and without rotation (different wavelengths)',
     lambda: starsim(mode=['spectroscopy'], flux_grid_qp=spectroscopy_grids()[0].grid,
                     flux_grid_sp=spectroscopy_grids(True)[1].grid, flux_grid_fc=spectroscopy_grids()[2].grid), ValueError),
    ('StarSim: spectroscopy grid with a wrong number of rows',
     lambda: starsim(mode=['spectroscopy'], flux_grid_qp=spectroscopy_grids()[0].grid[:-1],
                     flux_grid_sp=spectroscopy_grids()[1].grid, flux_grid_fc=spectroscopy_grids()[2].grid), ValueError),
    ('correct: CCF grid', lambda: ccf().built_grid(bisector=None), None),
    ('correct: CCF grid with the resolution of an instrument only', lambda: ccf(instrument='HARPS-N').built_grid(bisector=None), None),
    ('correct: add_resol with R as a number', lambda: add_resol(WV, lines()[1.0]['intensity'], 50000), None),
    ('correct: CCF grid by order', lambda: ccf(instrument=60000, blaze_function=([WV[100:-100:2]], [np.ones(len(WV[100:-100:2]))])).built_grid(bisector=None), None),
    ('correct: max_rings(64) = 10', lambda: None if max_rings(64) == 10 else 1 / 0, None),
    # photometry and spectroscopy in the same simulation
    ('photometry + spectroscopy without spec_grid', lambda: starsim(mode=['photometry', 'spectroscopy']).generate_timeseries(verbose=False),
     ValueError),
    ('correct: photometry + spectroscopy (spec_grid)', lambda: starsim(
        mode=['photometry', 'spectroscopy'], spec_grid_qp=spectroscopy_grids()[0].grid, spec_grid_sp=spectroscopy_grids()[1].grid,
        spec_grid_fc=spectroscopy_grids()[2].grid).generate_timeseries(verbose=False), None),
    # differential rotation laws
    ('differential rotation: law with NaN', lambda: starsim(differential_rotation=lambda lat: np.full_like(lat, np.nan)), ValueError),
    ('differential rotation: law not vectorised', lambda: starsim(differential_rotation=lambda lat: 1.0), ValueError),
    ('correct: solar differential rotation law', lambda: starsim(differential_rotation=solar_differential_rotation(-2.66)), None),
    # limb darkening, continuum, bisectors
    ('limb darkening: unknown law', lambda: limb_darkened_spectra(WV, np.ones_like(WV), 'cubic', (0.1,)), ValueError),
    ('limb darkening: coefficient of the wrong length', lambda: limb_darkened_spectra(WV, np.ones_like(WV), 'linear', (np.ones(5),)),
     ValueError),
    ('limb darkening: mu = 0', lambda: limb_darkened_spectra(WV, np.ones_like(WV), 'linear', (0.5,), mu=[0.0, 0.5]), ValueError),
    ('correct: limb-darkened spectra in flux_grid', lambda: flux_grid(N, limb_darkened_spectra(WV, np.ones_like(WV), 'quadratic', (0.4, 0.2)),
                                                                     'x', None, 5000, 5100).built_grid(), None),
    ('correct: flatten_spectra', lambda: flatten_spectra(spectrum()), None),
    ('correct: CIFIST bisector in a CCF grid', lambda: ccf().built_grid(bisector=cifist_bisector(mu_scaling=(0.5, 0.5))), None),
    ('correct: Dumusque bisector with a linear mu law', lambda: ccf().built_grid(bisector=dumusque_bisector(True, (0, 1))), None),
    ('CCF_grid: unknown medium', lambda: ccf(medium='water'), ValueError),
    ('mask_weights: unknown medium', lambda: mask_weights(bad_mask_file(TMP), mask_medium='glass'), ValueError),
    ('correct: CCF grid in vacuum (mask and spectra in vacuum)', lambda: ccf(medium='vacuum', mask_medium='vacuum').built_grid(bisector=None), None),
    ('correct: CCF grid in air (spectra in vacuum, converted)', lambda: ccf(medium='air', mask_medium='air').built_grid(bisector=None), None),
    ('StarSim: total_flux_qp (deprecated, ignored)', lambda: starsim(total_flux_qp=1.0), FutureWarning),
    ('correct: two identical planets cover the area of one', lambda: None if _overlap_check() < 1e-3 else 1 / 0, None),
    # SDO/HMI maps (sunsim.sdo): checks before reading any image, and the classification on arrays (no sunpy needed)
    ('sdo: end before start', lambda: sdo.make_sdo_maps(TMP, TMP, '2017-07-10', '2017-07-01'), ValueError),
    ('sdo: date in another format', lambda: sdo.make_sdo_maps(TMP, TMP, '01/07/2017', '2017-07-10'), ValueError),
    ('sdo: unknown epoch', lambda: sdo.make_sdo_maps(TMP, TMP, '2017-07-01', '2017-07-10', epoch='weeks'), ValueError),
    ('sdo: unknown file format', lambda: sdo.make_sdo_maps(TMP, TMP, '2017-07-01', '2017-07-10', fmt='fits'), ValueError),
    ('sdo: missing image folder', lambda: sdo.make_sdo_maps('/no/such/folder', TMP, '2017-07-01', '2017-07-10'), FileNotFoundError),
    ('sdo: download without email', lambda: sdo.download_hmi('2017-07-01', '2017-07-02', TMP, email=None), ValueError),
    ('sdo: arrays of different shapes', lambda: sdo.classify_pixels(np.ones((5, 5)), np.ones((5, 4)), np.ones((5, 5))), ValueError),
    ('correct: sdo classification of a synthetic Sun (spot, plage, network)', _sdo_check, None),
    # threshold sweeps
    ('correct: sdo parameter sets (product) and their labels', lambda: None if (
        len(sdo.parameter_sets(I_th=[0.85, 0.89], A_th=[20, 100], faculae=['plage', 'plage+network'])) == 8
        and sdo.set_label(sdo.parameter_sets()[0]) == 'noise8_B24_I0.89_A20_mu0.1'
        and sdo.set_label(dict(faculae='plage+network')).endswith('_network')) else 1 / 0, None),
    ('sdo: make_sdo_maps with several values of a threshold',
     lambda: sdo.make_sdo_maps(TMP, TMP, '2017-07-01', '2017-07-10', I_th=[0.85, 0.89]), ValueError),
    ('sdo: the same set twice in a sweep',
     lambda: sdo.make_sdo_sweep(TMP, TMP, '2017-07-01', '2017-07-10', I_th=[0.89, 0.89], n_rings=10), ValueError),
    ('sdo: negative threshold in a sweep', lambda: sdo.parameter_sets(A_th=[20, -1]), ValueError),
    ('sdo: unknown faculae in a sweep', lambda: sdo.parameter_sets(faculae=['plage', 'all']), ValueError),
    ('sdo: sweep without maps nor cells', lambda: sdo.make_sdo_sweep(TMP, TMP, '2017-07-01', '2017-07-10',
                                                                     I_th=[0.85, 0.89]), ValueError),
    ('correct: sdo noise threshold scaled with the size', lambda: None if sdo.noise_threshold_for(512) == 1.0 else 1 / 0, None),
    # paths of this computer
    ('local_path: unknown key', lambda: local_path('no_such_key_xyz'), FileNotFoundError),
    # spectroscopy with rotation computed on the fly (spec_cells_max_memory)
    ('StarSim: spec_cells_max_memory negative', lambda: starsim(spec_cells_max_memory=-1), ValueError),
    ('correct: rotating spectra on the fly = stored grids (inclined, differential rotation)', spectroscopy_on_the_fly, None),
    # comparison with the observations
    ('correct: observations (MAD, RMS, matching, periods)', _observations_check, None),
    ('observations: compare_rv with an unknown quantity', lambda: observations.compare_rv({'t': np.arange(3.0), 'rv': np.zeros(3)},
                                                                                   np.arange(3.0), np.zeros(3), quantity='xx'), KeyError),
    # CCF of the cells on the fly or kept, results saved
    ('StarSim: ccf_on_the_fly not a boolean', lambda: starsim(ccf_on_the_fly='yes'), ValueError),
    ('StarSim: save_results before generate_timeseries',
     lambda: StarSim(time=np.zeros(1), N_rings=N, inclination=0.0, rotation_period=25.0).save_results(TMP), RuntimeError),
    ('correct: ccf_on_the_fly True = False = auto, save_results (npz, txt)', _ccf_on_the_fly_check, None),
    # velocity of the quiet photosphere (supergranulation)
    ('StarSim: quiet_velocity without maps', lambda: starsim(quiet_velocity=np.zeros((10, 3))), ValueError),
    ('StarSim: quiet_velocity of the wrong shape', lambda: _quiet_velocity_run(np.zeros((3, 5))), ValueError),
    ('StarSim: quiet_velocity of the wrong number of cells', lambda: _quiet_velocity_run(np.zeros((4, 5))), ValueError),
    ('StarSim: quiet_velocity without ccf', lambda: _quiet_velocity_run(np.zeros((4, 5)), mode=['photometry']), ValueError),
    ('correct: quiet_velocity (zero, uniform shift, spotted cell)', _quiet_velocity_check, None),
    ('correct: dopplergram corrections (Haywood et al. 2016: eq. 10, 13, window)', _haywood_check, None),
    ('correct: local_path not required', lambda: None if local_path('no_such_key_xyz', required=False) is None else 1 / 0, None),
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
        warned = [w for w in caught if issubclass(w.category, (UserWarning, FutureWarning)) and not issubclass(w.category, DeprecationWarning)]
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
