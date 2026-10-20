"""
Facular fractions of the SDO identification (sdo, faculae='fraction'): every bright pixel (active, not a spot) is a
fraction min(|B| / mu / B_sat, 1) facula and the rest quiet Sun, and these fractions are kept on the SunSim grid.

    python tests/test_sdo_fraction.py              # synthetic cases, then 3 real days (path sdo_data of local_paths.ini)
    python tests/test_sdo_fraction.py --synthetic  # only the synthetic cases (no HMI image, a few seconds)

Checks that the total filling factors are kept: pixels -> cells of the grid -> filling factors of StarSim, and that
B_sat <= B_th gives exactly the binary maps of faculae='plage+network' (every bright pixel is then fully facular).
Needs sunpy for the real days (pip install "sunsim[sdo]").
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository (if not installed)
from sunsim import sdo, StarSim, flux_grid, local_path                             # noqa: E402
from sunsim.active_region_mask import active_region_mask, precomputed_filling_factors   # noqa: E402

N_RINGS = 60
DAYS = ('2015-09-01', '2015-09-04')            # 3 days (end excluded)
FAILED = []


def check(name, ok, detail=''):
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', name, (': ' + detail) if detail else ''))
    if not ok:
        FAILED.append(name)


def photometry_grids(n):
    """Flux grids of n rings from flat synthetic spectra (quiet, spot, facula): only the filling factors are tested."""
    wv = np.linspace(5000.0, 5100.0, 2001)
    out = {}
    for name, s in (('qp', 1.0), ('sp', 0.5), ('fc', 1.1)):
        spec = {mu: {'wav': wv, 'intensity': s * (0.4 + 0.6 * mu) * np.ones_like(wv)} for mu in (0.1, 0.4, 0.7, 1.0)}
        g = flux_grid(n, spec, name, None, 5000.0, 5100.0)
        g.built_grid()
        out['flux_grid_' + name] = g
    return out


def starsim_ff(times, ff_sp, ff_fc, n, grids):
    """ff_sp, ff_fc [% of the projected disc] of StarSim from the filling factors of the cells."""
    mask = precomputed_filling_factors(times, ff_sp, ff_fc, n)
    ss = StarSim(np.asarray(times, dtype=float), N_rings=n, inclination=0.0, rotation_period=25.05,
                 active_regions_mask=mask, mode=['photometry'], **grids)
    ss.generate_timeseries(verbose=False)
    return np.asarray(ss.ff_sp), np.asarray(ss.ff_fc)


def cell_totals(day, n):
    """Totals over the cells [% of the pixels of the cells]: sum of ff_cell x pixels of the cell / pixels of the cells,
    and the same sum over the pixels themselves (must be equal: the cell value is the mean of its pixels)."""
    ff_sp, ff_fc = sdo.cell_filling_factors(day, n)
    cells, npix = sdo.cell_projection(day['spot'].shape, n)
    frac = day['facula_fraction'] if day.get('facula_fraction') is not None else day['facula']
    inside = cells >= 0
    pix_fc = 100 * np.sum(np.minimum(frac.ravel()[inside], 1 - day['spot'].ravel()[inside]), dtype=np.float64)
    pix_sp = 100 * np.sum(day['spot'].ravel()[inside], dtype=np.float64)
    tot = npix.sum()
    return (100 * np.dot(ff_sp, npix) / tot, 100 * np.dot(ff_fc, npix) / tot, pix_sp / tot, pix_fc / tot, ff_sp, ff_fc)


# =============================================================================================
# synthetic
# =============================================================================================
def synthetic():
    print('--- synthetic')
    # 1. the fraction of a pixel
    blos = np.array([[0.0, 30.0, -60.0], [200.0, -500.0, 50.0]])
    mu = np.array([[1.0, 1.0, 0.5], [1.0, 0.25, 0.1]])
    bright = np.array([[False, True, True], [True, True, False]])
    f = sdo.facula_fraction(blos, mu, bright, B_sat=100.0)
    expected = np.array([[0, 0.3, 1.0], [1.0, 1.0, 0]], dtype=np.float32)    # |B|/mu/B_sat capped at 1, 0 off bright
    check('fraction = min(|B| / mu / B_sat, 1) on the bright pixels, 0 elsewhere', np.allclose(f, expected), str(f.tolist()))
    check('fraction float32', f.dtype == np.float32)
    try:
        sdo.facula_fraction(blos, mu, bright, B_sat=0)
        check('B_sat = 0 refused', False)
    except ValueError:
        check('B_sat = 0 refused', True)

    # 2. the sets of a sweep
    sets = sdo.parameter_sets(I_th=[0.85, 0.89], A_th=[20, 100], faculae=['plage', 'fraction'], B_sat=[50, 100, 200])
    labels = [sdo.set_label(p) for p in sets]
    check('sets: 2 x 2 plage + 2 x 3 fraction', len(sets) == 10 and len(set(labels)) == 10, '%d sets' % len(sets))
    check('label of a fraction set', 'noise8_B24_I0.85_Bsat50_fraction' in labels)
    check('label of a plage set unchanged', sdo.set_label(sdo.parameter_sets()[0]) == 'noise8_B24_I0.89_A20_mu0.1')
    try:
        sdo.make_sdo_sweep(tempfile.gettempdir(), tempfile.gettempdir(), '2017-07-01', '2017-07-02', faculae='fraction',
                           B_sat=[100, 100], n_rings=10, verbose=False)
        check('the same B_sat twice refused', False)
    except ValueError:
        check('the same B_sat twice refused', True)

    # 3. pixels -> cells: the total is kept (random fractions on a 512 px disc, a few spots)
    rng = np.random.default_rng(1)
    n = 512
    y, x = np.mgrid[:n, :n]
    r = np.hypot(x - (n - 1) / 2, y - (n - 1) / 2) / (n / 2)
    disc = r < 1
    bright = disc & (rng.random((n, n)) < 0.05)
    spot = disc & ~bright & (rng.random((n, n)) < 0.003)
    frac = np.where(bright, rng.random((n, n)), 0).astype(np.float32)
    day = {'spot': spot, 'facula': bright, 'facula_fraction': frac}
    a_sp, a_fc, p_sp, p_fc, ff_sp, ff_fc = cell_totals(day, 20)
    check('cells keep the total facular fraction (synthetic)', abs(a_fc - p_fc) < 1e-4 * p_fc,
          'cells %.6f %%, pixels %.6f %%' % (a_fc, p_fc))
    check('cells keep the total spot fraction (synthetic)', abs(a_sp - p_sp) < 1e-4 * p_sp)
    check('cell fractions in [0, 1]', ff_fc.min() >= 0 and ff_fc.max() <= 1)
    binary = sdo.cell_filling_factors({'spot': spot, 'facula': bright}, 20)[1]
    check('fractions below the binary faculae in every cell', np.all(ff_fc <= binary + 1e-7))


# =============================================================================================
# real days
# =============================================================================================
def real_days():
    try:
        folder = local_path('sdo_data')
        import sunpy  # noqa: F401
    except Exception as e:
        print('--- real days skipped (%s)' % str(e).splitlines()[0])
        return
    print('--- real days %s to %s (4096 px, %d rings), %s' % (DAYS[0], DAYS[1], N_RINGS, folder))
    grids = photometry_grids(N_RINGS)
    from datetime import datetime, timedelta
    d0 = datetime.strptime(DAYS[0], '%Y-%m-%d')
    days = [d0 + timedelta(days=k) for k in range((datetime.strptime(DAYS[1], '%Y-%m-%d') - d0).days)]
    rows = []
    for day in days:
        t = time.time()
        cont, los = sdo._resample(*sdo.read_hmi_day(folder, day), None)
        mu = sdo.mu_of_map(los)
        t_read = time.time() - t
        th = dict(noise_th=8.0, B_th=24.0, I_th=0.89, A_th=20.0, mu_area=0.1)
        c = sdo._classify(cont, los, mu, 8.0, 24.0, 0.89)
        both = sdo._identify_resampled(cont, los, mu, faculae='plage+network', classified=c, **th)
        same = sdo._identify_resampled(cont, los, mu, faculae='fraction', B_sat=24.0, classified=c, **th)
        t = time.time()
        frac = sdo._identify_resampled(cont, los, mu, faculae='fraction', B_sat=100.0, classified=c, **th)
        t_frac = time.time() - t
        fresh = sdo._identify_resampled(cont, los, mu, faculae='fraction', B_sat=100.0, **th)    # without the cache
        date = day.strftime('%Y-%m-%d')

        # B_sat = B_th: every bright pixel has |B| / mu > B_th / mu >= B_th, so it is fully facular
        check('%s B_sat = B_th gives plage+network' % date, abs(same['ff_fc'] - both['ff_fc']) < 1e-9
              and np.array_equal(same['facula_fraction'] == 1, both['facula']),
              'ff_fc %.4f %% vs %.4f %%' % (same['ff_fc'], both['ff_fc']))
        c1, c2 = sdo.cell_filling_factors(same, N_RINGS), sdo.cell_filling_factors(both, N_RINGS)
        check('%s B_sat = B_th: same cell filling factors' % date,
              np.allclose(c1[0], c2[0], atol=1e-6) and np.allclose(c1[1], c2[1], atol=1e-6))
        check('%s shared classification = recomputed' % date,
              np.array_equal(frac['facula_fraction'], fresh['facula_fraction']) and frac['ff_sp'] == fresh['ff_sp'])

        f = frac['facula_fraction']
        check('%s fractions in [0, 1], 0 off the bright pixels' % date,
              f.min() >= 0 and f.max() <= 1 and not np.any(f[~frac['bright']]))
        check('%s 0 < ff_fc(fraction) < ff_fc(plage+network)' % date, 0 < frac['ff_fc'] < both['ff_fc'],
              '%.4f %% < %.4f %%' % (frac['ff_fc'], both['ff_fc']))
        disc_fc = 100 * np.sum(f[frac['disc']], dtype=np.float64) / np.sum(frac['disc'])
        check('%s ff_fc = sum of the fractions / disc' % date, abs(disc_fc - frac['ff_fc']) < 1e-9)
        fmap = frac['facula_map']
        check('%s facula_map = the fractions, NaN outside the disc' % date,
              np.array_equal(np.isnan(fmap), ~frac['disc']) and np.array_equal(fmap[frac['disc']], f[frac['disc']]))

        a_sp, a_fc, p_sp, p_fc, ff_sp, ff_fc = cell_totals(frac, N_RINGS)
        check('%s cells keep the total facular fraction' % date, abs(a_fc - p_fc) < 1e-5 * p_fc,
              'cells %.6f %%, pixels %.6f %%' % (a_fc, p_fc))
        check('%s cells keep the total spot fraction' % date, abs(a_sp - p_sp) < 1e-5 * max(p_sp, 1e-9))
        # pixels of the disc -> cells: the maps are cut to a square around the disc from its central row, while HMI
        # puts the disc centre ~10 px off in y, so a thin crescent of the limb (mu < ~0.14) has no cell. This loses
        # the most where the faculae are at the limb: report it for the fractions and for the binary maps
        b_fc = cell_totals(both, N_RINGS)[1]
        loss, loss_b = 1 - a_fc / frac['ff_fc'], 1 - b_fc / both['ff_fc']
        check('%s disc -> cells: facular total kept to 2 %%' % date, abs(loss) < 0.02,
              'lost %.2f %% of the fractions (%.2f %% of plage+network)' % (100 * loss, 100 * loss_b))
        rows.append((date, frac, both, ff_sp, ff_fc, a_sp, a_fc))
        print('     %s: ff_sp %.4f %%, ff_fc %.4f %% (B_sat = 100 G) vs %.4f %% (plage+network), %.4f %% (plage); '
              'mean fraction of the bright pixels %.3f; read %.1f s, fraction set %.2f s'
              % (date, frac['ff_sp'], frac['ff_fc'], both['ff_fc'],
                 sdo._identify_resampled(cont, los, mu, faculae='plage', classified=c, **th)['ff_fc'],
                 f[frac['bright']].mean(), t_read, t_frac))
        del cont, los, mu, c, both, same, fresh

    # StarSim: the filling factors of the simulation (projected area of the cells) = those of the pixels of the cells
    times = np.arange(len(rows), dtype=float)
    sim_sp, sim_fc = starsim_ff(times, np.array([r[3] for r in rows]), np.array([r[4] for r in rows]), N_RINGS, grids)
    for k, (date, frac, both, _, _, a_sp, a_fc) in enumerate(rows):
        rel = abs(sim_fc[k] - a_fc) / a_fc
        check('%s StarSim ff_fc = pixels of the cells (< 0.1 %%)' % date, rel < 1e-3,
              'StarSim %.4f %%, pixels %.4f %% (%.1e relative: cell areas vs pixel counts)' % (sim_fc[k], a_fc, rel))
        rel = abs(sim_sp[k] - a_sp) / max(a_sp, 1e-9)
        check('%s StarSim ff_sp = pixels of the cells (< 1 %%)' % date, rel < 1e-2,
              'StarSim %.4f %%, pixels %.4f %%' % (sim_sp[k], a_sp))

    # the whole pipeline: make_sdo_sweep (npz maps + cells), maps read back by active_region_mask, StarSim
    with tempfile.TemporaryDirectory() as out:
        t = time.time()
        res = sdo.make_sdo_sweep(folder, out, DAYS[0], DAYS[1], size=None, faculae=['plage', 'fraction'],
                                 B_sat=[100.0, 300.0], fmt='npz', n_rings=N_RINGS, verbose=False)
        print('     make_sdo_sweep: 3 sets x %d days in %.0f s' % (len(rows), time.time() - t))
        lab = sdo.set_label(dict(faculae='fraction', B_sat=100.0))
        r = res[lab]
        check('sweep: summary ff_fc = identify_day', np.allclose(r['ff_fc'], [x[1]['ff_fc'] for x in rows], rtol=1e-6),
              str(np.round(r['ff_fc'], 4)))
        check('sweep: B_sat 300 G < 100 G', np.all(res[sdo.set_label(dict(faculae='fraction', B_sat=300.0))]['ff_fc']
                                                   < r['ff_fc']))
        ff = sdo.load_filling_factors(os.path.join(out, lab), N_RINGS)
        check('sweep: cells of filling_factors_N60.npz = cell_filling_factors',
              np.allclose(ff.ff_fc, np.array([x[4] for x in rows]), atol=1e-6))
        m = active_region_mask(os.path.join(out, lab), origin='lower')
        import sunsim.nbspectra as nb
        nin = np.asarray(nb.generate_grid_coordinates_nb(N_RINGS)[1], dtype=np.int64)
        sp_m, fc_m = m.filling_factors(m.times, N_RINGS, nin)
        check('sweep: npz maps read back by active_region_mask = the cells', np.allclose(fc_m, ff.ff_fc, atol=1e-6)
              and np.allclose(sp_m, ff.ff_sp, atol=1e-6))
        sets_csv = open(os.path.join(out, 'sets.csv')).read()
        check('sweep: sets.csv has B_sat', sets_csv.startswith('label,noise_th,B_th,I_th,A_th,mu_area,faculae,B_sat\n')
              and lab in sets_csv)
        print('     npz map of the fractions: %.0f kB per day' % (os.path.getsize(os.path.join(
            out, lab, 'faculae_map_0.0.npz')) / 1e3))


if __name__ == '__main__':
    synthetic()
    if '--synthetic' not in sys.argv:
        real_days()
    print('\n%s' % ('all ok' if not FAILED else '%d FAILED: %s' % (len(FAILED), ', '.join(FAILED))))
    sys.exit(1 if FAILED else 0)
