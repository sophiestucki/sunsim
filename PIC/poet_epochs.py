#!/usr/bin/env python
"""
SunSim at the times of the PoET exposures (poet.dag):

  day        One day (one Condor job): every PoET exposure of the day (BJD of poet_jdb.csv -> UTC at the Earth) takes
             the closest 720 s continuum of HMI (and the 45 s magnetogram closest to it); every epoch is downloaded
             from JSOC, resampled once, identified with every set of thresholds of the parameter file and its cell
             filling factors written; the FITS are deleted right after. A day can be started again: the epochs done
             are skipped.
  collect    All the days: one folder per set of thresholds (filling_factors_N<n>.npz, summary.csv, parameters.txt,
             what python -m sunsim.sdo --simulate-only reads; times = days since --start) and exposures.csv (the
             epoch of every exposure). The days without done.json are left out (incomplete_days.txt).
  exposures  The simulated time series of one set (rv.txt, ... of sunsim.sdo) at every exposure (the values of its
             epoch): <set folder>/exposures_simulated.csv.
  table      The exposures_simulated.csv of all the simulations in one csv (long format: one row per exposure and
             simulation).

  python poet_epochs.py day --times poet_jdb.csv --date 2026-04-06 --out OUT --params parameters_poet.txt
  python poet_epochs.py collect --times poet_jdb.csv --out OUT --params parameters_poet.txt --start 2026-04-06
  python poet_epochs.py exposures --out OUT --set OUT/sets/<name>
  python poet_epochs.py table --out OUT

Folders: OUT/days/<date>/epochs/<T_REC>__<set label>.npz (ff_sp, ff_fc, info), OUT/days/<date>/exposures.npz,
OUT/days/<date>/done.json, OUT/ident/<set label>/, OUT/exposures.csv, OUT/sets/<name>/, OUT/results/.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

COLUMNS = ('noise_th', 'I_th', 'A_th', 'Temp')     # columns of the parameter file (Temp: the simulation only)


def _log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _key(t_rec):
    """File name of a T_REC (no ':' or '.')."""
    return re.sub(r'[^0-9A-Za-z_]', '', t_rec.replace('.', '').replace(':', ''))


def threshold_sets(path):
    """The sets of thresholds of the parameter file (one per distinct noise_th, I_th, A_th; B_th, mu_area and faculae
    are the defaults of sunsim.sdo: 24 G, 0.1, plage): {label: thresholds}."""
    from sunsim import sdo
    sets = {}
    for line in open(path):
        v = re.split(r'[ ,\t]+', line.strip())
        if len(v) < len(COLUMNS) or v[0].startswith('#'):
            continue
        p = dict(zip(COLUMNS, (float(x) for x in v)))
        s = dict(noise_th=p['noise_th'], B_th=sdo.DEFAULTS['B_th'], I_th=p['I_th'], A_th=p['A_th'],
                 mu_area=sdo.DEFAULTS['mu_area'], faculae='plage', B_sat=sdo.DEFAULTS['B_sat'])
        sets[sdo.set_label(s)] = s
    if not sets:
        raise ValueError('no set of thresholds in %s' % path)
    return sets


def utc_of_bjd(bjd):
    """JD (UTC, at the Earth) of BJD (TDB) computed for a target in the direction of the Sun: BJD = JD_TDB + (r . n) / c
    with r the barycentric position of the Earth and n the direction from the Earth to the Sun (r . n / c ~ -500 s),
    inverted by iteration (the correction changes by < 1 ms in 500 s). The Earth is its centre (Paranal is < 25 ms of
    light from it). astropy's Time.light_travel_time cannot be used: it takes the direction of the target from the
    barycentre, which for the Sun is not the direction from the Earth."""
    import astropy.units as u
    from astropy.constants import c
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    bjd = Time(np.asarray(bjd, dtype=np.float64), format='jd', scale='tdb')
    t = bjd
    for _ in range(3):
        earth = get_body_barycentric('earth', t).xyz.to_value(u.m)
        sun = get_body_barycentric('sun', t).xyz.to_value(u.m)
        n = (sun - earth) / np.linalg.norm(sun - earth, axis=0)
        ltt = np.sum(earth * n, axis=0) / c.to_value(u.m / u.s)
        t = Time(bjd.jd1, bjd.jd2, format='jd', scale='tdb') - ltt * u.s
    return t.utc.jd


def load_exposures(path):
    """The PoET exposures: {'jdb', 'jd' (UTC at the Earth), 'date' (UTC day)}, sorted."""
    import pandas as pd
    from astropy.time import Time
    jdb = np.sort(pd.read_csv(path)['jdb'].values.astype(np.float64))
    jd = utc_of_bjd(jdb)
    return {'jdb': jdb, 'jd': jd, 'date': np.array(Time(jd, format='jd', scale='utc').iso).astype('U10')}


# =============================================================================================
# one day
# =============================================================================================
def run_day(a):
    from sunsim import sdo
    from sunsim import supergranulation as sg

    sets = threshold_sets(a.params)
    folder = os.path.join(a.out, 'days', a.date)
    os.makedirs(os.path.join(folder, 'epochs'), exist_ok=True)
    done_file = os.path.join(folder, 'done.json')
    if os.path.exists(done_file):
        os.remove(done_file)                         # written again at the end

    ex = load_exposures(a.times)
    day = ex['date'] == a.date
    jdb, jd = ex['jdb'][day], ex['jd'][day]
    _log('%s: %d exposures, %d sets of thresholds (%s)' % (a.date, len(jd), len(sets), ', '.join(sets)))
    if len(jd) == 0:
        json.dump({'date': a.date, 'n_exp': 0}, open(done_file, 'w'))
        return 0

    # ---- matching: every exposure takes the closest 720 s continuum (and the 45 s magnetogram closest to it)
    rec = {k: sg.query_records(sg.SERIES[k][0], jd.min() - a.max_dt / 86400, jd.max() + a.max_dt / 86400)
           for k in sg.SERIES}
    slots = rec['continuum']
    exp_trec = np.array([''] * len(jd), dtype=object)
    exp_dt = np.full(len(jd), np.inf)
    epochs = {}
    if len(slots) and len(rec['magnetogram']):
        k, dt = sg.closest(jd, slots['jd'].values)
        for i, (kk, ddt) in enumerate(zip(k, dt)):
            if ddt > a.max_dt:
                continue
            tr = slots['T_REC'].values[kk]
            if tr not in epochs:
                km, dm = sg.closest(slots['jd'].values[kk], rec['magnetogram']['jd'].values)
                epochs[tr] = {'continuum': tr, 'magnetogram': rec['magnetogram']['T_REC'].values[km[0]],
                              'jd': float(slots['jd'].values[kk]), 'dt_magnetogram': float(dm[0])}
            exp_trec[i], exp_dt[i] = tr, ddt
    epochs = sorted(epochs.values(), key=lambda e: e['jd'])
    n_match = int(np.sum(np.isfinite(exp_dt)))
    _log('%d of %d exposures matched to %d epochs (|dt| max %.0f s)' % (n_match, len(jd), len(epochs),
         np.nanmax(np.where(np.isfinite(exp_dt), exp_dt, np.nan)) if n_match else np.nan))
    np.savez(os.path.join(folder, 'exposures.npz'), jdb=jdb, jd=jd, exp_trec=exp_trec.astype(str), exp_dt=exp_dt,
             epochs=json.dumps(epochs))

    def path_of(e, label):
        return os.path.join(folder, 'epochs', '%s__%s.npz' % (_key(e['continuum']), label))

    todo = [e for e in epochs if not all(os.path.exists(path_of(e, lab)) for lab in sets)]
    if a.limit:
        todo = todo[:a.limit]
    _log('to do: %d of %d epochs (~%.1f GB downloaded, deleted after every epoch)' % (len(todo), len(epochs),
                                                                                   0.05 * len(todo)))
    fits_dir = tempfile.mkdtemp(prefix='hmi_', dir=a.tmp)
    failed = 0
    try:
        t0 = time.time()
        for k, (e, files) in enumerate(sg.prefetch(todo, lambda e: sg.fetch_epoch(e, fits_dir), workers=1, ahead=1), 1):
            if isinstance(files, Exception):
                failed += 1
                _log('[%d/%d] %s: download failed (%s)' % (k, len(todo), e['continuum'], str(files)[:150]))
                continue
            try:
                cont = sg.read_record(*files['continuum'])
                mag = sg.read_record(*files['magnetogram'])
                cont, mag = sdo._resample(cont, mag, a.size)          # once for all the sets
                mu = sdo.mu_of_map(mag)
                for label, s in sets.items():
                    if os.path.exists(path_of(e, label)):
                        continue
                    d = sdo._identify_resampled(cont, mag, mu, s['noise_th'], s['B_th'], s['I_th'], s['A_th'],
                                                s['mu_area'], s['faculae'], s['B_sat'])
                    ff_sp, ff_fc = sdo.cell_filling_factors(d, a.n_rings)
                    info = {key: float(d[key]) for key in ('ff_sp', 'ff_fc', 'ff_nt')}
                    info.update({'n_spots': int(d['n_spots']), 'n_plages': int(d['n_plages'])})
                    info.update({key: e[key] for key in ('continuum', 'magnetogram', 'jd', 'dt_magnetogram')})
                    np.savez(path_of(e, label), ff_sp=ff_sp, ff_fc=ff_fc, info=json.dumps(info))
                    del d
                del cont, mag, mu
            except Exception as err:                 # e.g. an image with missing data
                failed += 1
                _log('[%d/%d] %s: not processed (%s)' % (k, len(todo), e['continuum'], str(err)[:150]))
            finally:
                for p, _ in files.values():
                    if os.path.exists(p):
                        os.remove(p)
            el = time.time() - t0
            _log('[%d/%d] epoch %s, %.0f s, ~%.0f min left' % (k, len(todo), e['continuum'], el,
                                                               el / k * (len(todo) - k) / 60))
    finally:
        shutil.rmtree(fits_dir, ignore_errors=True)

    n_done = {lab: sum(os.path.exists(path_of(e, lab)) for e in epochs) for lab in sets}
    if a.limit:                                      # a test: the day is not complete
        _log('%s: %s of %d epochs (--limit %d: no done.json)' % (a.date, n_done, len(epochs), a.limit))
        return 1 if failed else 0
    json.dump({'date': a.date, 'n_exp': int(len(jd)), 'n_matched': n_match, 'n_epochs': len(epochs),
               'epochs_done': n_done, 'failed': failed, 'size': a.size, 'n_rings': a.n_rings, 'max_dt': a.max_dt,
               'sets': sets}, open(done_file, 'w'), indent=1)
    _log('%s done: %s of %d epochs, %d failed' % (a.date, n_done, len(epochs), failed))
    return 1 if failed else 0


# =============================================================================================
# all the days
# =============================================================================================
def run_collect(a):
    import pandas as pd
    from astropy.time import Time
    sets = threshold_sets(a.params)
    t_start = Time(a.start + 'T00:00:00', scale='utc').jd
    ex = load_exposures(a.times)
    days = sorted(set(ex['date']))
    rows_exp, epochs = [], {}
    incomplete = []
    for date in days:
        folder = os.path.join(a.out, 'days', date)
        if not os.path.exists(os.path.join(folder, 'done.json')):
            incomplete.append(date)
            continue
        if not os.path.exists(os.path.join(folder, 'exposures.npz')):
            continue                                 # no exposure that day
        with np.load(os.path.join(folder, 'exposures.npz')) as z:
            for e in json.loads(str(z['epochs'])):
                files = {lab: os.path.join(folder, 'epochs', '%s__%s.npz' % (_key(e['continuum']), lab)) for lab in sets}
                if all(os.path.exists(f) for f in files.values()):     # the epochs done for every set
                    epochs[e['continuum']] = dict(e, files=files)
            rows_exp.append(pd.DataFrame({'jdb': z['jdb'], 'jd_utc': z['jd'], 'T_REC': z['exp_trec'],
                                          'dt_s': z['exp_dt'], 'date': date}))
    with open(os.path.join(a.out, 'incomplete_days.txt'), 'w') as f:
        f.write(''.join(d + '\n' for d in incomplete))
    if incomplete:
        _log('WARNING: %d days not done (left out, listed in incomplete_days.txt): %s'
             % (len(incomplete), ' '.join(incomplete)))
    order = sorted(epochs, key=lambda tr: epochs[tr]['jd'])
    times = np.array([epochs[tr]['jd'] - t_start for tr in order])
    _log('%d epochs in %d days' % (len(order), len(days) - len(incomplete)))

    for label, s in sets.items():
        out = os.path.join(a.out, 'ident', label)
        os.makedirs(out, exist_ok=True)
        ff_sp, ff_fc, info = [], [], []
        for tr in order:
            with np.load(epochs[tr]['files'][label]) as z:
                ff_sp.append(z['ff_sp'])
                ff_fc.append(z['ff_fc'])
                info.append(json.loads(str(z['info'])))
        np.savez(os.path.join(out, 'filling_factors_N%d.npz' % a.n_rings), times=times, dates=np.array(order),
                 N_rings=a.n_rings, ff_sp=np.array(ff_sp, dtype=np.float32), ff_fc=np.array(ff_fc, dtype=np.float32))
        with open(os.path.join(out, 'summary.csv'), 'w') as f:
            f.write('t,date,ff_sp,ff_fc,ff_nt,n_spots,n_plages\n')
            for t, tr, r in zip(times, order, info):
                f.write('%.8f,%s,%.6f,%.6f,%.6f,%d,%d\n' % (t, tr, r['ff_sp'], r['ff_fc'], r['ff_nt'], r['n_spots'],
                                                          r['n_plages']))
        with open(os.path.join(out, 'parameters.txt'), 'w') as f:
            f.write('times = %s (PoET exposures, the closest 720 s continuum and 45 s magnetogram)\nstart = %s (t = 0)\n'
                    'size = %d\nn_rings = %d\n' % (os.path.abspath(a.times), a.start, a.size, a.n_rings))
            f.write(''.join('%s = %s\n' % (k, v) for k, v in s.items()))
        _log('%s: %d epochs, ff_sp %.3f-%.3f %%, ff_fc %.2f-%.2f %%' % (
            label, len(order), min(r['ff_sp'] for r in info), max(r['ff_sp'] for r in info),
            min(r['ff_fc'] for r in info), max(r['ff_fc'] for r in info)))

    exp = pd.concat(rows_exp, ignore_index=True) if rows_exp else pd.DataFrame()
    index = {tr: i for i, tr in enumerate(order)}
    exp['epoch'] = [index.get(tr, -1) for tr in exp['T_REC']]
    exp['t'] = [times[i] if i >= 0 else np.nan for i in exp['epoch']]
    exp.to_csv(os.path.join(a.out, 'exposures.csv'), index=False)
    _log('%s: %d exposures, %d with an epoch' % (os.path.join(a.out, 'exposures.csv'), len(exp),
                                                  int(np.sum(exp['epoch'] >= 0))))
    return 0


# =============================================================================================
# the simulations at the exposures
# =============================================================================================
OBSERVABLES = ('lc', 'rv', 'fwhm', 'bis', 'contrast')


def run_exposures(a):
    import pandas as pd
    exp = pd.read_csv(os.path.join(a.out, 'exposures.csv'))
    with np.load(os.path.join(a.set, 'filling_factors_N%d.npz' % a.n_rings)) as z:
        times = z['times']
    out = exp[['jdb', 'jd_utc', 'date', 'T_REC', 'dt_s', 'epoch', 't']].copy()
    ok = out['epoch'].values >= 0
    for q in OBSERVABLES + ('ff',):
        d = np.loadtxt(os.path.join(a.set, '%s.txt' % q), ndmin=2)
        if len(d) != len(times) or np.max(np.abs(d[:, 0] - times)) > 1e-6:
            sys.exit('%s/%s.txt is not on the epochs of the filling factors' % (a.set, q))
        names = ('ff_sp', 'ff_fc') if q == 'ff' else (q,)
        for j, name in enumerate(names, 1):
            v = np.full(len(out), np.nan)
            v[ok] = d[out['epoch'].values[ok], j]
            out[name] = v
    path = os.path.join(a.set, 'exposures_simulated.csv')
    out.to_csv(path, index=False)
    _log('%s: %d exposures (%d with an epoch)' % (path, len(out), int(ok.sum())))
    return 0


def run_table(a):
    import pandas as pd
    parts = []
    sets = os.path.join(a.out, 'sets')
    for name in sorted(os.listdir(sets)):
        path = os.path.join(sets, name, 'exposures_simulated.csv')
        if not os.path.exists(path):
            _log('%s: no exposures_simulated.csv (left out)' % name)
            continue
        d = pd.read_csv(path)
        meta = json.load(open(os.path.join(sets, name, 'poet_run.json')))
        for k, v in meta.items():
            d[k] = v
        parts.append(d)
    if not parts:
        sys.exit('no simulation in %s' % sets)
    res = os.path.join(a.out, 'results')
    os.makedirs(res, exist_ok=True)
    table = pd.concat(parts, ignore_index=True)
    table.to_csv(os.path.join(res, 'poet_simulations.csv'), index=False)
    # one line per simulation: standard deviation of the observables over the exposures
    keys = list(meta)
    summary = table.groupby(keys)[['rv', 'fwhm', 'bis', 'contrast', 'lc']].std().reset_index()
    summary.to_csv(os.path.join(res, 'poet_simulations_std.csv'), index=False)
    _log('%s: %d simulations, %d rows' % (os.path.join(res, 'poet_simulations.csv'), len(parts), len(table)))
    print(summary.to_string(index=False))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('day', help='one day: match, download, identify (every set), delete')
    d.add_argument('--times', required=True, help='csv with a column jdb (BJD TDB of the exposures)')
    d.add_argument('--date', required=True, help='YYYY-MM-DD (UTC day of the exposures)')
    d.add_argument('--out', required=True)
    d.add_argument('--params', required=True, help='parameter file: noise_th, I_th, A_th, Temp')
    d.add_argument('--size', type=int, default=4096)
    d.add_argument('--n-rings', type=int, default=60)
    d.add_argument('--max-dt', type=float, default=720.0, help='[s] exposures farther from a continuum are left out')
    d.add_argument('--tmp', default=None, help='folder of the FITS while they are processed')
    d.add_argument('--limit', type=int, default=0, help='only the first epochs to do (a test; the day stays incomplete)')
    c = sub.add_parser('collect', help='all the days: filling factors of every set and exposures.csv')
    c.add_argument('--times', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--params', required=True)
    c.add_argument('--start', required=True, help='t = 0 (YYYY-MM-DD, 00:00 UTC)')
    c.add_argument('--size', type=int, default=4096)
    c.add_argument('--n-rings', type=int, default=60)
    e = sub.add_parser('exposures', help='the simulation of one set at every exposure')
    e.add_argument('--out', required=True)
    e.add_argument('--set', required=True, help='folder of the simulation (python -m sunsim.sdo --simulate-only)')
    e.add_argument('--n-rings', type=int, default=60)
    t = sub.add_parser('table', help='all the simulations in one csv')
    t.add_argument('--out', required=True)
    a = p.parse_args(argv)
    return {'day': run_day, 'collect': run_collect, 'exposures': run_exposures, 'table': run_table}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
