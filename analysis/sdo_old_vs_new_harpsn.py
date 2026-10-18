"""
Old and new SDO/HMI identification against HARPS-N: RMS and MAD of the RV residuals.

Every day of HMI images (path 'sdo_data') from START to END is read once and classified
  - with the previous identification (analysis/sdo_old_classification.py: simu_rv_opt_fast.py + identification_utils.py),
  - with sunsim.sdo (the fixed one) for every set of NEW_SETS,
at SIZE pixels; the maps are projected on the cells of a grid of N rings (sdo.cell_filling_factors) and only these
filling factors are kept (OUT/days/<date>.npz: the run can be stopped and started again). Every identification is then
simulated with the same configuration and the same grids (sdo.simulate_sdo_sets) and compared with the HARPS-N daily
RVs (sunsim.observations.compare_rv: medians removed, one offset per instrumental period, best day shift) and with the
TSI of SORCE / TSIS-1. The RVs of the 10-year run of the old StarSim (REF_RV, the old identification at 4096 px) on the
same days are compared too, as a reference.

    python analysis/sdo_old_vs_new_harpsn.py [--workers 3] [--out folder]

Paths (local_paths.ini): sdo_data, old_sdo_identification, harpsn_sun, harpsn_interventions, sorce_tsi, tsis_tsi,
muram_* (the configuration of tests/old_version), and optionally ref_rv_10_years, ref_days_10_years (the RV and
the days of the 10-year run of the old StarSim).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

START, END = '2015-07-18', '2024-01-29'          # the 10-year run (t = 0 on START)
SIZE = 4096
N = 30
OLD = dict(noise_th=8.0, I_th=0.95, A_th=20.0)  # the best 10-year run: noise_th_8.0_I_th_0.95_A_th_20_T_300_size_4096
NEW_SETS = {'new_I0.95': dict(noise_th=8.0, B_th=24.0, I_th=0.95, A_th=20.0, mu_area=0.1, faculae='plage'),
            'new_I0.89': dict(noise_th=8.0, B_th=24.0, I_th=0.89, A_th=20.0, mu_area=0.1, faculae='plage')}
CONF_CHANGES = {'n_grid_rings': N, 'spot_T_contrast': 300.0, 'rotation_period': 25.4}


def process_day(task):
    """Old and new identifications of one day, as cell filling factors (saved in out/days/<date>.npz)."""
    import warnings
    from sunsim import sdo
    from sdo_old_classification import old_identify
    data, out, date, files = task
    path = os.path.join(out, 'days', date + '.npz')
    if os.path.exists(path):
        return date, 'done'
    t0 = time.perf_counter()
    try:
        cont, los = sdo._read_with_retries(data, date, files=files)
    except OSError as e:
        return date, 'not readable: %s' % str(e).splitlines()[0][:100]
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = old_identify(cont, los, size=SIZE, **OLD)
        res['old'] = (*sdo.cell_filling_factors(o, N), o['ff_sp'], o['ff_fc'])
        del o
        c, m = sdo._resample(cont, los, SIZE)
        mu = sdo.mu_of_map(m)
        for name, p in NEW_SETS.items():
            d = sdo._identify_resampled(c, m, mu, **p)
            res[name] = (*sdo.cell_filling_factors(d, N), d['ff_sp'], d['ff_fc'])
            del d
    np.savez(path + '.tmp.npz', **{'%s_%s' % (k, q): np.asarray(v[i]) for k, v in res.items()
                                   for i, q in enumerate(('cells_sp', 'cells_fc', 'ff_sp', 'ff_fc'))})
    os.replace(path + '.tmp.npz', path)
    return date, '%.1f s' % (time.perf_counter() - t0)


def gather(out, dates):
    """One folder per identification with filling_factors_N<N>.npz and summary.csv (for simulate_sdo_sets)."""
    t_ref = datetime.strptime(START, '%Y-%m-%d')
    days = [(d, np.load(os.path.join(out, 'days', d + '.npz'))) for d in dates
            if os.path.exists(os.path.join(out, 'days', d + '.npz'))]
    times = np.array([(datetime.strptime(d, '%Y-%m-%d') - t_ref).days for d, _ in days], dtype=float)
    folders = {}
    for name in ['old'] + list(NEW_SETS):
        folder = os.path.join(out, name)
        os.makedirs(folder, exist_ok=True)
        sp = np.array([z[name + '_cells_sp'] for _, z in days], dtype=np.float32)
        fc = np.array([z[name + '_cells_fc'] for _, z in days], dtype=np.float32)
        np.savez(os.path.join(folder, 'filling_factors_N%d.npz' % N), times=times,
                 dates=np.array([d for d, _ in days]), N_rings=N, ff_sp=sp, ff_fc=fc)
        with open(os.path.join(folder, 'summary.csv'), 'w') as f:
            f.write('t,date,ff_sp,ff_fc\n')
            for t, (d, z) in zip(times, days):
                f.write('%g,%s,%.6f,%.6f\n' % (t, d, float(z[name + '_ff_sp']), float(z[name + '_ff_fc'])))
        folders[name] = folder
    return folders


def write_conf(out):
    """The configuration of tests/old_version (MURaM, HARPS G2 mask, 2D maps) with CONF_CHANGES."""
    import re
    src = os.path.join(REPO, 'tests', 'old_version', 'reference', 'sdo_2015.conf')
    text = open(src).read()
    for k, v in CONF_CHANGES.items():
        text = re.sub(r'(?m)^%s = .*$' % k, '%s = %s' % (k, v), text)
    path = os.path.join(out, 'sdo_harpsn.conf')
    open(path, 'w').write(text)
    return path


def reference_rv(dates_sim):
    """The RV of the 10-year run of the old StarSim (paths ref_rv_10_years: its rv file, ref_days_10_years: its
    days_list.txt; optional) on the simulated days, and on all its days."""
    from sunsim import local_path
    rv_file, days_file = local_path('ref_rv_10_years', required=False), local_path('ref_days_10_years', required=False)
    if rv_file is None or days_file is None:
        return None
    rv, t = np.loadtxt(str(rv_file)), np.loadtxt(str(days_file))
    keep = np.isin(t, dates_sim)
    return t[keep], rv[keep], t, rv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--conf', default=None, help='configuration (sdo_input = 1, n_grid_rings = %d) instead of the one of '
                                                'tests/old_version with CONF_CHANGES' % N)
    p.add_argument('--out', default=os.path.join(REPO, 'analysis', 'results', 'sdo_old_vs_new_harpsn'))
    args = p.parse_args()
    from sunsim import sdo, local_path, observations as obs
    os.makedirs(os.path.join(args.out, 'days'), exist_ok=True)

    # 1. identifications (cell filling factors), every day read once
    data = str(local_path('sdo_data'))
    index = sdo._hmi_index(data)
    t0, t1 = datetime.strptime(START, '%Y-%m-%d'), datetime.strptime(END, '%Y-%m-%d')
    tasks = []
    day = t0
    while day < t1:
        files = index.get(day.strftime('%Y%m%d'))
        if files is not None:
            tasks.append((data, args.out, day.strftime('%Y-%m-%d'), files))
        day += timedelta(days=1)
    print('%d days with images' % len(tasks), flush=True)
    start = time.perf_counter()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, (date, msg) in enumerate(pool.map(process_day, tasks), 1):
            if msg != 'done':
                print('[%d/%d] %s %s (%.0f min)' % (k, len(tasks), date, msg, (time.perf_counter() - start) / 60),
                      flush=True)
    folders = gather(args.out, [t[2] for t in tasks])

    # 2. simulations, the grids computed once
    conf = args.conf or write_conf(args.out)
    tag = '' if args.conf is None else '_' + os.path.splitext(os.path.basename(args.conf))[0]
    sims = sdo.simulate_sdo_sets(conf, list(folders.values()), observables=['lc', 'rv'],
                                 grids_cache=os.path.join(args.out, 'grids%s.pkl' % tag), chunk_size=200)

    # 3. comparison with HARPS-N and the TSI
    harps = obs.load_harpsn(local_path('harpsn_sun'), START, end=END)
    cuts = obs.load_interventions(local_path('harpsn_interventions'), START)
    sorce = obs.load_sorce(local_path('sorce_tsi'), START, end=END)
    tsis = obs.load_tsis(local_path('tsis_tsi'), START, end=END)
    # the same days for all: days with a bad identification in any set (ff_fc > 20 %: a bad HMI image) removed
    t_all = sims[folders['old']]['time']
    bad = np.zeros(len(t_all), bool)
    for folder in folders.values():
        assert np.array_equal(sims[folder]['time'], t_all)
        bad |= sims[folder]['ff_fc'] > 20
    print('days removed (ff_fc > 20 %%): %s' % [str(datetime.strptime(START, '%Y-%m-%d') + timedelta(days=int(t)))[:10]
                                               for t in t_all[bad]])
    series = {name: (t_all[~bad], sims[f]['rv'][~bad], sims[f]['lc'][~bad], sims[f]['ff_sp'][~bad], sims[f]['ff_fc'][~bad])
              for name, f in folders.items()}
    ref = reference_rv(t_all[~bad])
    if ref is not None:
        series['old StarSim 10-year run'] = (ref[0], ref[1], None, None, None)
    summary = {}
    rv_cmp = {}
    for shifts, tag in (((0,), 'no shift'), ((-1, 0, 1), 'best shift')):
        for name, (t, rv, lc, ff_sp, ff_fc) in series.items():
            r = obs.compare_rv(harps, t, rv, interventions=cuts, shifts=shifts)
            row = {'n_days_rv': r['n'], 'rv_rms': r['rms'], 'rv_mad': r['mad'], 'harps_rms': r['rms_obs'],
                   'harps_mad': r['mad_obs'], 'reduction_%': r['reduction']}
            if lc is not None and tag == 'no shift':
                ph = [obs.compare_photometry(x, t, lc) for x in (sorce, tsis)]
                row.update(sorce_rms_ppm=ph[0]['rms'], sorce_mad_ppm=ph[0]['mad'], n_sorce=ph[0]['n'],
                           tsis_rms_ppm=ph[1]['rms'], tsis_mad_ppm=ph[1]['mad'], n_tsis=ph[1]['n'],
                           mean_ff_sp=float(np.mean(ff_sp)), mean_ff_fc=float(np.mean(ff_fc)))
            summary['%s, %s' % (name, tag)] = row
            rv_cmp[(name, tag)] = r
    if ref is not None:
        r = obs.compare_rv(harps, ref[2], ref[3], interventions=cuts)
        summary['old StarSim 10-year run, all 3117 days, best shift'] = {
            'n_days_rv': r['n'], 'rv_rms': r['rms'], 'rv_mad': r['mad'], 'harps_rms': r['rms_obs'],
            'harps_mad': r['mad_obs'], 'reduction_%': r['reduction']}
    json.dump(summary, open(os.path.join(args.out, 'summary%s.json' % tag), 'w'), indent=1)
    for name, v in summary.items():
        print('%-40s %s' % (name, ', '.join('%s %.4g' % (k, x) for k, x in v.items())))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = obs.plot_comparison(None, [rv_cmp[('old', 'no shift')], rv_cmp[('new_I0.95', 'no shift')]], start=START,
                                    figsize=(15, 10))
    axes[0].set_title('old identification (top) and new identification, I_th = 0.95 (bottom)')
    fig.savefig(os.path.join(args.out, 'rv_old_vs_new%s.png' % tag), dpi=110, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
