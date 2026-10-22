#!/usr/bin/env python
"""
Supergranulation and p modes of every HARPS-N solar exposure from SDO/HMI, for a whole period (the analysis of
analysis/supergranulation_harpsn_2021.ipynb, on PIC). Two steps:

  python supergranulation.py day --date 2021-05-05 --out DIR --harpsn FILE [--pmodes] [...]
      One day (one Condor job): the HMI epochs (720 s) closest to its HARPS-N exposures, identified (spots, faculae)
      and their dopplergram reduced to the velocity of the quiet photosphere of every cell (supergranulation.process_epoch);
      with --pmodes also every 45 s dopplergram around the exposures (supergranulation.frame_cell_velocities). The FITS
      files are deleted right after every epoch / frame (at most `workers` + 1 on disk). A day can be started again:
      what is done is skipped. --dry-run: only the matching and the size of the downloads.
  python supergranulation.py collect --out DIR --harpsn FILE --conf CONF [...]
      All the days: calibration of the HMI velocities over the whole period (rotation scale, north-south gradient, ring
      profile, polynomial of OBS_VR), p modes of every exposure, simulation of every exposure at its own time (maps;
      maps + dopplergram; maps + dopplergram + p modes), and the comparison with HARPS-N:
        - every day (intra-day: median of the day removed): RMS, Pearson r, slope; one figure per day;
        - the daily averages over the period (one offset per instrumental period): maps against maps + dopplergram;
        - the full dataset: every exposure of the period (one offset per instrumental period): RMS, Pearson r, slope.

Folder DIR:
  days/<date>/exposures.npz   the exposures of the day and their HMI epoch (T_REC, |dt|)
  days/<date>/epochs/*.npz    one file per 720 s epoch: cell filling factors and velocities, keywords
  days/<date>/frames45/*.npz  one file per 45 s frame: mean velocity of every cell
  days/<date>/done.json       the day is complete (numbers of exposures, epochs, frames, failures)
  results/                    exposures.csv, per_day.csv, intra_day_summary.csv, daily_means.csv, daily_stats.csv,
                              full_stats.csv, figures/ (days/<date>.png, daily_*.png, per_day_r.png, full_correlation.png)
                              (--results NAME: another folder, e.g. for collect with --min-obs-quality; then also
                              removed_exposures.csv, the exposures left out by the quality cut)

Quality of the exposures (collect --harpsn FILE --min-obs-quality Q): the flags of the release are matched to every
exposure (by the start of the exposure in file_root): data_quality (0/1: the exposures with 0 are always left out) and
obs_quality (0-1, the quality of the sky during the exposure: the release has only exposures with obs_quality >= 0.9);
the exposures with obs_quality < Q are left out of the comparison (the HMI calibration uses all the epochs).
--min-day-exp N: then also the days with fewer than N exposures left (all their exposures). The airmass of every
exposure (release) is in exposures.csv.

Needs sunsim with the sdo extras (sunpy, drms) and internet access to jsoc.stanford.edu (no JSOC account).
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

QUANTITIES = ('rv', 'fwhm', 'bis', 'contrast')
MODELS = ('maps', 'maps + dopplergram', 'maps + dopplergram + p modes')
COLOURS = {'maps': '#ea8642', 'maps + dopplergram': '#45a5aa', 'maps + dopplergram + p modes': '#c4a000'}
UNITS = {'rv': ('RV', 'm/s', '%.2f'), 'fwhm': ('FWHM', 'm/s', '%.2f'), 'bis': ('BIS', 'm/s', '%.2f'),
         'contrast': ('contrast', 'ppm', '%.0f')}


def _key(t_rec):
    """File name of a record: the digits of its T_REC (YYYYMMDDhhmmss)."""
    return re.sub(r'[^0-9]', '', str(t_rec))[:14]


def _log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# =============================================================================================
# one day
# =============================================================================================
def run_day(a):
    from sunsim import observations as obs
    from sunsim import supergranulation as sg

    folder = os.path.join(a.out, 'days', a.date)
    for sub in ('epochs', 'frames45'):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    done_file = os.path.join(folder, 'done.json')
    if os.path.exists(done_file) and not a.dry_run:
        os.remove(done_file)                         # written again at the end

    # ---- exposures of the day (middle of the exposure, UTC at the Earth)
    h = obs.load_harpsn(a.harpsn, a.date, end=a.date, daily=False, quality=a.quality)
    if 'jd_start' not in h:
        sys.exit('load_harpsn gives no jd_start: update sunsim (times from file_root + texp)')
    n_exp = len(h['jd'])
    _log('%s: %d exposures' % (a.date, n_exp))
    if n_exp == 0:
        if not a.dry_run:
            json.dump({'date': a.date, 'n_exp': 0}, open(done_file, 'w'))
        return 0

    # ---- matching: every exposure takes the closest 720 s slot with a good continuum and dopplergram (as the notebook)
    exp_file = os.path.join(folder, 'exposures.npz')
    names = {k: sg.SERIES[k][0] for k in sg.SERIES}
    jd0, jd1 = h['jd'].min() - a.max_dt / 86400, h['jd'].max() + a.max_dt / 86400
    rec = {k: sg.query_records(names[k], jd0, jd1) for k in names}
    slots = rec['continuum'][rec['continuum']['T_REC'].isin(rec['dopplergram']['T_REC'])].reset_index(drop=True)
    exp_trec = np.array([''] * n_exp, dtype=object)
    exp_dt = np.full(n_exp, np.inf)
    epochs = {}
    if len(slots) and len(rec['magnetogram']):
        k, dt = sg.closest(h['jd'], slots['jd'].values)
        for i, (kk, ddt) in enumerate(zip(k, dt)):
            if ddt > a.max_dt:
                continue
            tr = slots['T_REC'].values[kk]
            if tr not in epochs:
                km, _ = sg.closest(slots['jd'].values[kk], rec['magnetogram']['jd'].values)
                epochs[tr] = {'continuum': tr, 'dopplergram': tr, 'magnetogram': rec['magnetogram']['T_REC'].values[km[0]],
                              'jd': float(slots['jd'].values[kk])}
            exp_trec[i], exp_dt[i] = tr, ddt
    epochs = sorted(epochs.values(), key=lambda e: e['jd'])
    np.savez(exp_file, **{k: h[k] for k in ('jd', 'jd_start', 'texp', 'rv', 'rv_err', 'fwhm', 'fwhm_err', 'bis',
                                            'bis_err', 'contrast', 'contrast_err') if k in h},
             exp_trec=exp_trec.astype(str), exp_dt=exp_dt, epochs=json.dumps(epochs))
    _log('%d exposures matched to %d epochs (|dt| max %.0f s)' % (np.sum(exp_dt <= a.max_dt), len(epochs),
         exp_dt[np.isfinite(exp_dt)].max() if np.any(np.isfinite(exp_dt)) else np.nan))

    # ---- the 45 s frames needed: within ref_window / 2 (+ 2 frames) of the middle of an exposure
    frames = []
    if a.pmodes:
        half = (a.ref_window / 2 + 90) / 86400
        r45 = sg.query_records(sg.SERIES_45S[0], h['jd'].min() - half, h['jd'].max() + half)
        near = np.zeros(len(r45), bool)
        for m in h['jd']:
            near |= np.abs(r45['jd'].values - m) <= half
        frames = list(r45['T_REC'].values[near])
    todo_e = [e for e in epochs if not os.path.exists(os.path.join(folder, 'epochs', _key(e['continuum']) + '.npz'))]
    todo_f = [t for t in frames if not os.path.exists(os.path.join(folder, 'frames45', _key(t) + '.npz'))]
    if a.limit:
        todo_e, todo_f = todo_e[:a.limit], todo_f[:a.limit]
    _log('to do: %d of %d epochs (~%.1f GB), %d of %d frames of 45 s (~%.1f GB); all deleted after processing'
         % (len(todo_e), len(epochs), 0.05 * len(todo_e), len(todo_f), len(frames), 0.018 * len(todo_f)))
    if a.dry_run:
        return 0

    thresholds = dict(noise_th=a.noise_th, B_th=a.B_th, I_th=a.I_th, A_th=a.A_th, mu_area=a.mu_area, faculae=a.faculae)
    fits_dir = tempfile.mkdtemp(prefix='hmi_', dir=a.tmp)
    failed = 0
    try:
        # ---- 720 s epochs
        t0 = time.time()
        for k, (e, files) in enumerate(sg.prefetch(todo_e, lambda e: sg.fetch_epoch(e, fits_dir),
                                                   workers=a.workers, ahead=a.workers), 1):
            if isinstance(files, Exception):
                failed += 1
                _log('[%d/%d] %s: download failed (%s)' % (k, len(todo_e), e['continuum'], str(files)[:150]))
                continue
            try:
                r = sg.process_epoch(files, size=a.size, n_rings=a.n_rings, thresholds=thresholds)
                info = {key: (float(r[key]) if np.ndim(r[key]) == 0 else list(map(float, r[key])))
                        for key in ('ff_sp_disc', 'ff_fc_disc', 'ff_nt_disc', 'n_spots', 'n_plages', 'omega', 'rotation',
                                    'v_quiet_mean', 'obs_vr', 'B0', 'shift_px')}
                info.update({key: e[key] for key in ('continuum', 'magnetogram', 'dopplergram', 'jd')})
                np.savez(os.path.join(folder, 'epochs', _key(e['continuum']) + '.npz'), ff_sp=r['ff_sp'],
                         ff_fc=r['ff_fc'], v_cell=r['v_cell'], w_cell=r['w_cell'], info=json.dumps(info))
                del r
            except Exception as err:                 # e.g. an image with missing data
                failed += 1
                _log('[%d/%d] %s: not processed (%s)' % (k, len(todo_e), e['continuum'], str(err)[:150]))
            finally:
                for p, _ in files.values():
                    if os.path.exists(p):
                        os.remove(p)
            if k % 10 == 0 or k == len(todo_e):
                el = time.time() - t0
                _log('[%d/%d] epochs, %.0f s, ~%.0f min left' % (k, len(todo_e), el, el / k * (len(todo_e) - k) / 60))

        # ---- 45 s frames
        t0 = time.time()
        fetch45 = lambda t: sg.fetch_record(sg.SERIES_45S[0], t, sg.SERIES_45S[1], fits_dir)
        for k, (t_rec, files) in enumerate(sg.prefetch(todo_f, fetch45, workers=a.workers, ahead=a.workers), 1):
            if isinstance(files, Exception):
                failed += 1
                _log('[%d/%d] %s: download failed (%s)' % (k, len(todo_f), t_rec, str(files)[:150]))
                continue
            try:
                r = sg.frame_cell_velocities(files, a.n_rings)
                np.savez(os.path.join(folder, 'frames45', _key(t_rec) + '.npz'), v_cell=r['v_cell'], n_pix=r['n_pix'],
                         jd=r['jd'], obs_vr=r['obs_vr'], t_rec=t_rec)
            except Exception as err:
                failed += 1
                _log('[%d/%d] %s: not processed (%s)' % (k, len(todo_f), t_rec, str(err)[:150]))
            finally:
                if os.path.exists(files[0]):
                    os.remove(files[0])
            if k % 50 == 0 or k == len(todo_f):
                el = time.time() - t0
                _log('[%d/%d] frames, %.0f s, ~%.0f min left' % (k, len(todo_f), el, el / k * (len(todo_f) - k) / 60))
    finally:
        shutil.rmtree(fits_dir, ignore_errors=True)

    n_ep = len([f for f in os.listdir(os.path.join(folder, 'epochs')) if f.endswith('.npz')])
    n_fr = len([f for f in os.listdir(os.path.join(folder, 'frames45')) if f.endswith('.npz')])
    if not a.limit:
        json.dump({'date': a.date, 'n_exp': n_exp, 'n_epochs': len(epochs), 'epochs_done': n_ep,
                   'n_frames': len(frames), 'frames_done': n_fr, 'failed': failed, 'pmodes': bool(a.pmodes),
                   'n_rings': a.n_rings, 'size': a.size, 'thresholds': thresholds}, open(done_file, 'w'), indent=1)
    _log('%s done: %d / %d epochs, %d / %d frames, %d failed' % (a.date, n_ep, len(epochs), n_fr, len(frames), failed))
    return 1 if failed else 0


# =============================================================================================
# all the days
# =============================================================================================
def relative_ppm(x):
    x = np.asarray(x, dtype=np.float64)
    return 1e6 * (x / np.nanmedian(x) - 1)


def fit_line(x, y):
    """Pearson r, slope and its standard error of the least-squares line y = a + slope x."""
    x, y = np.asarray(x, float) - np.mean(x), np.asarray(y, float) - np.mean(y)
    if len(x) < 3 or np.sum(x * x) == 0:
        return np.nan, np.nan, np.nan
    slope = np.sum(x * y) / np.sum(x * x)
    err = np.sqrt(np.sum((y - slope * x) ** 2) / (len(x) - 2) / np.sum(x * x))
    return np.corrcoef(x, y)[0, 1], slope, err


def exposure_quality(ex, harpsn):
    """data_quality, obs_quality and airmass of the release (csv harpsn) for every exposure of ex (load_days), matched by
    the start of the exposure (file_root, to the second). NaN for an exposure that is not in the file."""
    import pandas as pd
    from astropy.time import Time
    d = pd.read_csv(harpsn, usecols=['file_root', 'data_quality', 'obs_quality', 'airmass'])
    iso = d['file_root'].astype(str).str.extract(r'(\d{4}-\d\d-\d\dT\d\d)-(\d\d)-(\d\d(?:\.\d+)?)')
    ok = ~iso.isna().any(axis=1)
    d, iso = d[ok], iso[ok]
    jd = Time(list(iso[0] + ':' + iso[1] + ':' + iso[2]), format='isot', scale='utc').jd
    row = dict(zip(np.round(jd * 86400).astype(np.int64), range(len(d))))
    idx = np.array([row.get(k, -1) for k in np.round(np.asarray(ex['jd_start']) * 86400).astype(np.int64)])
    out = {}
    for k in ('data_quality', 'obs_quality', 'airmass'):
        v = np.asarray(d[k], dtype=np.float64)
        out[k] = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    return out


def load_days(out, n_rings):
    """The products of the complete days: exposures (dict of arrays, one row per exposure with an epoch), epochs (dict
    of arrays), the 45 s frames of every day."""
    days = sorted(d for d in os.listdir(os.path.join(out, 'days'))
                  if os.path.exists(os.path.join(out, 'days', d, 'done.json')))
    ex, ep, frames = [], {k: [] for k in ('ff_sp', 'ff_fc', 'v_cell', 'w_cell', 'jd', 'obs_vr', 'a0', 'key')}, {}
    for d in days:
        folder = os.path.join(out, 'days', d)
        done = json.load(open(os.path.join(folder, 'done.json')))
        if done.get('n_exp', 0) == 0:
            continue
        if done.get('n_rings', n_rings) != n_rings:
            raise ValueError('%s: %d rings, collect asked for %d' % (d, done['n_rings'], n_rings))
        row_of = {}
        for f in sorted(os.listdir(os.path.join(folder, 'epochs'))):
            if not f.endswith('.npz'):
                continue
            with np.load(os.path.join(folder, 'epochs', f)) as z:
                info = json.loads(str(z['info']))
                row_of[_key(info['continuum'])] = len(ep['jd'])
                for k in ('ff_sp', 'ff_fc', 'v_cell', 'w_cell'):
                    ep[k].append(z[k])
            ep['jd'].append(info['jd']); ep['obs_vr'].append(info['obs_vr']); ep['a0'].append(info['rotation'][0])
            ep['key'].append(_key(info['continuum']))
        with np.load(os.path.join(folder, 'exposures.npz'), allow_pickle=True) as z:
            e = {k: z[k] for k in z.files if k != 'epochs'}
        e['epoch'] = np.array([row_of.get(_key(tr), -1) if tr else -1 for tr in e['exp_trec']])
        e['day'] = np.array([d] * len(e['jd']))
        ex.append(e)
        jd45, v45, n45 = [], [], []
        for f in sorted(os.listdir(os.path.join(folder, 'frames45'))):
            if f.endswith('.npz'):
                with np.load(os.path.join(folder, 'frames45', f)) as z:
                    jd45.append(float(z['jd'])); v45.append(z['v_cell']); n45.append(z['n_pix'])
        if jd45:
            frames[d] = (np.array(jd45), np.array(v45), np.array(n45))
    keys = set.intersection(*[set(e) for e in ex])
    ex = {k: np.concatenate([e[k] for e in ex]) for k in keys}
    keep = ex['epoch'] >= 0
    ex = {k: v[keep] for k, v in ex.items()}
    ep = {k: np.array(v) for k, v in ep.items()}
    return ex, ep, frames


def run_collect(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from sunsim import nbspectra, observations as obs, simulate_old_config, precomputed_filling_factors
    from sunsim import supergranulation as sg

    res_dir = os.path.join(a.out, getattr(a, 'results', 'results'))
    fig_dir = os.path.join(res_dir, 'figures')
    figures = getattr(a, 'figures', True)
    os.makedirs(os.path.join(fig_dir, 'days'), exist_ok=True)
    ex, ep, frames = load_days(a.out, a.n_rings)
    if getattr(a, 'harpsn', None):
        # quality flags of the release: data_quality = 1 always, obs_quality >= min_obs_quality if it is given
        qf = exposure_quality(ex, a.harpsn)
        ex.update(qf)
        keep = qf['data_quality'] == 1
        if getattr(a, 'min_obs_quality', None) is not None:
            keep &= qf['obs_quality'] >= a.min_obs_quality
        _log('quality: %d of %d exposures left out (data_quality != 1: %d, not in the release: %d, obs_quality < %s: %d)'
             % ((~keep).sum(), len(keep), np.sum(qf['data_quality'] == 0), np.sum(np.isnan(qf['data_quality'])),
                getattr(a, 'min_obs_quality', None), np.sum(qf['obs_quality'] < (getattr(a, 'min_obs_quality', None) or -1))))
        reason = np.where(keep, '', 'quality').astype(object)
        if getattr(a, 'min_day_exp', 0):
            # then the days with fewer than min_day_exp exposures left
            days, n_left = np.unique(ex['day'][keep], return_counts=True)
            few = np.isin(ex['day'], days[n_left < a.min_day_exp]) & keep
            _log('days with fewer than %d exposures left: %d (%d exposures)'
                 % (a.min_day_exp, len(np.unique(ex['day'][few])), few.sum()))
            reason[few] = 'day with < %d exposures' % a.min_day_exp
            keep &= ~few
        pd.DataFrame({'day': ex['day'][~keep], 'jd_utc': ex['jd'][~keep], 'data_quality': qf['data_quality'][~keep],
                      'obs_quality': qf['obs_quality'][~keep], 'airmass': qf['airmass'][~keep],
                      'reason': reason[~keep]}).to_csv(os.path.join(res_dir, 'removed_exposures.csv'), index=False)
        ex = {k: v[keep] for k, v in ex.items()}
        if not keep.any():
            raise ValueError('no exposures left after the quality cuts (min_obs_quality = %s, min_day_exp = %s; obs_quality '
                             'of the exposures: %.4f to %.4f)' % (getattr(a, 'min_obs_quality', None),
                                                                  getattr(a, 'min_day_exp', 0), np.nanmin(qf['obs_quality']),
                                                                  np.nanmax(qf['obs_quality'])))
    jd_start = obs.jd_of_date(a.start)
    ex['t'] = ex['jd'] - jd_start
    n_days = len(np.unique(ex['day']))
    _log('%d exposures on %d days, %d epochs, 45 s frames on %d days' % (len(ex['jd']), n_days, len(ep['jd']), len(frames)))

    # ---- calibration of the HMI velocities over the whole period (notebook, section 4.1)
    amu = np.asarray(nbspectra.generate_grid_coordinates_nb(a.n_rings)[3])
    ring = sg.ring_of_cells(a.n_rings)
    ld = (1 - 0.6 * (1 - amu))[ring]
    w_rv = ep['w_cell'] * ld
    no_data = ~np.isfinite(ep['v_cell'])
    v_scaled, scale = sg.rescale_by_rotation(ep['v_cell'], ep['a0'])
    v_flat, ns = sg.remove_large_scale(v_scaled, ep['w_cell'], a.n_rings, terms=('ns',))
    v_ring, _ = sg.remove_ring_profile(v_flat, ep['w_cell'], a.n_rings)
    v_sg, disc_raw, disc_sg, vr_coef = sg.remove_observer_trend(v_ring, w_rv, ep['obs_vr'], degree=a.vr_degree)
    v_sg[no_data] = 0.0
    _log('calibration: scale %.4f-%.4f, north-south %.0f to %.0f m/s/R, disc velocity std %.2f -> %.2f m/s (OBS_VR '
         'polynomial of degree %d)' % (scale.min(), scale.max(), ns.min(), ns.max(), np.std(disc_raw), np.std(disc_sg),
                                       a.vr_degree))

    # ---- p modes of every exposure (notebook, section 4.3)
    e_idx = ex['epoch']
    delta = np.zeros((len(e_idx), v_sg.shape[1]))
    ok45 = np.zeros(len(e_idx), bool)
    for d, (jd45, v45, _) in frames.items():
        m = np.flatnonzero(ex['day'] == d)
        if len(m) and len(jd45) > 1:
            dd, ok = sg.pmode_velocities(jd45, v45, ex['jd_start'][m], ex['jd_start'][m] + ex['texp'][m] / 86400,
                                         ref_window=a.ref_window)
            delta[m], ok45[m] = dd, ok
    _log('p modes for %d of %d exposures' % (ok45.sum(), len(ok45)))
    v_exp = v_sg[e_idx]
    v_pm = v_exp + delta
    v_pm[no_data[e_idx]] = 0.0

    # ---- simulations of every exposure at its own time
    conf = re.sub(r'(?m)^n_grid_rings\s*:\s*\d+', 'n_grid_rings : %d' % a.n_rings, open(a.conf).read())
    conf_file = os.path.join(res_dir, 'sun_N%d.conf' % a.n_rings)
    open(conf_file, 'w').write(conf)
    paths = {k: v for k, v in (('mps_folder', a.mps_folder), ('ccf_mask', a.mask)) if v}
    t = ex['t']
    ff = precomputed_filling_factors(t, ep['ff_sp'][e_idx], ep['ff_fc'][e_idx], a.n_rings)
    t0 = time.time()
    sims = {'maps': simulate_old_config(conf_file, times=t, observables=list(QUANTITIES), paths=paths,
                                        active_regions_mask=ff, verbose=False)}
    grids = sims['maps']['grids']
    for name, v in (('maps + dopplergram', v_exp), ('maps + dopplergram + p modes', v_pm)):
        sims[name] = simulate_old_config(conf_file, times=t, observables=list(QUANTITIES), paths=paths, grids=grids,
                                         active_regions_mask=ff, verbose=False,
                                         quiet_velocity=sg.quiet_velocity_field(t, v, a.n_rings))
    _log('3 simulations of %d exposures: %.0f s' % (len(t), time.time() - t0))
    sim = {name: {q: (relative_ppm(s[q]) if q == 'contrast' else np.asarray(s[q], float)) for q in QUANTITIES}
           for name, s in sims.items()}
    ob = {q: np.asarray(ex[q], float) for q in QUANTITIES}
    err = {q: np.asarray(ex.get(q + '_err', np.full(len(t), np.nan)), float) for q in QUANTITIES}
    err['contrast'] = 1e6 * err['contrast'] / np.nanmedian(ob['contrast'])
    ob['contrast'] = relative_ppm(ob['contrast'])

    table = pd.DataFrame({'day': ex['day'], 'jd_utc': ex['jd'], 't': t, 'sdo_t_rec': ex['exp_trec'], 'dt_sdo_s': ex['exp_dt'],
                          'pmodes': ok45})
    for k in ('obs_quality', 'airmass'):
        if k in ex:
            table[k] = ex[k]
    for q in QUANTITIES:
        table[q + '_obs'], table[q + '_err'] = ob[q], err[q]
        for name in MODELS:
            table['%s_%s' % (q, name.replace(' + ', '_').replace(' ', '_'))] = sim[name][q]
    table.to_csv(os.path.join(res_dir, 'exposures.csv'), index=False)

    # ---- every day: intra-day comparison (median of the day removed) and one figure
    rows = []
    days = np.unique(ex['day'])
    for d in days:
        m = ex['day'] == d
        hd = {'t': t[m]}
        for q in QUANTITIES:
            hd[q], hd[q + '_err'] = ob[q][m], err[q][m]
        cd = {q: {name: obs.compare_rv_exposures(hd, sim[name][q][m], quantity=q) for name in MODELS} for q in QUANTITIES}
        for q in QUANTITIES:
            for name in MODELS:
                c = cd[q][name]
                r, k, k_err = fit_line(c['sim'], c['obs'])
                rows.append({'day': d, 'n': c['n'], 'n_pmodes': int(ok45[m].sum()), 'quantity': q, 'model': name,
                             'RMS obs': c['rms_obs'], 'RMS obs - sim': c['rms'], 'r': r, 'slope': k,
                             'slope_err': k_err, 'sim std': np.std(c['sim'])})
        if figures and m.sum() >= a.min_exp:
            plot_day(plt, d, cd, os.path.join(fig_dir, 'days', '%s.png' % d))
    per_day = pd.DataFrame(rows)
    per_day.to_csv(os.path.join(res_dir, 'per_day.csv'), index=False)

    # summary of the intra-day comparison over all the days (days with >= min_exp exposures)
    summ = []
    good = per_day[per_day['n'] >= a.min_exp]
    for q in QUANTITIES:
        for name in MODELS:
            g = good[(good['quantity'] == q) & (good['model'] == name)]
            if len(g) == 0:
                continue
            o, res = [], []
            for d in g['day'].values:
                k = (ex['day'] == d) & np.isfinite(ob[q]) & np.isfinite(sim[name][q])
                o.append(ob[q][k] - np.median(ob[q][k]))
                res.append(o[-1] - (sim[name][q][k] - np.median(sim[name][q][k])))
            o, res = np.concatenate(o), np.concatenate(res)
            summ.append({'quantity': q, 'model': name, 'days': len(g), 'exposures': len(o),
                         'median r': g['r'].median(), 'days with r > 0 [%]': 100 * np.mean(g['r'] > 0),
                         'median slope': g['slope'].median(), 'RMS obs (intra-day)': np.sqrt(np.mean(o ** 2)),
                         'RMS obs - sim (intra-day)': np.sqrt(np.mean(res ** 2))})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(res_dir, 'intra_day_summary.csv'), index=False)
    print(summ.round(3).to_string(), flush=True)
    if figures:
        plot_per_day_r(plt, per_day[per_day['n'] >= a.min_exp], os.path.join(fig_dir, 'per_day_r.png'))

    # ---- daily averages over the period: maps against maps + dopplergram (one offset per instrumental period)
    cuts = obs.load_interventions(a.interventions, a.start) if a.interventions else None
    dm = []
    for d in days:
        m = ex['day'] == d
        if m.sum() < a.min_exp:
            continue
        row = {'day': d, 't': np.floor(np.median(t[m])) + 0.5, 'n': int(m.sum())}
        for q in QUANTITIES:
            row[q + '_obs'] = ob[q][m].mean()
            row[q + '_err'] = ob[q][m].std(ddof=1) / np.sqrt(m.sum())
            for name in MODELS:
                row['%s_%s' % (q, name)] = sim[name][q][m].mean()
        dm.append(row)
    dm = pd.DataFrame(dm)
    dm.to_csv(os.path.join(res_dir, 'daily_means.csv'), index=False)
    hd = {'t': dm['t'].values}
    for q in QUANTITIES:
        hd[q], hd[q + '_err'] = dm[q + '_obs'].values, dm[q + '_err'].values
    cmp_daily = {q: {name: obs.compare_rv_exposures(hd, dm['%s_%s' % (q, name)].values, quantity=q, interventions=cuts)
                     for name in MODELS} for q in QUANTITIES}
    rows = []
    for q in QUANTITIES:
        for name in MODELS:
            c = cmp_daily[q][name]
            r, k, k_err = fit_line(c['sim'], c['obs'])
            rows.append({'quantity': q, 'model': name, 'days': c['n'], 'RMS obs': c['rms_obs'], 'MAD obs': c['mad_obs'],
                         'RMS obs - sim': c['rms'], 'MAD obs - sim': c['mad'], 'r': r, 'slope': k, 'slope_err': k_err,
                         'sim std': np.std(c['sim'])})
    daily_stats = pd.DataFrame(rows)
    daily_stats.to_csv(os.path.join(res_dir, 'daily_stats.csv'), index=False)
    print(daily_stats.round(3).to_string(), flush=True)
    if figures:
        plot_daily(plt, cmp_daily, a.start, os.path.join(fig_dir, 'daily_timeseries.png'))
        plot_daily_correlation(plt, cmp_daily, os.path.join(fig_dir, 'daily_correlation_rv.png'))

    # ---- the full dataset: every exposure of the period (medians removed, one offset per instrumental period)
    hf = {'t': t}
    for q in QUANTITIES:
        hf[q], hf[q + '_err'] = ob[q], err[q]
    cmp_full = {q: {name: obs.compare_rv_exposures(hf, sim[name][q], quantity=q, interventions=cuts) for name in MODELS}
                for q in QUANTITIES}
    rows = []
    for q in QUANTITIES:
        for name in MODELS:
            c = cmp_full[q][name]
            s = c['obs'] - c['res']                               # simulation + offsets
            r, k, k_err = fit_line(s, c['obs'])
            rows.append({'quantity': q, 'model': name, 'exposures': c['n'], 'RMS obs': c['rms_obs'],
                         'MAD obs': c['mad_obs'], 'RMS obs - sim': c['rms'], 'MAD obs - sim': c['mad'], 'r': r,
                         'slope': k, 'slope_err': k_err, 'sim std': np.std(s)})
    full_stats = pd.DataFrame(rows)
    full_stats.to_csv(os.path.join(res_dir, 'full_stats.csv'), index=False)
    print(full_stats.round(3).to_string(), flush=True)
    if figures:
        plot_full_correlation(plt, cmp_full, os.path.join(fig_dir, 'full_correlation.png'))
    _log('results in %s' % res_dir)
    return 0


# =============================================================================================
# figures
# =============================================================================================
def plot_day(plt, day, cd, path):
    """One day: every quantity (median of the day removed), HARPS-N and the simulations, and the residuals."""
    fig = plt.figure(figsize=(14, 4.2 * len(QUANTITIES)))
    outer = fig.add_gridspec(len(QUANTITIES), 1, hspace=0.35)
    for k, q in enumerate(QUANTITIES):
        label, unit, fmt = UNITS[q]
        inner = outer[k].subgridspec(2, 1, height_ratios=[2, 1], hspace=0)
        ax = fig.add_subplot(inner[0])
        ar = fig.add_subplot(inner[1], sharex=ax)
        c0 = cd[q][MODELS[0]]
        t_day = np.floor(c0['t'].min())
        ax.errorbar((c0['t'] - t_day) * 24, c0['obs'], yerr=c0['obs_err'], fmt='o', ms=3, color='#102e37', alpha=0.5,
                    label=('HARPS-N: RMS ' + fmt + ' %s') % (c0['rms_obs'], unit))
        for name in MODELS:
            c = cd[q][name]
            h = (c['t'] - t_day) * 24
            r = np.corrcoef(c['obs'], c['sim'])[0, 1] if np.std(c['sim']) > 0 else np.nan
            ax.plot(h, c['sim'], '-', color=COLOURS[name], lw=1.3, label='%s (r = %.2f)' % (name, r))
            ar.plot(h, c['res'], 'o', ms=3, color=COLOURS[name], alpha=0.7,
                    label=('obs - %s: RMS ' + fmt) % (name, c['rms']))
        ar.axhline(0, color='k', alpha=0.2)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel('%s - median [%s]' % (label, unit))
        ar.set_ylabel('obs - sim')
        ax.set_title('%s: %s' % (day, label), fontsize=10)
        ax.legend(fontsize=7, ncols=2)
        ar.legend(fontsize=7, ncols=3)
    ar.set_xlabel('hour [UTC]')
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def plot_per_day_r(plt, per_day, path):
    """Pearson r and RMS obs - sim of every day (intra-day) for every quantity and simulation."""
    import pandas as pd
    fig, axes = plt.subplots(len(QUANTITIES), 2, figsize=(15, 3.2 * len(QUANTITIES)), sharex=True, squeeze=False)
    for row, q in zip(axes, QUANTITIES):
        label, unit, _ = UNITS[q]
        g = per_day[per_day['quantity'] == q]
        for name in MODELS:
            s = g[g['model'] == name]
            x = pd.to_datetime(s['day'])
            row[0].plot(x, s['r'], 'o-', ms=3, lw=0.6, color=COLOURS[name], label='%s: median %.2f' % (name, s['r'].median()))
            row[1].plot(x, s['RMS obs - sim'], 'o-', ms=3, lw=0.6, color=COLOURS[name], label=name)
        s = g[g['model'] == MODELS[0]]
        row[1].plot(pd.to_datetime(s['day']), s['RMS obs'], 'k.', label='HARPS-N')
        row[0].axhline(0, color='k', alpha=0.2)
        row[0].set_ylabel('r (%s, intra-day)' % label)
        row[1].set_ylabel('RMS [%s]' % unit)
        row[0].legend(fontsize=7)
        row[1].legend(fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_daily(plt, cmp_daily, start, path):
    """Daily averages over the period: HARPS-N and the simulations (offsets of compare_rv_exposures), and residuals."""
    import pandas as pd
    fig = plt.figure(figsize=(15, 4.5 * len(QUANTITIES)))
    outer = fig.add_gridspec(len(QUANTITIES), 1, hspace=0.35)
    for k, q in enumerate(QUANTITIES):
        label, unit, fmt = UNITS[q]
        inner = outer[k].subgridspec(2, 1, height_ratios=[2, 1], hspace=0)
        ax = fig.add_subplot(inner[0])
        ar = fig.add_subplot(inner[1], sharex=ax)
        c0 = cmp_daily[q][MODELS[0]]
        x = pd.Timestamp(start) + pd.to_timedelta(c0['t'], unit='D')
        ax.errorbar(x, c0['obs'], yerr=c0['obs_err'], fmt='o', ms=4, color='#102e37', alpha=0.6,
                    label=('HARPS-N: RMS ' + fmt + ' %s') % (c0['rms_obs'], unit))
        for name in MODELS[:2]:
            c = cmp_daily[q][name]
            x = pd.Timestamp(start) + pd.to_timedelta(c['t'], unit='D')
            ax.plot(x, c['obs'] - c['res'], 's-', ms=3, lw=1, color=COLOURS[name], label=name)
            ar.plot(x, c['res'], 'o-', ms=3, lw=0.6, color=COLOURS[name], label=('obs - %s: RMS ' + fmt) % (name, c['rms']))
        for cut in c0['interventions']:
            for a_ in (ax, ar):
                a_.axvline(pd.Timestamp(start) + pd.Timedelta(days=float(cut)), color='k', alpha=0.3)
        ar.axhline(0, color='k', alpha=0.2)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel('daily mean %s [%s]' % (label, unit))
        ar.set_ylabel('obs - sim')
        ax.set_title('daily averages: %s' % label, fontsize=10)
        ax.legend(fontsize=8)
        ar.legend(fontsize=8)
    fig.savefig(path, dpi=90, bbox_inches='tight')
    plt.close(fig)


def plot_daily_correlation(plt, cmp_daily, path):
    """Daily mean RV of HARPS-N against the simulations: maps, maps + dopplergram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), squeeze=False)
    for ax, name in zip(axes[0], MODELS[:2]):
        c = cmp_daily['rv'][name]
        s = c['obs'] - c['res']                                   # simulation + offsets
        r, k, k_err = fit_line(s, c['obs'])
        ax.errorbar(s, c['obs'], yerr=c['obs_err'], fmt='o', ms=4, color=COLOURS[name], alpha=0.6)
        lim = np.array([min(s.min(), c['obs'].min()), max(s.max(), c['obs'].max())])
        ax.plot(lim, lim, 'k-', alpha=0.3, label='1:1')
        xs = np.array([s.min(), s.max()])
        ax.plot(xs, np.mean(c['obs']) + k * (xs - np.mean(s)), '--', color='k', label='slope %.2f ± %.2f' % (k, k_err))
        ax.set_title('daily means, %s\nr = %.2f, obs - sim %.2f m/s' % (name, r, c['rms']), fontsize=10)
        ax.set_xlabel('simulated RV [m/s]')
        ax.set_ylabel('HARPS-N RV [m/s]')
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_full_correlation(plt, cmp_full, path):
    """Every exposure of the period: HARPS-N against every simulation (with the offsets), for every quantity."""
    fig, axes = plt.subplots(len(QUANTITIES), len(MODELS), figsize=(5 * len(MODELS), 4.4 * len(QUANTITIES)),
                             squeeze=False)
    for row, q in zip(axes, QUANTITIES):
        label, unit, fmt = UNITS[q]
        for ax, name in zip(row, MODELS):
            c = cmp_full[q][name]
            s = c['obs'] - c['res']
            r, k, k_err = fit_line(s, c['obs'])
            ax.plot(s, c['obs'], '.', ms=2, color=COLOURS[name], alpha=0.4)
            lim = np.array([min(s.min(), c['obs'].min()), max(s.max(), c['obs'].max())])
            ax.plot(lim, lim, 'k-', alpha=0.3, label='1:1')
            xs = np.array([s.min(), s.max()])
            ax.plot(xs, np.mean(c['obs']) + k * (xs - np.mean(s)), '--', color='k',
                    label='slope %.2f ± %.2f' % (k, k_err))
            ax.set_title(('%s, %s\nr = %.2f, RMS obs ' + fmt + ', obs - sim ' + fmt + ' %s')
                         % (label, name, r, c['rms_obs'], c['rms'], unit), fontsize=9)
            ax.set_xlabel('simulated %s [%s]' % (label, unit))
            ax.set_ylabel('HARPS-N %s [%s]' % (label, unit))
            ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# =============================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', required=True)

    d = sub.add_parser('day', help='one day: HMI epochs and 45 s frames of its exposures')
    d.add_argument('--date', required=True, help='YYYY-MM-DD (UTC day of the exposures)')
    d.add_argument('--out', required=True)
    d.add_argument('--harpsn', required=True, help='csv of the HARPS-N public release')
    d.add_argument('--quality', action='store_true', help='only the exposures with data_quality = 1')
    d.add_argument('--size', type=int, default=4096, help='pixels of the images (4096: full resolution)')
    d.add_argument('--n-rings', type=int, default=30)
    d.add_argument('--noise-th', type=float, default=8.0)
    d.add_argument('--B-th', type=float, default=24.0)
    d.add_argument('--I-th', type=float, default=0.89)
    d.add_argument('--A-th', type=float, default=20.0)
    d.add_argument('--mu-area', type=float, default=0.1)
    d.add_argument('--faculae', default='plage', choices=('plage', 'plage+network'))
    d.add_argument('--max-dt', type=float, default=900.0, help='[s] exposures farther from an HMI epoch are not used')
    d.add_argument('--pmodes', action='store_true', help='also the 45 s dopplergrams (p modes)')
    d.add_argument('--ref-window', type=float, default=1800.0, help='[s] p modes: exposure mean - mean over this window')
    d.add_argument('--workers', type=int, default=1, help='downloads at the same time')
    d.add_argument('--tmp', default=None, help='folder of the FITS files (deleted after every epoch / frame)')
    d.add_argument('--limit', type=int, default=0, help='test: at most this many epochs and frames')
    d.add_argument('--dry-run', action='store_true', help='only the matching and the size of the downloads')

    c = sub.add_parser('collect', help='all the days: calibration, simulations, comparison with HARPS-N')
    c.add_argument('--out', required=True)
    c.add_argument('--start', required=True, help='origin of the times (YYYY-MM-DD)')
    c.add_argument('--conf', required=True, help='configuration of the simulation (n_grid_rings replaced)')
    c.add_argument('--mps-folder', default=None, help='MURaM spectra (replaces the path of the configuration)')
    c.add_argument('--mask', default=None, help='CCF mask (replaces the path of the configuration)')
    c.add_argument('--interventions', default=None, help='HARPS-N interventions (one offset per instrumental period)')
    c.add_argument('--n-rings', type=int, default=30)
    c.add_argument('--vr-degree', type=int, default=2, help='polynomial of OBS_VR removed from the disc velocity')
    c.add_argument('--ref-window', type=float, default=1800.0)
    c.add_argument('--min-exp', type=int, default=5, help='days with fewer exposures: not in the statistics')
    c.add_argument('--harpsn', default=None, help='csv of the release: its quality flags (data_quality = 1 kept)')
    c.add_argument('--min-obs-quality', type=float, default=None,
                   help='with --harpsn: only the exposures with obs_quality >= this (e.g. 0.99)')
    c.add_argument('--min-day-exp', type=int, default=0,
                   help='with --harpsn: then also the days with fewer exposures left than this (e.g. 50)')
    c.add_argument('--results', default='results', help='folder of the results, in --out (default results)')
    c.add_argument('--no-figures', dest='figures', action='store_false', help='only the tables')

    a = p.parse_args(argv)
    return run_day(a) if a.step == 'day' else run_collect(a)


if __name__ == '__main__':
    sys.exit(main())
