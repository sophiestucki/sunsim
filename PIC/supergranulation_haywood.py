#!/usr/bin/env python
"""
Supergranulation of every HARPS-N solar exposure from SDO/HMI 45 s dopplergrams corrected as in Haywood et al. (2016),
for a whole period (on PIC). Two steps:

  python supergranulation_haywood.py day --date 2021-05-05 --out DIR --harpsn FILE [--window 720] [...]
      One day (one Condor job): the HMI epochs (720 s continuum, 45 s magnetogram) closest to its HARPS-N exposures,
      identified (spots, faculae, quiet pixels; supergranulation.identify_epoch), and every 45 s dopplergram within
      window / 2 of the middle of an exposure, corrected for the motion of SDO (eq. 10) and projected on the cells with
      the quiet pixels of its closest epoch, with the three images of the rotation of eq. 13
      (supergranulation.frame_cell_sums). The velocity of the Sun relative to the barycentre along the line of sight
      of every exposure from JPL Horizons (observations.sun_barycentric_rv, eq. 1). The FITS files are deleted right
      after every epoch / frame. A day can be started again: what is done is skipped. --dry-run: only the matching and
      the size of the downloads.
  python supergranulation_haywood.py collect --out DIR --start DATE --conf CONF [...]
      All the days: the rotation of eq. 12-13 fitted on the frames of every day (a1, a2, a3; supergranulation.fit_rotation)
      and removed, the corrected dopplergrams averaged over the window of every exposure (the p modes cancel) and
      projected (supergranulation.window_average; nothing else removed; with --remove-disc-mean the intensity weighted
      mean of the quiet cells of every exposure, v_quiet of Ervin et al. 2022, eq. 15: the zero point of HMI drifts
      over the orbit of SDO; with --rotation-obs-vr a1 + a4 OBS_VR in the rotation; with --clv-deg N the CLV of the
      quiet cells, shifted Legendre polynomials of mu up to degree N (Ervin et al. 2024, eq. 4), minus its average over
      all the exposures (--clv-remove-average: all of it), its coefficients fitted with OBS_VR + one offset per day,
      --clv-drift obs_vr, or as fitted on every exposure, --clv-drift exposure; with --north-south a north-south
      gradient ns + ns_vr OBS_VR fitted with the rotation), simulation of every exposure at its own time
      (maps; maps + dopplergram), and the comparison with the HARPS-N RV in the rest frame of the Sun, rv_raw (DRS,
      barycentric) - v_sun (JPL), with --diff-extinction also minus rv_diff_extinction of the release (the differential
      extinction across the disc in the atmosphere, Collier Cameron et al. 2019: a slow trend in the day, not in HMI):
        - every day (intra-day: median of the day removed): RMS, Pearson r, slope; one figure per day;
        - the daily averages over the period (one offset per instrumental period): maps against maps + dopplergram;
        - the full dataset: every exposure of the period (one offset per instrumental period): RMS, Pearson r, slope.

Folder DIR:
  days/<date>/exposures.npz   the exposures of the day, their HMI epoch (T_REC, |dt|) and JPL velocity
  days/<date>/epochs/*.npz    one file per epoch: cell filling factors, keywords
  days/<date>/frames45/*.npz  one file per 45 s dopplergram: sums of the quiet pixels of every cell (velocity, weights,
                              the three images of the rotation), keywords
  days/<date>/done.json       the day is complete (numbers of exposures, epochs, frames, failures)
  results/                    rotation.csv (a1, a2, a3 of every day), exposures.csv, frames.csv, per_day.csv, intra_day_summary.csv, daily_means.csv,
                              daily_stats.csv, full_stats.csv, figures/ (days/<date>.png, daily_*.png, per_day_r.png,
                              full_correlation.png) (--results NAME: another folder; with --min-obs-quality also
                              removed_exposures.csv)

Quality of the exposures (collect --harpsn FILE --min-obs-quality Q): as in supergranulation.py of the earlier runs
(data_quality = 1 always, obs_quality >= Q, --min-day-exp N).

Needs sunsim with the sdo extras (sunpy, drms) and internet access to jsoc.stanford.edu and ssd.jpl.nasa.gov.
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
MODELS = ('maps', 'maps + dopplergram')
COLOURS = {'maps': '#ea8642', 'maps + dopplergram': '#45a5aa'}
UNITS = {'rv': ('RV', 'm/s', '%.2f'), 'fwhm': ('FWHM', 'm/s', '%.2f'), 'bis': ('BIS', 'm/s', '%.2f'),
         'contrast': ('contrast', 'ppm', '%.0f')}
HARPSN_KEYS = ('jd', 'jd_start', 'texp', 'rv', 'rv_err', 'fwhm', 'fwhm_err', 'bis', 'bis_err', 'contrast',
               'contrast_err', 'rv_raw', 'berv', 'berv_bary_to_helio', 'rv_diff_extinction')
FRAME_INFO = ('jd', 'obs_vr', 'obs_vw', 'obs_vn', 'B0', 'dsun', 'shift_px', 'v_quiet_mean')


def _b0(jd):
    """B0 [deg] at jd (UTC)."""
    from astropy.time import Time
    import sunpy.coordinates.sun as sun
    return float(sun.B0(Time(jd, format='jd')).deg)


def move_mask(mask, dt, b0, omega=None):
    """A mask of the disc (frame of the maps of identify_day: cut to the disc, row 0 south, west on the right) moved by
    the differential rotation over dt [days]: every pixel takes the value of the point that was there dt before (False
    for a point that was behind the limb). omega(lat) [deg/day, synodic]: Snodgrass & Ulrich (1990) minus the motion of
    the Earth by default."""
    from sunsim import supergranulation as sg
    n = mask.shape[0]
    c, R = (n - 1) / 2.0, n / 2.0
    y, x = np.mgrid[:n, :n].astype(np.float32)
    x, y = (x - c) / R, (y - c) / R
    z2 = 1 - x ** 2 - y ** 2
    inside = z2 > 0
    z = np.sqrt(np.where(inside, z2, 0.0))
    b = np.deg2rad(b0)
    lat = np.arcsin(np.clip(y * np.cos(b) + z * np.sin(b), -1, 1))
    lon = np.arctan2(x, z * np.cos(b) - y * np.sin(b))
    del z
    s2 = np.sin(lat) ** 2
    a1, a2, a3 = sg.SNODGRASS_ULRICH
    w = (a1 + a2 * s2 + a3 * s2 ** 2 - 0.9856) if omega is None else omega(np.rad2deg(lat))
    lon0 = lon - np.deg2rad(w * dt)
    del lon, s2, w
    x0 = np.cos(lat) * np.sin(lon0)
    y0 = np.sin(lat) * np.cos(b) - np.cos(lat) * np.cos(lon0) * np.sin(b)
    z0 = np.sin(lat) * np.sin(b) + np.cos(lat) * np.cos(lon0) * np.cos(b)
    i0 = np.clip(np.rint(y0 * R + c).astype(np.int32), 0, n - 1)
    j0 = np.clip(np.rint(x0 * R + c).astype(np.int32), 0, n - 1)
    return mask[i0, j0] & inside & (z0 > 0)


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
    if 'jd_start' not in h or 'rv_raw' not in h:
        sys.exit('load_harpsn gives no jd_start / rv_raw: update sunsim')
    n_exp = len(h['jd'])
    _log('%s: %d exposures' % (a.date, n_exp))
    if n_exp == 0:
        if not a.dry_run:
            json.dump({'date': a.date, 'n_exp': 0}, open(done_file, 'w'))
        return 0

    # ---- matching: every exposure takes the closest 720 s continuum (and the 45 s magnetogram closest to it)
    exp_file = os.path.join(folder, 'exposures.npz')
    jd0, jd1 = h['jd'].min() - a.max_dt / 86400, h['jd'].max() + a.max_dt / 86400
    rec = {k: sg.query_records(sg.SERIES[k][0], jd0, jd1) for k in sg.SERIES}
    slots = rec['continuum']
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
                epochs[tr] = {'continuum': tr, 'magnetogram': rec['magnetogram']['T_REC'].values[km[0]],
                              'jd': float(slots['jd'].values[kk])}
            exp_trec[i], exp_dt[i] = tr, ddt
    epochs = sorted(epochs.values(), key=lambda e: e['jd'])
    _log('%d exposures matched to %d epochs (|dt| max %.0f s)' % (np.sum(exp_dt <= a.max_dt), len(epochs),
         exp_dt[np.isfinite(exp_dt)].max() if np.any(np.isfinite(exp_dt)) else np.nan))

    # ---- the 45 s dopplergrams within window / 2 of the middle of a matched exposure, given to their closest epoch
    matched = h['jd'][exp_dt <= a.max_dt]
    frames = {}
    if len(epochs) and len(matched):
        half = (a.window / 2 + sg.CADENCE_45S) / 86400
        r45 = sg.query_records(sg.SERIES_45S[0], matched.min() - half, matched.max() + half)
        near = np.zeros(len(r45), bool)
        for m in matched:
            near |= np.abs(r45['jd'].values - m) <= half
        r45 = r45[near]
        ke, _ = sg.closest(r45['jd'].values, [e['jd'] for e in epochs])
        for t_rec, kk in zip(r45['T_REC'].values, ke):
            frames.setdefault(epochs[kk]['continuum'], []).append(t_rec)
    n_frames = sum(len(f) for f in frames.values())

    def frame_done(t):
        """The frame is done with the images of the rotation (the frames of the first runs, Table 1 removed, are not)."""
        path = os.path.join(folder, 'frames45', _key(t) + '.npz')
        if not os.path.exists(path):
            return False
        with np.load(path) as z:
            return 'swg' in z.files

    def epoch_done(e):
        return (os.path.exists(os.path.join(folder, 'epochs', _key(e['continuum']) + '.npz'))
                and all(frame_done(t) for t in frames.get(e['continuum'], [])))

    todo = [e for e in epochs if not epoch_done(e)]
    if a.limit:
        todo = todo[:a.limit]
    n_todo_f = sum(len([t for t in frames.get(e['continuum'], []) if not frame_done(t)]) for e in todo)
    _log('to do: %d of %d epochs (~%.1f GB), %d of %d frames of 45 s (~%.1f GB); all deleted after processing'
         % (len(todo), len(epochs), 0.035 * len(todo), n_todo_f, n_frames, 0.018 * n_todo_f))

    # ---- velocity of the Sun relative to the barycentre along the line of sight (JPL Horizons, eq. 1)
    if not a.dry_run:
        jpl = obs.sun_barycentric_rv(h['jd'])
        np.savez(exp_file, **{k: h[k] for k in HARPSN_KEYS if k in h}, v_sun=jpl['v'], lt=jpl['lt'], rr=jpl['rr'],
                 exp_trec=exp_trec.astype(str), exp_dt=exp_dt, epochs=json.dumps(epochs))
        _log('JPL: v_sun %.3f to %.3f m/s, minus berv_bary_to_helio: %.4f m/s at most'
             % (jpl['v'].min(), jpl['v'].max(), np.nanmax(np.abs(jpl['v'] - h.get('berv_bary_to_helio', np.nan)))))
    if a.dry_run:
        return 0

    thresholds = dict(noise_th=a.noise_th, B_th=a.B_th, I_th=a.I_th, A_th=a.A_th, mu_area=a.mu_area, faculae=a.faculae)
    # variants of the quiet mask, the same frames reduced with each (one download for all), in <out>_<name>/days/<date>:
    # --margin-px m: the quiet pixels farther than m pixels from the spots and the faculae (margin<m>px);
    # --move-masks: the mask of the epoch moved to the time of every frame with the differential rotation (moved)
    variants = {'margin%dpx' % m: m for m in getattr(a, 'margin_px', None) or []}
    if getattr(a, 'move_masks', False):
        variants['moved'] = 'moved'
    margins = {name: os.path.join('%s_%s' % (a.out.rstrip('/'), name), 'days', a.date) for name in variants}
    for mf in margins.values():
        for sub in ('epochs', 'frames45'):
            os.makedirs(os.path.join(mf, sub), exist_ok=True)
    fits_dir = tempfile.mkdtemp(prefix='hmi_', dir=a.tmp)
    failed = 0
    fetch45 = lambda t: sg.fetch_record(sg.SERIES_45S[0], t, sg.SERIES_45S[1], fits_dir)
    try:
        t0 = time.time()
        for k, (e, files) in enumerate(sg.prefetch(todo, lambda e: sg.fetch_epoch(e, fits_dir), workers=1, ahead=1), 1):
            if isinstance(files, Exception):
                failed += 1
                _log('[%d/%d] %s: download failed (%s)' % (k, len(todo), e['continuum'], str(files)[:150]))
                continue
            # ---- identification of the epoch
            try:
                r = sg.identify_epoch(files, size=a.size, n_rings=a.n_rings, thresholds=thresholds,
                                      network_velocity=a.network_velocity)
                info = {key: float(r[key]) for key in ('ff_sp_disc', 'ff_fc_disc', 'ff_nt_disc', 'n_spots', 'n_plages')}
                info.update({key: e[key] for key in ('continuum', 'magnetogram', 'jd')})
                for f_ep in [folder] + list(margins.values()):
                    np.savez(os.path.join(f_ep, 'epochs', _key(e['continuum']) + '.npz'), ff_sp=r['ff_sp'],
                             ff_fc=r['ff_fc'], info=json.dumps(info))
                context = r['context']
                contexts = {}
                if any(v != 'moved' for v in variants.values()):
                    import scipy.ndimage as nd
                    dist = nd.distance_transform_edt(~(context['spot'] | context['facula']))
                    contexts = {name: dict(context, quiet=context['quiet'] & (dist > m))
                                for name, m in variants.items() if m != 'moved'}
                    del dist
                b0_epoch = _b0(e['jd']) if 'moved' in variants else None
                del r
            except Exception as err:                 # e.g. an image with missing data
                failed += 1
                _log('[%d/%d] %s: not processed (%s)' % (k, len(todo), e['continuum'], str(err)[:150]))
                continue
            finally:
                for p, _ in files.values():
                    if os.path.exists(p):
                        os.remove(p)

            # ---- its 45 s dopplergrams
            todo_f = [t for t in frames.get(e['continuum'], []) if not frame_done(t)]
            for t_rec, f45 in sg.prefetch(todo_f, fetch45, workers=a.workers, ahead=a.workers):
                if isinstance(f45, Exception):
                    failed += 1
                    _log('%s: download failed (%s)' % (t_rec, str(f45)[:150]))
                    continue
                try:
                    if 'moved' in variants:
                        dt = float(sg.jd_of_t_rec([t_rec])[0]) - e['jd']
                        contexts['moved'] = dict(context, quiet=move_mask(context['quiet'], dt, b0_epoch))
                    for f_ep, ctx in [(folder, context)] + [(margins[m], contexts[m]) for m in margins]:
                        fr = sg.frame_cell_sums(f45, ctx, n_rings=a.n_rings, mu_min=a.mu_min)
                        np.savez(os.path.join(f_ep, 'frames45', _key(t_rec) + '.npz'), swv=fr['swv'].astype(np.float32),
                                 sw=fr['sw'].astype(np.float32), swg=fr['swg'].astype(np.float32), t_rec=t_rec,
                                 epoch=e['continuum'],
                                 info=json.dumps({key: float(fr[key]) for key in FRAME_INFO}))
                        del fr
                except Exception as err:
                    failed += 1
                    _log('%s: not processed (%s)' % (t_rec, str(err)[:150]))
                finally:
                    if os.path.exists(f45[0]):
                        os.remove(f45[0])
            del context, contexts
            el = time.time() - t0
            _log('[%d/%d] epoch %s (%d frames), %.0f s, ~%.0f min left'
                 % (k, len(todo), e['continuum'], len(todo_f), el, el / k * (len(todo) - k) / 60))
    finally:
        shutil.rmtree(fits_dir, ignore_errors=True)

    n_ep = len([f for f in os.listdir(os.path.join(folder, 'epochs')) if f.endswith('.npz')])
    n_fr = len([f for f in os.listdir(os.path.join(folder, 'frames45')) if f.endswith('.npz')])
    if not a.limit:
        json.dump({'date': a.date, 'n_exp': n_exp, 'n_epochs': len(epochs), 'epochs_done': n_ep,
                   'n_frames': n_frames, 'frames_done': n_fr, 'failed': failed, 'window': a.window,
                   'mu_min': a.mu_min, 'network_velocity': a.network_velocity, 'rotation': 'fitted per day (collect)', 'n_rings': a.n_rings, 'size': a.size,
                   'thresholds': thresholds}, open(done_file, 'w'), indent=1)
        for name, mf in margins.items():
            shutil.copy(exp_file, os.path.join(mf, 'exposures.npz'))
            done = json.load(open(done_file))
            done['quiet_mask'] = name
            json.dump(done, open(os.path.join(mf, 'done.json'), 'w'), indent=1)
    _log('%s done: %d / %d epochs, %d / %d frames, %d failed' % (a.date, n_ep, len(epochs), n_fr, n_frames, failed))
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
    of arrays), the 45 s frames of every day ({day: (jd, swv, sw, swg, info)}) and the window of the days [s]."""
    days = sorted(d for d in os.listdir(os.path.join(out, 'days'))
                  if os.path.exists(os.path.join(out, 'days', d, 'done.json')))
    ex, ep, frames, windows = [], {k: [] for k in ('ff_sp', 'ff_fc', 'jd', 'key')}, {}, set()
    for d in days:
        folder = os.path.join(out, 'days', d)
        done = json.load(open(os.path.join(folder, 'done.json')))
        if done.get('n_exp', 0) == 0:
            continue
        if done.get('n_rings', n_rings) != n_rings:
            raise ValueError('%s: %d rings, collect asked for %d' % (d, done['n_rings'], n_rings))
        windows.add(float(done['window']))
        row_of = {}
        for f in sorted(os.listdir(os.path.join(folder, 'epochs'))):
            if not f.endswith('.npz'):
                continue
            with np.load(os.path.join(folder, 'epochs', f)) as z:
                info = json.loads(str(z['info']))
                row_of[_key(info['continuum'])] = len(ep['jd'])
                for k in ('ff_sp', 'ff_fc'):
                    ep[k].append(z[k])
            ep['jd'].append(info['jd'])
            ep['key'].append(_key(info['continuum']))
        with np.load(os.path.join(folder, 'exposures.npz'), allow_pickle=True) as z:
            e = {k: z[k] for k in z.files if k != 'epochs'}
        e['epoch'] = np.array([row_of.get(_key(tr), -1) if tr else -1 for tr in e['exp_trec']])
        e['day'] = np.array([d] * len(e['jd']))
        ex.append(e)
        jd45, swv, sw, swg, info = [], [], [], [], []
        for f in sorted(os.listdir(os.path.join(folder, 'frames45'))):
            if f.endswith('.npz'):
                with np.load(os.path.join(folder, 'frames45', f)) as z:
                    if 'swg' not in z.files:
                        raise ValueError('%s: frames without the images of the rotation (Table 1 removed): run the day '
                                         'step again' % d)
                    i = json.loads(str(z['info']))
                    jd45.append(i['jd']); swv.append(z['swv']); sw.append(z['sw']); swg.append(z['swg']); info.append(i)
        if jd45:
            frames[d] = (np.array(jd45), np.array(swv, dtype=np.float64), np.array(sw, dtype=np.float64),
                         np.array(swg, dtype=np.float64), info)
    keys = set.intersection(*[set(e) for e in ex])
    ex = {k: np.concatenate([e[k] for e in ex]) for k in keys}
    keep = ex['epoch'] >= 0
    ex = {k: v[keep] for k, v in ex.items()}
    ep = {k: np.array(v) for k, v in ep.items()}
    return ex, ep, frames, windows


def run_collect(a):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sunsim import observations as obs, simulate_old_config, precomputed_filling_factors
    from sunsim import supergranulation as sg

    res_dir = os.path.join(a.out, getattr(a, 'results', 'results'))
    fig_dir = os.path.join(res_dir, 'figures')
    figures = getattr(a, 'figures', True)
    os.makedirs(os.path.join(fig_dir, 'days'), exist_ok=True)
    ex, ep, frames, windows = load_days(a.out, a.n_rings)
    window = a.window if a.window else min(windows)
    if window > min(windows):
        raise ValueError('--window %.0f s: the days have the frames of %.0f s only' % (window, min(windows)))
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

    # ---- rotation fitted on the frames of every day and removed, the dopplergrams averaged over the window of every
    # exposure and projected (nothing else removed)
    n_cells = ep['ff_sp'].shape[1]
    v_exp, sw_exp = np.full((len(ex['jd']), n_cells), np.nan), np.zeros((len(ex['jd']), n_cells))
    n45, ok45 = np.zeros(len(ex['jd']), np.int64), np.zeros(len(ex['jd']), bool)
    ex['obs_vr'] = np.full(len(ex['jd']), np.nan)
    rows, rot = [], []
    for d, (jd45, swv, sw, swg, info) in frames.items():
        vr45 = np.array([i['obs_vr'] for i in info], dtype=np.float64)
        if getattr(a, 'rotation_obs_vr', False):
            swg = sg.rotation_obs_vr_basis(swg, vr45)
        if getattr(a, 'north_south', False):
            swg = sg.north_south_basis(swg, sw, vr45, a.n_rings)
        alphas, err, rms = sg.fit_rotation(swv, sw, swg, a.n_rings)
        names = (['a1', 'a2', 'a3'] + ['a4'] * getattr(a, 'rotation_obs_vr', False)
                 + ['ns', 'ns_vr'] * getattr(a, 'north_south', False))
        rot.append(dict({'day': d, 'frames': len(jd45)}, **dict(zip(names, alphas)),
                        **{n + '_err': e for n, e in zip(names, err)}, rms_residual=rms))
        m = np.flatnonzero(ex['day'] == d)
        if len(m):
            o = np.argsort(jd45)
            ex['obs_vr'][m] = np.interp(ex['jd'][m], jd45[o], vr45[o])
            v_exp[m], sw_exp[m], n45[m], ok45[m] = sg.window_average(jd45, sg.remove_rotation(swv, swg, alphas), sw,
                                                                     ex['jd'][m], window=window, min_fill=a.min_fill)
        rows += [dict(i, day=d) for i in info]
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, 'frames.csv'), index=False)
    rot = pd.DataFrame(rot)
    rot.to_csv(os.path.join(res_dir, 'rotation.csv'), index=False)
    _log('rotation of the days [deg/day]: a1 %.3f to %.3f, a2 %.2f to %.2f, a3 %.2f to %.2f (Table 1: %s)'
         % (rot['a1'].min(), rot['a1'].max(), rot['a2'].min(), rot['a2'].max(), rot['a3'].min(), rot['a3'].max(),
            sg.SNODGRASS_ULRICH))
    if getattr(a, 'rotation_obs_vr', False):
        _log('a1 + a4 OBS_VR: a4 %.3f to %.3f deg/day per km/s' % (rot['a4'].min(), rot['a4'].max()))
    if getattr(a, 'north_south', False):
        _log('north-south gradient ns + ns_vr OBS_VR: ns %.1f to %.1f m/s per solar radius, ns_vr %.1f to %.1f per km/s'
             % (rot['ns'].min(), rot['ns'].max(), rot['ns_vr'].min(), rot['ns_vr'].max()))
    _log('dopplergrams: %d frames, window %.0f s: %d of %d exposures with >= %.0f %% of the frames (left out: %d)'
         % (len(rows), window, ok45.sum(), len(ok45), 100 * a.min_fill, (~ok45).sum()))
    ex = {k: v[ok45] for k, v in ex.items()}
    v_exp, sw_exp, n45 = v_exp[ok45], sw_exp[ok45], n45[ok45]
    # disc mean of the quiet cells of every exposure (weighted by the intensity: v_quiet of Ervin et al. 2022, eq. 15),
    # removed with --remove-disc-mean: the zero point of HMI drifts over the orbit of SDO (with OBS_VR)
    k = np.isfinite(v_exp) & (sw_exp > 0)
    v_disc = np.sum(np.where(k, sw_exp * np.nan_to_num(v_exp), 0.0), axis=1) / np.sum(np.where(k, sw_exp, 0.0), axis=1)
    clv_deg = getattr(a, 'clv_deg', -1)
    if getattr(a, 'remove_disc_mean', False) and clv_deg >= 0:
        raise ValueError('--remove-disc-mean or --clv-deg, not both')
    if getattr(a, 'remove_disc_mean', False):
        v_exp = v_exp - v_disc[:, None]
        _log('disc mean of the quiet cells removed from every exposure: %.1f to %.1f m/s' % (v_disc.min(), v_disc.max()))
    clv = np.full((len(v_exp), max(clv_deg + 1, 0)), np.nan)
    if clv_deg >= 0:
        # centre-to-limb variation of the quiet cells of every exposure (shifted Legendre polynomials of mu, Ervin et al.
        # 2024, eq. 4), minus its average over all the exposures: removed, the average CLV kept. --clv-drift obs_vr:
        # the coefficients of every exposure taken from their fit with OBS_VR + one offset per day (the supergranulation
        # around it kept); exposure: the coefficients of every exposure as fitted
        clv = sg.fit_clv(v_exp, sw_exp, a.n_rings, clv_deg)
        # --clv-remove-average: the average CLV removed too (the spectra of the grids already have the convective
        # blueshift; only the supergranulation around the CLV stays)
        c = clv if a.clv_drift == 'exposure' else sg.obs_vr_model(clv, ex['obs_vr'], ex['day'], a.drift_deg, ex['jd'],
                                                                   getattr(a, 'drift_harmonics', 0))
        keep = 0.0 if getattr(a, 'clv_remove_average', False) else c.mean(axis=0)
        v_exp = v_exp - (c - keep) @ sg.clv_basis(a.n_rings, clv_deg).T
        _log('CLV of the quiet cells (degree %d, %s) removed%s: c0 %.1f to %.1f m/s'
             % (clv_deg, a.clv_drift, '' if getattr(a, 'clv_remove_average', False) else ' (its average kept)',
                c[:, 0].min(), c[:, 0].max()))
    v_exp, filled = sg.fill_empty_cells(v_exp, sw_exp, a.n_rings)
    if getattr(a, 'save_cells', False):
        # the velocity of every cell of every exposure given to the simulations, its weight (sum of the intensities of
        # the quiet pixels) and the exposures (cells.npz of the results)
        np.savez(os.path.join(res_dir, 'cells.npz'), v=v_exp.astype(np.float32), w=sw_exp.astype(np.float32),
                 filled=filled, jd=ex['jd'], day=ex['day'].astype(str), obs_vr=ex['obs_vr'], n_rings=a.n_rings)
    _log('cells without quiet pixels (ring mean): %.1f per exposure' % filled.sum(axis=1).mean())

    jd_start = obs.jd_of_date(a.start)
    ex['t'] = ex['jd'] - jd_start
    n_days = len(np.unique(ex['day']))
    _log('%d exposures on %d days, %d epochs' % (len(ex['jd']), n_days, len(ep['jd'])))
    # HARPS-N RV in the rest frame of the Sun (Haywood et al. 2016, eq. 1): the RV of the DRS (BERV applied) - v_sun (JPL);
    # --diff-extinction: minus the differential extinction of the release too (as its rv)
    ex['rv_release'] = ex['rv']
    ex['rv'] = ex['rv_raw'] - ex['v_sun']
    if getattr(a, 'diff_extinction', False):
        ex['rv'] = ex['rv'] - ex['rv_diff_extinction']
        _log('differential extinction removed from the HARPS-N RV: %.2f to %.2f m/s'
             % (np.nanmin(ex['rv_diff_extinction']), np.nanmax(ex['rv_diff_extinction'])))

    # ---- simulations of every exposure at its own time
    e_idx = ex['epoch']
    conf = re.sub(r'(?m)^n_grid_rings\s*:\s*\d+', 'n_grid_rings : %d' % a.n_rings, open(a.conf).read())
    conf_file = os.path.join(res_dir, 'sun_N%d.conf' % a.n_rings)
    open(conf_file, 'w').write(conf)
    paths = {k: v for k, v in (('mps_folder', a.mps_folder), ('ccf_mask', a.mask)) if v}
    t = ex['t']
    ff = precomputed_filling_factors(t, ep['ff_sp'][e_idx], ep['ff_fc'][e_idx], a.n_rings)
    t0 = time.time()
    sims = {'maps': simulate_old_config(conf_file, times=t, observables=list(QUANTITIES), paths=paths,
                                        active_regions_mask=ff, verbose=False)}
    sims['maps + dopplergram'] = simulate_old_config(conf_file, times=t, observables=list(QUANTITIES), paths=paths,
                                                     grids=sims['maps']['grids'], active_regions_mask=ff, verbose=False,
                                                     quiet_velocity=sg.quiet_velocity_field(t, v_exp, a.n_rings))
    _log('%d simulations of %d exposures: %.0f s' % (len(sims), len(t), time.time() - t0))
    sim = {name: {q: (relative_ppm(s[q]) if q == 'contrast' else np.asarray(s[q], float)) for q in QUANTITIES}
           for name, s in sims.items()}
    ob = {q: np.asarray(ex[q], float) for q in QUANTITIES}
    err = {q: np.asarray(ex.get(q + '_err', np.full(len(t), np.nan)), float) for q in QUANTITIES}
    err['contrast'] = 1e6 * err['contrast'] / np.nanmedian(ob['contrast'])
    ob['contrast'] = relative_ppm(ob['contrast'])

    table = pd.DataFrame({'day': ex['day'], 'jd_utc': ex['jd'], 't': t, 'sdo_t_rec': ex['exp_trec'], 'dt_sdo_s': ex['exp_dt'],
                          'n_frames': n45, 'obs_vr': ex['obs_vr'], 'v_disc_mean': v_disc, 'rv_raw': ex['rv_raw'], 'v_sun_jpl': ex['v_sun'],
                          'berv_bary_to_helio': ex['berv_bary_to_helio'], 'rv_diff_extinction': ex['rv_diff_extinction'],
                          'rv_release': ex['rv_release']})
    for k in ('obs_quality', 'airmass'):
        if k in ex:
            table[k] = ex[k]
    for n in range(clv.shape[1]):
        table['clv_c%d' % n] = clv[:, n]
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
                rows.append({'day': d, 'n': c['n'], 'quantity': q, 'model': name, 'RMS obs': c['rms_obs'],
                             'RMS obs - sim': c['rms'], 'r': r, 'slope': k, 'slope_err': k_err,
                             'sim std': np.std(c['sim'])})
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
    if len(dm) >= 3:
        hd = {'t': dm['t'].values}
        for q in QUANTITIES:
            hd[q], hd[q + '_err'] = dm[q + '_obs'].values, dm[q + '_err'].values
        cmp_daily = {q: {name: obs.compare_rv_exposures(hd, dm['%s_%s' % (q, name)].values, quantity=q,
                                                        interventions=cuts) for name in MODELS} for q in QUANTITIES}
        rows = []
        for q in QUANTITIES:
            for name in MODELS:
                c = cmp_daily[q][name]
                r, k, k_err = fit_line(c['sim'], c['obs'])
                rows.append({'quantity': q, 'model': name, 'days': c['n'], 'RMS obs': c['rms_obs'],
                             'MAD obs': c['mad_obs'], 'RMS obs - sim': c['rms'], 'MAD obs - sim': c['mad'], 'r': r,
                             'slope': k, 'slope_err': k_err, 'sim std': np.std(c['sim'])})
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
        for name in MODELS:
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
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 5), squeeze=False)
    for ax, name in zip(axes[0], MODELS):
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

    d = sub.add_parser('day', help='one day: HMI epochs and corrected 45 s dopplergrams of its exposures')
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
    d.add_argument('--no-network-velocity', dest='network_velocity', action='store_false',
                   help='quiet pixels: not active at all (default: neither spot nor facula)')
    d.add_argument('--margin-px', type=int, nargs='*', default=[],
                   help='the frames also with the quiet pixels farther than these numbers of pixels (of --size) from the '
                        'spots and faculae, in <out>_margin<m>px (same downloads)')
    d.add_argument('--move-masks', action='store_true',
                   help='the frames also with the mask of the epoch moved to their time by the differential rotation, '
                        'in <out>_moved (same downloads)')
    d.add_argument('--max-dt', type=float, default=900.0, help='[s] exposures farther from an HMI epoch are not used')
    d.add_argument('--window', type=float, default=720.0,
                   help='[s] the 45 s dopplergrams within window / 2 of the middle of every exposure are averaged')
    d.add_argument('--mu-min', type=float, default=0.1, help='only the pixels with mu > this are projected')
    d.add_argument('--workers', type=int, default=1, help='downloads at the same time')
    d.add_argument('--tmp', default=None, help='folder of the FITS files (deleted after every epoch / frame)')
    d.add_argument('--limit', type=int, default=0, help='test: at most this many epochs')
    d.add_argument('--dry-run', action='store_true', help='only the matching and the size of the downloads')

    c = sub.add_parser('collect', help='all the days: averages, simulations, comparison with HARPS-N')
    c.add_argument('--out', required=True)
    c.add_argument('--start', required=True, help='origin of the times (YYYY-MM-DD)')
    c.add_argument('--conf', required=True, help='configuration of the simulation (n_grid_rings replaced)')
    c.add_argument('--mps-folder', default=None, help='MURaM spectra (replaces the path of the configuration)')
    c.add_argument('--mask', default=None, help='CCF mask (replaces the path of the configuration)')
    c.add_argument('--interventions', default=None, help='HARPS-N interventions (one offset per instrumental period)')
    c.add_argument('--n-rings', type=int, default=30)
    c.add_argument('--window', type=float, default=None,
                   help='[s] averaging window (default: the window of the day step; at most that one)')
    c.add_argument('--min-fill', type=float, default=0.8,
                   help='exposures with fewer frames than this fraction of window / 45 s are left out')
    c.add_argument('--remove-disc-mean', action='store_true',
                   help='the disc mean of the quiet cells (intensity weighted) removed from every exposure')
    c.add_argument('--rotation-obs-vr', action='store_true',
                   help='a1 + a4 OBS_VR in the rotation of every day (the east-west gradient of HMI with the orbit)')
    c.add_argument('--clv-deg', type=int, default=-1,
                   help='CLV of the quiet cells of every exposure (shifted Legendre polynomials of mu up to this degree; '
                        '0: a constant) minus its average removed (default -1: not)')
    c.add_argument('--clv-drift', default='obs_vr', choices=('obs_vr', 'exposure'),
                   help='with --clv-deg: CLV coefficients fitted with OBS_VR + one offset per day (obs_vr) or as fitted '
                        'on every exposure (exposure)')
    c.add_argument('--drift-deg', type=int, default=2, help='degree of the polynomial in OBS_VR (--clv-drift obs_vr)')
    c.add_argument('--drift-harmonics', type=int, default=0,
                   help='--clv-drift obs_vr: also the harmonics 1..N of the orbit of SDO (24 h, 12 h, ...; default 0)')
    c.add_argument('--clv-remove-average', action='store_true',
                   help='with --clv-deg: the average CLV removed too (already in the spectra of the grids)')
    c.add_argument('--north-south', action='store_true',
                   help='a north-south gradient ns + ns_vr OBS_VR fitted with the rotation of every day and removed')
    c.add_argument('--diff-extinction', action='store_true',
                   help='HARPS-N RV also minus rv_diff_extinction of the release (differential extinction)')
    c.add_argument('--min-exp', type=int, default=5, help='days with fewer exposures: not in the statistics')
    c.add_argument('--harpsn', default=None, help='csv of the release: its quality flags (data_quality = 1 kept)')
    c.add_argument('--min-obs-quality', type=float, default=None,
                   help='with --harpsn: only the exposures with obs_quality >= this (e.g. 0.99)')
    c.add_argument('--min-day-exp', type=int, default=0,
                   help='with --harpsn: then also the days with fewer exposures left than this (e.g. 50)')
    c.add_argument('--results', default='results', help='folder of the results, in --out (default results)')
    c.add_argument('--no-figures', dest='figures', action='store_false', help='only the tables')
    c.add_argument('--save-cells', action='store_true',
                   help='also the velocity of every cell of every exposure given to the simulations (cells.npz)')

    a = p.parse_args(argv)
    if a.step == 'day':
        return run_day(a)
    import matplotlib
    matplotlib.use('Agg')                            # no display on PIC (the notebooks keep their backend)
    return run_collect(a)


if __name__ == '__main__':
    sys.exit(main())
