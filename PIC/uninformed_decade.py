#!/usr/bin/env python
"""
The model of uninformed_masks_model.ipynb (starsim codes+tests) trained on the decade of HARPS-N (2015-07-18 to
2024-01-29), with the generator uninformed_masks.py (a copy of the one of codes+tests). On PIC: uninformed_decade.dag
(maps-local, train, synth); on a laptop: maps (download), train, synth.

  python uninformed_decade.py maps-local --data SDO_DATA --out DIR --block K --n-blocks N [--workers 2]
      As maps, from the HMI images already in SDO_DATA (sunsim.sdo._hmi_index: the continuum and the hmi.M_45s
      magnetogram of every day), the K-th of N blocks of days, every day in its own process (~4 GB each).

  python uninformed_decade.py maps  --out DIR [--start 2015-07-18 --end 2024-01-29] [--workers 2]
      Every day: the hmi.Ic_noLimbDark_720s continuum closest to 12:00 TAI and the hmi.M_45s magnetogram closest to it
      are downloaded from JSOC (sunsim.supergranulation.fetch_epoch: no email), identified at SIZE px with the
      thresholds THRESHOLDS (sunsim.sdo.identify_day), the masks written in DIR/maps as sunsim.sdo writes them
      (spot_map_{t}.npz, faculae_map_{t}.npz, bit-packed; t = days since --start; days/<date>.json, summary.csv) and
      the FITS files deleted after every epoch (at most `workers` epochs downloaded in advance). A day already done is
      skipped, so the run can be started again.
  python uninformed_decade.py train --out DIR [--maps DIR/maps] [--decay-sat A_sat] [--workers 8]
      Features of every day, regions tracked over the decade, statistics; the parameters of the generator
      (uninformed_masks.py) fitted: sizes, decay (below A_sat MSH only, constant decay rate above it, if given),
      nesting, spots of a region on the whole decade; emergence rate and
      butterfly latitudes of cycles 24 and 25; pores, plage-only regions and background plage scaled with the activity
      of the cycles; then the closed loop on the spots and the calibration of the plage on windows spread over the
      decade. Written in DIR/model: model.json (the parameters, uninformed_masks.params(json.load(...)['P'])),
      statistics.pkl, bins.csv and train.log.
  python uninformed_decade.py synth --out DIR [--min 2019-11-01] [--max auto] [--days 91] [--n-real 3]
      Synthetic masks of 3 months at the solar minimum and maximum (the 91 days of highest mean spot filling factor of
      the decade if --max auto) at the dates and sizes of the SDO masks, n-real realizations, written in
      DIR/synthetic/<min|max>_<k>/ (as sunsim.sdo writes them), and the statistics of SDO and of the synthetic masks
      through the same extraction in DIR/synthetic/statistics.pkl.
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import scipy.ndimage as nd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

import uninformed_masks as um

START, END = '2015-07-18', '2024-01-29'
SIZE = 1024
THRESHOLDS = dict(noise_th=8.0, I_th=0.8, A_th=10.0, faculae='plage')


def _log(msg, path=None):
    line = '%s  %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    if path:
        with open(path, 'a') as f:
            f.write(line + '\n')


# =============================================================================================
# maps: download, identification, FITS deleted
# =============================================================================================
def _epoch(date, client):
    """{'continuum': T_REC, 'magnetogram': T_REC} of a day: the continuum closest to 12:00 TAI (11:00 to 13:00) and the
    45 s magnetogram closest to it (within 5 min); None if there is none (good quality)."""
    from sunsim import supergranulation as sg
    from astropy.time import Time
    noon = Time(date + 'T12:00:00', scale='tai').utc.jd
    rc = sg.query_records(sg.SERIES['continuum'][0], noon - 1 / 24, noon + 1 / 24, client=client)
    k, dt = sg.closest(noon, rc['jd'].values)
    if dt[0] > 3600:
        return None
    jd_c = rc['jd'].values[k[0]]
    rm = sg.query_records(sg.SERIES['magnetogram'][0], jd_c - 5 / 1440, jd_c + 5 / 1440, client=client)
    km, dtm = sg.closest(jd_c, rm['jd'].values)
    if dtm[0] > 300:
        return None
    return {'continuum': rc['T_REC'].values[k[0]], 'magnetogram': rm['T_REC'].values[km[0]]}


def _write_summary(folder):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(folder, 'days', '*.json')))]
    rows.sort(key=lambda r: r['t'])
    with open(os.path.join(folder, 'summary.csv'), 'w') as f:
        f.write('t,date,ff_sp,ff_fc,ff_nt,n_spots,n_plages\n')
        for r in rows:
            f.write('%g,%s,%.6f,%.6f,%.6f,%d,%d\n' % (r['t'], r['date'], r['ff_sp'], r['ff_fc'], r['ff_nt'],
                                                     r['n_spots'], r['n_plages']))
    return len(rows)


def _maps_folder(a, source):
    """DIR/maps with its parameters.txt, and the days not done yet from a.start to a.end."""
    folder = os.path.join(a.out, 'maps')
    os.makedirs(os.path.join(folder, 'days'), exist_ok=True)
    with open(os.path.join(folder, 'parameters.txt'), 'w') as f:
        f.write('start = %s\nend = %s\nsize = %d\nmagnetogram = 45s\nimages = %s\n' % (a.start, a.end, SIZE, source))
        for k, v in THRESHOLDS.items():
            f.write('%s = %s\n' % (k, v))
    done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, 'days', '*.json'))}
    dates = []
    day = datetime.strptime(a.start, '%Y-%m-%d')
    while day < datetime.strptime(a.end, '%Y-%m-%d'):
        if day.strftime('%Y-%m-%d') not in done:
            dates.append(day.strftime('%Y-%m-%d'))
        day += timedelta(days=1)
    return folder, dates


def _save_day(folder, date, start, day, epoch):
    """The masks of an identified day (sunsim.sdo.identify_day) and its row (days/<date>.json)."""
    t = float((datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days)
    for kind, mask in (('spot', day['spot']), ('faculae', day['facula'])):
        np.savez_compressed(os.path.join(folder, '%s_map_%.1f.npz' % (kind, t)),
                            packed=np.packbits(mask), shape=np.array(mask.shape))
    row = {'t': t, 'date': date, 'ff_sp': float(day['ff_sp']), 'ff_fc': float(day['ff_fc']),
           'ff_nt': float(day['ff_nt']), 'n_spots': int(day['n_spots']), 'n_plages': int(day['n_plages']),
           'continuum': epoch['continuum'], 'magnetogram': epoch['magnetogram']}
    with open(os.path.join(folder, 'days', date + '.json'), 'w') as f:
        json.dump(row, f)
    return row


def run_maps(a):
    from sunsim import sdo
    from sunsim import supergranulation as sg
    folder, dates = _maps_folder(a, 'JSOC: hmi.Ic_noLimbDark_720s closest to 12:00 TAI, hmi.M_45s closest to it')
    log = os.path.join(folder, 'maps.log')
    _log('%d days to do' % len(dates), log)
    tmp = tempfile.mkdtemp(prefix='hmi_', dir=a.tmp)

    def fetch(date):
        epoch = _epoch(date, sg._client())
        if epoch is None:
            raise FileNotFoundError('no good continuum near 12:00 TAI or no magnetogram next to it')
        d = os.path.join(tmp, date)
        os.makedirs(d, exist_ok=True)
        return epoch, sg.fetch_epoch(epoch, d)

    n_ok = 0
    t_run = time.time()
    try:
        for k, (date, res) in enumerate(sg.prefetch(dates, fetch, workers=a.workers, ahead=a.workers), 1):
            try:
                if isinstance(res, Exception):
                    raise res
                epoch, files = res
                cont, mag = sg.read_record(*files['continuum']), sg.read_record(*files['magnetogram'])
                day = sdo.identify_day(cont, mag, size=SIZE, **THRESHOLDS)
                del cont, mag
                row = _save_day(folder, date, a.start, day, epoch)
                n_ok += 1
                rate = (time.time() - t_run) / k
                _log('[%d/%d] %s ff_sp %.3f%% ff_fc %.2f%% (%.0f s / day, %.1f h left)'
                     % (k, len(dates), date, row['ff_sp'], row['ff_fc'], rate, rate * (len(dates) - k) / 3600), log)
            except Exception as e:      # noqa: BLE001 (a day that fails is skipped, the run goes on)
                _log('%s: skipped (%s)' % (date, str(e).splitlines()[0][:150] if str(e) else type(e).__name__), log)
            finally:
                shutil.rmtree(os.path.join(tmp, date), ignore_errors=True)       # the FITS of this epoch
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        n = _write_summary(folder)
        _log('%d days done in this run, %d in %s' % (n_ok, n, folder), log)
    return 0


def _maps_local_day(task):
    """One day of maps-local (in a worker process): (date, row) or (date, reason it was skipped)."""
    from sunsim import sdo
    folder, date, start, files = task
    try:
        cont, mag = sdo._read_with_retries(None, None, files=files)
        day = sdo.identify_day(cont, mag, size=SIZE, **THRESHOLDS)
        del cont, mag
        return date, _save_day(folder, date, start, day, {'continuum': os.path.basename(files[0]),
                                                          'magnetogram': os.path.basename(files[1])})
    except Exception as e:      # noqa: BLE001 (a day that fails is skipped)
        return date, str(e).splitlines()[0][:150] if str(e) else type(e).__name__


def run_maps_local(a):
    from sunsim import sdo
    folder, dates = _maps_folder(a, 'local: %s' % a.data)
    log = os.path.join(folder, 'maps_block_%d.log' % a.block)
    index = sdo._hmi_index(a.data, '45s')
    mine = [dates[k] for k in np.array_split(np.arange(len(dates)), a.n_blocks)[a.block]] if dates else []
    tasks = [(folder, d, a.start, index[d.replace('-', '')]) for d in mine if d.replace('-', '') in index]
    _log('block %d / %d: %d days to do, %d of them without images' % (a.block, a.n_blocks, len(mine),
                                                                     len(mine) - len(tasks)), log)
    t_run = time.time()
    with _pool(a.workers) as pool:
        for k, (date, res) in enumerate(pool.imap_unordered(_maps_local_day, tasks), 1):
            if isinstance(res, dict):
                _log('[%d/%d] %s ff_sp %.3f%% ff_fc %.2f%% (%.0f s / day)' % (k, len(tasks), date, res['ff_sp'],
                                                                             res['ff_fc'], (time.time() - t_run) / k), log)
            else:
                _log('%s: skipped (%s)' % (date, res), log)
    return 0


# =============================================================================================
# extraction (uninformed_masks_model.ipynb, sections 1-5)
# =============================================================================================
DEG = um.DEG
NEST_DEG = 10.0     # nesting: a new region (or a pore) closer than this to an existing region
NEST_DAYS = 27.0    # ... seen in the last NEST_DAYS days
LINK_DEG = 5.0      # spots of the same region are closer than this [deg]
MU_MIN = 0.3        # areas and contrasts only where mu > MU_MIN
A_DET = 3.0         # regions smaller than this [MSH] (a few pixels) are counted as pores, not tracked
EIGHT = np.ones((3, 3), dtype=int)
PATCH_BINS = (1, 3, 10, 30, 100, 300, 1e5)          # plage patch areas [MSH]
SMALL_MSH = 30.0                                    # plage patches smaller than this: the small-scale plage
LAT_BINS = np.array([0, 10, 20, 30, 40, 60, 90])    # |lat| [deg] of the small-scale plage (mu > 0.7)
MU_BINS = np.array([0.3, 0.5, 0.7, 0.85, 1.0])      # mu of the small-scale plage (|lat| < 35 deg)
GATE_DEG = 4.0
MAX_GAP = 2.5
VISIBLE = 120 / 360
PROF_W = 20.0       # weight of the plage profile around regions in the cost of the faculae
um.A_TH = THRESHOLDS['A_th']                        # plage / network cut of the synthetic masks = that of the SDO masks


def ang_dist(v1, v2):
    """Angle [deg] between unit vectors (broadcast on the last axis)."""
    return np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1, 1)) / DEG


def lon_rel(lon, L0):
    """Longitude from the central meridian [deg], -180..180 (negative = east)."""
    return (np.asarray(lon) - L0 + 180) % 360 - 180


def b0_l0(date):
    from astropy.time import Time
    import sunpy.coordinates.sun as sun
    t = Time(date)
    return sun.B0(t).deg, sun.L0(t).deg


def drift_carrington(lat, dt):
    return um.drift(lat, dt, um.SUN)


def read_days(folder, start=START):
    """Days of a folder of sunsim.sdo masks (npz or npy), from its summary.csv: dicts with date (12:00), mjd, t [days
    since start] and the paths of the masks (not read: load_masks)."""
    from astropy.time import Time
    t_start = datetime.strptime(start, '%Y-%m-%d')
    rows = np.genfromtxt(os.path.join(folder, 'summary.csv'), delimiter=',', names=True, dtype=None, encoding=None)
    days = []
    for r in np.atleast_1d(rows):
        date = '%sT12:00:00' % r['date']
        paths = []
        for kind in ('spot', 'faculae'):
            base = os.path.join(folder, '%s_map_%.1f' % (kind, float(r['t'])))
            paths.append(base + '.npz' if os.path.exists(base + '.npz') else base + '.npy')
        days.append(dict(date=date, mjd=Time(date).mjd, t=float((datetime.strptime(str(r['date']), '%Y-%m-%d')
                                                                  - t_start).days), paths=paths))
    return days


def load_mask(path):
    if path.endswith('.npz'):
        d = np.load(path)
        shape = tuple(d['shape'])
        return np.unpackbits(d['packed'], count=shape[0] * shape[1]).reshape(shape).astype(bool)
    return np.load(path).astype(bool)


def mask_size(path):
    if path.endswith('.npz'):
        return int(np.load(path)['shape'][0])
    return int(np.load(path, mmap_mode='r').shape[0])


def load_masks(day):
    return dict(day, spot=load_mask(day['paths'][0]), facula=load_mask(day['paths'][1]))


def day_features(day):
    n = day['spot'].shape[0]
    B0, L0 = b0_l0(day['date'])
    g = um.disc_geometry(n, B0, L0)
    good = g['inside'] & (g['mu'] > MU_MIN)
    msh = np.where(good, g['msh'], 0.0)
    sp = day['spot'] & good
    fc = day['facula'] & good & ~day['spot']

    # spots
    lab, ns = nd.label(sp, structure=EIGHT)
    spots = []
    if ns:
        idx = np.arange(1, ns + 1)
        area = nd.sum(msh, lab, idx)
        vec = np.stack([nd.sum(msh * g['vec'][..., k], lab, idx) for k in range(3)], axis=-1)
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        lat = np.arcsin(vec[:, 2]) / DEG
        lon = np.arctan2(vec[:, 1], vec[:, 0]) / DEG % 360
        mu = nd.mean(np.nan_to_num(g['mu']), lab, idx)
        spots = [dict(area=a, lat=b, lon=c, mu=m, vec=v) for a, b, c, m, v in zip(area, lat, lon, mu, vec)]

    # regions = single-linkage clusters of the spots
    regions = []
    if len(spots) == 1:
        clusters = np.array([1])
    elif len(spots) > 1:
        V = np.array([s['vec'] for s in spots])
        chord = 2 * np.sin(LINK_DEG * DEG / 2)
        clusters = fcluster(linkage(V, method='single', metric='euclidean'), chord, criterion='distance')
    pores = []
    for k in (np.unique(clusters) if spots else []):
        mem = [s for s, c in zip(spots, clusters) if c == k]
        a = np.array([s['area'] for s in mem])
        v = np.sum([s['area'] * s['vec'] for s in mem], axis=0)
        v /= np.linalg.norm(v)
        lat, lon = np.arcsin(v[2]) / DEG, np.arctan2(v[1], v[0]) / DEG % 360
        if a.sum() < A_DET:
            pores.append(dict(area=a.sum(), vec=v, lat=lat))
            continue
        regions.append(dict(area=a.sum(), n_spots=len(mem), leader=a.max() / a.sum(), lat=lat, lon=lon, vec=v,
                            lonrel=lon_rel(lon, L0), mu=np.mean([s['mu'] for s in mem]), spot_areas=a))

    # plage patches; spotless = no spot pixel touches the patch (spots included in the patch for the test)
    labp, npl = nd.label(fc | sp, structure=EIGHT)
    spotless = []
    patch_area = np.zeros(0)
    small = np.zeros_like(fc)
    if npl:
        idx = np.arange(1, npl + 1)
        has_spot = nd.maximum(sp, labp, idx) > 0
        pa = nd.sum(np.where(fc, msh, 0), labp, idx)
        spotless = pa[~has_spot]
        patch_area = pa[pa > 0]
        small = fc & np.concatenate(([False], pa < SMALL_MSH))[labp]

    # small-scale plage against |lat| (mu > 0.7) and against mu (|lat| < 35 deg): sums of plage and of surface
    w = np.where(good & ~sp, msh, 0.0)
    alat, mu = np.abs(np.nan_to_num(g['lat'])), np.nan_to_num(g['mu'])
    cen, low = (w > 0) & (mu > 0.7), (w > 0) & (alat < 35)
    small_lat = (np.histogram(alat[cen], LAT_BINS, weights=(w * small)[cen])[0],
                 np.histogram(alat[cen], LAT_BINS, weights=w[cen])[0])
    small_mu = (np.histogram(mu[low], MU_BINS, weights=(w * small)[low])[0],
                np.histogram(mu[low], MU_BINS, weights=w[low])[0])

    surface = msh.sum()                                    # deprojected area of the disc with mu > MU_MIN [MSH]
    rings, profile = facula_profile(g, sp, fc, regions)
    # pores next to a region (closer than NEST_DEG)
    near = [min([ang_dist(p['vec'], r['vec']) for r in regions] + [180.0]) < NEST_DEG for p in pores]
    return dict(t=day['t'], date=day['date'], B0=B0, L0=L0, regions=regions, pores=pores,
                pores_near=np.mean(near) if near else np.nan,
                ff_sp=(msh * sp).sum() / surface, ff_fc=(msh * fc).sum() / surface, surface=surface,
                spotless_plage=np.asarray(spotless), rings=rings, profile=profile, patch_area=patch_area,
                ff_small=(msh * small).sum() / surface, small_lat=small_lat, small_mu=small_mu)


def facula_profile(g, sp, fc, regions, rings=np.arange(0, 21, 1.0)):
    """Mean plage fraction (deprojected) in rings of angular distance [deg] around every region of the day."""
    good = g['inside'] & (g['mu'] > MU_MIN) & ~sp
    V = g['vec'][good]
    w = g['msh'][good]
    f = fc[good]
    out = []
    for r in regions:
        d = ang_dist(V, r['vec'][None, :].astype(np.float32))
        k = np.digitize(d, rings) - 1
        ok = (k >= 0) & (k < len(rings) - 1)
        num = np.bincount(k[ok], weights=(w * f)[ok], minlength=len(rings) - 1)
        den = np.bincount(k[ok], weights=w[ok], minlength=len(rings) - 1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out.append((r['area'], num / den))
    return rings, out


def _features_of_day(day):
    return day_features(load_masks(day))


def track_regions(feats):
    tracks = []
    active = []
    for f in feats:
        regs = f['regions']
        active = [k for k in active if f['t'] - tracks[k]['t'][-1] <= MAX_GAP]
        cost = np.full((len(active), len(regs)), 1e9)
        for a, k in enumerate(active):
            tr = tracks[k]
            dt = f['t'] - tr['t'][-1]
            pv = um.unit_vector(tr['lat'][-1], tr['lon'][-1] + drift_carrington(tr['lat'][-1], dt))
            for b, r in enumerate(regs):
                d = ang_dist(pv, r['vec'])
                if d < GATE_DEG + 1.0 * dt:
                    cost[a, b] = d
        used = set()
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < 1e9:
                    tr = tracks[active[a]]
                    r = regs[b]
                    for key, val in (('t', f['t']), ('lat', r['lat']), ('lon', r['lon']), ('area', r['area']),
                                     ('lonrel', r['lonrel']), ('n_spots', r['n_spots']), ('leader', r['leader'])):
                        tr[key].append(val)
                    used.add(b)
        for b, r in enumerate(regs):
            if b not in used:
                tracks.append(dict(t=[f['t']], lat=[r['lat']], lon=[r['lon']], area=[r['area']], lonrel=[r['lonrel']],
                                   n_spots=[r['n_spots']], leader=[r['leader']]))
                active.append(len(tracks) - 1)
    for tr in tracks:
        for key in list(tr):
            tr[key] = np.array(tr[key])
    return tracks


def plage_summary(feats):
    """Plage patches per day in PATCH_BINS, small-scale plage filling factor and its profiles in |lat| and mu [%]."""
    out = dict(n_patches=np.mean([np.histogram(f['patch_area'], PATCH_BINS)[0] for f in feats], axis=0),
               ff_small=np.array([f['ff_small'] for f in feats]))
    for key in ('small_lat', 'small_mu'):
        num = np.sum([f[key][0] for f in feats], axis=0)
        den = np.sum([f[key][1] for f in feats], axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[key] = 100 * num / den
    return out


def near_fractions(births, tracks, rng, n_random=20):
    """Fraction of the births closer than NEST_DEG to a region of the last NEST_DAYS days (moved with the differential
    rotation), and the mean of the same fraction with the birth longitudes drawn at random n_random times (the
    near_fraction of the notebook, with the regions of the last NEST_DAYS found from the time span of every track)."""
    if not births:
        return np.nan, np.nan
    first = np.array([o['t'][0] for o in tracks])
    last = np.array([o['t'][-1] for o in tracks])
    obs, rand = [], []
    for tr in births:
        tb, lat_b, lon_b = tr['t'][0], tr['lat'][0], tr['lon'][0]
        vb = um.unit_vector(np.full(n_random + 1, lat_b), np.r_[lon_b, rng.uniform(0, 360, n_random)])
        vo = []
        for j in np.nonzero((first < tb) & (last >= tb - NEST_DAYS))[0]:
            o = tracks[j]
            if o is tr:
                continue
            sel = np.nonzero((o['t'] < tb) & (o['t'] >= tb - NEST_DAYS))[0]
            if len(sel):
                i = sel[-1]
                vo.append(um.unit_vector(o['lat'][i], o['lon'][i] + drift_carrington(o['lat'][i], tb - o['t'][i])))
        if vo:
            hit = (ang_dist(vb[:, None, :], np.array(vo)[None, :, :]).min(axis=1) < NEST_DEG)
        else:
            hit = np.zeros(n_random + 1, bool)
        obs.append(hit[0])
        rand.append(hit[1:].mean())
    return float(np.mean(obs)), float(np.mean(rand))


def statistics(feats, tracks, seed=1):
    """The statistics of the notebook (section 5)."""
    rng = np.random.default_rng(seed)
    t0, t1 = feats[0]['t'], feats[-1]['t']
    S = {}
    S['duration'] = t1 - t0
    S['ff_sp'] = np.array([f['ff_sp'] for f in feats])
    S['ff_fc'] = np.array([f['ff_fc'] for f in feats])
    S['t'] = np.array([f['t'] for f in feats])
    S['n_regions'] = np.array([len(f['regions']) for f in feats])
    S['n_pores'] = np.array([len(f['pores']) for f in feats])
    S['pores_near'] = np.nanmean([f['pores_near'] for f in feats]) if any(f['pores'] for f in feats) else np.nan

    births = [tr for tr in tracks if tr['t'][0] > t0 + 0.5 and tr['lonrel'][0] > -60]
    S['birth_lat'] = np.array([tr['lat'][0] for tr in births])
    S['birth_t'] = np.array([tr['t'][0] for tr in births])
    S['rate'] = len(births) / max(S['duration'], 1.0) / VISIBLE                  # regions / day over the whole Sun
    born = {id(tr) for tr in births}

    peaks, decays, growth, lifetimes = [], [], [], []
    for tr in tracks:
        k = np.argmax(tr['area'])
        if abs(tr['lonrel'][k]) < 60:
            peaks.append(tr['area'][k])
        # decay: points after the peak far from the limb (near mu = MU_MIN the cut of the disc mimics a decay)
        after = (np.arange(len(tr['t'])) >= k) & (np.abs(tr['lonrel']) < 60)
        if after.sum() >= 3 and abs(tr['lonrel'][k]) < 60:
            p = np.polyfit(tr['t'][after], tr['area'][after], 1)
            decays.append((tr['area'][k], -p[0]))
        if id(tr) in born and k > 0:
            growth.append(tr['t'][k] - tr['t'][0])
        if id(tr) in born and tr['t'][-1] < t1 - 0.5 and tr['lonrel'][-1] < 60:
            lifetimes.append((tr['area'].max(), tr['t'][-1] - tr['t'][0] + 1))
    S['peak_area'] = np.array(peaks)
    S['decay'] = np.array(decays).reshape(-1, 2)
    S['growth_days'] = np.array(growth)
    S['lifetime'] = np.array(lifetimes).reshape(-1, 2)

    # spots per region
    S['region_area'] = np.array([r['area'] for f in feats for r in f['regions']])
    S['region_nspots'] = np.array([r['n_spots'] for f in feats for r in f['regions']])
    S['region_leader'] = np.array([r['leader'] for f in feats for r in f['regions']])
    S['spotless_plage'] = np.concatenate([np.asarray(f['spotless_plage'], float) for f in feats])
    S.update(plage_summary(feats))

    late = [tr for tr in births if tr['t'][0] > t0 + NEST_DAYS]
    S['nest_obs'], S['nest_rand'] = near_fractions(late, tracks, rng)
    S['p_nest'] = (S['nest_obs'] - S['nest_rand']) / (1 - S['nest_rand'])
    return S


def spotless_share(feats):
    tot = sum(np.sum(f['spotless_plage']) for f in feats)
    allp = sum(f['ff_fc'] * f['surface'] for f in feats)
    return tot / allp


def region_profiles(feats, n_rings=10):
    """Mean plage fraction in the first n_rings rings around the regions of 3-100 MSH and of > 100 MSH."""
    out = []
    for lo, hi in ((A_DET, 100), (100, 1e5)):
        prof = [p for f in feats for a, p in f['profile'] if lo <= a < hi]
        out.append(np.nanmean(prof, axis=0)[:n_rings] if prof else np.full(n_rings, np.nan))
    return np.array(out)


# =============================================================================================
# train: statistics of the decade, fit of the generator
# =============================================================================================
def _pool(workers):
    import multiprocessing as mp
    return mp.get_context('spawn').Pool(workers)


def sdo_features(days, cache, workers, log):
    """Features of every day (cached in `cache`: only the new days are done)."""
    feats = {}
    if os.path.exists(cache):
        feats = pickle.load(open(cache, 'rb'))
    todo = [d for d in days if d['date'] not in feats]
    _log('features: %d days to do, %d cached' % (len(todo), len(feats)), log)
    if todo:
        t_run = time.time()
        with _pool(workers) as pool:
            for k, f in enumerate(pool.imap(_features_of_day, todo, chunksize=4), 1):
                feats[f['date']] = f
                if k % 200 == 0:
                    _log('  %d / %d (%.1f s / day)' % (k, len(todo), (time.time() - t_run) / k), log)
        pickle.dump(feats, open(cache, 'wb'))
    return [feats[d['date']] for d in days]


def _cycle_terms(years, P):
    """Vectorised uninformed_masks._cycle_terms: rates relative to rate_max (2 x N: the current and the previous
    cycle) and the time since the start of each cycle [years] (2 x N)."""
    per = P['cycle_period']
    b = P['cycle_width'] * per / 11.0
    tt = np.linspace(0.01, 2 * per, 2000)
    peak = np.max(um._shape(tt, b))
    years = np.asarray(years, dtype=float)
    k = np.floor((years - P['cycle_start']) / per)
    x = np.array([years - (P['cycle_start'] + kk * per) for kk in (k, k - 1)])
    return um._shape(x, b) / peak, x


def bins_table(feats, tracks, year0, width=91.0):
    """Statistics in bins of `width` days: days with a map, births (tracks born on the disc, the day before them
    observed: no false births after a gap), births / day over the Sun, mean |lat| of the births, regions, pores and
    the filling factors per day."""
    t = np.array([f['t'] for f in feats])
    prev_gap = np.r_[np.inf, np.diff(t)]
    ok_day = dict(zip(t, prev_gap <= 1.5))
    births = [tr for tr in tracks if ok_day.get(tr['t'][0], False) and tr['lonrel'][0] > -60]
    bt = np.array([tr['t'][0] for tr in births])
    bl = np.array([tr['lat'][0] for tr in births])
    edges = np.arange(t[0], t[-1] + width, width)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        n_eff = int(np.sum(sel & (prev_gap <= 1.5)))
        if sel.sum() < width / 3 or n_eff == 0:
            continue
        fb = (bt >= a) & (bt < b)
        fs = [f for f, s in zip(feats, sel) if s]
        rows.append(dict(t_mid=float(t[sel].mean()), year=year0 + float(t[sel].mean()) / 365.25, n_days=int(sel.sum()),
                         n_eff=n_eff, births=int(fb.sum()), rate=fb.sum() / n_eff / VISIBLE,
                         abs_lat=float(np.mean(np.abs(bl[fb]))) if fb.any() else np.nan,
                         n_regions=float(np.mean([len(f['regions']) for f in fs])),
                         n_pores=float(np.mean([len(f['pores']) for f in fs])),
                         ff_sp=float(np.mean([f['ff_sp'] for f in fs])), ff_fc=float(np.mean([f['ff_fc'] for f in fs])),
                         ff_small=float(np.mean([f['ff_small'] for f in fs]))))
    return rows, np.array([year0 + x / 365.25 for x in bt]), np.abs(bl)


def fit_parameters(S, P, A_sat=None):
    """fit_parameters of the notebook for what does not follow the cycle: sizes, growth, decay, nesting, spots of a
    region, pores next to regions. With A_sat [MSH] the decay law D0 (A / 100)^p is fitted on the regions below A_sat
    only, and D is constant above it (the decay of the large regions, seen over a few days, is too short to fit)."""
    P = dict(P, A_sat=A_sat)
    lp = np.log(S['peak_area'])
    P['logA_mean'], P['logA_sigma'] = float(lp.mean()), float(lp.std())
    P['A_min'] = float(S['peak_area'].min())
    if len(S['growth_days']):
        P['growth_days'] = float(max(np.median(S['growth_days']), 0.5))
    A, D = S['decay'][:, 0], S['decay'][:, 1]
    ok = (D > 0) & (A > 0)
    if A_sat is not None:
        ok &= A < A_sat
    if ok.sum() >= 3:
        p, lnD0 = np.polyfit(np.log(A[ok] / 100), np.log(D[ok]), 1)
        P['decay_p'], P['D0'] = float(np.clip(p, 0.0, 1.0)), float(np.exp(lnD0))
    P['p_nest'] = float(np.clip(S['p_nest'], 0, 1)) if np.isfinite(S['p_nest']) else P['p_nest']
    P['pore_near'] = float(S['pores_near']) if np.isfinite(S['pores_near']) else P['pore_near']
    m = S['region_nspots'] > 1
    P['leader_frac'] = float(np.median(S['region_leader'][m])) if m.any() else P['leader_frac']
    A_r, N_r = S['region_area'], S['region_nspots']
    if len(A_r) > 5:
        P['follower_scale'] = float(np.sum(A_r) / max(np.sum(N_r - 1), 1))
    return P


def fit_cycles(rows, birth_year, birth_abslat, P):
    """Emergence rate (rate_max, cycle_start, cycle_width; Poisson on the births of the bins), butterfly latitudes
    (lat_start, lat_tau, lat_sigma; folded gaussians of the two cycles weighted by their rates, on every birth) and the
    pores against the activity (pore_rate_min, pore_rate: straight line through the pores per day of the bins)."""
    from scipy.optimize import minimize
    P = dict(P, rate=None, lat_mean=None)
    yr = np.array([r['year'] for r in rows])
    nb = np.array([r['births'] for r in rows])
    ne = np.array([r['n_eff'] for r in rows])

    def nll_rate(x):
        Q = dict(P, cycle_start=x[1], cycle_width=np.exp(x[2]))
        mu = np.exp(x[0]) * _cycle_terms(yr, Q)[0].sum(axis=0) * ne * VISIBLE + 1e-9
        return np.sum(mu - nb * np.log(mu))

    # bounds: cycle 24 started at 2008.96 (cycle 25 at 2019.96 with cycle_period = 11); widths and drifts of the Sun
    best = min((minimize(nll_rate, [np.log(r0), s0, np.log(4.0)], method='Nelder-Mead',
                         bounds=[(np.log(0.05), np.log(100.0)), (2006.0, 2012.0), (np.log(2.0), np.log(8.0))],
                         options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
                for r0 in (1.0, 3.0, 10.0) for s0 in (2008.0, 2009.0, 2010.0)), key=lambda r: r.fun)
    P['rate_max'], P['cycle_start'], P['cycle_width'] = float(np.exp(best.x[0])), float(best.x[1]), float(np.exp(best.x[2]))

    r, x = _cycle_terms(birth_year, P)
    w = r / np.maximum(r.sum(axis=0), 1e-12)

    def nll_lat(y):
        m = y[0] * np.exp(-x / np.exp(y[1]))
        s = np.exp(y[2])
        g = (np.exp(-0.5 * ((birth_abslat - m) / s)**2) + np.exp(-0.5 * ((birth_abslat + m) / s)**2)) / s
        return -np.sum(np.log(np.sum(w * g, axis=0) + 1e-300))

    lat = minimize(nll_lat, [28.0, np.log(7.5), np.log(5.0)], method='Nelder-Mead', options=dict(maxiter=4000),
                   bounds=[(10.0, 45.0), (np.log(2.0), np.log(30.0)), (np.log(1.0), np.log(15.0))])
    P['lat_start'], P['lat_tau'], P['lat_sigma'] = float(lat.x[0]), float(np.exp(lat.x[1])), float(np.exp(lat.x[2]))

    # pores live about a day: the number on the disc (mu > MU_MIN, (1 - MU_MIN) / 2 of the sphere) is the daily rate
    act = _cycle_terms(yr, P)[0].sum(axis=0)
    n_pores = np.array([r['n_pores'] for r in rows])
    b, a = np.polyfit(act, n_pores, 1, w=np.sqrt(ne)) if np.ptp(act) > 0.05 else (0.0, np.average(n_pores, weights=ne))
    if a <= 0 or a + b <= 0:                             # no sensible line (too little of the cycle): constant
        a, b = np.average(n_pores, weights=ne), 0.0
    P['pore_rate_min'] = float(a / ((1 - MU_MIN) / 2))
    P['pore_rate'] = float((a + b) / ((1 - MU_MIN) / 2))
    return P, dict(rate_nll=float(best.fun), lat_nll=float(lat.fun), pores_line=(float(a), float(b)))


def windows(feats, n_win, length):
    """Indices of the days of n_win windows of `length` days spread evenly over the period (windows with fewer than
    length / 2 days left out)."""
    t = np.array([f['t'] for f in feats])
    out = []
    for c in np.linspace(t[0] + length / 2, t[-1] - length / 2, n_win):
        idx = np.nonzero((t >= c - length / 2) & (t < c + length / 2))[0]
        if len(idx) >= length / 2:
            out.append(idx)
    return out


def _synthetic_window(task):
    """Synthetic masks at the days of one window (task: P, year0, days [(t, date, n)], seed, folder or None) through
    the extraction: their features (and the masks written in folder as sunsim.sdo writes them, if given)."""
    P, year0, days, seed, folder = task
    ts = [d[0] for d in days]
    regs = um.generate_regions(P, um.SUN, min(ts), max(ts), year0=year0, seed=seed)
    feats, rows = [], []
    if folder:
        os.makedirs(folder, exist_ok=True)
    for t, date, n in days:
        B0, L0 = b0_l0(date)
        Pt = dict(P, bg_level=um.scaled(P, 'bg_level', year0 + t / 365.25))
        s, f, _, _ = um.render(regs, t, n, Pt, um.SUN, B0, L0, seed=seed)
        fe = day_features(dict(date=date, t=t, spot=s, facula=f))
        feats.append(fe)
        if folder:
            for kind, m in (('spot', s), ('faculae', f)):
                np.savez_compressed(os.path.join(folder, '%s_map_%.1f.npz' % (kind, t)), packed=np.packbits(m),
                                    shape=np.array(m.shape))
            rows.append('%g,%s,%.6f,%.6f' % (t, date[:10], 100 * fe['ff_sp'], 100 * fe['ff_fc']))
    if folder:
        with open(os.path.join(folder, 'summary.csv'), 'w') as fh:
            fh.write('t,date,ffs_sp,ffs_fc\n' + '\n'.join(rows) + '\n')
    return feats


def _day_tuples(feats, idx, sizes):
    return [(feats[i]['t'], feats[i]['date'], sizes[i]) for i in idx]


def closed_loop(P, feats, wins, sizes, year0, workers, log, n_iter=4, seeds=(51, 52)):
    """Closed loop of the notebook (8b) on the windows: rate_max, the pore rates and D0 rescaled until the extracted
    regions and pores per day and the median decay of the synthetic masks (spots only) match those of SDO."""
    sdo = [statistics([feats[i] for i in w], track_regions([feats[i] for i in w])) for w in wins]
    n_reg_t = np.mean(np.concatenate([S['n_regions'] for S in sdo]))
    n_por_t = np.mean(np.concatenate([S['n_pores'] for S in sdo]))
    dec_t = np.median(np.concatenate([S['decay'][:, 1] for S in sdo]))
    history = []
    for it in range(n_iter):
        Q = dict(P, bg_level=0.0, bg_level_min=None, speck_sigma=0.0, k_fac=0.0, spotless_rate=0.0,
                 spotless_rate_min=None)
        tasks = [(Q, year0, _day_tuples(feats, w, sizes), 1000 * s + j, None) for s in seeds for j, w in enumerate(wins)]
        with _pool(workers) as pool:
            res = pool.map(_synthetic_window, tasks)
        Ss = [statistics(fe, track_regions(fe)) for fe in res]
        n_reg = np.mean(np.concatenate([S['n_regions'] for S in Ss]))
        n_por = np.mean(np.concatenate([S['n_pores'] for S in Ss]))
        dec = np.median(np.concatenate([S['decay'][:, 1] for S in Ss]))
        P = dict(P)
        f_rate = float(np.clip(n_reg_t / max(n_reg, 0.1), 0.5, 2))**0.5
        f_pore = float(np.clip(n_por_t / max(n_por, 0.1), 0.5, 2))**0.5
        f_dec = float(np.clip(dec_t / dec, 0.5, 2))**0.5 if np.isfinite(dec) and dec > 0 else 1.0
        P['rate_max'] *= f_rate
        P['pore_rate'] *= f_pore
        P['pore_rate_min'] *= f_pore
        P['D0'] *= f_dec
        history.append(dict(n_regions=n_reg, n_pores=n_por, decay=dec, rate_max=P['rate_max'],
                            pore_rate=P['pore_rate'], D0=P['D0']))
        _log('closed loop %d: regions/day %.2f (SDO %.2f), pores/day %.2f (%.2f), decay %.1f (%.1f) MSH/day -> '
             'rate_max %.2f, pore_rate %.2f (min %.2f), D0 %.1f' % (it, n_reg, n_reg_t, n_por, n_por_t, dec, dec_t,
                                                                  P['rate_max'], P['pore_rate'], P['pore_rate_min'],
                                                                  P['D0']), log)
    return P, sdo, history


def _plage_of_days(task):
    P, year0, days, seed = task
    return _synthetic_window((P, year0, days, seed, None))


def calibrate_background(P, feats, wins, sizes, year0, workers, log, level_grid, sigma_grid, every=3, seed=1):
    """Background plage and specks (8c of the notebook) at low and at high activity: the windows split in two halves
    by the activity of the cycles; per half, bg_level and speck_sigma on the plage patches of 1-30 MSH per day and the
    small-scale plage filling factor; speck_sigma the best of both halves together, bg_level a straight line in the
    activity through the two halves (bg_level_min at zero activity, bg_level at activity 1)."""
    act_w = np.array([um.activity(year0 + feats[w[len(w) // 2]]['t'] / 365.25, P) for w in wins])
    order = np.argsort(act_w)
    halves = [np.concatenate([wins[k][::every] for k in order[:len(order) // 2]]),
              np.concatenate([wins[k][::every] for k in order[len(order) // 2:]])]
    acts, targets = [], []
    for idx in halves:
        acts.append(float(np.mean([um.activity(year0 + feats[i]['t'] / 365.25, P) for i in idx])))
        ps = plage_summary([feats[i] for i in idx])
        targets.append(np.log(np.r_[ps['n_patches'][:3], ps['ff_small'].mean()]))
    grid = [(h, lv, sg) for h in range(2) for lv in level_grid for sg in sigma_grid]
    tasks = [(dict(P, bg_level=lv, bg_level_min=None, speck_sigma=sg), year0, _day_tuples(feats, halves[h], sizes), seed)
             for h, lv, sg in grid]
    with _pool(workers) as pool:
        res = pool.map(_plage_of_days, tasks)
    cost, rows = {}, []
    for (h, lv, sg), fe in zip(grid, res):
        ps = plage_summary(fe)
        got = np.r_[ps['n_patches'][:3], ps['ff_small'].mean()]
        cost[h, lv, sg] = float(np.sum((np.log(np.maximum(got, 1e-9)) - targets[h])**2))
        rows.append(dict(half=h, activity=acts[h], bg_level=lv, speck_sigma=sg, n_patches=got[:3].tolist(),
                         ff_small=float(got[3]), target=np.exp(targets[h]).tolist(), cost=cost[h, lv, sg]))
    sg_best = min(sigma_grid, key=lambda sg: sum(min(cost[h, lv, sg] for lv in level_grid) for h in range(2)))
    bg = [min(level_grid, key=lambda lv: cost[h, lv, sg_best]) for h in range(2)]
    P = dict(P, speck_sigma=float(sg_best))
    if acts[1] - acts[0] > 0.05:
        slope = (bg[1] - bg[0]) / (acts[1] - acts[0])
        P['bg_level_min'] = float(max(bg[0] - slope * acts[0], 0.0))
        P['bg_level'] = float(max(P['bg_level_min'] + slope, 0.0))
    else:
        P['bg_level_min'], P['bg_level'] = None, float(np.mean(bg))
    _log('background: activity %.2f -> bg_level %.2f, activity %.2f -> %.2f, speck_sigma %.1f: bg_level_min %s, '
         'bg_level %.3f' % (acts[0], bg[0], acts[1], bg[1], sg_best, P['bg_level_min'], P['bg_level']), log)
    return P, rows


def calibrate_faculae(P, feats, wins, sizes, year0, workers, log, k_grid, tau_grid, rate_grid, every=4, seeds=(1, 2)):
    """Faculae (9 of the notebook): k_fac, tau_fac and spotless_rate (plage-only regions at activity 1, scaled with
    the activity: spotless_rate_min = 0) on the mean plage filling factor of every window, the share of the plage in
    patches without spots and the plage profile around the regions."""
    subs = [w[::every] for w in wins]
    ff_t = np.array([np.mean([feats[i]['ff_fc'] for i in w]) for w in subs])
    allt = [feats[i] for w in subs for i in w]
    sh_t, pr_t = spotless_share(allt), region_profiles(allt)
    days = _day_tuples(feats, np.concatenate(subs), sizes)
    t_win = [set(feats[i]['t'] for i in w) for w in subs]
    grid = [(k, tau, r) for k in k_grid for tau in tau_grid for r in rate_grid]
    tasks = [(dict(P, k_fac=k, tau_fac=tau, spotless_rate=r, spotless_rate_min=0.0), year0, days, s)
             for k, tau, r in grid for s in seeds]
    with _pool(workers) as pool:
        res = pool.map(_plage_of_days, tasks)
    best, rows = None, []
    for j, (k, tau, r) in enumerate(grid):
        fe = [f for s in range(len(seeds)) for f in res[j * len(seeds) + s]]
        ff = np.array([np.mean([f['ff_fc'] for f in fe if f['t'] in tw]) for tw in t_win])
        sh = spotless_share(fe)
        dp = float(np.nanmean((region_profiles(fe) - pr_t)**2))
        c = float(np.mean(np.log(ff / ff_t)**2) + (sh - sh_t)**2 + PROF_W * dp)
        rows.append(dict(k_fac=k, tau_fac=tau, spotless_rate=r, ff_plage=ff.tolist(), ff_target=ff_t.tolist(),
                         spotless_share=sh, spotless_target=sh_t, profile_rms=np.sqrt(dp), cost=c))
        _log('  k_fac %4.0f tau_fac %3.0f d spotless_rate %.1f /day -> ff_plage %.2f%% (SDO %.2f%%), spotless share '
             '%.2f (%.2f), profile rms %.1f%%, cost %.3f' % (k, tau, r, 100 * ff.mean(), 100 * ff_t.mean(), sh, sh_t,
                                                           100 * np.sqrt(dp), c), log)
        if best is None or c < best[-1]:
            best = (k, tau, r, c)
    _log('faculae: k_fac %g, tau_fac %g d, spotless_rate %g /day' % best[:3], log)
    return dict(P, k_fac=float(best[0]), tau_fac=float(best[1]), spotless_rate=float(best[2]),
                spotless_rate_min=0.0), rows


def _jsonable(P):
    return {k: (None if v is None else float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
            for k, v in P.items()}


def run_train(a):
    from astropy.time import Time
    out = os.path.join(a.out, 'model')
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, 'train.log')
    maps = a.maps or os.path.join(a.out, 'maps')
    if os.path.isdir(os.path.join(maps, 'days')):
        _write_summary(maps)                                     # all the blocks of maps / maps-local together
    days = read_days(maps, a.start)
    year0 = Time(a.start).decimalyear
    _log('%d days of SDO masks, %s to %s' % (len(days), days[0]['date'][:10], days[-1]['date'][:10]), log)
    feats = sdo_features(days, os.path.join(out, 'features.pkl'), a.workers, log)
    sizes = [mask_size(d['paths'][0]) for d in days]
    tracks = track_regions(feats)
    S = statistics(feats, tracks)
    _log('%d tracked regions, %d births over the period; peak area <A> %.0f MSH, p_nest %.2f, %.1f regions and %.1f '
         'pores per day' % (len(tracks), len(S['birth_lat']), np.exp(np.log(S['peak_area']).mean()), S['p_nest'],
                            S['n_regions'].mean(), S['n_pores'].mean()), log)

    P = fit_parameters(S, um.params(um.DEFAULTS), a.decay_sat)
    _log('decay: D0 %.1f MSH/day, decay_p %.3f, A_sat %s MSH' % (P['D0'], P['decay_p'], P['A_sat']), log)
    rows, b_year, b_lat = bins_table(feats, tracks, year0, a.bin_days)
    P, fit = fit_cycles(rows, b_year, b_lat, P)
    _log('cycles: rate_max %.2f /day (detected births), cycle_start %.2f, cycle_width %.2f yr; butterfly lat_start '
         '%.1f deg, lat_tau %.1f yr, lat_sigma %.1f deg; pores %.1f /day at activity 0, %.1f at 1'
         % (P['rate_max'], P['cycle_start'], P['cycle_width'], P['lat_start'], P['lat_tau'], P['lat_sigma'],
            P['pore_rate_min'], P['pore_rate']), log)
    P0 = dict(P)

    wins = windows(feats, a.n_windows, a.window_days)
    _log('%d windows of %d days for the calibrations' % (len(wins), a.window_days), log)
    P, S_win, loop = closed_loop(P, feats, wins, sizes, year0, a.workers, log, n_iter=a.n_iter)
    P, bg_rows = calibrate_background(P, feats, wins, sizes, year0, a.workers, log, a.bg_grid, a.speck_grid)
    P, fac_rows = calibrate_faculae(P, feats, wins, sizes, year0, a.workers, log, a.k_grid, a.tau_grid, a.spotless_grid)

    model = dict(P=_jsonable(P), P_fit=_jsonable(P0), fit=fit, thresholds=THRESHOLDS, size=SIZE, start=a.start,
                 year0=year0, days=len(days), first=days[0]['date'][:10], last=days[-1]['date'][:10],
                 windows=[[float(feats[w[0]]['t']), float(feats[w[-1]]['t'])] for w in wins],
                 trained=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(model, open(os.path.join(out, 'model.json'), 'w'), indent=1)
    pickle.dump(dict(S=S, bins=rows, birth_year=b_year, birth_abslat=b_lat, S_windows=S_win, closed_loop=loop,
                     background=bg_rows, faculae=fac_rows, model=model), open(os.path.join(out, 'statistics.pkl'), 'wb'))
    with open(os.path.join(out, 'bins.csv'), 'w') as f:
        keys = list(rows[0])
        f.write(','.join(keys + ['model_rate', 'model_lat']) + '\n')
        for r in rows:
            terms = um._cycle_terms(r['year'], P)
            f.write(','.join('%.6g' % r[k] for k in keys) + ',%.6g,%.6g\n'
                    % (um.cycle_rate(r['year'], P), sum(x * l for x, l in terms) / max(sum(x for x, _ in terms), 1e-9)))
    _log('model written in %s' % out, log)
    return 0


# =============================================================================================
# synth: 3 months at the minimum and at the maximum
# =============================================================================================
def run_synth(a):
    from astropy.time import Time
    model = json.load(open(os.path.join(a.out, 'model', 'model.json')))
    P = um.params(model['P'])
    year0 = model['year0']
    folder = os.path.join(a.out, 'synthetic')
    os.makedirs(folder, exist_ok=True)
    log = os.path.join(folder, 'synth.log')
    days = read_days(a.maps or os.path.join(a.out, 'maps'), model['start'])
    feats_all = pickle.load(open(os.path.join(a.out, 'model', 'features.pkl'), 'rb'))
    t = np.array([d['t'] for d in days])
    t_start = datetime.strptime(model['start'], '%Y-%m-%d')
    periods = {}
    for name, start in (('min', a.min), ('max', a.max)):
        if start == 'auto':
            # the window of a.days days with the highest mean spot filling factor (at least 2/3 of the days observed)
            ff = np.array([feats_all[d['date']]['ff_sp'] if d['date'] in feats_all else np.nan for d in days])
            best = max((s for s in t if np.sum((t >= s) & (t < s + a.days)) >= 2 * a.days / 3),
                       key=lambda s: np.nanmean(ff[(t >= s) & (t < s + a.days)]))
            start = (t_start + timedelta(days=float(best))).strftime('%Y-%m-%d')
        t0 = float((datetime.strptime(start, '%Y-%m-%d') - t_start).days)
        periods[name] = (start, [d for d in days if t0 <= d['t'] < t0 + a.days])
        _log('%s: %s, %d days of SDO masks' % (name, start, len(periods[name][1])), log)

    tasks, keys = [], []
    for name, (start, pd) in periods.items():
        sizes = [mask_size(d['paths'][0]) for d in pd]
        dts = [(d['t'], d['date'], n) for d, n in zip(pd, sizes)]
        for k in range(a.n_real):
            tasks.append((P, year0, dts, 100 + k, os.path.join(folder, '%s_%d' % (name, k))))
            keys.append((name, k))
    with _pool(a.workers) as pool:
        res = pool.map(_synthetic_window, tasks)
    out = {}
    for name, (start, pd) in periods.items():
        fe = [feats_all[d['date']] for d in pd]
        S_sdo = statistics(fe, track_regions(fe))
        syn = [statistics(r, track_regions(r), seed=100 + k) for (n, k), r in zip(keys, res) if n == name]
        year = Time(pd[len(pd) // 2]['date']).decimalyear
        out[name] = dict(start=start, dates=[d['date'] for d in pd], t=[d['t'] for d in pd], sdo=S_sdo, syn=syn,
                         activity=um.activity(year, P), rate=um.cycle_rate(year, P),
                         folders=[os.path.join(folder, '%s_%d' % (name, k)) for k in range(a.n_real)])
        _log('%s (%s, activity %.2f): ff spot SDO %.4f%%, synthetic %s; ff plage SDO %.2f%%, synthetic %s'
             % (name, start, out[name]['activity'], 100 * S_sdo['ff_sp'].mean(),
                ', '.join('%.4f%%' % (100 * S['ff_sp'].mean()) for S in syn), 100 * S_sdo['ff_fc'].mean(),
                ', '.join('%.2f%%' % (100 * S['ff_fc'].mean()) for S in syn)), log)
    pickle.dump(out, open(os.path.join(folder, 'statistics.pkl'), 'wb'))
    return 0


# =============================================================================================
# main
# =============================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('maps')
    m.add_argument('--out', required=True)
    m.add_argument('--start', default=START)
    m.add_argument('--end', default=END)
    m.add_argument('--workers', type=int, default=2, help='epochs downloaded at the same time (and in advance)')
    m.add_argument('--tmp', default=None, help='folder of the FITS files (deleted after every epoch)')
    ml = sub.add_parser('maps-local')
    ml.add_argument('--data', required=True, help='folder of the HMI images (continuum and hmi.M_45s)')
    ml.add_argument('--out', required=True)
    ml.add_argument('--start', default=START)
    ml.add_argument('--end', default=END)
    ml.add_argument('--block', type=int, default=0)
    ml.add_argument('--n-blocks', type=int, default=1)
    ml.add_argument('--workers', type=int, default=2, help='days at the same time (~4 GB each)')
    t = sub.add_parser('train')
    t.add_argument('--out', required=True)
    t.add_argument('--maps', default=None, help='folder of the SDO masks (default OUT/maps)')
    t.add_argument('--start', default=START, help='t = 0 of the masks')
    t.add_argument('--workers', type=int, default=8)
    t.add_argument('--bin-days', type=float, default=91.0, help='bins of the cycle fit [days]')
    t.add_argument('--n-windows', type=int, default=12, help='windows of the calibrations, spread over the period')
    t.add_argument('--window-days', type=float, default=30.0)
    t.add_argument('--n-iter', type=int, default=4, help='iterations of the closed loop')
    t.add_argument('--decay-sat', type=float, default=None,
                   help='A_sat [MSH]: decay rate constant above this peak area (default: no limit)')
    t.add_argument('--bg-grid', type=float, nargs='+', default=[0.08, 0.12, 0.16, 0.20, 0.24, 0.28])
    t.add_argument('--speck-grid', type=float, nargs='+', default=[1.0, 1.2])
    t.add_argument('--k-grid', type=float, nargs='+', default=[10.0, 20.0, 40.0])
    t.add_argument('--tau-grid', type=float, nargs='+', default=[10.0, 20.0, 30.0])
    t.add_argument('--spotless-grid', type=float, nargs='+', default=[0.0, 1.0, 2.0])
    s = sub.add_parser('synth')
    s.add_argument('--out', required=True)
    s.add_argument('--maps', default=None)
    s.add_argument('--min', default='2019-11-01', help='first day of the 3 months at the minimum')
    s.add_argument('--max', default='auto', help="first day at the maximum ('auto': highest spot filling factor)")
    s.add_argument('--days', type=float, default=91.0)
    s.add_argument('--n-real', type=int, default=3)
    s.add_argument('--workers', type=int, default=6)
    a = p.parse_args(argv)
    return {'maps': run_maps, 'maps-local': run_maps_local, 'train': run_train, 'synth': run_synth}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
