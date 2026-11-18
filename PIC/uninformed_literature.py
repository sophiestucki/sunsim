#!/usr/bin/env python
"""
The uninformed-mask generator of the decade (uninformed_masks.py, model.json of uninformed_decade.py) against the
literature on the filling factors, latitudes, sizes, decay and nesting of spots and faculae
(solar_filling_factors_literature.md):
  1. the parameters fitted on the SDO masks against the published values (part1),
  2. simulations in the conditions of every reference, measured as the reference measured them.

Simulations (cached in OUT, then read by uninformed_literature.ipynb):

  python uninformed_literature.py cycle --out OUT [--scale 1] [--every 1] [--size 1024] [--workers 8] [--seed 1]
      Masks of one cycle of the model (cycle_start to cycle_start + cycle_period), every `every` days, at the dates
      of the Sun (B0, L0), the amplitude multiplied by `scale` (scale_amplitude); per day the projected filling
      factors, the corrected areas, the spot components and groups, |latitude| histograms of the spot and plage
      areas and the plage poleward of 60 deg. OUT/cycle_s<scale>_e<every>_n<size>_seed<seed>.npz
  python uninformed_literature.py satire --out OUT --satire satire_ff.csv [--n-real 3]
      Masks at the dates of the SATIRE-S masks of satire_ff.csv (2017 and 2021). OUT/satire_n<size>.npz
  python uninformed_literature.py emergence --out OUT [--scales 1 4 16] [--size 512]
      Masks at the maximum of the model with the emergence multiplied by s (Nemec et al. 2022), with the background
      plage of the model extrapolated (bg 'model') or kept at its value at activity 1 (bg 'fixed').
      OUT/emergence_n<size>.npz

The region catalogues (latitudes, sizes, decay, nesting) are cheap and made in the notebook (catalog()).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.ndimage as nd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import curve_fit, minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import poisson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uninformed_masks as um                      # noqa: E402
import uninformed_decade as ud                     # noqa: E402  (sets um.A_TH to the A_th of the SDO masks)

DECADE = os.environ.get('UM_DECADE', os.path.expanduser('~/Documents/uninformed_decade'))
OUT = os.path.join(DECADE, 'literature')
DEG = um.DEG
EIGHT = np.ones((3, 3), dtype=int)
LAT_EDGES = np.arange(0, 91, 5.0)
SYNODIC = 27.2753                                  # days, Carrington synodic rotation
CARR_RATE = 360 / SYNODIC                          # deg/day, drift of the central meridian in Carrington longitude


# =============================================================================================
# references (solar_filling_factors_literature.md)
# =============================================================================================
REF = dict(
    shapiro2014=dict(A_S=(0.105, -1.315, 4.102), A_F=(-0.233, 1.400), S_min=0.169, S_max=0.188,
                     min=dict(A_S=0.003e-2, A_F=0.36e-2), max=dict(A_S=0.28e-2, A_F=3e-2), bin_days=58),
    chatzistergos2022=dict(daily_sa=(-0.004, 0.342, 0.35), daily_sa_err=(0.008, 0.093, 0.08),
                           daily_gn=(0.004, 0.005, 0.86), daily_gn_err=(0.001, 0.001, 0.06),
                           annual_sa=(0.007, 0.0176, 0.78), annual_gn=(0.005, 0.0029, 1.12),
                           sa_min=5e-4, sa_bin=1e-4,
                           cycle_sa={19: 1.98e-3, 21: 1.73e-3, 22: 1.66e-3, 23: 1.32e-3, 24: 0.88e-3},
                           cycle_gn={19: 6.85, 21: 6.24, 22: 6.04, 23: 5.01}),
    jiang2011=dict(lat=(26.4, -34.2, 16.1), lat_n=(12.2, 0.022), lat_12_20=14.6, sigma=(0.14, 1.05, -0.78),
                   area=(115, 396, -426), groups_per_RG=1 / 2.1, bins=30, used=(4, 27)),
    hathaway2011=dict(lat0=28.0, tau_months=90.0, start_offset_months=(-8, 3), stop_lat=7.0),
    hathaway1994=dict(b=lambda R: 27.12 + 25.15 * (100 / R)**0.25, c=0.71),      # b in months, R = Rmax (ISN v1)
    hathaway2015=dict(area_per_R_v1=16.7),
    baumann2005=dict(max_total=(62.2, 2.45), snap_total=(58.6, 2.49), A_min=60.0, bin_frac=0.2),
    martinez1993=dict(lnD_mean=1.75, lnD_var=2.0),
    castenmiller1986=dict(frac=(0.30, 0.40), dlon=7.5, dlat=5.0, rotations=4, n_min=3, phase_years=(4.7, 10.7)),
    brouwer1990=dict(frac_min=1 / 3),
    karapinar2026=dict(D=(0.61, 0.12), D_c24=(0.664, 0.063), D_c25=(0.627, 0.101), eps=0.11, min_samples=3,
                       N_ref=350, band=5.0, lat_max=35.0,
                       D_lat={(0, 5): 0.582, (10, 15): 0.690, (15, 20): 0.688, (30, 35): 0.416}),
    isik2018=dict(p=0.7, nest_lat=2.0, nest_lon=3.0),
    pojoga2002=dict(nest_lat=2.0, nest_lon=3.0),
    jha2024=dict(plage_over_spot_lat=(1.205, 0.020)),
    chapman2001=dict(ratio=(5, 40), bin_days=100),
    nemec2022=dict(scales=(16, 64), ratio_t0=(14, 50)),
)
# Values not in the papers above, needed to put the references in the conditions of the model [background knowledge,
# SILSO]: minimum of cycle 24 (2008.96), maximum of the smoothed ISN v1 of cycle 24 (81.9) and of cycle 22 (158.5).
SILSO = dict(min_c24=2008.96, Rmax_v1_c24=81.9, Rmax_v1_c22=158.5)


def shapiro_AS(S):
    a, b, c = REF['shapiro2014']['A_S']
    return a + b * S + c * S**2


def shapiro_AF(S):
    a, b = REF['shapiro2014']['A_F']
    return a + b * S


def shapiro_AS_of_AF(A_F):
    """Spot coverage of Shapiro et al. (2014) at the facular coverage A_F (S eliminated between their Eqs. 1-2)."""
    a, b = REF['shapiro2014']['A_F']
    return shapiro_AS((np.asarray(A_F) - a) / b)


def shapiro_AF_of_AS(A_S):
    """Facular coverage of Shapiro et al. (2014) at the spot coverage A_S (the branch of Eq. 1 above its minimum)."""
    a, b, c = REF['shapiro2014']['A_S']
    S0 = -b / (2 * c)
    S = S0 + np.sqrt(np.maximum(np.asarray(A_S) - shapiro_AS(S0), 0) / c)
    return shapiro_AF(S)


def plaw(x, a1, a2, a3):
    return a1 + a2 * np.asarray(x, float)**a3


def jiang_lat(x, lat_n=None):
    a, b, c = REF['jiang2011']['lat']
    lam = a + b * x + c * x**2
    return lam if lat_n is None else lam * lat_n / REF['jiang2011']['lat_12_20']


def jiang_sigma(x, lat_n=None):
    a, b, c = REF['jiang2011']['sigma']
    return (a + b * x + c * x**2) * jiang_lat(x, lat_n)


def jiang_area(x):
    a, b, c = REF['jiang2011']['area']
    return a + b * x + c * x**2


# =============================================================================================
# model
# =============================================================================================
def load_model(path=None):
    """model.json of the decade: (model, P = generator, P_fit = fit on the SDO masks)."""
    model = json.load(open(path or os.path.join(DECADE, 'model', 'model.json')))
    return model, um.params(model['P']), um.params(model['P_fit'])


def scale_amplitude(P, s, background=True):
    """The model of a cycle s times stronger: the emergence of spotted and plage-only regions multiplied by s, and the
    pores and background plage, linear in the activity in the model (x_min + (x - x_min) activity), extrapolated to
    activity s (background=False: background plage as at activity 1)."""
    Q = dict(P, rate_max=P['rate_max'] * s, spotless_rate=P['spotless_rate'] * s)
    for key in ('pore_rate', 'bg_level'):
        if key == 'bg_level' and not background:
            continue
        lo = P.get(key + '_min')
        if lo is not None:
            Q[key] = lo + (P[key] - lo) * s
    return Q


def lognormal_bs(mu, sigma):
    """A lognormal of the areas (ln A ~ N(mu, sigma)) in the convention of Baumann & Solanki (2005),
    ln dN/dA = -(ln A - ln<A>)^2 / (2 ln sigma_A) + c: <A> = exp(mu - sigma^2) (vertex of dN/dA), sigma_A = exp(sigma^2)."""
    return np.exp(mu - sigma**2), np.exp(sigma**2)


def bs_slope(A, A_mean, sigma_A):
    """d ln(dN/dA) / d ln A of a Baumann & Solanki lognormal at A."""
    return -(np.log(A) - np.log(A_mean)) / np.log(sigma_A)


def cycle_minima(P, y0, y1, step=1 / 365.25):
    """Decimal years of the minima of the total emergence rate (sum of the overlapping cycles) between y0 and y1."""
    yr = np.arange(y0, y1, step)
    r = np.array([um.cycle_rate(y, P) for y in yr])
    k = np.nonzero((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:]))[0] + 1
    return yr[k]


# =============================================================================================
# region catalogues
# =============================================================================================
def catalog(P, y0, y1, year0, seed=1):
    """Regions emerging between the decimal years y0 and y1 (uninformed_masks.generate_regions, Carrington frame):
    dict of arrays for the spotted regions (not the pores): t0 [days from year0], lat, lon (Carrington at
    emergence), A (peak), D, tg, k (number of the cycle: 0 = the one starting at cycle_start); 'pores': t0, lat, lon;
    'regs': all the regions."""
    t0, t1 = (y0 - year0) * 365.25, (y1 - year0) * 365.25
    regs = um.generate_regions(P, um.SUN, t0, t1, year0=year0, seed=seed)
    sp = [r for r in regs if r['spots'] and not r['pore'] and t0 <= r['t0'] <= t1]
    po = [r for r in regs if r['pore'] and t0 <= r['t0'] <= t1]
    yr = year0 + np.array([r['t0'] for r in sp]) / 365.25
    k = np.floor((yr - P['cycle_start']) / P['cycle_period']) - np.array([r['cycle'] for r in sp])
    out = {key: np.array([r[key] for r in sp]) for key in ('t0', 'lat', 'lon', 'A', 'D', 'tg')}
    out.update(k=k.astype(int), year=yr, pores={key: np.array([r[key] for r in po]) for key in ('t0', 'lat', 'lon')},
               regs=regs, P=P, year0=year0, span=(t0, t1))
    return out


def daily_areas(cat, L0_start=0.0):
    """Every region on every day it has spots: index of the region, day, area [MSH], Carrington longitude, distance
    from the central meridian seen from the Earth (B0 = 0; CMD) and visibility (on the Earth side)."""
    tb = np.floor(cat['t0']) + 1
    te = cat['t0'] + cat['tg'] + cat['A'] / cat['D']
    n = np.maximum(np.floor(te) - tb + 1, 0).astype(int)
    idx = np.repeat(np.arange(len(n)), n)
    day = tb[idx] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    dt = day - cat['t0'][idx]
    tg, A, D = cat['tg'][idx], cat['A'][idx], cat['D'][idx]
    area = np.where(dt < tg, A * dt / tg, np.maximum(A - D * (dt - tg), 0.0))
    lat = cat['lat'][idx]
    lon = (cat['lon'][idx] + um.drift(lat, dt, um.SUN)) % 360
    L0 = (L0_start - CARR_RATE * day) % 360
    cmd = (lon - L0 + 180) % 360 - 180
    keep = area > 0
    return dict(i=idx[keep], day=day[keep], area=area[keep], lat=lat[keep], lon=lon[keep], cmd=cmd[keep],
                visible=np.abs(cmd[keep]) < 90)


def recorded_groups(cat, daily, cmd_max=90.0):
    """Regions seen from the Earth (|CMD| < cmd_max on a day with spots), each once at its maximum recorded area:
    index, day, area, |lat|, lat, Carrington longitude of that day."""
    v = np.abs(daily['cmd']) < cmd_max
    i, a = daily['i'][v], daily['area'][v]
    order = np.lexsort((-a, i))
    first = np.r_[True, i[order][1:] != i[order][:-1]]
    sel = np.nonzero(v)[0][order[first]]
    return dict(i=daily['i'][sel], day=daily['day'][sel], area=daily['area'][sel], lat=daily['lat'][sel],
                lon=daily['lon'][sel])


def groups_on_disc(cat, daily, with_pores=True):
    """Daily number of groups on the Earth side of the Sun (regions with spots; + the one-day pores), per integer day."""
    v = daily['visible']
    days = np.arange(np.floor(cat['span'][0]), np.ceil(cat['span'][1]) + 1)
    n = np.bincount((daily['day'][v] - days[0]).astype(int), minlength=len(days))[:len(days)].astype(float)
    if with_pores and len(cat['pores']['t0']):
        pd_ = np.floor(cat['pores']['t0'])
        L0 = (-CARR_RATE * pd_) % 360
        vis = np.abs((cat['pores']['lon'] - L0 + 180) % 360 - 180) < 90
        n += np.bincount((pd_[vis] - days[0]).astype(int), minlength=len(days))[:len(days)]
    return days, n


# =============================================================================================
# mask simulations
# =============================================================================================
def measure_day(s, f, g):
    """Per-day measurements of a spot and plage mask (geometry g of uninformed_masks.disc_geometry)."""
    ins = g['inside']
    mu = np.nan_to_num(g['mu'])
    lat = np.nan_to_num(g['lat'])
    R = g['R']
    msh = 1e6 / (2 * np.pi * R**2 * np.maximum(mu, 0.05))
    npx = ins.sum()
    row = dict(ff_sp=s.sum() / npx, ff_fc=f.sum() / npx)
    ok = ins & (mu > 0.1)
    row['A_sp'] = msh[ok & s].sum()
    row['A_fc'] = msh[ok & f].sum()
    good = ins & (mu > 0.3)
    alat = np.abs(lat)
    for key, m in (('h_sp', s), ('h_fc', f), ('h_surf', ins)):
        sel = good & m
        row[key] = np.histogram(alat[sel], LAT_EDGES, weights=msh[sel])[0]
    pol = ok & (alat > 60)
    row['pol_fc'], row['pol_surf'] = msh[pol & f].sum(), msh[pol].sum()
    lab, nc = nd.label(s, structure=EIGHT)
    row['n_comp'] = nc
    row['n_groups'] = row['n_groups3'] = 0
    if nc:
        idx = np.arange(1, nc + 1)
        vec = np.stack([nd.sum(np.nan_to_num(g['vec'][..., k]), lab, idx) for k in range(3)], axis=-1)
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        area = nd.sum(np.where(ins, msh, 0), lab, idx)
        if nc == 1:
            cl = np.array([1])
        else:
            cl = fcluster(linkage(vec, method='single', metric='euclidean'), 2 * np.sin(ud.LINK_DEG * DEG / 2),
                          criterion='distance')
        ga = np.bincount(cl, weights=area)[1:]
        row['n_groups'] = int(np.sum(ga > 0))
        row['n_groups3'] = int(np.sum(ga >= ud.A_DET))
    return row


def _render_chunk(task):
    """Masks of some days (task: P, regions alive in the chunk, year0, [(t, date)], size, seed): measure_day rows."""
    P, regs, year0, days, size, seed = task
    rows = []
    for t, date in days:
        B0, L0 = ud.b0_l0(date)
        Pt = dict(P, bg_level=um.scaled(P, 'bg_level', year0 + t / 365.25))
        s, f, _, g = um.render(regs, t, size, Pt, um.SUN, B0, L0, seed=seed)
        rows.append(dict(measure_day(s, f, g), t=t, date=date))
    return rows


def simulate_days(P, year0, times, size, seed, workers, regs=None, chunk=20):
    """measure_day of the masks at the times [days from year0] (dates of the Sun), in parallel, the regions generated
    once for the whole span (so that the chunks are the same realization)."""
    from astropy.time import Time
    times = np.sort(np.asarray(times, float))
    dates = [Time(year0 + t / 365.25, format='decimalyear').isot[:19] for t in times]
    if regs is None:
        regs = um.generate_regions(P, um.SUN, times.min(), times.max(), year0=year0, seed=seed)
    tasks = []
    for k in range(0, len(times), chunk):
        tt = times[k:k + chunk]
        alive = [r for r in regs if r['t0'] <= tt[-1] and r['t_end'] >= tt[0]]
        tasks.append((P, alive, year0, list(zip(tt, dates[k:k + chunk])), size, seed))
    with ud._pool(workers) as pool:
        res = pool.map(_render_chunk, tasks)
    rows = [r for c in res for r in c]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out['year'] = year0 + out['t'] / 365.25
    return out, regs


def _save(path, d, **meta):
    np.savez_compressed(path, **{k: v for k, v in d.items()}, meta=json.dumps(meta, default=str))


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != 'meta'}
    d['meta'] = json.loads(str(z['meta']))
    return d


def cycle_path(out, scale, every, size, seed, p_nest=None):
    tag = '' if p_nest is None else '_pnest%g' % p_nest
    return os.path.join(out, 'cycle_s%g_e%g_n%d_seed%d%s.npz' % (scale, every, size, seed, tag))


def run_cycle(a):
    model, P, _ = load_model(a.model)
    Q = scale_amplitude(P, a.scale)
    if a.p_nest is not None:
        Q['p_nest'] = a.p_nest
    y0 = P['cycle_start']
    times = np.arange((y0 - model['year0']) * 365.25, (y0 + P['cycle_period'] - model['year0']) * 365.25, a.every)
    t = time.time()
    d, _ = simulate_days(Q, model['year0'], times, a.size, a.seed, a.workers)
    os.makedirs(a.out, exist_ok=True)
    path = cycle_path(a.out, a.scale, a.every, a.size, a.seed, a.p_nest)
    _save(path, d, scale=a.scale, every=a.every, size=a.size, seed=a.seed, P=Q, year0=model['year0'])
    print('%s: %d days in %.0f s' % (path, len(times), time.time() - t), flush=True)
    return 0


def run_satire(a):
    import pandas as pd
    from astropy.time import Time
    model, P, _ = load_model(a.model)
    sat = pd.read_csv(a.satire)
    times = np.array([(Time(d + 'T12:00:00').decimalyear - model['year0']) * 365.25 for d in sat['date']])
    out = {}
    for k in range(a.n_real):
        d, _ = simulate_days(P, model['year0'], times, a.size, 200 + k, a.workers)
        for key in ('ff_sp', 'ff_fc'):
            out['%s_%d' % (key, k)] = d[key]
    out['t'], out['date'] = d['t'], np.array([x[:10] for x in d['date']])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'satire_n%d.npz' % a.size)
    _save(path, out, n_real=a.n_real, size=a.size)
    print(path, flush=True)
    return 0


def run_emergence(a):
    model, P, _ = load_model(a.model)
    y_max = minimize_scalar(lambda y: -um.cycle_rate(y, P), bounds=(P['cycle_start'], P['cycle_start'] + 8),
                            method='bounded').x
    t_max = (y_max - model['year0']) * 365.25
    times = t_max + np.arange(0, a.days, a.every)
    out = dict(times=times)
    for bg in ('model', 'fixed'):
        for s in a.scales:
            if bg == 'fixed' and s == 1:
                d = out['ff_sp_model_1'], out['ff_fc_model_1']
                out['ff_sp_fixed_1'], out['ff_fc_fixed_1'] = d
                continue
            t = time.time()
            d, _ = simulate_days(scale_amplitude(P, s, background=(bg == 'model')), model['year0'], times, a.size,
                                 a.seed, a.workers, chunk=2)
            out['ff_sp_%s_%g' % (bg, s)], out['ff_fc_%s_%g' % (bg, s)] = d['ff_sp'], d['ff_fc']
            print('bg %s, x%g: ff_sp %.4f%%, ff_fc %.2f%% (%.0f s)' % (bg, s, 100 * d['ff_sp'].mean(),
                                                                  100 * d['ff_fc'].mean(), time.time() - t), flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'emergence_n%d.npz' % a.size)
    _save(path, out, scales=a.scales, size=a.size, year_max=y_max)
    print(path, flush=True)
    return 0


# =============================================================================================
# analyses (the notebook)
# =============================================================================================
def bins_by(x, y, width, n_min=1):
    """Days sorted by x, cut in consecutive bins of `width` days: mean x and y per bin (Shapiro et al. 2014)."""
    o = np.argsort(x)
    k = np.arange(len(o)) // width
    keep = np.bincount(k) >= max(n_min, width // 2)
    mx = np.bincount(k, x[o]) / np.bincount(k)
    my = np.bincount(k, y[o]) / np.bincount(k)
    return mx[keep], my[keep]


def binned_mean(x, y, width, x_min):
    """Mean y in bins of x of `width` above x_min (the PDF matrices of Chatzistergos et al. 2022): centres, means, n."""
    sel = x > x_min
    k = np.floor((x[sel] - x_min) / width).astype(int)
    n = np.bincount(k)
    m = np.bincount(k, y[sel]) / np.maximum(n, 1)
    c = x_min + (np.arange(len(n)) + 0.5) * width
    return c[n > 0], m[n > 0], n[n > 0]


def fit_plaw(x, y, w=None, p0=(0.0, 0.3, 0.4)):
    """p = a1 + a2 x^a3 (least squares, weights w = counts): parameters and 1-sigma errors."""
    sig = None if w is None else 1 / np.sqrt(w)
    p, c = curve_fit(plaw, x, y, p0=p0, sigma=sig, maxfev=20000, bounds=([-0.05, 0.0, 0.05], [0.05, np.inf, 2.0]))
    return p, np.sqrt(np.diag(c))


def annual(year, x, stat=np.median):
    yr = np.floor(year)
    u = np.unique(yr)
    return u, np.array([stat(x[yr == v]) for v in u])


def running_mean(x, n):
    return np.convolve(x, np.ones(n) / n, mode='valid')


def bs_fit(areas, A_min=60.0, frac=0.2):
    """Baumann & Solanki (2005): bins of width frac x geometric mean area above A_min, dN/dA normalised to the number
    above A_min, parabola in log-log: <A>, sigma_A (and the number of areas)."""
    a = areas[areas > A_min]
    if len(a) < 20:
        return np.nan, np.nan, len(a)
    q = (1 + frac / 2) / (1 - frac / 2)
    edges = A_min * q**np.arange(0, int(np.log(a.max() / A_min) / np.log(q)) + 2)
    n, _ = np.histogram(a, edges)
    c = np.sqrt(edges[1:] * edges[:-1])
    ok = n > 0
    y = np.log(n[ok] / np.diff(edges)[ok] / len(a))
    p = np.polyfit(np.log(c[ok]), y, 2, w=np.sqrt(n[ok]))
    if p[0] >= 0:
        return np.nan, np.nan, len(a)
    return float(np.exp(-p[1] / (2 * p[0]))), float(np.exp(-1 / (2 * p[0]))), len(a)


def bs_fit_slopes(areas, A_eval=(100, 300, 1000), A_min=60.0, frac=0.2):
    """Slopes d ln(dN/dA) / d ln A at A_eval of the Baumann & Solanki parabola fitted above A_min (better
    constrained than <A> and sigma_A when the curvature is small)."""
    A, s, _ = bs_fit(areas, A_min, frac)
    return [bs_slope(x, A, s) for x in A_eval]


def hathaway_centroids(cat, daily, hemisphere=1):
    """Area-weighted centroid latitude and width per Carrington rotation of every cycle of one hemisphere (Hathaway
    2011, whole Sun): time since the start of the cycle [months], centroid, width, total area."""
    P = cat['P']
    out = []
    for k in np.unique(cat['k']):
        sel = (cat['k'][daily['i']] == k) & (np.sign(daily['lat']) == hemisphere)
        start = (P['cycle_start'] + k * P['cycle_period'] - cat['year0']) * 365.25
        rot = np.floor((daily['day'][sel] - start) / SYNODIC)
        w, lat = daily['area'][sel], np.abs(daily['lat'][sel])
        for r in np.unique(rot):
            m = rot == r
            W = w[m].sum()
            c = np.sum(w[m] * lat[m]) / W
            out.append((k, (r + 0.5) * SYNODIC / 30.4375, c, np.sqrt(np.sum(w[m] * (lat[m] - c)**2) / W), W))
    return np.array(out).reshape(-1, 5)


def fit_hathaway(cen, min_area_frac=0.05):
    """lambda = lat0 exp(-t / tau) on the centroids (weights: total area; rotations above min_area_frac of the largest
    total area): (lat0, tau [months]) free, and the offset t0 [months] with Hathaway's 28 deg and 90 months."""
    t, c, W = cen[:, 1], cen[:, 2], cen[:, 4]
    ok = W > min_area_frac * W.max()
    f = lambda x, a, tau: a * np.exp(-x / tau)                                     # noqa: E731
    (a, tau), cov = curve_fit(f, t[ok], c[ok], p0=(28, 90), sigma=1 / np.sqrt(W[ok]))
    g = lambda x, t0: 28.0 * np.exp(-(x - t0) / 90.0)                              # noqa: E731
    (t0,), cov0 = curve_fit(g, t[ok], c[ok], p0=(0,), sigma=1 / np.sqrt(W[ok]))
    return dict(lat0=a, lat0_err=np.sqrt(cov[0, 0]), tau=tau, tau_err=np.sqrt(cov[1, 1]), t0=t0,
                t0_err=np.sqrt(cov0[0, 0]), n=int(ok.sum()))


def jiang_phases(cat, daily, minima, nbins=30):
    """Jiang et al. (2011) per cycle (minimum to minimum, 30 phases): recorded groups once at their maximum area;
    mean and std of |lat| and mean area per phase; lambda_n (mean of the phase means 4-27); S_n (maximum of the
    12-month running mean of R_G = 12.08 x groups on the disc per day, monthly); groups per month against R_G."""
    rec = recorded_groups(cat, daily)
    days, n_disc = groups_on_disc(cat, daily)
    days_np, n_disc_np = groups_on_disc(cat, daily, with_pores=False)
    yr_days = cat['year0'] + days / 365.25
    out = []
    for m0, m1 in zip(minima[:-1], minima[1:]):
        d0, d1 = (m0 - cat['year0']) * 365.25, (m1 - cat['year0']) * 365.25
        sel = (rec['day'] >= d0) & (rec['day'] < d1)
        x = (rec['day'][sel] - d0) / (d1 - d0)
        i = np.minimum((x * nbins).astype(int), nbins - 1) + 1                       # 1..30
        alat, A = np.abs(rec['lat'][sel]), rec['area'][sel]
        lam = np.array([alat[i == j].mean() if np.any(i == j) else np.nan for j in range(1, nbins + 1)])
        sig = np.array([alat[i == j].std() if np.sum(i == j) > 1 else np.nan for j in range(1, nbins + 1)])
        area = np.array([A[i == j].mean() if np.any(i == j) else np.nan for j in range(1, nbins + 1)])
        a, b = REF['jiang2011']['used']
        dsel = (days >= d0) & (days < d1)
        month = np.floor((days[dsel] - d0) / 30.4375).astype(int)
        RG = 12.08 * np.bincount(month, n_disc[dsel]) / np.bincount(month)
        RG_np = 12.08 * np.bincount(month, n_disc_np[dsel]) / np.bincount(month)
        S_n = running_mean(RG, 12).max()
        # groups recorded per month (each once, regions + pores on the Earth side) against R_G / 2.1
        gm = np.bincount(np.floor((rec['day'][sel] - d0) / 30.4375).astype(int), minlength=len(RG))[:len(RG)]
        pd_ = cat['pores']['t0']
        L0 = (-CARR_RATE * np.floor(pd_)) % 360
        pvis = (np.abs((cat['pores']['lon'] - L0 + 180) % 360 - 180) < 90) & (pd_ >= d0) & (pd_ < d1)
        pm = np.bincount(np.floor((pd_[pvis] - d0) / 30.4375).astype(int), minlength=len(RG))[:len(RG)]
        out.append(dict(min0=m0, min1=m1, lam=lam, sigma=sig, area=area, lam_n=np.nanmean(lam[a - 1:b]),
                        S_n=S_n, S_n_no_pores=running_mean(RG_np, 12).max(), RG=RG, RG_no_pores=RG_np,
                        groups_month=gm + pm, groups_month_no_pores=gm, n_groups=int(sel.sum())))
    return out


def castenmiller_fraction(t, lat, lon, dlon=7.5, dlat=5.0, dt=4 * SYNODIC, n_min=3):
    """Fraction of the regions with at least n_min - 1 others within |dlon| <= dlon (Carrington, wrapped),
    |dlat| <= dlat and |dt| <= dt (Castenmiller et al. 1986: >= 3 active regions within 4 rotations, +-7.5 deg in
    longitude and +-5 deg in latitude)."""
    pts = np.column_stack([t / dt, lat / dlat, lon / dlon])
    per = 360 / dlon
    # wrap in longitude: copies at +-360 deg
    allp = np.vstack([pts, pts + [0, 0, per], pts - [0, 0, per]])
    tree = cKDTree(allp)
    nb = tree.query_ball_point(pts, r=1.0, p=np.inf, return_length=True)
    return float(np.mean(nb >= n_min))


def dbscan(X, eps, min_samples):
    """DBSCAN (labels, -1 = noise), Euclidean, min_samples counting the point itself (as scikit-learn)."""
    tree = cKDTree(X)
    nbrs = tree.query_ball_point(X, eps)
    core = np.array([len(n) >= min_samples for n in nbrs])
    lab = np.full(len(X), -1)
    c = 0
    for i in range(len(X)):
        if lab[i] != -1 or not core[i]:
            continue
        lab[i] = c
        stack = [i]
        while stack:
            j = stack.pop()
            for q in nbrs[j]:
                if lab[q] == -1:
                    lab[q] = c
                    if core[q]:
                        stack.append(q)
        c += 1
    return lab


def karapinar_D(t, lon, eps0=0.11, min_samples=3, N_ref=350, alpha=0.05):
    """Nesting degree of Karapinar et al. (2026) in one window (one cycle, one latitude band): DBSCAN in the
    standardised (Carrington longitude, time) plane with the wrap-around copies, clusters significant against a
    uniform Poisson emergence (p < alpha in their bounding box), D = groups (original longitudes) in significant
    clusters / groups."""
    N = len(t)
    if N < min_samples:
        return np.nan, N
    eps = eps0 * min(1.5, np.sqrt(N_ref / N)) if N < N_ref else eps0
    lo = np.r_[lon, lon[lon > 275] - 360, lon[lon < 85] + 360]
    tt = np.r_[t, t[lon > 275], t[lon < 85]]
    orig = np.r_[np.arange(N), np.nonzero(lon > 275)[0], np.nonzero(lon < 85)[0]]
    X = np.column_stack([(lo - lon.mean()) / lon.std(), (tt - t.mean()) / t.std()])
    lab = dbscan(X, eps, min_samples)
    T = t.max() - t.min()
    nested = np.zeros(N, bool)
    for c in np.unique(lab[lab >= 0]):
        m = lab == c
        area = max(np.ptp(lo[m]), 1e-3) * max(np.ptp(tt[m]), 1e-3)
        n_exp = N / (360.0 * T) * area
        if poisson.sf(m.sum() - 1, n_exp) < alpha:
            nested[orig[m]] = True                  # a group is nested if it or its wrap-around copy is
    return float(nested.mean()), N


def karapinar_cycles(cat, daily, minima, pooled=True):
    """D per cycle (minimum to minimum) and per 5-deg |latitude| band (0-35 deg; pooled hemispheres, or the two
    hemispheres separately), recorded groups once at their maximum area (time and Carrington longitude of that
    day): rows (cycle index, band, hemisphere, D, N)."""
    rec = recorded_groups(cat, daily)
    rows = []
    for j, (m0, m1) in enumerate(zip(minima[:-1], minima[1:])):
        d0, d1 = (m0 - cat['year0']) * 365.25, (m1 - cat['year0']) * 365.25
        sel = (rec['day'] >= d0) & (rec['day'] < d1)
        for b0 in np.arange(0, REF['karapinar2026']['lat_max'], REF['karapinar2026']['band']):
            for h in ((0,) if pooled else (1, -1)):
                m = sel & (np.abs(rec['lat']) >= b0) & (np.abs(rec['lat']) < b0 + 5)
                if h:
                    m &= np.sign(rec['lat']) == h
                D, N = karapinar_D(rec['day'][m], rec['lon'][m])
                rows.append((j, b0, h, D, N))
    return np.array(rows, dtype=float).reshape(-1, 5)


# =============================================================================================
# reports: rows (reference, quantity, reference value, SDO fit, simulation, note)
# =============================================================================================
def row(ref, quantity, ref_val, sdo=None, sim=None, note=''):
    f = lambda v: '' if v is None else v if isinstance(v, str) else '%.3g' % v          # noqa: E731
    return dict(reference=ref, quantity=quantity, reference_value=f(ref_val), SDO_fit=f(sdo), simulation=f(sim),
                note=note)


def part1(P_fit, P):
    """The parameters fitted on the SDO masks (P_fit; generator P after the closed loop where it differs) against
    the published values."""
    R = []
    # --- butterfly: Hathaway (2011), Hathaway et al. (1994), Jiang et al. (2011)
    R.append(row('Hathaway 2011', 'latitude at the cycle start [deg]', REF['hathaway2011']['lat0'], P_fit['lat_start']))
    R.append(row('Hathaway 2011', 'e-folding time of the drift [months]', REF['hathaway2011']['tau_months'],
                 12 * P_fit['lat_tau']))
    off = 12 * (P_fit['cycle_start'] - SILSO['min_c24'])
    R.append(row('Hathaway 2011', 'cycle start - minimum (SILSO 2008.96) [months]', '-8 to +3', off,
                 note='minimum of cycle 24 from SILSO'))
    b = REF['hathaway1994']['b'](SILSO['Rmax_v1_c24']) / 12
    R.append(row('Hathaway et al. 1994', 'cycle width b [yr] (b(Rmax) with Rmax v1 = 81.9)', b, P_fit['cycle_width'],
                 note='Rmax of cycle 24 from SILSO'))
    x = np.arange(4, 28) / 30
    lam_m = P_fit['lat_start'] * np.exp(-x * P_fit['cycle_period'] / P_fit['lat_tau'])
    for xx in (4 / 30, 0.5, 27 / 30):
        R.append(row('Jiang et al. 2011', 'mean |lat| at phase %.2f [deg]' % xx, jiang_lat(xx),
                     P_fit['lat_start'] * np.exp(-xx * P_fit['cycle_period'] / P_fit['lat_tau'])))
    R.append(row('Jiang et al. 2011', 'rms(model - Jiang) over phases 4-27 [deg]', 0.3,
                 np.sqrt(np.mean((lam_m - jiang_lat(x))**2)), note='0.3 = rms of the Jiang fit'))
    R.append(row('Jiang et al. 2011', 'cycle mean latitude lambda_n [deg]', '%.1f (cycles 12-20)' % REF['jiang2011'][
        'lat_12_20'], lam_m.mean()))
    R.append(row('Jiang et al. 2011', 'latitude spread sigma [deg] (mean of phases 4-27)',
                 '%.1f (%.1f-%.1f)' % (jiang_sigma(x).mean(), jiang_sigma(x).min(), jiang_sigma(x).max()),
                 P_fit['lat_sigma'], note='constant in the model'))
    # --- sizes: Baumann & Solanki (2005), Jiang et al. (2011)
    a, s = lognormal_bs(P_fit['logA_mean'], P_fit['logA_sigma'])
    R.append(row('Baumann & Solanki 2005', '<A> of dN/dA, maximum area, total [MSH]', REF['baumann2005']['max_total'][0], a,
                 note='lognormal of the SDO peak areas in their convention'))
    R.append(row('Baumann & Solanki 2005', 'sigma_A', REF['baumann2005']['max_total'][1], s))
    for A in (100, 300, 1000):
        R.append(row('Baumann & Solanki 2005', 'slope d ln(dN/dA)/d ln A at %d MSH' % A,
                     bs_slope(A, *REF['baumann2005']['max_total']), bs_slope(A, a, s)))
    for A in (10, 30, 100):
        R.append(row('Jiang et al. 2011', 'slope of n(A) at %d MSH (power law A^-1.1 below 300)' % A, -1.1,
                     bs_slope(A, a, s)))
    # --- decay: Martinez Pillet et al. (1993) as used by Baumann & Solanki (2005)
    for lab, Q in (('fit', P_fit), ('generator', P)):
        m = np.log(Q['D0']) + Q['decay_p'] * (Q['logA_mean'] - np.log(100))
        v = Q['decay_p']**2 * Q['logA_sigma']**2
        R.append(row('Martinez Pillet et al. 1993', 'mean ln D [ln MSH/day] (%s, D0 = %.1f)' % (lab, Q['D0']),
                     REF['martinez1993']['lnD_mean'], m if lab == 'fit' else None, m if lab != 'fit' else None))
        R.append(row('Martinez Pillet et al. 1993', 'variance of ln D (%s)' % lab, REF['martinez1993']['lnD_var'],
                     v if lab == 'fit' else None, v if lab != 'fit' else None))
    # --- nesting
    R.append(row('Castenmiller et al. 1986', 'fraction of regions in nests', '0.30-0.40', P_fit['p_nest'],
                 note='p_nest = excess over random of births within 10 deg / 27 d (not the same definition)'))
    R.append(row('Brouwer & Zwaan 1990', 'fraction in clusters', '>= 0.33', P_fit['p_nest']))
    R.append(row('Karapinar et al. 2026', 'nesting degree D', '%.2f +- %.2f' % REF['karapinar2026']['D'], P_fit['p_nest']))
    R.append(row('Isik et al. 2018', 'nesting probability p (model input)', REF['isik2018']['p'], P_fit['p_nest']))
    R.append(row('Pojoga & Cudnik 2002 / Isik 2018', 'scatter of nest members, lat / lon [deg]', '2 / 3',
                 '%g / %g' % (P_fit['nest_sigma'], P_fit['nest_sigma']), note='nest_sigma fixed, not fitted'))
    R.append(row('Castenmiller et al. 1986', 'time window of a nest', '4 rotations (109 d)',
                 '%g d' % P_fit['nest_days'], note='fixed'))
    return R


def sdo_daily(path):
    """Daily SDO masks of the decade (maps/summary.csv of uninformed_decade.dag, fractions of the disc [%]) as
    fractions: dict year, ff_sp, ff_fc, n_spots; None if the file is not there."""
    if not os.path.exists(path):
        return None
    import pandas as pd
    from astropy.time import Time
    d = pd.read_csv(path)
    d = d[(d['ff_fc'] < 20) & (d['ff_sp'] < 5)]                  # bad HMI images, as sunsim.observations
    return dict(year=Time(list(d['date'] + 'T12:00:00')).decimalyear, ff_sp=d['ff_sp'].values / 100,
                ff_fc=d['ff_fc'].values / 100, n_spots=d['n_spots'].values)


def chatzistergos_tests(d, label, gn=None):
    """Chatzistergos et al. (2022) on daily projected spot (s_a) and plage (p_a) fractions: power law on the plage
    averaged in spot bins of 1e-4 above 5e-4, annual medians, cycle-averaged spot fraction and the plage predicted by
    their Eqs. 1 and A.1 from our spot fractions."""
    C = REF['chatzistergos2022']
    R = []
    x, y, n = binned_mean(d['ff_sp'], d['ff_fc'], C['sa_bin'], C['sa_min'])
    try:
        p, e = fit_plaw(x, y, n)
        for j, nm in enumerate(('a1', 'a2', 'a3')):
            R.append(row('Chatzistergos et al. 2022', 'daily p_a = a1 + a2 s_a^a3: %s' % nm, '%.3g +- %.2g' % (
                C['daily_sa'][j], C['daily_sa_err'][j]), *((p[j], None) if label == 'SDO' else (None, p[j])),
                note=label))
    except RuntimeError:
        pass
    try:
        q, _ = curve_fit(lambda s_, a2, a3: a2 * s_**a3, x, y, p0=(0.3, 0.35), sigma=1 / np.sqrt(n), maxfev=20000)
        R.append(row('Chatzistergos et al. 2022', 'daily p_a = a2 s_a^a3 (a1 = 0): a3', C['daily_sa'][2],
                     *((q[1], None) if label == 'SDO' else (None, q[1])), note=label))
    except RuntimeError:
        pass
    pa_mean = d['ff_fc'].mean()
    pred1 = np.mean(plaw(d['ff_sp'], *C['daily_sa']))
    m = d['ff_sp'] > C['sa_min']
    vals = [('mean plage fraction / Eq. 1 prediction (all days)', pa_mean / pred1),
            ('same, days with spots > 5e-4 of the disc', d['ff_fc'][m].mean() / np.mean(plaw(d['ff_sp'][m],
                                                                                          *C['daily_sa'])))]
    for q, v in vals:
        R.append(row('Chatzistergos et al. 2022', q, 1.0, *((v, None) if label == 'SDO' else (None, v)), note=label))
    yr, sa = annual(d['year'], d['ff_sp'])
    _, pa = annual(d['year'], d['ff_fc'])
    try:
        p, _ = curve_fit(plaw, sa, pa, p0=(0.007, 0.02, 0.8), maxfev=20000,
                         bounds=([-0.05, 0.0, 0.05], [0.05, np.inf, 2.0]))
        R.append(row('Chatzistergos et al. 2022', 'annual medians: exponent a3', C['annual_sa'][2],
                     *((p[2], None) if label == 'SDO' else (None, p[2])), note=label))
    except (RuntimeError, TypeError):
        pass
    if gn is not None:
        x, y, n = binned_mean(gn.astype(float), d['ff_fc'], 1.0, 0.5)
        try:
            p, e = fit_plaw(x, y, n, p0=(0.004, 0.005, 0.9))
            R.append(row('Chatzistergos et al. 2022', 'daily p_a vs group number: exponent a3', C['daily_gn'][2],
                         *((p[2], None) if label == 'SDO' else (None, p[2])), note=label))
        except RuntimeError:
            pass
    return R


def shapiro_tests(d, label, width=None):
    """Shapiro et al. (2014): days sorted by the facular coverage (their S-index sorting, A_F linear in S), bins of 58
    days: spot coverage against the A_S(A_F) of their Eqs. 1-2."""
    width = width or REF['shapiro2014']['bin_days']
    AF, AS = bins_by(d['ff_fc'], d['ff_sp'], width)
    pred = shapiro_AS_of_AF(AF)
    ok = pred > 2e-4
    R = [row('Shapiro et al. 2014', 'spot coverage / A_S(A_F) of Eqs. 1-2 (median over 58-d bins with A_S > 0.02%)',
             1.0, *((np.median(AS[ok] / pred[ok]), None) if label == 'SDO' else (None, np.median(AS[ok] / pred[ok]))),
             note=label + ' (%d bins)' % ok.sum())]
    ok = AS > 2e-4
    r_f = np.median(AF[ok] / shapiro_AF_of_AS(AS[ok]))
    R.append(row('Shapiro et al. 2014', 'facular coverage / A_F(A_S) of Eqs. 1-2 (median over the same bins)', 1.0,
                 *((r_f, None) if label == 'SDO' else (None, r_f)), note=label))
    c = np.polyfit(AF, AS, 2)
    R.append(row('Shapiro et al. 2014', 'A_S quadratic in A_F: curvature (d2 A_S / d A_F^2 / 2)',
                 REF['shapiro2014']['A_S'][2] / REF['shapiro2014']['A_F'][1]**2,
                 *((c[0], None) if label == 'SDO' else (None, c[0])), note=label))
    return R, (AF, AS)


def wolf_tests(d, label):
    """Hathaway (2015): total corrected spot area ~ 16.7 R (ISN v1): monthly means of the area against R = 10 g + f of
    the masks (raw count, ~ ISN v2 = ISN v1 / 0.6)."""
    m = np.floor((d['year'] - d['year'][0]) * 12)
    A = np.bincount(m.astype(int), d['A_sp']) / np.bincount(m.astype(int))
    Rr = 10 * d['n_groups'] + d['n_comp']
    R_ = np.bincount(m.astype(int), Rr) / np.bincount(m.astype(int))
    ok = np.isfinite(A) & (R_ > 0)
    slope = np.sum(A[ok] * R_[ok]) / np.sum(R_[ok]**2)
    return [row('Hathaway 2015', 'spot area / R, R = 10 g + f of the masks [MSH] (ISN v1: 16.7; x 0.6 for a raw count)',
                '%.1f (v1) / %.1f (raw)' % (REF['hathaway2015']['area_per_R_v1'],
                                            0.6 * REF['hathaway2015']['area_per_R_v1']), None, slope, note=label)]


def scenario(P, year0, scale=1.0, p_nest=None, n_cycles=3, seed=1):
    """Catalogue of n_cycles cycles of the model (amplitude x scale, p_nest changed if given), its daily areas and the
    minima of the total emergence rate."""
    Q = scale_amplitude(P, scale)
    if p_nest is not None:
        Q['p_nest'] = p_nest
    y0, y1 = Q['cycle_start'] - 1, Q['cycle_start'] + n_cycles * Q['cycle_period'] + 1
    cat = catalog(Q, y0, y1, year0, seed=seed)
    return dict(cat=cat, daily=daily_areas(cat), minima=cycle_minima(Q, Q['cycle_start'] - 0.5, y1 - 0.5), P=Q,
                scale=scale)


def catalog_tests(sc, label, rng=None):
    """The tests on a region catalogue (rows) and what the plots need."""
    rng = rng or np.random.default_rng(0)
    cat, daily, mins = sc['cat'], sc['daily'], sc['minima']
    R, out = [], {}
    # Hathaway (2011): centroids of both hemispheres and all complete cycles together
    cen = np.vstack([hathaway_centroids(cat, daily, h) for h in (1, -1)])
    cen = cen[(cen[:, 0] >= 0) & (cen[:, 0] < np.max(cen[:, 0]))]
    h = fit_hathaway(cen)
    out['hathaway'] = (cen, h)
    R.append(row('Hathaway 2011', 'centroid at the cycle start, fit [deg]', REF['hathaway2011']['lat0'], None,
                 h['lat0'], note=label))
    R.append(row('Hathaway 2011', 'e-folding time, fit [months]', REF['hathaway2011']['tau_months'], None, h['tau'],
                 note=label))
    R.append(row('Hathaway 2011', 't0 - cycle start with 28 deg / 90 months [months]', '-8 to +3 (vs minimum)', None,
                 h['t0'], note=label))
    tail = cen[(cen[:, 1] > 100) & (cen[:, 4] > 0.05 * cen[:, 4].max())]
    R.append(row('Hathaway 2011', 'latitude where the drift stops [deg] (centroids after 100 months)',
                 REF['hathaway2011']['stop_lat'], None, np.average(tail[:, 2], weights=tail[:, 4]) if len(tail) else None,
                 note=label))
    # Jiang et al. (2011)
    J = jiang_phases(cat, daily, mins)
    out['jiang'] = J
    x = np.arange(1, 31) / 30
    a, b = REF['jiang2011']['used']
    lam = np.nanmean([j['lam'] for j in J], axis=0)
    sig = np.nanmean([j['sigma'] for j in J], axis=0)
    area = np.nanmean([j['area'] for j in J], axis=0)
    S_n = np.mean([j['S_n'] for j in J])
    lam_n = np.mean([j['lam_n'] for j in J])
    R.append(row('Jiang et al. 2011', 'rms(simulated - Eq. 1) over phases 4-27 [deg]', 0.3, None,
                 np.sqrt(np.nanmean((lam[a - 1:b] - jiang_lat(x[a - 1:b]))**2)), note=label))
    pred = REF['jiang2011']['lat_n'][0] + REF['jiang2011']['lat_n'][1] * S_n
    R.append(row('Jiang et al. 2011', 'lambda_n [deg] at S_n = %.0f (12-month R_G max of the simulation)' % S_n,
                 pred, None, lam_n, note=label))
    R.append(row('Jiang et al. 2011', 'sigma of |lat| [deg], phases 4-27 mean', jiang_sigma(x[a - 1:b], lam_n).mean(),
                 None, np.nanmean(sig[a - 1:b]), note=label))
    R.append(row('Jiang et al. 2011', 'sigma late/early (phase 24-27 / 4-7)',
                 jiang_sigma(x[23:27]).mean() / jiang_sigma(x[3:7]).mean(), None,
                 np.nanmean(sig[23:27]) / np.nanmean(sig[3:7]), note=label))
    R.append(row('Jiang et al. 2011', 'mean group area at maximum, phases 4-27 [MSH]', jiang_area(x[a - 1:b]).mean(),
                 None, np.nanmean(area[a - 1:b]), note=label))
    R.append(row('Jiang et al. 2011', 'mean area at phase 0.5 / at phases 0.1 and 0.9',
                 jiang_area(0.5) / np.mean(jiang_area(np.array([0.1, 0.9]))), None,
                 np.nanmean(area[13:16]) / np.nanmean(np.r_[area[1:4], area[25:28]]), note=label))
    gm = np.mean([j['groups_month_no_pores'].sum() / len(j['RG']) for j in J])
    rg = np.mean([j['RG_no_pores'].mean() for j in J])
    R.append(row('Jiang et al. 2011', 'groups per month / (R_G / 2.1), without pores', 1.0, None, gm / (rg / 2.1),
                 note=label))
    gm = np.mean([j['groups_month'].sum() / len(j['RG']) for j in J])
    rg = np.mean([j['RG'].mean() for j in J])
    R.append(row('Jiang et al. 2011', 'groups per month / (R_G / 2.1), with pores', 1.0, None, gm / (rg / 2.1),
                 note=label))
    out['jiang_mean'] = dict(lam=lam, sigma=sig, area=area, S_n=S_n, lam_n=lam_n)
    # Baumann & Solanki (2005) and Martinez Pillet et al. (1993)
    rec30 = recorded_groups(cat, daily, cmd_max=30)
    v = np.abs(daily['cmd']) < 30
    for nm, ar, ref in (('maximum area', rec30['area'], REF['baumann2005']['max_total']),
                        ('snapshot', daily['area'][v], REF['baumann2005']['snap_total'])):
        A, s, n = bs_fit(ar)
        R.append(row('Baumann & Solanki 2005', '%s (|CMD| < 30, A > 60): <A> [MSH]' % nm, ref[0], None, A, note=label))
        R.append(row('Baumann & Solanki 2005', '%s: sigma_A' % nm, ref[1], None, s, note=label))
        for x, sl in zip((100, 300, 1000), bs_fit_slopes(ar)):
            R.append(row('Baumann & Solanki 2005', '%s: slope of dN/dA at %d MSH' % (nm, x), bs_slope(x, *ref), None, sl,
                         note=label))
    for amin in (0, 60):
        lnD = np.log(cat['D'][cat['A'] > amin])
        R.append(row('Martinez Pillet et al. 1993', 'mean ln D [ln MSH/day], regions > %d MSH' % amin,
                     REF['martinez1993']['lnD_mean'], None, lnD.mean(), note=label))
        R.append(row('Martinez Pillet et al. 1993', 'variance ln D, regions > %d MSH' % amin,
                     REF['martinez1993']['lnD_var'], None, lnD.var(), note=label))
    # nesting: Castenmiller et al. (1986), Karapinar et al. (2026)
    rec = recorded_groups(cat, daily)
    C = REF['castenmiller1986']
    for amin in (0, 30, 100):
        fr, frr = [], []
        for k in range(int(np.max(cat['k'])) + 1):
            st = (sc['P']['cycle_start'] + k * sc['P']['cycle_period'] - cat['year0']) * 365.25
            m = (rec['day'] >= st + C['phase_years'][0] * 365.25) & (rec['day'] < st + C['phase_years'][1] * 365.25) & (
                rec['area'] >= amin)
            if m.sum() < 10:
                continue
            fr.append(castenmiller_fraction(rec['day'][m], rec['lat'][m], rec['lon'][m]))
            frr.append(np.mean([castenmiller_fraction(rec['day'][m], rec['lat'][m], rng.uniform(0, 360, m.sum()))
                                for _ in range(5)]))
        R.append(row('Castenmiller et al. 1986', 'fraction in nests, groups >= %d MSH (random longitudes)' % amin,
                     '0.30-0.40', None, '%.2f (%.2f)' % (np.mean(fr), np.mean(frr)), note=label))
    K = karapinar_cycles(cat, daily, mins)
    Kr = karapinar_cycles(cat, dict(daily, lon=rng.uniform(0, 360, len(daily['lon']))), mins)
    out['karapinar'] = (K, Kr)
    R.append(row('Karapinar et al. 2026', 'nesting degree D (random longitudes)', '%.2f +- %.2f' % REF['karapinar2026']['D'],
                 None, '%.2f +- %.2f (%.2f)' % (np.nanmean(K[:, 3]), np.nanstd(K[:, 3]), np.nanmean(Kr[:, 3])), note=label))
    for (lo, hi), ref in REF['karapinar2026']['D_lat'].items():
        R.append(row('Karapinar et al. 2026', 'D at |lat| %d-%d deg' % (lo, hi), ref, None,
                     np.nanmean(K[K[:, 1] == lo, 3]), note=label))
    return R, out


def mask_tests(d, label):
    """The tests on the daily masks of a simulated cycle (rows) and what the plots need."""
    R = chatzistergos_tests(d, label, gn=d['n_groups'])
    r, sh = shapiro_tests(d, label)
    R += r
    R += wolf_tests(d, label)
    out = dict(shapiro=sh)
    C = REF['chatzistergos2022']
    R.append(row('Chatzistergos et al. 2022 (MEA20)', 'cycle-mean spot fraction of the disc x 1e3',
                 '%.2f (cycle 24)' % (1e3 * C['cycle_sa'][24]), None, 1e3 * d['ff_sp'].mean(), note=label))
    # Chapman et al. (2001): plage / spot area ratio in 100-day bins
    k = np.floor((d['year'] - d['year'][0]) * 365.25 / REF['chapman2001']['bin_days']).astype(int)
    rs = np.bincount(k, d['ff_fc']) / np.maximum(np.bincount(k, d['ff_sp']), 1e-12)
    ok = np.bincount(k, d['ff_sp']) / np.bincount(k) > 2e-4
    R.append(row('Chapman et al. 2001', 'plage / spot area, 100-d bins with spots > 0.02 %: 10th-90th percentiles',
                 '5-40', None, '%.0f-%.0f' % tuple(np.percentile(rs[ok], [10, 90])), note=label))
    # Jha et al. (2024): mean |lat| of the plage / of the spots (deprojected areas, mu > 0.3)
    c = (LAT_EDGES[1:] + LAT_EDGES[:-1]) / 2
    hs, hf = d['h_sp'].sum(0), d['h_fc'].sum(0)
    R.append(row('Jha et al. 2024', 'mean |lat| of plage / of spots (whole cycle)', '%.3f +- %.3f' % REF['jha2024'][
        'plage_over_spot_lat'], None, np.sum(hf * c) / hf.sum() / (np.sum(hs * c) / hs.sum()), note=label))
    yr = np.floor(d['year'])
    per_year = []
    for y in np.unique(yr):
        m = yr == y
        a, b = d['h_sp'][m].sum(0), d['h_fc'][m].sum(0)
        if a.sum() > 0:
            per_year.append((y, np.sum(b * c) / b.sum() / (np.sum(a * c) / a.sum())))
    per_year = np.array(per_year)
    out['jha'] = per_year
    R.append(row('Jha et al. 2024', 'same, range of the yearly values', 'independent of phase', None,
                 '%.2f-%.2f' % (per_year[:, 1].min(), per_year[:, 1].max()), note=label))
    # polar faculae (Sheeley; Munoz-Jaramillo et al. 2012): plage poleward of 60 deg against the spot area
    pol = d['pol_fc'] / np.maximum(d['pol_surf'], 1)
    _, pa = annual(d['year'], pol, np.mean)
    _, sa = annual(d['year'], d['ff_sp'], np.mean)
    out['polar'] = (np.unique(yr), pa, sa)
    R.append(row('Sheeley; Munoz-Jaramillo et al. 2012', 'correlation of the yearly polar (>60 deg) plage with the spots',
                 '< 0 (polar faculae peak at minimum)', None, np.corrcoef(pa, sa)[0, 1], note=label))
    # Nemec et al. (2022): A_f = a sqrt(A_s + c) + d on the 58-day bins
    AF, AS = sh
    f = lambda x, a_, c_, d_: a_ * np.sqrt(np.maximum(x + c_, 0)) + d_                   # noqa: E731
    try:
        p, _ = curve_fit(f, AS, AF, p0=(0.5, 1e-4, 0.0), maxfev=20000)
        res = AF - f(AS, *p)
        q = np.polyfit(AS, AF, 1)
        out['nemec_fit'] = p
        R.append(row('Nemec et al. 2022', 'A_f = a sqrt(A_s + c) + d: rms / rms of a straight line',
                     '< 1 (sub-linear)', None, np.sqrt(np.mean(res**2)) / np.sqrt(np.mean((AF - np.polyval(q, AS))**2)),
                     note=label))
    except RuntimeError:
        pass
    return R, out


# =============================================================================================
# main
# =============================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('cycle', 'satire', 'emergence'):
        s = sub.add_parser(name)
        s.add_argument('--out', default=OUT)
        s.add_argument('--model', default=None)
        s.add_argument('--workers', type=int, default=8)
        s.add_argument('--size', type=int, default=1024 if name != 'emergence' else 512)
        s.add_argument('--seed', type=int, default=1)
        if name == 'cycle':
            s.add_argument('--scale', type=float, default=1.0)
            s.add_argument('--every', type=float, default=1.0)
            s.add_argument('--p-nest', type=float, default=None)
        if name == 'satire':
            s.add_argument('--satire', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'satire_ff.csv'))
            s.add_argument('--n-real', type=int, default=3)
        if name == 'emergence':
            s.add_argument('--scales', type=float, nargs='+', default=[1.0, 4.0, 16.0])
            s.add_argument('--days', type=float, default=30.0)
            s.add_argument('--every', type=float, default=3.0)
    a = p.parse_args(argv)
    return {'cycle': run_cycle, 'satire': run_satire, 'emergence': run_emergence}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
