"""
Comparison of a simulation of the Sun (from the SDO/HMI maps, sunsim.sdo) with the observations:
  photometry   : total solar irradiance of SORCE/TIM (2003-2020, IDL .sav) and TSIS-1/TIM (2018-, csv)
  spectroscopy : HARPS-N solar telescope (csv of the public release: rv, fwhm, bis_span, contrast), averaged per day

Times. The simulation is given in days since `start` (the epochs of make_sdo_maps(epoch='days'), t = 0 the day `start`
at 00:00 UTC); the observations in JD (UTC). Simulation and observations are matched by day: the day number of the
simulation is floor(t), the one of an observation floor(JD - JD(start 00:00)) (+ shift, see best_shift_match).

Statistics (as in the 10-year comparison, Stucki et al.):
  mad(x) = median(|x - median(x)|) (not scaled by 1.4826), rms(x) = sqrt(mean((x - mean(x))^2)) (= np.std(x)).
  Photometry: the observed TSI and the simulated flux are both divided by their mean over the matched days, the
  residual is 1e6 (obs/<obs> - sim/<sim>) [ppm].
  RV (and fwhm, bis, contrast): the median of all the simulation and of all the daily HARPS-N values is removed; the
  series are cut at the HARPS-N interventions (instrumental periods) and, in every period, the shift of the days
  (-1, 0, +1) that gives the smallest rms of obs - sim is kept and the residual is obs - sim - median(obs - sim) (one
  offset per period).

Typical use (see tutorial/tutorial_sdo.ipynb):

    from sunsim import observations as obs
    harps = obs.load_harpsn(path_csv, start='2015-07-18')
    sorce, tsis = obs.load_sorce(path_sav, start='2015-07-18'), obs.load_tsis(path_csv, start='2015-07-18')
    cuts = obs.load_interventions(path_txt, start='2015-07-18')
    phot = [obs.compare_photometry(sorce, t_sim, flux_sim), obs.compare_photometry(tsis, t_sim, flux_sim)]
    rv = obs.compare_rv(harps, t_sim, rv_sim, interventions=cuts)
    obs.print_summary(phot, rv)
    obs.plot_comparison(phot, rv)

The files are found with sunsim.local_path: keys 'harpsn_sun', 'sorce_tsi', 'tsis_tsi', 'harpsn_interventions'.
"""
from datetime import datetime

import numpy as np

__all__ = ['mad', 'rms', 'jd_of_date', 'load_harpsn', 'daily_average', 'load_sorce', 'load_tsis', 'load_interventions',
           'match_days', 'best_shift_match', 'compare_photometry', 'compare_rv', 'compare_rv_exposures', 'print_summary',
           'plot_photometry_comparison', 'plot_rv_comparison', 'plot_comparison', 'plot_correlation',
           'sweep_stats', 'load_sweep_stats', 'best_sets', 'plot_sweep_heatmaps', 'plot_sweep_parameters']

COLOURS = {'sim': '#ea8642', 'harpsn': '#102e37', 'sorce': '#102e37', 'tsis': '#918AE5', 'res': '#4d4d4d'}


# ----------------------------------------------------------------------------------------------------------------------
# statistics and times
# ----------------------------------------------------------------------------------------------------------------------

def mad(x):
    """Median absolute deviation, median(|x - median(x)|) (not scaled to a gaussian sigma). NaN are ignored."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def rms(x):
    """Root mean square around the mean, sqrt(mean((x - mean(x))^2)) (= np.std(x)). NaN are ignored."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean((x - np.mean(x)) ** 2))) if len(x) else np.nan


def jd_of_date(date):
    """JD (UTC) of a date at 00:00: 'YYYY-MM-DD', a datetime / date, or a JD (number, returned as it is)."""
    if isinstance(date, (int, float, np.floating, np.integer)):
        return float(date)
    if isinstance(date, str):
        date = datetime.strptime(date[:10], '%Y-%m-%d')
    # JD of 1970-01-01 00:00 = 2440587.5
    return 2440587.5 + (datetime(date.year, date.month, date.day) - datetime(1970, 1, 1)).days


def _days(jd, start):
    """Days since start (00:00 UTC) of JD values."""
    return np.asarray(jd, dtype=np.float64) - jd_of_date(start)


def _cut(d, start, end):
    """Keep the epochs of the dict d (key 'jd', all the arrays of the same length) between start and end (JD or dates)."""
    jd = d['jd']
    keep = np.ones(len(jd), bool)
    if start is not None:
        keep &= jd >= jd_of_date(start)
    if end is not None:
        keep &= jd < jd_of_date(end) + 1                  # the whole last day
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == jd.shape else v) for k, v in d.items()}


# ----------------------------------------------------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------------------------------------------------

def load_harpsn(path, start, end=None, t_start=None, daily=True, quality=False):
    """HARPS-N solar telescope, csv of the public release (columns bjd [BJD - 2400000], rv, fwhm, bis_span [km/s],
    contrast, obs_quality, data_quality, ...).
    start : origin of the days (the `start` of the simulation, t = 0), 'YYYY-MM-DD'
    end, t_start : keep only the observations between t_start (default start) and end (dates; default all)
    daily : True (default) returns the averages per day (daily_average), False every observation
    quality : True keeps only the observations with data_quality = 1 (False: all, as in the 10-year comparison)
    Returns a dict: 'name', 'jd' (UTC at the Earth, middle of the exposure), 't' (days since start; daily: the middle of
    the day, day + 0.5), 'rv', 'fwhm', 'bis' [m/s], 'contrast' and, daily, their standard deviations in the day 'rv_err',
    'fwhm_err', 'bis_err', 'contrast_err' and the number of observations 'n' (not daily: the errors of every exposure,
    same names, and 'jd_start' (UTC) and 'texp' [s] of every exposure).
    The bjd of the release is barycentric for a target in the direction of the Sun: ~500 s (1 AU / c) before the
    exposure at the Earth. The times come from the start of the exposure in file_root (HARPN.YYYY-MM-DDTHH-MM-SS.sss,
    UTC) and texp when they are there, otherwise from bjd (-(TDB - UTC) only: ~8 min early)."""
    import pandas as pd
    data = pd.read_csv(path)
    if quality and 'data_quality' in data:
        data = data[data['data_quality'] == 1]
    data = data.sort_values('bjd')
    jd_start = texp = None
    if 'file_root' in data and 'texp' in data:
        iso = data['file_root'].astype(str).str.extract(r'(\d{4}-\d\d-\d\dT\d\d)-(\d\d)-(\d\d(?:\.\d+)?)')
        if not iso.isna().any().any():
            from astropy.time import Time
            jd_start = Time(list(iso[0] + ':' + iso[1] + ':' + iso[2]), format='isot', scale='utc').jd
            texp = np.asarray(data['texp'], dtype=np.float64)
    if jd_start is not None:
        jd = jd_start + texp / 2 / 86400
    else:
        bjd = np.asarray(data['bjd'], dtype=np.float64) + 2400000.0
        # BJD (TDB) -> JD (UTC): -(TDB - UTC), 69.184 s since 2017 (68.184 s before)
        try:
            from astropy.time import Time
            jd = Time(bjd, format='jd', scale='tdb').utc.jd
        except Exception:                               # no astropy / no leap-second table: the 2017 value
            jd = bjd - 69.184 / 86400
    out = {'name': 'HARPS-N', 'jd': jd,
           'rv': 1e3 * np.asarray(data['rv'], dtype=np.float64),
           'fwhm': 1e3 * np.asarray(data['fwhm'], dtype=np.float64),
           'bis': 1e3 * np.asarray(data['bis_span'], dtype=np.float64),
           'contrast': np.asarray(data['contrast'], dtype=np.float64)}
    if not daily:                                       # the errors of every exposure (daily: the scatter in the day)
        for k, col, scale in (('rv', 'rv_err', 1e3), ('fwhm', 'fwhm_err', 1e3), ('bis', 'bis_span_err', 1e3),
                              ('contrast', 'contrast_err', 1.0)):
            if col in data:
                out[k + '_err'] = scale * np.asarray(data[col], dtype=np.float64)
        if jd_start is not None:
            out['jd_start'], out['texp'] = jd_start, texp
    out = _cut(out, t_start if t_start is not None else start, end)
    out['t'] = _days(out['jd'], start)
    return daily_average(out, start) if daily else out


def daily_average(obs, start, keys=('rv', 'fwhm', 'bis', 'contrast')):
    """Average per UTC day of the observations obs (load_harpsn(daily=False)): the mean of every key, its standard
    deviation in the day (ddof=1, NaN with one observation) as key + '_err', the number of observations 'n', 't' the
    middle of the day (days since start, day + 0.5) and 'jd' its JD."""
    day = np.floor(_days(obs['jd'], start)).astype(np.int64)
    days, inv, n = np.unique(day, return_inverse=True, return_counts=True)
    out = {'name': obs.get('name', ''), 't': days + 0.5, 'jd': jd_of_date(start) + days + 0.5, 'n': n}
    for k in keys:
        x = obs[k]
        s = np.bincount(inv, weights=x, minlength=len(days))
        s2 = np.bincount(inv, weights=x * x, minlength=len(days))
        mean = s / n
        with np.errstate(invalid='ignore', divide='ignore'):
            var = (s2 - n * mean ** 2) / (n - 1)
        out[k] = mean
        out[k + '_err'] = np.where(n > 1, np.sqrt(np.maximum(var, 0)), np.nan)
    return out


def load_sorce(path, start, end=None, t_start=None):
    """SORCE/TIM total solar irradiance, IDL .sav (variables time [JD], tsi [W/m^2]); values <= 0 removed.
    Returns a dict 'name', 'jd', 't' (days since start), 'tsi'."""
    from scipy.io import readsav
    d = readsav(path)
    jd, tsi = np.asarray(d['time'], dtype=np.float64), np.asarray(d['tsi'], dtype=np.float64)
    keep = np.isfinite(tsi) & (tsi > 0)
    out = _cut({'name': 'SORCE/TIM', 'jd': jd[keep], 'tsi': tsi[keep]}, t_start if t_start is not None else start, end)
    out['t'] = _days(out['jd'], start)
    return out


def load_tsis(path, start, end=None, t_start=None):
    """TSIS-1/TIM total solar irradiance, csv (columns 'time (Julian Date)', 'tsi_1au (W/m^2)'); values <= 0 removed.
    Returns a dict 'name', 'jd', 't' (days since start), 'tsi'."""
    import pandas as pd
    d = pd.read_csv(path)
    jd = np.asarray(d['time (Julian Date)'], dtype=np.float64)
    tsi = np.asarray(d['tsi_1au (W/m^2)'], dtype=np.float64)
    keep = np.isfinite(tsi) & (tsi > 0)
    out = _cut({'name': 'TSIS-1', 'jd': jd[keep], 'tsi': tsi[keep]}, t_start if t_start is not None else start, end)
    out['t'] = _days(out['jd'], start)
    return out


def load_interventions(path, start):
    """Dates of the HARPS-N interventions (text file, one 'YYYY-MM-DD' per line) in days since start (sorted)."""
    dates = np.atleast_1d(np.loadtxt(path, dtype=str))
    return np.sort(np.array([jd_of_date(d) for d in dates]) - jd_of_date(start))


# ----------------------------------------------------------------------------------------------------------------------
# matching and comparison
# ----------------------------------------------------------------------------------------------------------------------

def match_days(t_obs, t_sim, shift=0):
    """Indices (i_obs, i_sim) of the observations and the simulated epochs of the same day: floor(t_sim) =
    floor(t_obs + shift) (times in days since start). The first epoch of a day is used when there are several."""
    d_obs = np.floor(np.asarray(t_obs, dtype=np.float64) + shift).astype(np.int64)
    d_sim = np.floor(np.asarray(t_sim, dtype=np.float64)).astype(np.int64)
    _, i_obs, i_sim = np.intersect1d(d_obs, d_sim, assume_unique=False, return_indices=True)
    return i_obs, i_sim


def best_shift_match(t_obs, y_obs, t_sim, y_sim, shifts=(-1, 0, 1)):
    """match_days for every shift of the observed days and the one with the smallest rms(y_obs - y_sim).
    Returns i_obs, i_sim, shift (empty arrays and NaN if no day matches)."""
    best = (np.zeros(0, int), np.zeros(0, int), np.nan, np.inf)
    for s in shifts:
        io, i_s = match_days(t_obs, t_sim, s)
        if len(io) == 0:
            continue
        r = rms(np.asarray(y_obs)[io] - np.asarray(y_sim)[i_s])
        if r < best[3]:
            best = (io, i_s, s, r)
    return best[:3]


def compare_photometry(tsi, t_sim, flux_sim, shift=0):
    """Observed TSI (load_sorce / load_tsis) against the simulated flux (t_sim in days since start), matched by day.
    Both are divided by their mean over the matched days. Returns a dict: 'name', 't' (simulated epochs of the matched
    days), 'obs', 'sim' [ppm, 1e6 (x/<x> - 1)], 'res' = obs - sim [ppm], 'mad', 'rms' (of res), 'mad_obs', 'rms_obs'
    (of obs), 'n', and 'offset' [ppm]: 1e6 (<sim matched> / <sim all> - 1), added to obs and sim by the plots so that
    several instruments are on the same scale (the mean of the whole simulation)."""
    t_sim, flux_sim = np.asarray(t_sim, dtype=np.float64), np.asarray(flux_sim, dtype=np.float64)
    io, i_s = match_days(tsi['t'], t_sim, shift)
    out = {'name': tsi.get('name', 'TSI'), 'n': len(io)}
    if len(io) == 0:
        return {**out, 't': np.zeros(0), 'obs': np.zeros(0), 'sim': np.zeros(0), 'res': np.zeros(0), 'mad': np.nan,
                'rms': np.nan, 'mad_obs': np.nan, 'rms_obs': np.nan, 'offset': 0.0}
    o, s = tsi['tsi'][io], flux_sim[i_s]
    obs, sim = 1e6 * (o / o.mean() - 1), 1e6 * (s / s.mean() - 1)
    res = obs - sim
    return {**out, 't': t_sim[i_s], 'obs': obs, 'sim': sim, 'res': res, 'mad': mad(res), 'rms': rms(res),
            'mad_obs': mad(obs), 'rms_obs': rms(obs), 'offset': 1e6 * (s.mean() / np.nanmean(flux_sim) - 1)}


def _periods(t, cuts):
    """Index of the instrumental period of every time t (cuts: days of the interventions)."""
    return np.searchsorted(np.sort(np.asarray(cuts, dtype=np.float64)), t, side='right')


def compare_rv(harps, t_sim, y_sim, quantity='rv', interventions=None, shifts=(-1, 0, 1)):
    """Daily HARPS-N values (load_harpsn) of quantity ('rv', 'fwhm', 'bis' [m/s] or 'contrast') against the simulation
    (t_sim in days since start, y_sim in the same unit), see the module docstring: medians removed, one period per
    interval between the interventions, the best day shift and one offset per period.
    Returns a dict: 'quantity', 't' (days of the observations + shift), 'obs', 'obs_err', 'sim', 'res', 'mad', 'rms'
    (of res), 'mad_obs', 'rms_obs' (of obs), 'reduction' [%] = 100 (rms_obs - rms) / rms_obs, 'n', 'interventions' (the
    ones inside the observations) and 'periods': a list of dicts (start, end, n_sim, n_obs, n, shift, offset, rms, mad)."""
    t_sim, y_sim = np.asarray(t_sim, dtype=np.float64), np.asarray(y_sim, dtype=np.float64)
    t_obs, y_obs = np.asarray(harps['t'], dtype=np.float64), np.asarray(harps[quantity], dtype=np.float64)
    err = np.asarray(harps.get(quantity + '_err', np.full(len(t_obs), np.nan)), dtype=np.float64)
    ok = np.isfinite(y_obs)
    t_obs, y_obs, err = t_obs[ok], y_obs[ok], err[ok]
    ok = np.isfinite(y_sim)
    t_sim, y_sim = t_sim[ok], y_sim[ok]
    y_obs = y_obs - np.median(y_obs)
    y_sim = y_sim - np.median(y_sim)

    cuts = np.zeros(0) if interventions is None else np.sort(np.asarray(interventions, dtype=np.float64))
    if len(t_obs):
        cuts = cuts[(cuts > t_obs.min()) & (cuts < t_obs.max())]
    p_obs, p_sim = _periods(t_obs, cuts), _periods(t_sim, cuts)
    edges = np.concatenate(([t_obs.min() if len(t_obs) else 0], cuts, [t_obs.max() if len(t_obs) else 0]))

    cols = {k: [] for k in ('t', 'obs', 'obs_err', 'sim', 'res')}
    periods = []
    for p in range(len(cuts) + 1):
        mo = np.where(p_obs == p)[0]
        ms = np.where((p_sim == p) & (t_sim >= edges[0]) & (t_sim <= edges[-1]))[0]   # inside the observations
        io, i_s, shift = best_shift_match(t_obs[mo], y_obs[mo], t_sim[ms], y_sim[ms], shifts)
        info = {'start': edges[p], 'end': edges[p + 1], 'n_sim': len(ms), 'n_obs': len(mo), 'n': len(io),
                'shift': shift, 'offset': np.nan, 'rms': np.nan, 'mad': np.nan}
        if len(io):
            o, s = y_obs[mo][io], y_sim[ms][i_s]
            d = o - s
            info.update(offset=float(np.median(d)), rms=rms(d), mad=mad(d))
            cols['t'].append(t_obs[mo][io] + shift)
            cols['obs'].append(o)
            cols['obs_err'].append(err[mo][io])
            cols['sim'].append(s)
            cols['res'].append(d - np.median(d))
        periods.append(info)
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    out.update(quantity=quantity, n=len(out['res']), interventions=cuts, periods=periods,
               mad=mad(out['res']), rms=rms(out['res']), mad_obs=mad(out['obs']), rms_obs=rms(out['obs']))
    out['reduction'] = 100 * (out['rms_obs'] - out['rms']) / out['rms_obs'] if out['n'] else np.nan
    return out


def compare_rv_exposures(obs, y_sim, quantity='rv', interventions=None):
    """Every HARPS-N exposure (load_harpsn(daily=False), or any dict with 't' and quantity) against the simulated value
    of the same exposure, y_sim (same length as obs['t']: e.g. the simulation at the SDO epoch closest to every
    exposure), without averaging per day. Medians removed, one offset per instrumental period (interventions, days
    since start), as compare_rv but without day shifts. Returns the dict of compare_rv ('t', 'obs', 'obs_err', 'sim',
    'res', 'mad', 'rms', 'mad_obs', 'rms_obs', 'reduction', 'n', 'quantity', 'interventions', 'periods'), so that
    print_summary, plot_rv_comparison and plot_correlation work with it."""
    t = np.asarray(obs['t'], dtype=np.float64)
    y_obs = np.asarray(obs[quantity], dtype=np.float64)
    y_sim = np.asarray(y_sim, dtype=np.float64)
    if len(y_sim) != len(t):
        raise ValueError('y_sim must have one value per exposure (%d), got %d' % (len(t), len(y_sim)))
    err = np.asarray(obs.get(quantity + '_err', np.full(len(t), np.nan)), dtype=np.float64)
    ok = np.isfinite(y_obs) & np.isfinite(y_sim)
    t, y_obs, y_sim, err = t[ok], y_obs[ok] - np.median(y_obs[ok]), y_sim[ok] - np.median(y_sim[ok]), err[ok]
    cuts = np.zeros(0) if interventions is None else np.sort(np.asarray(interventions, dtype=np.float64))
    if len(t):
        cuts = cuts[(cuts > t.min()) & (cuts < t.max())]
    period = _periods(t, cuts)
    edges = np.concatenate(([t.min() if len(t) else 0], cuts, [t.max() if len(t) else 0]))
    res = np.full(len(t), np.nan)
    periods = []
    for p in range(len(cuts) + 1):
        m = period == p
        info = {'start': edges[p], 'end': edges[p + 1], 'n_sim': int(m.sum()), 'n_obs': int(m.sum()), 'n': int(m.sum()),
                'shift': 0, 'offset': np.nan, 'rms': np.nan, 'mad': np.nan}
        if m.any():
            d = y_obs[m] - y_sim[m]
            res[m] = d - np.median(d)
            info.update(offset=float(np.median(d)), rms=rms(d), mad=mad(d))
        periods.append(info)
    out = {'t': t, 'obs': y_obs, 'obs_err': err, 'sim': y_sim, 'res': res, 'quantity': quantity, 'n': len(t),
           'interventions': cuts, 'periods': periods, 'mad': mad(res), 'rms': rms(res), 'mad_obs': mad(y_obs),
           'rms_obs': rms(y_obs)}
    out['reduction'] = 100 * (out['rms_obs'] - out['rms']) / out['rms_obs'] if out['n'] else np.nan
    return out


def print_summary(photometry=(), rv=None):
    """Prints the MAD and RMS of the residuals (and of the observations) of compare_photometry (one dict or a list) and
    compare_rv (one dict or a list), and the periods of compare_rv."""
    if isinstance(photometry, dict):
        photometry = [photometry]
    for ph in photometry:
        print('%-10s %4d days  residuals: MAD %7.1f ppm  RMS %7.1f ppm   (observed: MAD %7.1f  RMS %7.1f ppm)'
              % (ph['name'], ph['n'], ph['mad'], ph['rms'], ph['mad_obs'], ph['rms_obs']))
    for r in ([] if rv is None else [rv] if isinstance(rv, dict) else rv):
        unit = '' if r['quantity'] == 'contrast' else ' m/s'
        for i, p in enumerate(r['periods']):
            print('  period %2d: %7.1f -> %7.1f  sim %4d  obs %4d  matched %4d  shift %4s  RMS %.3f%s'
                  % (i, p['start'], p['end'], p['n_sim'], p['n_obs'], p['n'], p['shift'], p['rms'], unit))
        print('%-10s %4d days  residuals: MAD %.3f%s  RMS %.3f%s   (observed: MAD %.3f  RMS %.3f%s; RMS reduced by %.1f %%)'
              % (r['quantity'].upper(), r['n'], r['mad'], unit, r['rms'], unit, r['mad_obs'], r['rms_obs'], unit,
                 r['reduction']))


# ----------------------------------------------------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------------------------------------------------

def _x(t, t0):
    return np.asarray(t, dtype=np.float64) - t0


def plot_photometry_comparison(photometry, ax=None, ax_res=None, t0=0.0, colours=None, align=True, sim_label='SunSim',
                               stats=True):
    """Observed and simulated TSI variations [ppm] (compare_photometry: one dict or a list, one per instrument) on ax
    and the residuals on ax_res (optional). align: adds the 'offset' of every instrument (common scale: the mean of the
    whole simulation). t0: subtracted from the times (days since start) on the x axis. Returns ax, ax_res."""
    import matplotlib.pyplot as plt
    if isinstance(photometry, dict):
        photometry = [photometry]
    if ax is None:
        _, (ax, ax_res) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
    defaults = [COLOURS['sorce'], COLOURS['tsis'], '#2a9d8f', '#e76f51']
    for k, ph in enumerate(photometry):
        if ph['n'] == 0:
            continue
        c = (colours or {}).get(ph['name'], defaults[k % len(defaults)])
        off = ph['offset'] if align else 0.0
        x = _x(ph['t'], t0)
        label = ph['name'] + ('  (res. RMS %.0f ppm, MAD %.0f ppm)' % (ph['rms'], ph['mad']) if stats else '')
        ax.plot(x, ph['obs'] + off, 'o-', color=c, alpha=0.5, label='TSI: ' + label)
        ax.plot(x, ph['sim'] + off, 's-', color=COLOURS['sim'], alpha=0.5, label=sim_label if k == 0 else None)
        if ax_res is not None:
            ax_res.plot(x, ph['res'], 'o', color=c, alpha=0.6)
    ax.axhline(0, color='k', alpha=0.1)
    ax.set_ylabel('TSI variability [ppm]')
    ax.legend(ncols=2)
    if ax_res is not None:
        ax_res.axhline(0, color='k', alpha=0.2)
        ax_res.set_ylabel('Res. [ppm]')
    return ax, ax_res


def plot_rv_comparison(rv, ax=None, ax_res=None, t0=0.0, obs_label='HARPS-N', sim_label='SunSim', stats=True,
                       interventions=True):
    """Daily HARPS-N and simulated values (compare_rv) on ax, residuals on ax_res (optional); vertical lines at the
    interventions. Returns ax, ax_res."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, (ax, ax_res) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'height_ratios': [2, 1]})
    q = rv['quantity']
    unit = '' if q == 'contrast' else ' [m/s]'
    x = _x(rv['t'], t0)
    err = np.where(np.isfinite(rv['obs_err']), rv['obs_err'], 0)
    ax.errorbar(x, rv['obs'], yerr=err, marker='o', linestyle='-', color=COLOURS['harpsn'], alpha=0.3, label=obs_label)
    ax.plot(x, rv['sim'], 's-', color=COLOURS['sim'], alpha=0.9, label=sim_label)
    ax.axhline(0, color='k', alpha=0.1)
    ax.set_ylabel('%s variability%s' % ('RV' if q == 'rv' else q.upper() if q != 'contrast' else 'Contrast', unit))
    ax.legend(ncols=2)
    if stats and rv['n']:
        scale, u = (100, 'cm/s') if q != 'contrast' else (1, '')
        ax.text(0.01, 0.04, 'Res. RMS: %.2f %s\nRes. MAD: %.2f %s' % (rv['rms'] * scale, u, rv['mad'] * scale, u),
                transform=ax.transAxes, color=COLOURS['sim'], fontsize='large',
                bbox=dict(facecolor='white', edgecolor='none', pad=3))
    axes = [ax] + ([ax_res] if ax_res is not None else [])
    if ax_res is not None:
        ax_res.plot(x, rv['res'], 'o', color=COLOURS['res'], alpha=0.7)
        ax_res.axhline(0, color='k', alpha=0.2)
        ax_res.set_ylabel('Res.%s' % unit)
    if interventions:
        for a in axes:
            for c in rv['interventions']:
                a.axvline(c - t0, color='k', alpha=0.2, linewidth=1)
    return ax, ax_res


def plot_comparison(photometry=None, rv=None, start=None, t0=0.0, figsize=(15, 9), save=None, **kw):
    """Figure of the comparison: TSI and its residuals (photometry: compare_photometry, one dict or a list) and RV (or
    another quantity) and its residuals (rv: compare_rv, one dict or a list), panels sharing the time axis.
    start: the date of t = 0, written in the label of the time axis. save: file name. Returns fig, axes."""
    import matplotlib.pyplot as plt
    blocks = []
    if photometry is not None and len(photometry if not isinstance(photometry, dict) else [photometry]):
        blocks.append(('ph', photometry))
    for r in ([] if rv is None else [rv] if isinstance(rv, dict) else rv):
        blocks.append(('rv', r))
    if not blocks:
        raise ValueError('nothing to plot: give photometry and/or rv')
    fig, axes = plt.subplots(2 * len(blocks), 1, sharex=True, figsize=figsize,
                             gridspec_kw={'hspace': 0, 'height_ratios': [2, 1] * len(blocks)})
    for k, (kind, data) in enumerate(blocks):
        a, ar = axes[2 * k], axes[2 * k + 1]
        if kind == 'ph':
            plot_photometry_comparison(data, a, ar, t0=t0, **kw)
        else:
            plot_rv_comparison(data, a, ar, t0=t0, **kw)
    axes[-1].set_xlabel('Time [days since %s]' % start if start is not None else 'Time [days]')
    fig.align_ylabels(axes)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')
    return fig, axes


def plot_correlation(rv, ax=None, label=None):
    """Simulated against observed values (compare_rv) with the 1:1 line. Returns ax."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rv['obs'], rv['sim'], 'o', color=COLOURS['sim'], alpha=0.5)
    lo = np.nanmin([rv['obs'].min(), rv['sim'].min()]) if rv['n'] else 0
    hi = np.nanmax([rv['obs'].max(), rv['sim'].max()]) if rv['n'] else 1
    ax.plot([lo, hi], [lo, hi], 'k-', alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    unit = '' if rv['quantity'] == 'contrast' else ' [m/s]'
    ax.set_xlabel('%s observed%s' % (label or rv['quantity'], unit))
    ax.set_ylabel('%s simulated%s' % (label or rv['quantity'], unit))
    if rv['n'] > 2:
        ax.set_title('Pearson r = %.3f' % np.corrcoef(rv['obs'], rv['sim'])[0, 1])
    return ax


# ----------------------------------------------------------------------------------------------------------------------
# parameter sweeps: statistics of every set, table of all the sets, figures (as the optimisation of the thresholds)
# ----------------------------------------------------------------------------------------------------------------------

def _read_series(folder):
    """The time series written by sdo.simulate_sdo_sets in folder: {'time', 'ff_sp', 'ff_fc', 'lc', 'rv', ...}."""
    import os
    ff = np.loadtxt(os.path.join(folder, 'ff.txt'), ndmin=2)
    out = {'time': ff[:, 0], 'ff_sp': ff[:, 1], 'ff_fc': ff[:, 2]}
    for k in ('lc', 'rv', 'fwhm', 'bis', 'contrast'):
        path = os.path.join(folder, '%s.txt' % k)
        if os.path.exists(path):
            out[k] = np.loadtxt(path, ndmin=2)[:, 1]
    return out


def sweep_stats(folder, start, harps=None, sorce=None, tsis=None, interventions=None, window=None, max_ff_fc=20.0,
                max_ff_sp=5.0, shifts=(-1, 0, 1), params=None, save=True):
    """MAD and RMS of the residuals of one simulated set (a folder of simulate_sdo_sets) against the observations given
    (load_harpsn, load_sorce, load_tsis; None: not compared): RV, FWHM and BIS (compare_rv: one offset per instrumental
    period of interventions, best day shift among shifts) and the TSI (compare_photometry).
    window    : (first, last) dates: only the epochs in this window (e.g. ('2021-01-01', '2024-01-01'))
    max_ff_fc, max_ff_sp : the epochs with more faculae / spots [% of the disc] are left out (bad HMI images)
    params    : dict saved with the statistics (the thresholds of the set, T_sp, ...)
    save      : write folder/stats.json
    Returns a dict: the params, 'n_epochs', 'n_removed', and for every observable '<q>_rms', '<q>_mad', '<q>_n',
    '<q>_obs_rms' (q = rv, fwhm, bis, sorce, tsis; [m/s] or [ppm])."""
    import json
    import os
    s = _read_series(folder)
    keep = np.ones(len(s['time']), bool)
    if window is not None:
        keep &= (s['time'] >= jd_of_date(window[0]) - jd_of_date(start)) & \
                (s['time'] < jd_of_date(window[1]) - jd_of_date(start) + 1)
    bad = (s['ff_fc'] > max_ff_fc) | (s['ff_sp'] > max_ff_sp)
    out = dict(params or {})
    out.update(n_epochs=int(np.sum(keep & ~bad)), n_removed=int(np.sum(keep & bad)))
    keep &= ~bad
    t = s['time'][keep]
    if harps is not None:
        for q in ('rv', 'fwhm', 'bis'):
            if q in s:
                r = compare_rv(harps, t, s[q][keep], quantity=q, interventions=interventions, shifts=shifts)
                out.update({'%s_rms' % q: r['rms'], '%s_mad' % q: r['mad'], '%s_n' % q: int(r['n']),
                            '%s_obs_rms' % q: r['rms_obs']})
    for name, tsi in (('sorce', sorce), ('tsis', tsis)):
        if tsi is not None and 'lc' in s:
            ph = compare_photometry(tsi, t, s['lc'][keep])
            out.update({'%s_rms' % name: ph['rms'], '%s_mad' % name: ph['mad'], '%s_n' % name: int(ph['n']),
                        '%s_obs_rms' % name: ph['rms_obs']})
    out = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
    if save:
        with open(os.path.join(folder, 'stats.json'), 'w') as f:
            json.dump(out, f, indent=1)
    return out


def load_sweep_stats(base):
    """Table (pandas DataFrame) of the stats.json of every set under base (base/*/stats.json, or base/sets/*/...)."""
    import glob
    import json
    import os
    import pandas as pd
    files = sorted(glob.glob(os.path.join(str(base), '*', 'stats.json')) +
                   glob.glob(os.path.join(str(base), 'sets', '*', 'stats.json')))
    if not files:
        raise FileNotFoundError('no stats.json under %s' % base)
    rows = []
    for f in files:
        d = json.load(open(f))
        d['folder'] = os.path.basename(os.path.dirname(f))
        rows.append(d)
    return pd.DataFrame(rows)


def best_sets(df, value='rv_rms', n=5):
    """The n sets with the smallest value (a column of load_sweep_stats)."""
    return df.dropna(subset=[value]).sort_values(value).head(n)


def _draw_outline(ax, mask, **kw):
    """Outline of the True cells of a 2D mask (imshow cells centred on integers)."""
    ny, nx = mask.shape
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x]:
                continue
            if x == 0 or not mask[y, x - 1]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], **kw)
            if x == nx - 1 or not mask[y, x + 1]:
                ax.plot([x + 0.5, x + 0.5], [y - 0.5, y + 0.5], **kw)
            if y == 0 or not mask[y - 1, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], **kw)
            if y == ny - 1 or not mask[y + 1, x]:
                ax.plot([x - 0.5, x + 0.5], [y + 0.5, y + 0.5], **kw)


def plot_sweep_heatmaps(df, value='rv_rms', x='A_th', y='I_th', panel='T_sp', fixed=None, levels=None, title=None,
                        unit=None, cmap=None, save=None):
    """One heatmap of value (y against x) per value of panel, as the figures of the optimisation of the thresholds;
    the other parameters (fixed: {name: value}; otherwise the minimum over them) are reduced by the minimum.
    levels : values whose regions (value < level) are outlined in red (solid, dashed, dotted from the smallest).
    Returns fig, axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = df.copy()
    for k, v in (fixed or {}).items():
        d = d[np.isclose(d[k], v)]
    if cmap is None:
        try:
            import cmasher as cmr
            cmap = cmr.torch
        except ImportError:
            cmap = 'inferno'
    panels = sorted(d[panel].unique()) if panel else [None]
    xs, ys = sorted(d[x].unique()), sorted(d[y].unique())
    vmin, vmax = d[value].min(), d[value].max()
    fig, axes = plt.subplots(1, len(panels), figsize=(max(4.5 * len(panels), 6), 4.5), sharex=True, sharey=True,
                             squeeze=False, gridspec_kw={'wspace': 0.05})
    axes = axes[0]
    styles = [('-', 3), ('--', 2.5), (':', 2.5)]
    for ax, p in zip(axes, panels):
        sub = d if p is None else d[d[panel] == p]
        grid = sub.groupby([y, x])[value].min().unstack(x).reindex(index=ys, columns=xs).values
        im = ax.imshow(grid, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.95)
        for lev, (ls, lw) in zip(sorted(levels or [])[:3], styles):
            _draw_outline(ax, np.nan_to_num(grid, nan=np.inf) < lev, color='red', linestyle=ls, linewidth=lw)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(['%g' % v for v in xs], rotation=45)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(['%g' % v for v in ys])
        ax.set_xlabel(x)
        if p is not None:
            ax.set_title('%s = %g' % (panel, p))
    axes[0].set_ylabel(y)
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.01)
    cbar.set_label('%s%s' % (value, ' [%s]' % unit if unit else ''))
    if levels:
        axes[-1].legend(handles=[Line2D([0], [0], color='red', linestyle=ls, lw=lw, label='< %g%s' % (lev, ' ' + unit if unit else ''))
                                 for lev, (ls, lw) in zip(sorted(levels)[:3], styles)], fontsize=8)
    if title:
        fig.suptitle(title, y=1.02)
    if save:
        fig.savefig(save, bbox_inches='tight')
    return fig, axes


def plot_sweep_parameters(df, value='rv_rms', parameters=('T_sp', 'noise_th', 'I_th', 'A_th'), unit=None, save=None):
    """value against every parameter (all the sets, one point each) with the minimum as a dashed line."""
    import matplotlib.pyplot as plt
    parameters = [p for p in parameters if p in df and df[p].nunique() > 0]
    n = len(parameters)
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots((n + 1) // cols, cols, figsize=(6.5 * cols, 4.5 * ((n + 1) // cols)), squeeze=False)
    axes = axes.ravel()
    for ax, p in zip(axes, parameters):
        ax.plot(df[p], df[value], 'o', alpha=0.35, markersize=3)
        ax.axhline(df[value].min(), ls='--', alpha=0.7)
        ax.set_xlabel(p)
        ax.set_ylabel('%s%s' % (value, ' [%s]' % unit if unit else ''))
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')
    return fig, axes


def _main(argv=None):
    """python -m sunsim.observations stats|collect ...  (see --help)"""
    import argparse
    import os
    p = argparse.ArgumentParser(prog='python -m sunsim.observations', description=(
        'stats: MAD and RMS of a simulated set (folder of sunsim-sdo-maps --conf) against HARPS-N, SORCE, TSIS-1 '
        '(stats.json); collect: table of all the sets (sweep_stats.csv) and the figures of the sweep'))
    sub = p.add_subparsers(dest='what', required=True)
    s = sub.add_parser('stats')
    s.add_argument('folder')
    s.add_argument('--start', required=True, help='date of t = 0 of the simulation (the --start of sunsim-sdo-maps)')
    s.add_argument('--harpsn'), s.add_argument('--sorce'), s.add_argument('--tsis')
    s.add_argument('--interventions', help='HARPS-N interventions (one offset per instrumental period)')
    s.add_argument('--window', nargs=2, default=None, help='first and last dates of the epochs compared')
    s.add_argument('--param', nargs='*', default=[], help='key=value saved with the statistics (e.g. T_sp=300)')
    c = sub.add_parser('collect')
    c.add_argument('base', help='folder of the sets')
    c.add_argument('--out', default=None, help='folder of the table and the figures (default base)')
    c.add_argument('--rv-levels', type=float, nargs='*', default=[0.85, 0.90, 0.95])
    c.add_argument('--tsi-levels', type=float, nargs='*', default=[130, 150, 200])
    a = p.parse_args(argv)
    if a.what == 'stats':
        end = None
        harps = load_harpsn(a.harpsn, a.start, end=end) if a.harpsn else None
        sorce = load_sorce(a.sorce, a.start) if a.sorce else None
        tsis = load_tsis(a.tsis, a.start) if a.tsis else None
        cuts = load_interventions(a.interventions, a.start) if a.interventions else None
        params = {}
        for kv in a.param:
            k, v = kv.split('=', 1)
            try:
                v = float(v)
            except ValueError:
                pass
            params[k] = v
        r = sweep_stats(a.folder, a.start, harps, sorce, tsis, cuts, window=a.window, params=params)
        print(' '.join('%s=%.4g' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v) for k, v in r.items()))
        return 0
    import matplotlib
    matplotlib.use('Agg')
    out = a.out or a.base
    os.makedirs(out, exist_ok=True)
    df = load_sweep_stats(a.base)
    df.to_csv(os.path.join(out, 'sweep_stats.csv'), index=False)
    print('%d sets -> %s' % (len(df), os.path.join(out, 'sweep_stats.csv')))
    for value, unit, levels, title in (('rv_rms', 'm/s', a.rv_levels, 'RV - HARPS-N'),
                                       ('sorce_rms', 'ppm', a.tsi_levels, 'TSI - SORCE/TIM'),
                                       ('tsis_rms', 'ppm', a.tsi_levels, 'TSI - TSIS-1/TIM')):
        if value not in df or df[value].isna().all():
            continue
        print('best %s: %s' % (value, best_sets(df, value, 1).iloc[0].to_dict()))
        for noise in sorted(df['noise_th'].unique()) if 'noise_th' in df else [None]:
            fixed = {} if noise is None else {'noise_th': noise}
            tag = '' if noise is None else '_noise%g' % noise
            plot_sweep_heatmaps(df, value, 'A_th', 'I_th', 'T_sp', fixed=fixed, levels=levels, unit=unit, title=title,
                                save=os.path.join(out, '%s_I_A%s.pdf' % (value, tag)))
            plot_sweep_heatmaps(df, value, 'T_sp', 'I_th', 'A_th', fixed=fixed, unit=unit, title=title,
                                save=os.path.join(out, '%s_I_T%s.pdf' % (value, tag)))
        plot_sweep_parameters(df, value, unit=unit, save=os.path.join(out, '%s_parameters.pdf' % value))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_main())
