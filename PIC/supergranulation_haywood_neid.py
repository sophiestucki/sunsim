#!/usr/bin/env python
"""
SunSim at the times of the NEID exposures (solar feed of WIYN, Kitt Peak), with the two steps of
supergranulation_haywood.py (not changed: its day and collect steps are run with the NEID exposures in place of the
HARPS-N ones, the JPL velocity of the Sun at Kitt Peak and the NEID quality cut; all its options are taken):

  python supergranulation_haywood_neid.py day --neid FILE --date 2021-05-25 --out DIR [options of its day step]
  python supergranulation_haywood_neid.py collect --out DIR --start DATE --conf CONF --diff-extinction
         [--min-snr-ratio 0.98] [--min-day-exp 50] [options of its collect step]

FILE: csv of the solarl2 table of the NEID archive (TAP table neidsolarl2 or the csv of the web interface), or a folder
of <YYYY-MM-DD>.csv (observations.load_neid).

The NEID RV as the HARPS-N RV (rv_raw - v_sun (JPL) - differential extinction, observations.load_neid): ccfrvmod is
already in the rest frame of the Sun (SSBRV of the L2 headers = -(range rate of the Sun from JPL Horizons at Kitt Peak)
- GRAV_NEID); collect is given rv_raw = ccfrvmod + GRAV_NEID + v_sun (JPL at Kitt Peak, day step), the RV in the
barycentric frame, so that its rv_raw - v_sun is ccfrvmod + GRAV_NEID, and rv_diff_extinction of load_neid
(--diff-extinction removes it). Quality (clouds, as obs_quality of HARPS-N): flagged = 0 and snr_ratio >=
--min-snr-ratio (extsnr / the highest within +- 10 min); then the days with fewer than --min-day-exp exposures left.
The day step takes all the exposures (the cuts are made in collect).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supergranulation_haywood as sgh  # noqa: E402

GRAV_NEID = 631.78                   # [m/s] -(SSBRV + range rate of JPL) of the L2 headers of 2021-05-25
NEID_KEYS = ('extsnr', 'snr_ratio', 'flagged')


def load_neid_exposures(path, date, end=None):
    """observations.load_neid as load_harpsn(daily=False) for the day step: 'rv' = ccfrvmod (the differential extinction
    in 'rv_diff_extinction'), fwhm, bis, contrast and the other keys of the release NaN."""
    from sunsim import observations as obs
    n = obs.load_neid(path, date, end)
    out = {k: n[k] for k in ('jd', 'jd_start', 'texp', 'rv_err', 'rv_diff_extinction') + NEID_KEYS}
    out['rv'] = n['rv_drs']
    nan = np.full(len(n['jd']), np.nan)
    for k in ('fwhm', 'bis', 'contrast', 'fwhm_err', 'bis_err', 'contrast_err', 'rv_raw', 'berv', 'berv_bary_to_helio'):
        out[k] = nan
    return out


def _cut(ex, keep, min_day_exp):
    """ex with the exposures keep, then without the days with fewer than min_day_exp exposures left."""
    days, n_left = np.unique(ex['day'][keep], return_counts=True)
    few = np.isin(ex['day'], days[n_left < min_day_exp]) & keep
    return {k: v[keep & ~few] for k, v in ex.items()}, few.sum()


def merge_days(a, b):
    """The products of load_days of two folders together: exposures and epochs one after the other (the epochs of b
    renumbered), the 45 s frames of a day of both folders together (each frame once)."""
    ex_a, ep_a, fr_a, win_a = a
    ex_b, ep_b, fr_b, win_b = b
    ex_b = dict(ex_b, epoch=ex_b['epoch'] + len(ep_a['jd']))
    ex = {k: np.concatenate([ex_a[k], ex_b[k]]) for k in set(ex_a) & set(ex_b)}
    ep = {k: np.concatenate([ep_a[k], ep_b[k]]) for k in ep_a}
    frames = dict(fr_a)
    for d, f in fr_b.items():
        if d in frames:
            jd = np.concatenate([frames[d][0], f[0]])
            _, i = np.unique(jd, return_index=True)
            info = frames[d][4] + f[4]
            frames[d] = tuple(np.concatenate([x, y])[i] for x, y in zip(frames[d][:4], f[:4])) + ([info[k] for k in i],)
        else:
            frames[d] = f
    return ex, ep, frames, win_a | win_b


def run(step, argv, neid=None, min_snr_ratio=0.98, min_day_exp=50, harpsn_out=None, harpsn=None, min_obs_quality=0.99):
    """supergranulation_haywood.py <step> argv with the NEID exposures (day: file or folder neid). collect with
    harpsn_out (the folder of supergranulation_haywood.py): its days too, with the cuts of the release harpsn
    (data_quality = 1, obs_quality >= min_obs_quality, days with >= min_day_exp exposures), so that the CLV drift
    (--clv-drift obs_vr: OBS_VR, harmonics of the orbit, one offset per day) is fitted on the same exposures as the HARPS-N
    run and the NEID ones (on one day it absorbs the supergranulation: r of HARPS-N 0.35 on 2021-05-25, 0.92 with the 26
    days). exposures.csv then has both: the NEID exposures are those of load_neid (jd_utc)."""
    from sunsim import observations as obs
    saved = (obs.load_harpsn, obs.sun_barycentric_rv, sgh.HARPSN_KEYS, sgh.load_days)
    load_days, sun_rv = sgh.load_days, obs.sun_barycentric_rv

    def neid_days(out, n_rings):
        ex, ep, frames, windows = load_days(out, n_rings)
        keep = (ex['flagged'] == 0) & (ex['snr_ratio'] >= min_snr_ratio)
        n_exp, n_flagged, n_snr = len(keep), np.sum(ex['flagged'] != 0), np.sum(ex['snr_ratio'] < min_snr_ratio)
        ex, few = _cut(ex, keep, min_day_exp)
        sgh._log('NEID quality: %d of %d exposures left out (flagged: %d, snr_ratio < %.3f: %d; days with fewer than %d '
                 'exposures left: %d)' % (n_exp - len(ex['jd']), n_exp, n_flagged, min_snr_ratio, n_snr, min_day_exp, few))
        ex['rv_raw'] = ex['rv'] + GRAV_NEID + ex['v_sun']
        if not harpsn_out:
            return ex, ep, frames, windows
        exh, eph, frh, winh = load_days(harpsn_out, n_rings)
        qf = sgh.exposure_quality(exh, harpsn)
        n_h = len(exh['jd'])
        exh, few = _cut(exh, (qf['data_quality'] == 1) & (qf['obs_quality'] >= min_obs_quality), min_day_exp)
        sgh._log('HARPS-N days of %s with them: %d of %d exposures (data_quality = 1, obs_quality >= %s, days with >= %d '
                 'exposures), %d days' % (harpsn_out, len(exh['jd']), n_h, min_obs_quality, min_day_exp,
                                          len(np.unique(exh['day']))))
        return merge_days((exh, eph, frh, winh), (ex, ep, frames, windows))

    obs.load_harpsn = lambda path, date, end=None, daily=False, quality=False: load_neid_exposures(path, date, end)
    obs.sun_barycentric_rv = lambda jd_utc, site=None: sun_rv(jd_utc, site=obs.KPNO_SITE)
    sgh.HARPSN_KEYS = tuple(saved[2]) + NEID_KEYS
    sgh.load_days = neid_days
    try:
        return sgh.main([step] + argv + (['--harpsn', neid] if step == 'day' else []))
    finally:
        obs.load_harpsn, obs.sun_barycentric_rv, sgh.HARPSN_KEYS, sgh.load_days = saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], add_help=False)
    p.add_argument('step', choices=('day', 'collect'))
    p.add_argument('--neid', default=None, help='day: csv of the NEID archive (solarl2) or a folder of <YYYY-MM-DD>.csv')
    p.add_argument('--min-snr-ratio', type=float, default=0.98,
                   help='collect: exposures with extsnr >= this x the highest extsnr within +- 10 min')
    p.add_argument('--min-day-exp', type=int, default=50, help='collect: days with fewer exposures left are left out')
    p.add_argument('--harpsn-out', default=None,
                   help='collect: folder of supergranulation_haywood.py, its days with the NEID ones (the CLV drift fit)')
    p.add_argument('--harpsn', default=None, help='collect with --harpsn-out: csv of the release (its quality flags)')
    p.add_argument('--min-obs-quality', type=float, default=0.99, help='collect with --harpsn-out: obs_quality >= this')
    a, rest = p.parse_known_args(argv)
    if a.step == 'day' and not a.neid:
        p.error('day needs --neid')
    if a.harpsn_out and not a.harpsn:
        p.error('--harpsn-out needs --harpsn')
    return run(a.step, rest, a.neid, a.min_snr_ratio, a.min_day_exp, a.harpsn_out, a.harpsn, a.min_obs_quality)


if __name__ == '__main__':
    sys.exit(main())
