#!/usr/bin/env python
"""
Disc filling factors of the spots and faculae of SATIRE-S against those of the sets of a sweep, on the days with a
SATIRE mask (2017-07/08 and 2021-04/07), as satire_map.ipynb (starsim codes+tests):

  python satire_ff.py satire --masks <folder of *_TAI_bmu_mask.sav> --out satire_ff.csv
      one row per mask: t_rec, date, n_disc, ff_umb, ff_pen, ff_sp, and ff_fc_<B_sat> for every --B-sat [% of the disc]
      (spots: umb_mask + pen_mask; faculae: min(fac_bmu_mask / B_sat, 1), fac_bmu_mask being |B| / mu of the facular
      pixels). Run on the computer with the masks (~200 MB each), then copy the table to PIC (SATIRE_FF of
      settings_noise.sh).
  python satire_ff.py compare --ident <folder of the sets> --satire satire_ff.csv --out satire_ff_stats.csv
      every <set>/summary.csv (and parameters.txt) of --ident against SATIRE, the same days (date of the HMI images
      and of the SATIRE masks): n, mean of the set and of SATIRE, ratio of the means, RMS of the difference and
      correlation, for the spots (ff_sp) and the faculae (ff_fc, against SATIRE with --B-sat-ref; for the sets with
      faculae = fraction also against SATIRE with the B_sat of the set, *_same). The epochs left out of the statistics
      of the RV and TSI (ff_fc > 20 %, ff_sp > 5 %: bad HMI images) are left out here too.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

MAX_FF_FC, MAX_FF_SP = 20.0, 5.0                 # as sunsim.observations.sweep_stats


def satire_table(masks, B_sats):
    """One row per SATIRE mask of the folder: t_rec, date, n_disc, ff_umb, ff_pen, ff_sp, ff_fc_<B_sat> [%]."""
    import scipy.io as sio
    rows = []
    files = sorted(glob.glob(os.path.join(masks, '*_TAI_bmu_mask.sav')))
    for k, file in enumerate(files, 1):
        t_rec = os.path.basename(file).replace('_bmu_mask.sav', '')
        d = sio.readsav(file)
        ny, nx = d.fac_bmu_mask.shape
        yy, xx = np.ogrid[:ny, :nx]
        disc = (xx - d.x0) ** 2 + (yy - d.y0) ** 2 <= d.rs ** 2
        n_disc = np.sum(disc)
        umb, pen = d.umb_mask[disc].astype(float), d.pen_mask[disc].astype(float)
        bmu = d.fac_bmu_mask[disc]
        row = dict(t_rec=t_rec, date=t_rec[:10].replace('.', '-'), n_disc=n_disc, ff_umb=100 * np.sum(umb) / n_disc,
                   ff_pen=100 * np.sum(pen) / n_disc, ff_sp=100 * np.sum(umb + pen) / n_disc)
        for b in B_sats:
            row['ff_fc_%g' % b] = 100 * np.sum(np.minimum(bmu / b, 1)) / n_disc
        rows.append(row)
        print('[%d/%d] %s: ff_sp %.4f %%, ff_fc %s' % (k, len(files), t_rec, row['ff_sp'],
              ', '.join('%.3f %% (%g G)' % (row['ff_fc_%g' % b], b) for b in B_sats)), flush=True)
    return pd.DataFrame(rows)


def read_parameters(path):
    """The thresholds of parameters.txt of a set (sunsim.sdo): 'key = value' lines."""
    p = {}
    for line in open(path):
        if '=' in line:
            k, v = (s.strip() for s in line.split('=', 1))
            v = v.split()[0] if v else v
            try:
                p[k] = float(v)
            except ValueError:
                p[k] = v
    return p


def differences(sim, sat):
    """n, means, ratio of the means, RMS of sim - sat and correlation of two arrays of the same days."""
    ok = np.isfinite(sim) & np.isfinite(sat)
    sim, sat = sim[ok], sat[ok]
    if len(sim) < 2:
        return dict(n=len(sim), mean=np.nan, mean_satire=np.nan, ratio=np.nan, rms=np.nan, corr=np.nan)
    return dict(n=len(sim), mean=np.mean(sim), mean_satire=np.mean(sat), ratio=np.mean(sim) / np.mean(sat),
                rms=np.sqrt(np.mean((sim - sat) ** 2)), corr=np.corrcoef(sim, sat)[0, 1])


def compare(ident, satire, B_sat_ref):
    """One row per set of ident (folders with summary.csv and parameters.txt) against the SATIRE table."""
    sat = satire.drop_duplicates('date').set_index('date')
    ref = 'ff_fc_%g' % B_sat_ref
    if ref not in sat:
        raise ValueError('no column %s in the SATIRE table (columns: %s)' % (ref, ', '.join(sat.columns)))
    rows = []
    for summary in sorted(glob.glob(os.path.join(ident, '*', 'summary.csv'))):
        folder = os.path.dirname(summary)
        s = pd.read_csv(summary)
        s = s[(s['ff_fc'] <= MAX_FF_FC) & (s['ff_sp'] <= MAX_FF_SP)]
        s = s.assign(date=s['date'].astype(str).str[:10]).drop_duplicates('date').set_index('date')
        days = s.index.intersection(sat.index)
        params = read_parameters(os.path.join(folder, 'parameters.txt'))
        row = dict(set=os.path.basename(folder), **{k: params.get(k) for k in
                                                    ('noise_th', 'B_th', 'I_th', 'A_th', 'faculae', 'B_sat')})
        if params.get('faculae') != 'fraction':
            row['B_sat'] = np.nan
        else:
            row['A_th'] = np.nan
        for q, sat_col in (('sp', 'ff_sp'), ('fc', ref)):
            d = differences(s.loc[days, 'ff_' + q].values, sat.loc[days, sat_col].values)
            row.update({'%s_%s' % (k, q): v for k, v in d.items()})
        same = 'ff_fc_%g' % row['B_sat'] if np.isfinite(row['B_sat']) else None
        if same in sat:
            d = differences(s.loc[days, 'ff_fc'].values, sat.loc[days, same].values)
            row.update({'%s_fc_same' % k: v for k, v in d.items()})
        rows.append(row)
    if not rows:
        raise FileNotFoundError('no <set>/summary.csv under %s' % ident)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='what', required=True)
    s = sub.add_parser('satire', help='the filling factors of the SATIRE masks')
    s.add_argument('--masks', required=True, help='folder of the SATIRE masks (<t_rec>_bmu_mask.sav)')
    s.add_argument('--out', required=True, help='csv file')
    s.add_argument('--B-sat', type=float, nargs='+', default=[25, 50, 100, 200, 300, 305, 500],
                   help='|B| / mu of a pixel fully covered by faculae [G], one column each')
    c = sub.add_parser('compare', help='the sets of a sweep against SATIRE')
    c.add_argument('--ident', required=True, help='folder of the sets (<set>/summary.csv, <set>/parameters.txt)')
    c.add_argument('--satire', required=True, help='csv file of satire_ff.py satire')
    c.add_argument('--out', required=True, help='csv file')
    c.add_argument('--B-sat-ref', type=float, default=305, help='B_sat of SATIRE for ff_fc (default 305 G)')
    a = p.parse_args(argv)
    if a.what == 'satire':
        df = satire_table(a.masks, a.B_sat)
    else:
        df = compare(a.ident, pd.read_csv(a.satire), a.B_sat_ref)
        cols = ['set', 'rms_sp', 'ratio_sp', 'rms_fc', 'ratio_fc', 'corr_fc'] + (['rms_fc_same'] if 'rms_fc_same' in df else [])
        print(df.sort_values('rms_fc')[cols].to_string(index=False, float_format='%.4g'))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print('%d rows -> %s' % (len(df), a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
