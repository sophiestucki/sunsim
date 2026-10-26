#!/usr/bin/env python
"""
The spot bisectors of PIC/bisectors_components.txt (mu_0_0, mu_1_0, palumbo: the Dumusque bisector times the
polynomial in mu fitted on Palumbo et al. 2024, spot_dumusque_palumbo.py --pipeline) against HARPS-N.

The cell filling factors of the old identification (noise_th 8, I_th 0.95, A_th 20 at 4096 px: the identification of
the 10-year run), N = 30 rings, of sdo_old_vs_new_harpsn.py (FF: one day a week, 2015-07-18 to 2023-01-25) are
simulated with the configuration of the component pipeline (write_conf of PIC/settings.sh, Temp, N rings, the spot
bisector of every row) and the local MURaM spectra and mask. The RV, FWHM, BIS and contrast of every simulated CCF are
measured as in CAUP/ccf_yarara_harpn.ipynb (gaussian fit on the window of the median CCF) and compared with the HARPS-N
DRS, with the CCFs of the YARARA spectra and with the CCFs of the 10-year SunSim run (yarara_ccf_parameters.csv and
sunsim_ccf_parameters.csv of that notebook), on the same days (sunsim.observations.compare_rv: medians removed, one
offset per instrumental period, best day shift -1, 0, 1 for the simulations).

    python spot_dumusque_palumbo_harpsn.py [--bisectors mu_0_0 mu_1_0 palumbo] [--temp 300] [--out folder]
"""
import argparse
import json
import os
import shutil
import sys

os.environ.setdefault('KMP_WARNINGS', '0')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
PIC = os.path.join(REPO, 'PIC')
START = '2015-07-18'                                                         # t = 0
N = 30
FF = os.path.join(HERE, 'results', 'sdo_old_vs_new_harpsn', 'old')         # filling_factors_N30.npz, summary.csv
RV_GRID = np.arange(-18000, 18000 + 250, 250.0)                              # [m/s] ccf_rv_range 18, ccf_rv_step 0.25
VSINI = 1000 * 2 * np.pi * 696342 / (25.05 * 86400)                          # [m/s] as ccf_yarara_harpn.ipynb
QUANTITIES = (('rv', 'RV [m/s]'), ('fwhm', 'FWHM [m/s]'), ('bis', 'BIS [m/s]'), ('contrast', 'contrast [%]'))


def bisector_row(name):
    """remove_bis_spot, convective_shift, dumusque_mu_scaling of the row name of bisectors_components.txt."""
    for line in open(os.path.join(PIC, 'bisectors_components.txt')):
        f = line.split()
        if len(f) >= 4 and not line.startswith('#') and f[0] == name:
            return f[1], f[2], '' if f[3] == '-' else f[3]
    raise ValueError('no bisector %r in PIC/bisectors_components.txt' % name)


def conf_for(path, temp, name):
    """write_conf of PIC/settings.sh with the bisector name (spot_dumusque_palumbo.pipeline_conf), the local spectra
    and mask in place of the PIC ones."""
    from sunsim import local_path
    from spot_dumusque_palumbo import pipeline_conf
    pipeline_conf(path, temp, N, *bisector_row(name))
    local = {'mps_spectra_folder': str(local_path('muram_spectra')) + '/',
             'ccf_mask': str(local_path('ccf_masks') / 'HARPS_G20_mps_friendly.mas')}
    lines = []
    for line in open(path):
        key = line.split(':')[0].strip()
        lines.append('%s : %s\n' % (key, local[key]) if key in local else line)
    open(path, 'w').writelines(lines)
    return path


def ccf_parameters(ccfs):
    """rv, fwhm, bis [m/s], contrast [%] of every CCF and the failed fits (ccf_parameters of ccf_yarara_harpn.ipynb)."""
    import pandas as pd
    from sunsim import nbspectra
    ref = np.median(ccfs, axis=0)
    c0 = ref - ref.min() + 1e-6
    cutleft, cutright, _, _ = nbspectra.speed_bisector_nb(RV_GRID, c0 / c0.max(), True)
    rv, contrast, fwhm, bis, _, _, failed = nbspectra.ccf_params_nb(RV_GRID, np.ascontiguousarray(ccfs), int(cutleft),
                                                                    int(cutright), VSINI, True)
    return pd.DataFrame({'rv': rv, 'fwhm': fwhm, 'bis': bis, 'contrast': 100 * contrast}), failed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--bisectors', nargs='+', default=['mu_0_0', 'mu_1_0', 'palumbo'])
    p.add_argument('--temp', type=int, default=300, help='spot_T_contrast [K] (300: the 10-year run)')
    p.add_argument('--out', default=os.path.join(HERE, 'results', 'spot_dumusque_palumbo_harpsn'))
    a = p.parse_args(argv)
    import pandas as pd
    from sunsim import sdo, local_path, observations as obs
    os.makedirs(a.out, exist_ok=True)

    # 1. simulations: one folder per bisector with the filling factors of FF
    sims = {}
    for b in a.bisectors:
        folder = os.path.join(a.out, '%s_T_%d' % (b, a.temp))
        os.makedirs(folder, exist_ok=True)
        for f in ('filling_factors_N%d.npz' % N, 'summary.csv'):
            if not os.path.exists(os.path.join(folder, f)):
                shutil.copy(os.path.join(FF, f), folder)
        conf = conf_for(os.path.join(a.out, '%s_T_%d.conf' % (b, a.temp)), a.temp, b)
        r = sdo.simulate_sdo_sets(conf, folder, grids_cache=os.path.join(a.out, 'grids_%s_T_%d.pkl' % (b, a.temp)),
                                  chunk_size=200)[folder]
        assert np.allclose(r['ccf_rv'], RV_GRID / 1000) or np.allclose(r['ccf_rv'], RV_GRID)
        par, failed = ccf_parameters(np.asarray(r['ccf']))
        sims[b] = pd.concat([pd.DataFrame({'t': r['time'], 'ff_sp': r['ff_sp'], 'ff_fc': r['ff_fc']}), par],
                            axis=1)[~failed]
        sims[b].to_csv(os.path.join(folder, 'ccf_parameters.csv'), index=False)
        print('%s: %d days, %d failed fits' % (b, len(r['time']), failed.sum()), flush=True)

    # 2. the same days for all: no bad HMI image (ff_fc > 20 %), in the 10-year run
    yarara_dir = local_path('poet_data') / 'output' / 'yarara_harpn_all_corr_except_activity_ccf'
    yarara = pd.read_csv(yarara_dir / 'yarara_ccf_parameters.csv')
    ten = pd.read_csv(yarara_dir / 'sunsim_ccf_parameters.csv')
    days = set(ten['t'])
    for s in sims.values():
        days &= set(s['t'][s['ff_fc'] <= 20])
    days = np.array(sorted(days))
    series = {'SunSim 10-year run': ten[ten['t'].isin(days)].reset_index(drop=True)}
    series.update({'SunSim %s' % b: s[s['t'].isin(days)].reset_index(drop=True) for b, s in sims.items()})
    print('%d days (%s to %s)' % (len(days), *(str(pd.Timestamp(START) + pd.Timedelta(days=x))[:10]
                                               for x in (days.min(), days.max()))))

    harps = obs.load_harpsn(local_path('harpsn_sun'), start=START)
    cuts = obs.load_interventions(local_path('harpsn_interventions'), START)
    data = {'HARPS-N': harps, 'YARARA': yarara}
    rows, cmp = [], {}
    for q, _ in QUANTITIES:
        for d, tab in data.items():
            for name, s in series.items():
                r = obs.compare_rv(tab, s['t'], s[q], quantity=q, interventions=cuts, shifts=(-1, 0, 1))
                fit = r['obs'] - r['res']
                cmp[d, name, q] = r
                rows.append({'indicator': q, 'data': d, 'simulation': name, 'days': r['n'], 'rms data': r['rms_obs'],
                             'rms sim (+ offsets)': obs.rms(fit), 'rms residuals': r['rms'],
                             'mad residuals': r['mad'], 'pearson r': np.corrcoef(r['obs'], fit)[0, 1],
                             'slope': np.polyfit(r['obs'], fit, 1)[0]})
    summary = pd.DataFrame(rows).set_index(['indicator', 'data', 'simulation'])
    summary.to_csv(os.path.join(a.out, 'comparison_T_%d.csv' % a.temp))
    with pd.option_context('display.float_format', '{:.3f}'.format, 'display.width', 200, 'display.max_columns', None):
        print(summary)

    # 3. figures: time series of the residuals and the simulated indicators against HARPS-N
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = pd.Timestamp(START)
    date = lambda t: t0 + pd.to_timedelta(np.asarray(t), unit='D')
    colours = dict(zip(series, ['0.5'] + ['C%d' % k for k in range(len(sims))]))
    fig, axes = plt.subplots(len(QUANTITIES), 2, figsize=(17, 3.6 * len(QUANTITIES)),
                             gridspec_kw=dict(width_ratios=[3, 1]))
    for ax in axes[1:, 0]:
        ax.sharex(axes[0, 0])
    for k, (q, label) in enumerate(QUANTITIES):
        ax, axc = axes[k]
        r0 = cmp['HARPS-N', 'SunSim 10-year run', q]
        ax.plot(date(r0['t']), r0['obs'], 'k.', ms=3, alpha=0.6, label='HARPS-N DRS: rms %.3g' % r0['rms_obs'])
        for name in series:
            r = cmp['HARPS-N', name, q]
            fit = r['obs'] - r['res']
            ax.plot(date(r['t']), fit, '.-', ms=2.5, lw=0.6, color=colours[name],
                    label='%s: rms res. %.3g, r = %.2f' % (name, r['rms'], np.corrcoef(r['obs'], fit)[0, 1]))
            axc.plot(r['obs'], fit, '.', ms=3, color=colours[name], alpha=0.6)
        lo, hi = np.percentile(r0['obs'], [0.5, 99.5])
        axc.plot([lo, hi], [lo, hi], 'k-', lw=0.8)
        for cut in cuts[(cuts > 0) & (cuts < days.max())]:
            ax.axvline(date(cut), color='0.8', lw=0.6, zorder=0)
        ax.set_ylabel(label + ' (+ offset per interval)')
        axc.set_xlabel('HARPS-N: ' + label)
        axc.set_ylabel('SunSim: ' + label)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
        for c in (ax, axc):
            c.grid(alpha=0.3, lw=0.5)
    axes[0, 0].set_title('spot T contrast %d K, N = %d rings, old identification' % (a.temp, N))
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, 'harpsn_T_%d.png' % a.temp), dpi=130, bbox_inches='tight')
    json.dump({'days': days.tolist(), 'bisectors': {b: bisector_row(b) for b in a.bisectors}},
              open(os.path.join(a.out, 'run_T_%d.json' % a.temp), 'w'), indent=1)
    print('written: %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
