"""
Injection-recovery of planets in the RV residuals HARPS-N - SunSim of several 10-year simulations (rv-X.txt in
local_path('personal_data')), and in the HARPS-N RVs (median of every instrumental period removed), as in
analysis/10_years_injection.ipynb. For every series: <OUT>/<name>.npz (ResidualAnalysis.load), the detection maps,
the ratio to HARPS-N and the recovered semi-amplitudes and periods (pdf), and summary.txt.

    python analysis/10_years_injection_runs.py [X ...]          (default X = 2 3 4)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sunsim import local_path, observations as obs
from sunsim.residuals import ResidualAnalysis

ORIGIN = '2015-07-18'
FIRST, LAST = '2015-07-18', '2024-01-28'
PERIODS = np.linspace(5, 500, 10)
AMPLITUDES = np.linspace(0.1, 0.5, 21)
N_PHASES = 100
FAP = 1e-3
PERIOD_TOL = 0.1
ROTATION = 26
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', '10_years_injection')


def comparison(x, harps, cuts):
    t_s, rv_s = np.loadtxt(os.path.join(local_path('personal_data'), 'rv-%s.txt' % x)).T
    window = (t_s >= obs.jd_of_date(FIRST) - obs.jd_of_date(ORIGIN)) & (t_s < obs.jd_of_date(LAST) - obs.jd_of_date(ORIGIN) + 1)
    return obs.compare_rv(harps, t_s[window], rv_s[window], interventions=cuts)


def detection_maps(ra, name):
    kinds = {'FAP <= %g' % FAP: dict(),
             'and |P - P_inj| / P_inj <= %g' % PERIOD_TOL: dict(period_tol=PERIOD_TOL),
             'at the rotation (%g d) instead' % ROTATION: dict(period_tol=PERIOD_TOL, at_period=ROTATION)}
    fig, axes = plt.subplots(1, len(kinds), figsize=(8 * len(kinds), 6))
    for ax, (title, kw) in zip(axes, kinds.items()):
        ra.plot_detection_map(ra.detection_rate(**kw), ax, title='%s: %s' % (ra.name, title))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'detection_maps_%s.pdf' % name), bbox_inches='tight')
    plt.close(fig)

    axes = ra.plot_recovered()
    axes[0].figure.suptitle(ra.name)
    axes[0].figure.tight_layout()
    axes[0].figure.savefig(os.path.join(OUT, 'recovered_%s.pdf' % name), bbox_inches='tight')
    plt.close(axes[0].figure)


def ratio_map(ra, ref, name):
    rate, rate_ref = ra.detection_rate(period_tol=PERIOD_TOL), ref.detection_rate(period_tol=PERIOD_TOL)
    ratio = np.divide(rate, rate_ref, out=np.full_like(rate, np.nan), where=rate_ref > 0)
    ax = ra.plot_detection_map(ratio, vmin=None, vmax=None, label='Detections: %s / %s' % (ra.name, ref.name),
                               title='|P - P_inj| / P_inj <= %g, FAP <= %g' % (PERIOD_TOL, FAP))
    ax.figure.savefig(os.path.join(OUT, 'ratio_%s.pdf' % name), bbox_inches='tight')
    plt.close(ax.figure)


def summary_line(ra, rv=None):
    pg = ra.periodogram()
    rate = ra.detection_rate(period_tol=PERIOD_TOL)
    i50 = [np.flatnonzero(rate[:, j] >= 50) for j in range(len(PERIODS))]
    k50 = ' '.join('%5s' % ('%.2f' % AMPLITUDES[i[0]] if len(i) else '>%.1f' % AMPLITUDES[-1]) for i in i50)
    head = '%-22s N %4d  std %.3f m/s' % (ra.name, ra.N, ra.y.std())
    if rv is not None:
        head += '  (RMS %.3f, MAD %.3f m/s)' % (rv['rms'], rv['mad'])
    return ('%s\n    GLS peak P = %.2f d, amp %.3f m/s, FAP %.1e; recovered at the period: %.1f %% of all; '
            'rotation instead: %.1f %%\n    K [m/s] with >= 50 %% recovered, per period: %s'
            % (head, pg['P'], pg['amp'], pg['fap'], rate.mean(),
               ra.detection_rate(period_tol=PERIOD_TOL, at_period=ROTATION).mean(), k50))


def main(xs):
    os.makedirs(OUT, exist_ok=True)
    harps = obs.load_harpsn(local_path('harpsn_sun'), ORIGIN, end=LAST, t_start=FIRST)
    cuts = obs.load_interventions(local_path('harpsn_interventions'), ORIGIN)
    lines = ['periods [d]: %s' % ' '.join('%5.0f' % p for p in PERIODS),
             'semi-amplitudes: %g to %g m/s (%d), %d phases, FAP <= %g, period tolerance %g'
             % (AMPLITUDES[0], AMPLITUDES[-1], len(AMPLITUDES), N_PHASES, FAP, PERIOD_TOL), '']

    ref_path = os.path.join(OUT, 'harpsn.npz')
    rv = None
    for x in xs:
        rv = comparison(x, harps, cuts)
        print('rv-%s: RMS %.3f m/s' % (x, rv['rms']))
        ra = ResidualAnalysis.from_comparison(rv, 'res', name='HARPS-N - SunSim (rv-%s)' % x)
        ra.inject(PERIODS, AMPLITUDES, n_phases=N_PHASES, fap=FAP)
        ra.save(os.path.join(OUT, 'residuals_rv-%s.npz' % x))
        detection_maps(ra, 'residuals_rv-%s' % x)
        lines.append(summary_line(ra, rv))

        if os.path.exists(ref_path):
            ref = ResidualAnalysis.load(ref_path)
        else:
            print('HARPS-N')
            ref = ResidualAnalysis.from_comparison(rv, 'obs', period_medians=True, name='HARPS-N')
            ref.inject(PERIODS, AMPLITUDES, n_phases=N_PHASES, fap=FAP)
            ref.save(ref_path)
            detection_maps(ref, 'harpsn')
        ratio_map(ra, ref, 'residuals_rv-%s' % x)
    lines.append(summary_line(ResidualAnalysis.load(ref_path)))

    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:] or ['2', '3', '4'])
