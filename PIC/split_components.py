#!/usr/bin/env python
"""
Cell filling factors of the parts of a set of thresholds of the area threshold method, each simulated alone by
run_components.sh (sweep_components.dag), from the identifications of the set with faculae='plage' (ff_fc = plage)
and faculae='plage+network' (ff_fc = plage + network), made on the same days by identify_components.sh:
  spot               : ff_sp, no facula
  plage              : the plage, no spot
  network            : the network (ff_fc of plage+network - ff_fc of plage: spots and bright pixels never overlap,
                       sunsim.sdo.classify_pixels), no spot
  spot_plage         : ff_sp and the plage (the filling factors of faculae='plage')
  spot_plage_network : ff_sp, the plage and the network (those of faculae='plage+network')

  python split_components.py IDENT/<label> IDENT/<label>_network OUT --n-rings 60

writes OUT/<part>/filling_factors_N<n>.npz, summary.csv (t, date, ff_sp, ff_fc, ff_nt, n_spots, n_plages of the part
[% of the disc]) and parameters.txt. Only the epochs of both identifications are kept.
"""
import argparse
import os

import numpy as np

PARTS = ('spot', 'plage', 'network', 'spot_plage', 'spot_plage_network')


def read_summary(folder):
    s = np.genfromtxt(os.path.join(folder, 'summary.csv'), delimiter=',', names=True, dtype=None, encoding=None)
    return np.atleast_1d(s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('plage', help='identification of the set with faculae=plage')
    p.add_argument('network', help='identification of the same set with faculae=plage+network')
    p.add_argument('out', help='folder of the parts')
    p.add_argument('--n-rings', type=int, required=True)
    a = p.parse_args(argv)

    name = 'filling_factors_N%d.npz' % a.n_rings
    fp, fn = np.load(os.path.join(a.plage, name)), np.load(os.path.join(a.network, name))
    times = np.intersect1d(fp['times'], fn['times'])
    ip, i_n = np.searchsorted(fp['times'], times), np.searchsorted(fn['times'], times)
    if len(times) < max(len(fp['times']), len(fn['times'])):
        print('%d epochs in %s, %d in %s: the %d of both kept' % (len(fp['times']), a.plage, len(fn['times']),
                                                                   a.network, len(times)))
    sp = fp['ff_sp'][ip]
    plage = fp['ff_fc'][ip]
    plage_network = fn['ff_fc'][i_n]
    network = np.clip(plage_network - plage, 0.0, None).astype(np.float32)
    zero = np.zeros_like(sp)
    cells = {'spot': (sp, zero), 'plage': (zero, plage), 'network': (zero, network), 'spot_plage': (sp, plage),
             'spot_plage_network': (sp, plage_network)}

    s = read_summary(a.plage)
    s = s[np.searchsorted(s['t'], times)]
    nt = s['ff_nt']
    z = np.zeros(len(s))
    disc = {'spot': (s['ff_sp'], z, z), 'plage': (z, s['ff_fc'], z), 'network': (z, nt, nt),
            'spot_plage': (s['ff_sp'], s['ff_fc'], z), 'spot_plage_network': (s['ff_sp'], s['ff_fc'] + nt, nt)}
    params = open(os.path.join(a.plage, 'parameters.txt')).read()

    for part in PARTS:
        folder = os.path.join(a.out, part)
        os.makedirs(folder, exist_ok=True)
        ff_sp, ff_fc = cells[part]
        np.savez(os.path.join(folder, name + '.tmp.npz'), times=times, dates=fp['dates'][ip], N_rings=a.n_rings,
                 ff_sp=ff_sp, ff_fc=ff_fc)
        os.replace(os.path.join(folder, name + '.tmp.npz'), os.path.join(folder, name))
        d_sp, d_fc, d_nt = disc[part]
        with open(os.path.join(folder, 'summary.csv'), 'w') as f:
            f.write('t,date,ff_sp,ff_fc,ff_nt,n_spots,n_plages\n')
            for k in range(len(s)):
                f.write('%g,%s,%.6f,%.6f,%.6f,%d,%d\n' % (s['t'][k], s['date'][k], d_sp[k], d_fc[k], d_nt[k],
                                                         s['n_spots'][k], s['n_plages'][k]))
        with open(os.path.join(folder, 'parameters.txt'), 'w') as f:
            f.write(params + 'part = %s (split_components.py: %s, %s)\n' % (part, a.plage, a.network))
        print('%s: %d epochs, mean ff_sp = %.4f %%, ff_fc = %.4f %% of the disc'
              % (folder, len(times), np.mean(d_sp), np.mean(d_fc)))


if __name__ == '__main__':
    main()
