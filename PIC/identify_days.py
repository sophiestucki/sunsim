#!/usr/bin/env python
"""
Identification of a block of days for all the sets of thresholds at once (the sweep of sunsim-sdo-maps for a part of
the period, so that many jobs can share it): every day is read, resampled and its mu computed once, then classified
for every set (sunsim.sdo._process_day), and the cell filling factors written in <out>/<set label>/cells_N<n>/ and
days/<date>.json, as python -m sunsim.sdo --format none does. The times are the days since --start (the whole period),
so all the blocks write in the same folders without clash. The days already done are skipped.

  python identify_days.py --data SDO_DATA --out FOLDER --start 2015-07-18 --end 2024-01-29 --size 4096 --n-rings 60 \
         --noise-th 8 --I-th 0.55 0.6 ... --A-th 1 10 ... --block 3 --n-blocks 40
  python identify_days.py ... --faculae fraction --B-sat 50 100 ...       (facular fractions, no area threshold)

The sets are all the combinations of the thresholds given (sunsim.sdo.parameter_sets). After all the blocks, one run of
python -m sunsim.sdo with the same thresholds over the whole period does the days still missing and writes
filling_factors_N<n>.npz and summary.csv of every set (aggregate_4096.sh).
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data', required=True, help='folder of the HMI images')
    p.add_argument('--out', required=True, help='folder of the sets (one sub-folder per set of thresholds)')
    p.add_argument('--start', required=True, help='first day of the whole period (t = 0), YYYY-MM-DD')
    p.add_argument('--end', required=True, help='last day of the whole period (excluded), YYYY-MM-DD')
    p.add_argument('--size', type=int, default=4096)
    p.add_argument('--n-rings', type=int, nargs='+', required=True)
    for k, default in (('noise-th', 8.0), ('B-th', 24.0), ('I-th', 0.89), ('A-th', 20.0), ('mu-area', 0.1)):
        p.add_argument('--' + k, type=float, nargs='+', default=[default])
    p.add_argument('--faculae', nargs='+', default=['plage'], choices=['plage', 'plage+network', 'fraction'])
    p.add_argument('--B-sat', type=float, nargs='+', default=[100.0], help="with --faculae fraction [G]")
    p.add_argument('--block', type=int, required=True, help='index of this block (0 ... n_blocks - 1)')
    p.add_argument('--n-blocks', type=int, required=True)
    p.add_argument('--workers', type=int, default=1, help='processes working on different days')
    a = p.parse_args(argv)

    from sunsim import sdo
    sets = sdo.parameter_sets(a.noise_th, a.B_th, a.I_th, a.A_th, a.mu_area, a.faculae, a.B_sat)
    folders = [(os.path.join(a.out, sdo.set_label(s)), s) for s in sets]
    for folder, _ in folders:
        for sub in ['days'] + ['cells_N%d' % n for n in a.n_rings]:
            os.makedirs(os.path.join(folder, sub), exist_ok=True)

    t0, t1 = datetime.strptime(a.start, '%Y-%m-%d'), datetime.strptime(a.end, '%Y-%m-%d')
    index = sdo._hmi_index(a.data)
    days = []
    day = t0
    while day < t1:
        files = index.get(day.strftime('%Y%m%d'))
        if files is not None:
            days.append((day, float((day - t0).days), files))
        day += timedelta(days=1)
    mine = np.array_split(np.arange(len(days)), a.n_blocks)[a.block]
    todo = []
    for k in mine:
        day, t, files = days[k]
        left = [st for st in folders if not sdo._day_done(st[0], t, day.strftime('%Y-%m-%d'), None, a.n_rings)]
        if left:
            todo.append((a.data, day, t, files, a.size, None, list(a.n_rings), left))
    print('block %d / %d: %d days with images (%s to %s), %d to do; %d sets of thresholds, %d worker(s)'
          % (a.block, a.n_blocks, len(mine), days[mine[0]][0].date() if len(mine) else '-',
             days[mine[-1]][0].date() if len(mine) else '-', len(todo), len(folders), a.workers), flush=True)

    t_start, skipped = time.time(), 0

    def report(r, k):
        nonlocal skipped
        if isinstance(r, tuple):                          # ('skipped', date, reason)
            skipped += 1
            print('%s: %s, skipped' % (r[1], r[2]), flush=True)
        elif r:
            el = time.time() - t_start
            print('[%d/%d] %s: %d sets, ff_sp %.3f %%, ff_fc %.3f %% (first set) | %.0f s, ~%.0f min left'
                  % (k, len(todo), r[0]['date'], len(r), r[0]['ff_sp'], r[0]['ff_fc'], el,
                     el / k * (len(todo) - k) / 60), flush=True)

    if a.workers == 1:
        for k, task in enumerate(todo, 1):
            report(sdo._process_day(task), k)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for k, r in enumerate(pool.map(sdo._process_day, todo), 1):
                report(r, k)
    print('block %d done: %d days in %.1f h, %d skipped' % (a.block, len(todo), (time.time() - t_start) / 3600, skipped))
    return 1 if skipped else 0


if __name__ == '__main__':
    sys.exit(main())
