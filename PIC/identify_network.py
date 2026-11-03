#!/usr/bin/env python
"""
Identification of sweep_network.dag: for a block of days, every set of thresholds of parameters_network.txt (noise_th,
B_th, I_th, A_th, min_px, B_sat; Temp ignored) and the noise diagnostics of every (noise_th, B_th, I_th), every day
read once (identify_days.py for the parts of the network).

Sets: the bright pixels (sunsim.sdo.classify_pixels) in connected groups (4-connectivity, as split_plage_network):
  plage            : the groups larger than A_th micro-hemispheres (projected at mu_area = 0.1)
  network          : the other groups of at least min_px pixels (4096 px images), full pixels
  network_fraction : the same pixels, min(|B| / mu / B_sat, 1) facula (sunsim.sdo.facula_fraction)
  removed          : the groups of less than min_px pixels (quiet Sun)
written as <out>/<set label>/cells_N<n>/ff_<t>.npy (4, n_cells): the cell filling factors of the spots, the plage, the
network and the network fraction (sunsim.sdo.cell_filling_factors), and days/<date>.json (disc filling factors [%]).

Noise diagnostics, <out>/diag/<key label>/days/<date>.npz for every (noise_th, B_th, I_th) of the sets and of
--diag-B-th (with --diag-noise-th, --diag-I-th): the number of groups and of pixels of the bright groups of every size
(1 ... S_MAX - 1 pixels, the last bin >= S_MAX), their summed facular fraction (B_sat = --diag-B-sat), and the same for
two realisations of pure noise classified as the data on the quiet pixels of the day (the real intensity, |B| replaced
by the noise): 'white' (Gaussian, sigma(mu) of the day) and 'corr' (the same smoothed by a Gaussian kernel giving the
lag-1 correlation of the weak fields of the day); sigma(mu) is 1.4826 MAD of B in 10 bins of mu, rho1 the correlation
of B / sigma(mu) of horizontal neighbours with |B| < 3 sigma(mu).

  python identify_network.py --data SDO_DATA --out IDENT --start 2015-07-18 --end 2024-01-29 --n-rings 60 \
         --params parameters_network.txt --block 3 --n-blocks 80 --workers 4
  python identify_network.py ... --block 0 --n-blocks 1 --aggregate --parts PARTS_DIR
The second form does the days still missing, then writes for every set <parts>/<set label>/<part>/ (part: spot,
plage, network, network_fraction, spot_plage, spot_plage_network, spot_plage_network_fraction) with
filling_factors_N<n>.npz, summary.csv and parameters.txt (as split_components.py), and <out>/diag/<key label>/
histograms.npz (all the days). The days already done are skipped.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import numpy as np

COLUMNS = ('noise_th', 'B_th', 'I_th', 'A_th', 'min_px', 'B_sat', 'Temp')
S_MAX = 100                                   # group sizes of the histograms: 1 ... S_MAX - 1 pixels, >= S_MAX
N_MU = 10                                     # bins of mu of sigma(mu)
PARTS = ('spot', 'plage', 'network', 'network_fraction', 'spot_plage', 'spot_plage_network',
         'spot_plage_network_fraction')
# part -> (spot, facula): indices of the rows of the cell files (0 spot, 1 plage, 2 network, 3 network fraction)
PART_ROWS = {'spot': ((0,), ()), 'plage': ((), (1,)), 'network': ((), (2,)), 'network_fraction': ((), (3,)),
             'spot_plage': ((0,), (1,)), 'spot_plage_network': ((0,), (1, 2)),
             'spot_plage_network_fraction': ((0,), (1, 3))}
DISC_KEYS = ('ff_sp', 'ff_plage', 'ff_net', 'ff_net_frac')


def set_label(s):
    return 'noise%g_B%g_I%g_A%g_min%g_Bsat%g' % (s['noise_th'], s['B_th'], s['I_th'], s['A_th'], s['min_px'], s['B_sat'])


def key_label(k):
    return 'noise%g_B%g_I%g' % k


def read_sets(path):
    """The distinct sets of thresholds of the lines of the parameter file (columns COLUMNS, Temp not used)."""
    sets = {}
    for line in open(path):
        v = re.split(r'[ ,\t]+', line.strip())
        if len(v) < len(COLUMNS) or v[0].startswith('#'):
            continue
        s = {k: float(x) for k, x in zip(COLUMNS[:-1], v)}
        sets[set_label(s)] = s
    if not sets:
        raise ValueError('no set in %s (columns %s)' % (path, ' '.join(COLUMNS)))
    return list(sets.values())


def noise_kernel_table(n=512, seed=0):
    """Lag-1 correlation of white noise smoothed by a Gaussian of s pixels, for s on a grid: (s, rho)."""
    from scipy.ndimage import gaussian_filter
    w = np.random.default_rng(seed).standard_normal((n, n)).astype(np.float32)
    s = np.concatenate(([0.0], np.geomspace(0.2, 4.0, 30)))
    rho = []
    for x in s:
        c = gaussian_filter(w, x) if x > 0 else w
        rho.append(float(np.corrcoef(c[:, :-1].ravel(), c[:, 1:].ravel())[0, 1]))
    return s, np.maximum.accumulate(np.array(rho))


def noise_level(blos, mu, disc):
    """sigma(mu) [G] (1.4826 MAD of B in N_MU bins of mu, every 3rd pixel) and rho1 (lag-1 correlation of B / sigma of
    horizontal neighbours with |B| < 3 sigma, every 4th row)."""
    sub = (slice(None, None, 3), slice(None, None, 3))
    b, m, d = blos[sub], mu[sub], disc[sub]
    edges = np.linspace(0, 1, N_MU + 1)
    centres, sigma = 0.5 * (edges[1:] + edges[:-1]), np.full(N_MU, np.nan)
    for i in range(N_MU):
        x = b[d & (m > edges[i]) & (m <= edges[i + 1])]
        x = x[np.isfinite(x)]
        if len(x) > 100:
            sigma[i] = 1.4826 * np.median(np.abs(x - np.median(x)))
    ok = np.isfinite(sigma)
    z = blos[::4] / np.interp(mu[::4], centres[ok], sigma[ok]).astype(np.float32)
    q = disc[::4] & (np.abs(z) < 3)
    pair = q[:, :-1] & q[:, 1:]
    a, c = z[:, :-1][pair], z[:, 1:][pair]
    rho1 = float(np.sum((a - a.mean()) * (c - c.mean())) / np.sqrt(np.sum((a - a.mean())**2) * np.sum((c - c.mean())**2)))
    return centres, sigma, rho1


def size_histogram(labels, n, weights=None):
    """Groups and pixels of every size (1 ... S_MAX - 1, >= S_MAX) of a labelled image, and the sum of weights (an
    image) per size. Returns (sizes of the groups, n_groups, n_pixels, summed weights or None)."""
    idx = np.flatnonzero(labels)
    lab = labels.ravel()[idx]
    sizes = np.bincount(lab, minlength=n + 1)[1:]
    b = np.minimum(sizes, S_MAX) - 1
    n_groups = np.bincount(b, minlength=S_MAX)
    n_pix = np.bincount(b, weights=sizes, minlength=S_MAX)
    w = None
    if weights is not None:
        per_group = np.bincount(lab, weights=weights.ravel()[idx], minlength=n + 1)[1:]
        w = np.bincount(b, weights=per_group, minlength=S_MAX)
    return sizes, n_groups, n_pix, w


def _done(out, sets, keys, t, date, n_rings):
    from sunsim import sdo
    label = sdo._time_label(t)
    for s in sets:
        f = os.path.join(out, set_label(s))
        if not (os.path.exists(os.path.join(f, 'days', date + '.json'))
                and all(os.path.exists(os.path.join(f, 'cells_N%d' % n, 'ff_%s.npy' % label)) for n in n_rings)):
            return False
    return all(os.path.exists(os.path.join(out, 'diag', key_label(k), 'days', date + '.npz')) for k in keys)


def process_day(task):
    """Every set and every noise diagnostic of one day. Returns a summary row of the first set, or ('skipped', date,
    reason)."""
    import scipy.ndimage as nd
    from sunsim import sdo
    data, day, t, files, n_rings, out, sets, keys, diag_B_sat, kernel = task
    date = day.strftime('%Y-%m-%d')
    try:
        cont_map, los_map = sdo._resample(*sdo._read_with_retries(data, day, files=files), None)
    except OSError as e:
        return ('skipped', date, 'images not readable (%s)' % str(e).splitlines()[0][:120])
    if cont_map.data.shape[0] != sdo.FULL_SIZE:
        return ('skipped', date, 'images of %d px, not %d' % (cont_map.data.shape[0], sdo.FULL_SIZE))
    mu = sdo.mu_of_map(los_map)
    blos = np.asarray(los_map.data, dtype=np.float32)
    intensity = cont_map.data
    lo, hi = sdo.disc_square(intensity)
    label_t = sdo._time_label(t)

    def frame(a):
        return np.flip(np.asarray(a)[lo:hi, lo:hi])

    # pure noise of the day: unit white noise, and smoothed to the lag-1 correlation of the weak fields
    disc0 = (mu > 0) & np.isfinite(intensity)
    centres, sigma, rho1 = noise_level(blos, mu, disc0)
    ok = np.isfinite(sigma)
    sig_map = np.interp(mu, centres[ok], sigma[ok]).astype(np.float32)
    rng = np.random.default_rng(day.toordinal())
    white = rng.standard_normal(blos.shape, dtype=np.float32)
    s_kernel = float(np.interp(min(max(rho1, 0.0), kernel[1][-1]), kernel[1], kernel[0]))
    corr = nd.gaussian_filter(white, s_kernel) if s_kernel > 0 else white.copy()
    corr /= np.std(corr[::8, ::8])
    white *= sig_map
    corr *= sig_map
    del sig_map

    first = None
    for k in keys:
        noise_th, B_th, I_th = k
        c = sdo.classify_pixels(intensity, blos, mu, noise_th=noise_th, B_th=B_th, I_th=I_th)
        labels, n = nd.label(c['bright'])
        frac = sdo.facula_fraction(blos, mu, c['bright'], diag_B_sat)
        sizes, n_groups, n_pix, w = size_histogram(labels, n, frac)
        del frac
        diag = dict(t=t, date=date, n_groups=n_groups, n_pix=n_pix, frac=w, n_disc=int(np.sum(c['disc'])),
                    n_quiet=int(np.sum(c['disc'] & ~c['active'])), mu_centres=centres, sigma=sigma, rho1=rho1,
                    s_kernel=s_kernel, I_quiet=c['I_quiet'])
        quiet = c['disc'] & ~c['active']
        for name, noise in (('white', white), ('corr', corr)):
            absn = np.abs(noise)
            bright_n = quiet & (absn >= noise_th) & (absn > B_th) & ~(intensity < I_th * c['I_quiet'])
            del absn
            lab_n, n_n = nd.label(bright_n)
            _, g, p, _ = size_histogram(lab_n, n_n)
            diag[name + '_groups'], diag[name + '_pix'] = g, p
            del lab_n, bright_n
        folder = os.path.join(out, 'diag', key_label(k), 'days')
        np.savez(os.path.join(folder, date + '.tmp.npz'), **diag)
        os.replace(os.path.join(folder, date + '.tmp.npz'), os.path.join(folder, date + '.npz'))

        mine = [s for s in sets if (s['noise_th'], s['B_th'], s['I_th']) == k]
        if not mine:
            del labels
            continue
        gsize = np.concatenate(([0], sizes)).astype(np.int32)[labels]
        del labels
        disc, spot = frame(c['disc']), frame(c['spot'])
        n_disc = max(int(np.sum(disc)), 1)
        n_spots = sdo.count_spots(c['spot'])[0]
        zero = np.zeros_like(spot)
        for s in mine:
            area_px = sdo.area_threshold_pixels(los_map, s['A_th'])
            plage = gsize > area_px
            network = (gsize > 0) & ~plage & (gsize >= s['min_px'])
            removed = (gsize > 0) & (gsize < s['min_px'])
            fraction = sdo.facula_fraction(blos, mu, network, s['B_sat'])
            plage, network, removed, fraction = frame(plage), frame(network), frame(removed), frame(fraction)
            ff_sp, ff_plage = sdo.cell_filling_factors(dict(spot=spot, facula=plage), n_rings)
            _, ff_net = sdo.cell_filling_factors(dict(spot=zero, facula=network), n_rings)
            _, ff_frac = sdo.cell_filling_factors(dict(spot=zero, facula=network, facula_fraction=fraction), n_rings)
            folder = os.path.join(out, set_label(s))
            np.save(os.path.join(folder, 'cells_N%d' % n_rings, 'ff_%s.npy' % label_t),
                    np.stack([ff_sp, ff_plage, ff_net, ff_frac]))
            n_net = int(np.sum((sizes <= area_px) & (sizes >= s['min_px'])))
            row = {'t': t, 'date': date, 'ff_sp': 100.0 * np.sum(spot) / n_disc,
                   'ff_plage': 100.0 * np.sum(plage) / n_disc, 'ff_net': 100.0 * np.sum(network) / n_disc,
                   'ff_net_frac': 100.0 * np.sum(fraction, dtype=np.float64) / n_disc,
                   'ff_removed': 100.0 * np.sum(removed) / n_disc, 'n_spots': int(n_spots),
                   'n_plages': int(np.sum(sizes > area_px)), 'n_net': n_net,
                   'n_removed': int(np.sum(sizes < s['min_px'])), 'area_th_pixels': float(area_px)}
            with open(os.path.join(folder, 'days', date + '.json'), 'w') as f:
                json.dump(row, f)
            first = first or row
        del gsize
    return first or {'t': t, 'date': date}


def aggregate(out, parts_dir, sets, keys, days, n_rings, header):
    """filling_factors_N<n>.npz, summary.csv, parameters.txt of every part of every set, and histograms.npz of every
    noise diagnostic, from the files of the days."""
    from sunsim import sdo
    name = 'filling_factors_N%d.npz' % n_rings
    for s in sets:
        folder = os.path.join(out, set_label(s))
        rows = []
        for day, t, _ in days:
            p = os.path.join(folder, 'days', day.strftime('%Y-%m-%d') + '.json')
            if os.path.exists(p):
                rows.append(json.load(open(p)))
        rows.sort(key=lambda r: r['t'])
        if not rows:
            print('%s: no day' % folder)
            continue
        cells = np.array([np.load(os.path.join(folder, 'cells_N%d' % n_rings, 'ff_%s.npy' % sdo._time_label(r['t'])))
                          for r in rows], dtype=np.float32)
        times, dates = np.array([r['t'] for r in rows]), np.array([r['date'] for r in rows])
        disc = {k: np.array([r[k] for r in rows]) for k in DISC_KEYS}
        thresholds = ''.join('%s = %g\n' % (k, s[k]) for k in COLUMNS[:-1])
        for part in PARTS:
            rows_sp, rows_fc = PART_ROWS[part]
            zero = np.zeros(cells.shape[::2], dtype=np.float32)
            ff_sp = cells[:, rows_sp].sum(axis=1) if rows_sp else zero
            ff_fc = cells[:, rows_fc].sum(axis=1) if rows_fc else zero
            d_sp = disc['ff_sp'] if rows_sp else np.zeros(len(rows))
            d_fc = sum((disc[DISC_KEYS[i]] for i in rows_fc), np.zeros(len(rows)))
            d_nt = sum((disc[DISC_KEYS[i]] for i in rows_fc if i >= 2), np.zeros(len(rows)))
            f = os.path.join(parts_dir, set_label(s), part)
            os.makedirs(f, exist_ok=True)
            np.savez(os.path.join(f, name + '.tmp.npz'), times=times, dates=dates, N_rings=n_rings,
                     ff_sp=ff_sp.astype(np.float32), ff_fc=ff_fc.astype(np.float32))
            os.replace(os.path.join(f, name + '.tmp.npz'), os.path.join(f, name))
            with open(os.path.join(f, 'summary.csv'), 'w') as fs:
                fs.write('t,date,ff_sp,ff_fc,ff_nt,n_spots,n_plages\n')
                for k, r in enumerate(rows):
                    fs.write('%g,%s,%.6f,%.6f,%.6f,%d,%d\n' % (r['t'], r['date'], d_sp[k], d_fc[k], d_nt[k],
                                                             r['n_spots'], r['n_plages']))
            with open(os.path.join(f, 'parameters.txt'), 'w') as fp:
                fp.write(header + thresholds + 'part = %s (identify_network.py)\n' % part)
        with open(os.path.join(folder, 'summary.csv'), 'w') as fs:
            cols = list(rows[0])
            fs.write(','.join(cols) + '\n')
            for r in rows:
                fs.write(','.join(str(r[c]) for c in cols) + '\n')
        print('%s: %d days, mean ff_sp %.3f, plage %.3f, network %.3f, network fraction %.3f, removed %.3f %%'
              % (set_label(s), len(rows), disc['ff_sp'].mean(), disc['ff_plage'].mean(), disc['ff_net'].mean(),
                 disc['ff_net_frac'].mean(), np.mean([r['ff_removed'] for r in rows])))
    for k in keys:
        folder = os.path.join(out, 'diag', key_label(k))
        files = [os.path.join(folder, 'days', d.strftime('%Y-%m-%d') + '.npz') for d, _, _ in days]
        loaded = [dict(np.load(f)) for f in files if os.path.exists(f)]
        if not loaded:
            continue
        stacked = {key: np.array([x[key] for x in loaded]) for key in loaded[0]}
        np.savez(os.path.join(folder, 'histograms.npz'), S_MAX=S_MAX, noise_th=k[0], B_th=k[1], I_th=k[2], **stacked)
        print('%s: %d days' % (folder, len(loaded)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--data', required=True, help='folder of the HMI images')
    p.add_argument('--out', required=True, help='folder of the identification (one sub-folder per set, diag/)')
    p.add_argument('--start', required=True, help='first day of the whole period (t = 0), YYYY-MM-DD')
    p.add_argument('--end', required=True, help='last day of the whole period (excluded), YYYY-MM-DD')
    p.add_argument('--n-rings', type=int, required=True)
    p.add_argument('--params', required=True, help='parameter file (columns %s)' % ' '.join(COLUMNS))
    p.add_argument('--diag-noise-th', type=float, default=8.0)
    p.add_argument('--diag-B-th', type=float, nargs='*', default=[], help='more B_th of the noise diagnostics [G]')
    p.add_argument('--diag-I-th', type=float, default=0.9)
    p.add_argument('--diag-B-sat', type=float, default=200.0, help='facular fraction of the diagnostics [G]')
    p.add_argument('--magnetogram', choices=['45s', '720s'], default='45s')
    p.add_argument('--block', type=int, required=True)
    p.add_argument('--n-blocks', type=int, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--aggregate', action='store_true', help='then write the parts of every set and the histograms')
    p.add_argument('--parts', default=None, help='with --aggregate: folder of the parts of the sets')
    a = p.parse_args(argv)
    if a.aggregate and not a.parts:
        p.error('--aggregate needs --parts')

    from sunsim import sdo
    sets = read_sets(a.params)
    keys = sorted({(s['noise_th'], s['B_th'], s['I_th']) for s in sets}
                  | {(a.diag_noise_th, b, a.diag_I_th) for b in a.diag_B_th})
    for s in sets:
        for sub in ('days', 'cells_N%d' % a.n_rings):
            os.makedirs(os.path.join(a.out, set_label(s), sub), exist_ok=True)
    for k in keys:
        os.makedirs(os.path.join(a.out, 'diag', key_label(k), 'days'), exist_ok=True)

    t0, t1 = datetime.strptime(a.start, '%Y-%m-%d'), datetime.strptime(a.end, '%Y-%m-%d')
    index = sdo._hmi_index(a.data, a.magnetogram)
    days = []
    day = t0
    while day < t1:
        files = index.get(day.strftime('%Y%m%d'))
        if files is not None:
            days.append((day, float((day - t0).days), files))
        day += timedelta(days=1)
    mine = np.array_split(np.arange(len(days)), a.n_blocks)[a.block]
    kernel = noise_kernel_table()
    todo = [(a.data, days[k][0], days[k][1], days[k][2], a.n_rings, a.out, sets, keys, a.diag_B_sat, kernel)
            for k in mine if not _done(a.out, sets, keys, days[k][1], days[k][0].strftime('%Y-%m-%d'), [a.n_rings])]
    print('block %d / %d: %d days with images, %d to do; %d sets, %d noise diagnostics (%s), %d worker(s)'
          % (a.block, a.n_blocks, len(mine), len(todo), len(sets), len(keys), ' '.join(map(key_label, keys)),
             a.workers), flush=True)

    t_start, skipped = time.time(), 0

    def report(r, k):
        nonlocal skipped
        if isinstance(r, tuple):
            skipped += 1
            print('%s: %s, skipped' % (r[1], r[2]), flush=True)
        else:
            el = time.time() - t_start
            info = ('ff_sp %.3f, plage %.3f, network %.3f (fraction %.3f), removed %.3f %% (first set)'
                    % tuple(r[x] for x in DISC_KEYS + ('ff_removed',))) if 'ff_sp' in r else ''
            print('[%d/%d] %s: %s | %.0f s, ~%.0f min left' % (k, len(todo), r['date'], info, el,
                                                             el / k * (len(todo) - k) / 60), flush=True)

    if a.workers == 1:
        for k, task in enumerate(todo, 1):
            report(process_day(task), k)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for k, r in enumerate(pool.map(process_day, todo), 1):
                report(r, k)
    print('block %d done: %d days in %.1f h, %d skipped' % (a.block, len(todo), (time.time() - t_start) / 3600, skipped))
    if a.aggregate:
        header = ('data_folder = %s\nmagnetogram = %s\nstart = %s\nend = %s\nsize = 4096\nmu_area = 0.1\n'
                  % (a.data, sdo._magnetogram(a.magnetogram)[0], a.start, a.end))
        aggregate(a.out, a.parts, sets, keys, days, a.n_rings, header)
    return 1 if skipped else 0


if __name__ == '__main__':
    sys.exit(main())
