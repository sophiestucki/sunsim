"""
End-to-end benchmark of the SDO pipeline: from the HMI images to the light curve and the RV.

  old : the previous identification (simu_rv_opt_fast.py + identification_utils.py, analysis/sdo_old_classification.py),
        the maps written as text files as that script did, then the old StarSim (compute_forward lc, then rv) on them
        (tests/old_version/run_old_starsim.py, the version of the path old_starsim). Two processes, as the old script
        identification + StarSim.
  new : one command, python -m sunsim.sdo --format none --conf ...: identification (sunsim.sdo), filling factors of the
        cells, grids and simulation (simulate_sdo_sets, the grids computed: no cache), with 1 worker and with --workers.
The same days (HMI images of the 2015 test, tests/old_version/reference/sdo_2015/hmi, or the path sdo_data), the same
configuration (tests/old_version/reference/sdo_2015.conf: MURaM, 15 rings, HARPS G2 mask), the same size of the images.
Every step is a new process (imports, Numba caches already compiled); wall time and peak memory of the process and all
its children (sum of the resident memory, sampled every 0.1 s with psutil).

    python benchmarks/sdo_pipeline_benchmark.py [--sizes 512 1024] [--workers 4] [--start 2015-09-01 --end 2015-12-01]

Results: benchmarks/results/sdo_pipeline_<size>.json and a table.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / 'work' / 'sdo_pipeline'
RESULTS = HERE / 'results'
CONF_TEMPLATE = ROOT / 'tests' / 'old_version' / 'reference' / 'sdo_2015.conf'
LOCAL_HMI = ROOT / 'tests' / 'old_version' / 'reference' / 'sdo_2015' / 'hmi'


def measure(cmd, env=None, cwd=None):
    """Runs cmd; returns wall time [s], peak memory of the process tree [MB] (psutil, every 0.1 s), and the output."""
    import psutil
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=cwd)
    proc = psutil.Process(p.pid)
    peak = 0
    out = []
    import threading
    reader = threading.Thread(target=lambda: out.extend(p.stdout), daemon=True)
    reader.start()
    while p.poll() is None:
        try:
            procs = [proc] + proc.children(recursive=True)
            rss = 0
            for q in procs:
                try:
                    rss += q.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.1)
    reader.join()
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError('%s failed:\n%s' % (' '.join(map(str, cmd)), ''.join(out)[-3000:]))
    return dt, peak / 1e6, ''.join(out)


def write_conf(folder, path):
    """The configuration of the 2015 test with SDO_input_path = folder."""
    text = CONF_TEMPLATE.read_text()
    text = re.sub(r'(?m)^SDO_input_path = .*$', 'SDO_input_path = %s/' % folder, text)
    Path(path).write_text(text)
    return path


def child_old_identification(data, out, start, end, size):
    """The loop of simu_rv_opt_fast.py: every day, the previous identification and the two text maps."""
    import numpy as np
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'analysis'))
    from sunsim import sdo
    from sdo_old_classification import old_identify
    os.makedirs(out, exist_ok=True)
    index = sdo._hmi_index(data)
    day, t1 = datetime.strptime(start, '%Y-%m-%d'), datetime.strptime(end, '%Y-%m-%d')
    i = 0
    while day < t1:
        files = index.get(day.strftime('%Y%m%d'))
        if files is not None:
            try:
                cont, los = sdo._read_with_retries(data, day, files=files)
            except OSError:
                cont = None
            if cont is not None:
                r = old_identify(cont, los, size=size, noise_th=8.0, I_th=0.89, A_th=20.0)
                np.savetxt(os.path.join(out, 'faculae_map_%.1f.txt' % i), r['facula_map'])
                np.savetxt(os.path.join(out, 'spot_map_%.1f.txt' % i), r['spot_map'])
                i += 1
        day += timedelta(days=1)
    print('%d days' % i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+', default=[512, 1024])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--start', default='2015-09-01')
    ap.add_argument('--end', default='2015-12-01')
    ap.add_argument('--data', default=None, help='HMI images (default: the local copy of the 2015 test, else sdo_data)')
    ap.add_argument('--_child', nargs='*', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._child:
        data, out, start, end, size = args._child
        child_old_identification(data, out, start, end, int(size))
        return

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'tests' / 'old_version'))
    from sunsim import local_path
    import test_old_version as tov
    data = args.data or (str(LOCAL_HMI) if LOCAL_HMI.is_dir() else str(local_path('sdo_data')))
    py = sys.executable
    # the caches of the old code of the tests (compiled already, as the Numba caches of SunSim): both codes run warm
    env_old = dict(os.environ, PYTHONPYCACHEPREFIX=str(tov.REFERENCE / '_cache' / 'pycache'),
                   NUMBA_CACHE_DIR=str(tov.REFERENCE / '_cache' / 'numba'), MPLBACKEND='Agg')
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for size in args.sizes:
        base = WORK / ('size_%d' % size)
        if base.exists():
            import shutil
            shutil.rmtree(base)                           # only the outputs of this benchmark
        base.mkdir(parents=True)
        res = {'size': size, 'start': args.start, 'end': args.end, 'data': data}

        # old: identification (text maps), then the old StarSim
        maps = base / 'old_maps'
        dt, mem, out = measure([py, __file__, '--_child', data, str(maps), args.start, args.end, str(size)])
        res['old_identification'] = {'time': dt, 'memory': mem}
        n_days = int(re.search(r'(\d+) days', out).group(1))
        conf = write_conf(maps, base / 'old.conf')
        job = {'old_package': str(tov.OLD_PACKAGE), 'conf': str(conf), 'out': str(base / 'old.npz'), 'times': None,
               'observables': ['lc', 'rv'], 'regions': None, 'patch': None}
        (base / 'old_job.json').write_text(json.dumps(job))
        dt, mem, _ = measure([tov.OLD_PYTHON, str(ROOT / 'tests' / 'old_version' / 'run_old_starsim.py'),
                              str(base / 'old_job.json')], env=env_old)
        res['old_starsim'] = {'time': dt, 'memory': mem}
        disk = sum(f.stat().st_size for f in maps.iterdir()) / 1e6
        res['old_maps_MB'] = disk

        # new: one command, 1 worker and --workers
        for w in (1, args.workers):
            out_dir = base / ('new_w%d' % w)
            conf_new = write_conf(out_dir, base / ('new_w%d.conf' % w))
            dt, mem, out = measure([py, '-m', 'sunsim.sdo', '--data', data, '--out', str(out_dir), '--start', args.start,
                                    '--end', args.end, '--size', str(size), '--format', 'none', '--conf', str(conf_new),
                                    '--observables', 'lc', 'rv', '--no-grids-cache', '--workers', str(w), '--quiet'],
                                   cwd=str(ROOT))
            res['new_w%d' % w] = {'time': dt, 'memory': mem,
                                  'cells_MB': sum(f.stat().st_size for f in (out_dir / 'cells_N15').iterdir()) / 1e6}
        res['n_days'] = n_days
        json.dump(res, open(RESULTS / ('sdo_pipeline_%d.json' % size), 'w'), indent=1)
        old_t = res['old_identification']['time'] + res['old_starsim']['time']
        old_m = max(res['old_identification']['memory'], res['old_starsim']['memory'])
        rows.append((size, n_days, res['old_identification']['time'], res['old_starsim']['time'], old_t, old_m,
                     res['new_w1']['time'], res['new_w1']['memory'], res['new_w%d' % args.workers]['time'],
                     res['new_w%d' % args.workers]['memory']))
        print('size %d done' % size, flush=True)

    print('\n%d days, light curve + RV (15 rings); time [s], peak memory [MB]' % rows[0][1])
    print('%5s | %9s %9s %9s %7s | %9s %7s %8s | %9s %7s %8s' % ('size', 'old ident', 'old sim', 'old total', 'MB',
                                                                  'new 1 w', 'MB', 'speed-up', 'new %d w' % args.workers,
                                                                  'MB', 'speed-up'))
    for r in rows:
        print('%5d | %9.1f %9.1f %9.1f %7.0f | %9.1f %7.0f %7.1fx | %9.1f %7.0f %7.1fx'
              % (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[4] / r[6], r[8], r[9], r[4] / r[8]))


if __name__ == '__main__':
    main()
