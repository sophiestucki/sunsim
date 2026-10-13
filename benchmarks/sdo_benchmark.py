"""
Benchmark of the 2D maps: the real SDO maps of spots and faculae (89 epochs, ~950 x 950 pixels), with the old StarSim and
with SunSim, in photometry and in RV (RV, FWHM, BIS, contrast). Every run is a new process (imports included); the time
of every step and the peak memory of the process are recorded.

    python benchmarks/sdo_benchmark.py run <variant> [--repeat n]     one variant (results in benchmarks/results/)
    python benchmarks/sdo_benchmark.py all [--repeat n]               every variant

Variants
    old           the old StarSim (environment env_sunsim), compute_forward(['lc']) then (['rv', 'bis', 'fwhm', 'contrast'])
    new_old_bugs  SunSim (old_config.simulate_old_config, reproduce_old_bugs=True): the same physics as the old code
                  (faculae of the maps as spots in photometry and quiet photosphere in the CCFs, old projection), so
                  that the outputs can be compared number by number
    new           SunSim with its own physics (faculae with their own MURaM spectra, projection of active_region_mask)
    new_npy       'new' with the maps saved once as uint8 .npy (in benchmarks/work/, not in the folder of the maps)
All read the text maps of SDO_MAPS (not modified: no .npy or cache file is written there), MURaM low resolution spectra,
15 rings, the mask and the velocity grid of the old configuration.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from configparser import ConfigParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = HERE / 'work'
RESULTS = HERE / 'results'
SDO_MAPS = Path(os.environ.get('SDO_REAL_MAPS', '/Users/sophiestucki/Library/CloudStorage/GoogleDrive-stucki@ieec.cat/'
                                                'My Drive/storage/maps_active_2015_09_12_ccf_notnorm'))
OLD_PACKAGE = Path(os.environ.get('OLD_STARSIM_PACKAGE', '/Users/sophiestucki/starsim/starsim'))
OLD_PYTHON = os.environ.get('OLD_STARSIM_PYTHON', '/Users/sophiestucki/miniforge3/envs/env_sunsim/bin/python')
NEW_PYTHON = os.environ.get('SUNSIM_PYTHON', '/Users/sophiestucki/miniforge3/envs/sunsim/bin/python')
N_RINGS = 15
OBSERVABLES = ['lc', 'rv', 'bis', 'fwhm', 'contrast']
VARIANTS = ['old', 'new_old_bugs', 'new', 'new_npy']


def peak_rss_mb():
    """Peak resident memory of this process [MB] (ru_maxrss is in bytes on macOS, in kB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == 'darwin' else r / 1e3


def write_conf(maps_folder):
    """Configuration of the old code for the benchmark (the reference case of tests/old_version with MURaM and maps)."""
    sys.path.insert(0, str(ROOT / 'tests' / 'old_version'))
    from test_old_version import BASE
    conf = ConfigParser()
    conf.optionxform = str
    for section, kv in BASE.items():
        conf[section] = {k: str(v) for k, v in kv.items()}
    conf['general']['spectral_library'] = 'mps'
    conf['general']['n_grid_rings'] = str(N_RINGS)
    conf['spots']['sdo_input'] = '1'
    conf['spots']['SDO_input_path'] = str(maps_folder) + '/'
    conf['spots']['plot_grid_map'] = '0'                    # the old code writes no plot file
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / 'sdo_benchmark.conf'
    with open(path, 'w') as f:
        conf.write(f)
    return path


def npy_maps():
    """The text maps of SDO_MAPS saved once as uint8 .npy in WORK/maps_npy (0/1; the NaN outside the disc -> 0)."""
    import numpy as np
    import pandas as pd
    out = WORK / 'maps_npy'
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    n = 0
    for f in sorted(SDO_MAPS.glob('*_map_*.txt')):
        dst = out / (f.stem + '.npy')
        if not dst.exists():
            a = pd.read_csv(f, sep=r'\s+', header=None, dtype=np.float32).to_numpy()
            np.save(dst, np.nan_to_num(a, nan=0.0).astype(np.uint8))
            n += 1
    return out, n, time.perf_counter() - t


# ------------------------------------------------------------------------------------------- one run (child process)
def child_old(conf, out):
    import contextlib
    import io
    t0 = time.perf_counter()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    sys.path.insert(0, str(OLD_PACKAGE.parent))
    with contextlib.redirect_stdout(io.StringIO()):
        from starsim import StarSim
    steps = {'import': time.perf_counter() - t0}
    with contextlib.redirect_stdout(io.StringIO()):
        t = time.perf_counter()
        ss = StarSim(conf_file_path=str(conf))                # reads the 89 x 2 maps
        steps['init (reads the maps)'] = time.perf_counter() - t
        t = time.perf_counter()
        ss.compute_forward(observables=['lc'], t=None)
        steps['photometry'] = time.perf_counter() - t
        res = {'lc': ss.results['lc'], 'ff_sp': ss.results['ff_sp'], 'ff_fc': ss.results['ff_fc']}
        t = time.perf_counter()
        ss.compute_forward(observables=['rv', 'bis', 'fwhm', 'contrast'], t=None)
        steps['rv'] = time.perf_counter() - t
        res.update(rv=ss.results['rv'], fwhm=ss.results['fwhm'], bis=ss.results['bis'], contrast=ss.results['contrast'])
    res['time'] = ss.obs_times
    return res, steps


def child_new(conf, out, variant):
    t0 = time.perf_counter()
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from sunsim import simulate_old_config
    steps = {'import': time.perf_counter() - t0}
    paths = {'sdo_folder': str(WORK / 'maps_npy')} if variant == 'new_npy' else None
    t = time.perf_counter()
    r = simulate_old_config(conf, observables=OBSERVABLES, old_package=OLD_PACKAGE, paths=paths,
                            reproduce_old_bugs=(variant == 'new_old_bugs'), verbose=False)
    total = time.perf_counter() - t
    timing = dict(getattr(r['starsim'], 'timing', {}))
    sim = timing.pop('total', sum(timing.values()))
    steps['spectra and grids (old_config)'] = total - sim
    for k, v in timing.items():
        steps[k] = v
    res = {k: r[k] for k in ('lc', 'ff_sp', 'ff_fc', 'rv', 'fwhm', 'bis', 'contrast')}
    res['time'] = r['time']
    return res, steps


def child(variant, conf, out):
    import numpy as np
    t0 = time.perf_counter()
    res, steps = child_old(conf, out) if variant == 'old' else child_new(conf, out, variant)
    steps['total'] = time.perf_counter() - t0
    np.savez(out, **{k: np.asarray(v, dtype=float) for k, v in res.items()})
    with open(str(out) + '.json', 'w') as f:
        json.dump({'variant': variant, 'steps': steps, 'peak_rss_mb': peak_rss_mb()}, f)


# ------------------------------------------------------------------------------------------- driver
def run(variant, repeat=1):
    """Run a variant `repeat` times, each in a new process; returns the list of {steps, peak_rss_mb}."""
    conf = write_conf(SDO_MAPS)
    if variant == 'new_npy':
        folder, n, t = npy_maps()
        if n:
            print('  %d text maps saved as uint8 .npy in %.0f s (once)' % (n, t), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    python = OLD_PYTHON if variant == 'old' else NEW_PYTHON
    env = dict(os.environ, PYTHONPYCACHEPREFIX=str(WORK / 'cache' / ('pycache_' + variant)),
               NUMBA_CACHE_DIR=str(WORK / 'cache' / ('numba_' + ('old' if variant == 'old' else 'new'))),
               MPLBACKEND='Agg', KMP_WARNINGS='0')
    out = []
    for k in range(repeat):
        dst = RESULTS / ('%s_%d.npz' % (variant, k))
        p = subprocess.run([python, str(Path(__file__).resolve()), '_child', variant, str(conf), str(dst)],
                           env=env, capture_output=True, text=True, cwd=str(WORK))
        if p.returncode != 0:
            raise RuntimeError('%s failed:\n%s' % (variant, p.stderr[-3000:]))
        info = json.load(open(str(dst) + '.json'))
        out.append(info)
        print('  %-13s run %d: %7.1f s, peak memory %6.0f MB   (%s)' % (
            variant, k + 1, info['steps']['total'], info['peak_rss_mb'],
            ', '.join('%s %.1f s' % (s, v) for s, v in info['steps'].items() if s != 'total')), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['run', 'all', '_child'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--repeat', type=int, default=1)
    a = ap.parse_args()
    if a.what == '_child':
        variant, conf, out = a.args
        child(variant, Path(conf), Path(out))
    elif a.what == 'run':
        for v in a.args:
            run(v, a.repeat)
    else:
        for v in VARIANTS:
            run(v, a.repeat)


if __name__ == '__main__':
    main()
