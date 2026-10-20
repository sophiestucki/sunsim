"""
Convergence of the quiet-Sun CCF with the number of rings (test_disc_spectrum_vs_ccf.ipynb, section 5).

One run = one method and one number of rings, in its own process, so that its time and its peak memory are measured
alone:
    python ring_convergence.py <method> <N_rings>        method: 'ccf' (CCF mode) or 'spectrum' (spectrum first)
prints one line of JSON: RV, FWHM, contrast, BIS of the CCF, the CCF itself, the number of cells, the time of the grid
and of the whole run [s] and the peak memory of the process [GB].
run(method, N) does it from python (a subprocess).
"""
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROTATION_PERIOD = 24.47              # [days] at the equator
INCLINATION_DEG = 7.25
STELLAR_RADIUS = 1.0
RV = 1000 * np.arange(-20, 20.25, 0.25)
MASK = 'HARPS_G20_mps_friendly.mas'
MURAM = 'av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid.npy'


def _peak_memory_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == 'darwin' else peak / 1e6     # bytes on macOS, kB on Linux


def _compute(method, n_rings):
    from sunsim import (local_path, StarSim, SOLAR_DIFFERENTIAL_ROTATION, flux_grid, CCF_grid, mask_weights, nbspectra,
                        vacuum2air)
    t_start = time.perf_counter()
    raw = np.load(str(local_path('muram_spectra') / MURAM), allow_pickle=True).item()
    spectra = {float(mu): {'wav': vacuum2air(np.asarray(d['wav'], dtype=np.float64)),
                           'intensity': np.asarray(d['intensity'], dtype=np.float64)} for mu, d in raw.items()}
    del raw
    wav = spectra[1.0]['wav']
    wvm, fm = mask_weights(str(local_path('ccf_masks') / MASK))
    keep = (wvm > wav[0] + 1) & (wvm < wav[-1] - 1)
    wvm, fm = wvm[keep], fm[keep]
    inc = np.deg2rad(INCLINATION_DEG)
    t0 = time.perf_counter()
    if method == 'spectrum':
        g = flux_grid(n_rings, spectra, 'quiet', None, wav[0], wav[-1], rotation_period=ROTATION_PERIOD, inclination=inc,
                      stellar_radius=STELLAR_RADIUS, stellar_rotation=True, mode='spectroscopy',
                      differential_rotation=SOLAR_DIFFERENTIAL_ROTATION)
        g.built_grid()
        ccf = nbspectra.cross_correlation_mask(RV, g.wv, g.disc_spectrum, wvm, fm, False, 2)
        n_cells = int(np.sum(g.Ngrid_in_ring)) if hasattr(g, 'Ngrid_in_ring') else -1
        del g
    else:
        g = CCF_grid(n_rings, spectra, RV, wvm, fm, normalization='none', interpolation='spectrum', spectrum_medium='air')
        g.built_grid(bisector=None)
        ss = StarSim(np.array([0.0]), n_rings, inclination=inc, rotation_period=ROTATION_PERIOD,
                     differential_rotation=SOLAR_DIFFERENTIAL_ROTATION, stellar_radius=STELLAR_RADIUS,
                     active_regions_map=[], mode=['ccf'], ccf_grid_qp=g, ccf_grid_sp=g, ccf_grid_fc=g)
        ss.generate_timeseries(verbose=False)
        ccf = ss.ccf_var[0]
        n_cells = int(np.sum(ss._Nin_arr)) if hasattr(ss, '_Nin_arr') else -1
    t_grid = time.perf_counter() - t0
    fitter = StarSim(np.array([0.0]), 4, inclination=inc, rotation_period=ROTATION_PERIOD,
                     differential_rotation=SOLAR_DIFFERENTIAL_ROTATION, stellar_radius=STELLAR_RADIUS,
                     active_regions_map=[], mode=['ccf'])
    r, contrast, fwhm, bis = (float(x[0]) for x in fitter.compute_ccf_params(rv=RV, ccf=[ccf])[:4])
    return {'method': method, 'N_rings': n_rings, 'n_cells': n_cells, 'rv': r, 'fwhm': fwhm, 'contrast': contrast,
            'bis': bis, 'ccf': list(map(float, ccf)), 't_grid': t_grid, 't_total': time.perf_counter() - t_start,
            'peak_memory_gb': _peak_memory_gb()}


def run(method, n_rings, python=sys.executable, timeout=7200):
    """One run in a subprocess (the time and memory of this run alone). Returns the dict of _compute."""
    env = dict(os.environ, KMP_WARNINGS='0')
    out = subprocess.run([python, os.path.abspath(__file__), method, str(n_rings)], capture_output=True, text=True,
                         timeout=timeout, env=env)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith('{')]
    if out.returncode != 0 or not lines:
        raise RuntimeError('%s, %d rings failed:\n%s' % (method, n_rings, out.stderr[-2000:]))
    return json.loads(lines[-1])


if __name__ == '__main__':
    print(json.dumps(_compute(sys.argv[1], int(sys.argv[2]))), flush=True)
