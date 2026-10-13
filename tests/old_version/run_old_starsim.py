"""
Run the OLD StarSim on one test case and save its results (used by test_old_version.py; run with the Python of the old
version, env_sunsim). Usage: python run_old_starsim.py job.json

job.json: {"old_package": folder of the old starsim code, "conf": configuration file, "out": .npz to write,
           "times": list of epochs or null (2D maps: the epochs of the maps), "observables": [...],
           "regions": rows of the spot map or null, "patch": null or 'efficiency64' (instrument efficiency in float64,
           as the old code computes it with numpy >= 2)}
The old code is imported from old_package and is not modified: the configuration given has plot_grid_map = 0 (no plot
files), and the caller sets PYTHONPYCACHEPREFIX / NUMBA_CACHE_DIR outside the old package.
"""
import contextlib
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None          # the old code shows a plot of every bisector it fits

job = json.load(open(sys.argv[1]))
sys.path.insert(0, os.path.dirname(os.path.abspath(job['old_package'])))
from starsim import StarSim                      # noqa: E402  (the old package)

if job.get('patch') == 'efficiency64':
    # numpy >= 2 behaviour of the line `eff = np.poly1d(coeff_eff)(wv_orders[i])` of the old instrument option: the
    # polynomial is evaluated in float64 (numpy 1.x keeps the float32 of the wavelength file and the result is
    # meaningless). Only the `np` seen by the old starsim module is wrapped; its files are not modified.
    import types
    import starsim.starsim as _old_module

    class _Poly64(np.poly1d):
        def __call__(self, val):
            return np.poly1d.__call__(self, np.asarray(val, dtype=np.float64))

    _np = types.ModuleType('numpy_poly64')
    _np.__dict__.update(np.__dict__)
    _np.poly1d = _Poly64
    _old_module.np = _np

with contextlib.redirect_stdout(io.StringIO()):
    old = StarSim(conf_file_path=job['conf'])
    if job['regions']:
        rows = np.array(job['regions'], dtype=float)
        old.spot_map = rows[np.argsort(rows[:, 0], kind='stable')]          # spots first, as the old code sorts them
        old.active_region_types = list(old.spot_map[:, 0])
        old.facular_area_ratio = [0.0] * len(old.spot_map)
    times = None if old.sdo_input else np.asarray(job['times'], dtype=float)
    res = {}
    obs = job['observables']
    if 'lc' in obs:
        old.compute_forward(observables=['lc'], t=times)
        res.update(lc=old.results['lc'], ff_sp=old.results['ff_sp'], ff_fc=old.results['ff_fc'], ff_pl=old.results['ff_pl'])
    rv_obs = [o for o in obs if o != 'lc']
    if rv_obs:
        old.compute_forward(observables=rv_obs, t=times)
        res.update(rv=old.results['rv'], fwhm=old.results['fwhm'], bis=old.results['bis'],
                   contrast=old.results['contrast'], ff_sp_rv=old.results['ff_sp_rv'],
                   ff_fc_rv=old.results['ff_fc_rv'], CCF=old.results['CCF'], ccf_quiet=old.results['ccf_quiet'])
res['time'] = np.asarray(old.obs_times, dtype=float)
np.savez(job['out'], **{k: np.asarray(v, dtype=float) for k, v in res.items()})
print('ok')
