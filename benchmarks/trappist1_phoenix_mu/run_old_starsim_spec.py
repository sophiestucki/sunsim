"""
Run the old StarSim (observables=['spec']) on this benchmark, as SunSim_PHOENIX_mu_test_sophie.ipynb (O. Porqueras),
and write its outputs in the same format (spectra.npz, filling_factors.csv, timings.csv).

    /path/to/env_sunsim/bin/python run_old_starsim_spec.py [old package folder] [output folder]

Defaults: the path 'old_starsim' (sunsim.local_path: environment variable SUNSIM_OLD_STARSIM or local_paths.ini, see
local_paths.example.ini), and results_phoenix_mu_old_local/. The old package
needs the Phoenix SPECINT models 2300-2600 K, logg 5.0-5.5 in models/Phoenix_mu.
The configuration file of the benchmark is read through a temporary copy that adds the keys this version of the old
code requires but does not use for 'spec' (highres_mu, ...). Side effects of the old code are disabled: the FITS files
written by compute_immaculate_spec (fixed paths of another computer) and the grid plots (plot_grid_map = 0).
"""
import os
import sys
import tempfile
import configparser
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from astropy.io import fits

HERE = Path(__file__).resolve().parent


def _local_path_module():
    """sunsim/local_paths.py of this repository, loaded alone (without importing sunsim, which this process may not
    have: the old code runs in its own environment)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('sunsim_local_paths', HERE.parents[1] / 'sunsim' / 'local_paths.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_package = Path(sys.argv[1]) if len(sys.argv) > 1 else _local_path_module().local_path('old_starsim')
output = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'results_phoenix_mu_old_local'

# keys read by the old StarSim.__init__ but missing in the benchmark file (not used by 'spec')
EXTRA_KEYS = {'general': {'spectral_library': 'phoenix', 'mps_spectra_folder': 'MPS-ATLAS_spectra/',
                          'stagger_spectra_folder': 'stagger/', 'mps_mu_grid': '0', 'bb_ratio': '1', 'ccf_norm': '1',
                          'flux_flat': '1', 'highres_mu': '1', 'smoothing_kernel': '0',
                          'remove_bis_spot': '1', 'add_median_diff': '0.0'},
              'rv': {'instrument': 'None', 'ccf_interpolation_spectra': '0'},
              'LD': {'limb_extrapolation': 'linear', 'limb_lim': '0.0', 'meunier': '0'},
              'spots': {'sdo_input': '0', 'sdo_input_path': 'SDO_input/'}}

conf = configparser.ConfigParser(inline_comment_prefixes='#', strict=False)
conf.read(HERE / 'TRAPPIST1_spec_PHOENIX_mu.conf')
for section, keys in EXTRA_KEYS.items():
    for key, value in keys.items():
        if not conf.has_option(section, key):
            conf.set(section, key, value)
conf.set('spots', 'plot_grid_map', '0')
tmp = Path(tempfile.mkdtemp()) / 'TRAPPIST1_spec_PHOENIX_mu.conf'
with open(tmp, 'w') as f:
    conf.write(f)

fits.HDUList.writeto = lambda self, *a, **k: None      # compute_immaculate_spec writes FITS files at fixed paths
sys.path.insert(0, str(old_package.parent))
import starsim
print('Using:', starsim.__file__)

ss = starsim.StarSim(str(tmp))
spots = np.loadtxt(HERE / 'spotmap.dat', ndmin=2)
ss.spot_map = spots
ss.active_region_types = spots[:, 0].astype(int).tolist()
ss.facular_area_ratio = [0.0] * len(spots)
ss.convolve_spec_with_filt = False
times = np.linspace(0, ss.rotation_period, 5, endpoint=False)

elapsed = []
for label in ['First call', 'Repeat']:
    start = perf_counter()
    ss.compute_forward(observables=['spec'], t=times)
    elapsed.append(perf_counter() - start)
    print(f'\n{label}: {elapsed[-1]:.3f} s')

output.mkdir(exist_ok=True)
pd.DataFrame({'time_day': times, **{f'{k}_percent': ss.results[k] for k in ['ff_ph', 'ff_sp', 'ff_fc', 'ff_pl']}}).to_csv(
    output / 'filling_factors.csv', index=False)
pd.DataFrame({'call': ['first', 'repeat'], 'seconds': elapsed}).to_csv(output / 'timings.csv', index=False)
np.savez_compressed(output / 'spectra.npz', time=times, spec_wv=ss.results['spec_wv'], spec=ss.results['spec'])
print('Saved to:', output)
