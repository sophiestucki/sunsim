"""MURaM vxdrift-corrected spectra (quiet: ssd, faculae: 200G) at the 10 mu, resampled (flux conserving, specutils) to
the PHOENIX high resolution grid and to the PHOENIX low resolution specific intensity grid, 3800-8000 A, as in
~/Downloads/resampling_muram_spectra.ipynb.

    python analysis/resample_muram_spectra.py SOURCE_FOLDER OUTPUT_FOLDER

SOURCE_FOLDER holds vxdrift_corrected_ssd_spectra/ and vxdrift_corrected_200G_spectra/ (the .dat files: wavelength
[nm], intensity). The high resolution grid is the 'wav' of PHOENIX_GRID_FILE (the wavelengths of
WAVE_PHOENIX-ACES-AGSS-COND-2011.fits between 3800 and 8000 A); the low resolution grid is 1 A steps (the grid of the
PHOENIX SPECINT files). Output: the file names read through local_path('muram_spectra') and
local_path('muram_lowres_spectra'); 'std_intensity' is a copy of 'intensity', as in the notebook.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from specutils.manipulation import FluxConservingResampler

try:
    from specutils import Spectrum
except ImportError:                       # specutils < 2
    from specutils import Spectrum1D as Spectrum

PHOENIX_GRID_FILE = Path('/Users/sophiestucki/Documents/poet/mps/av_solar_ssd_200G_10_mu_angles_PHOENIX_high_res_grid_cleaned.npy')
MUS = ['%02d' % i for i in range(1, 11)]              # file index 01 ... 10 -> mu 0.1 ... 1.0
SURFACES = {'ssd': ('vxdrift_corrected_ssd_spectra', 'ssd_corrected_average_{}_90sec_cleaned.dat', 'av_solar_ssd_200G'),
            '200G': ('vxdrift_corrected_200G_spectra', '200G_corrected_average_{}_90sec_cleaned.dat', 'av_solar_faculae_200G')}
OUT_NAMES = {'high': '{}_10_mu_angles_PHOENIX_high_res_grid.npy',
             'low': '{}_10_mu_angles_PHOENIX_low_res_specint_grid_cleaned.npy'}


def grids():
    high = np.asarray(next(iter(np.load(PHOENIX_GRID_FILE, allow_pickle=True).item().values()))['wav'], dtype=np.float64)
    low = np.arange(500, 26000)
    low = low[(low > 3800) & (low < 8000)].astype(np.float64)
    return {'high': high, 'low': low}


def read_muram(folder, pattern):
    """{mu: (wavelength [A], intensity)} of the 10 .dat files."""
    out = {}
    for i, mu in enumerate(MUS):
        data = pd.read_csv(folder / pattern.format(mu), sep=r'\s+', header=None, names=['wav', 'intensity'])
        out['%.1f' % ((i + 1) / 10)] = (data['wav'].values * 10, data['intensity'].values)
    return out


def resample(spectra, grid):
    fluxcon = FluxConservingResampler()
    new_grid = grid * u.AA
    out = {}
    for mu, (w, f) in spectra.items():
        ok = ~np.isnan(f)
        spec = Spectrum(spectral_axis=w[ok] * u.AA, flux=f[ok] * u.Unit('erg s-1 cm-2 Hz-1'))
        new = np.asarray(fluxcon(spec, new_grid).flux.value, dtype=np.float64)
        out[mu] = {'wav': grid.copy(), 'intensity': new, 'std_intensity': new.copy()}
    return out


if __name__ == '__main__':
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    g = grids()
    for surface, (folder, pattern, prefix) in SURFACES.items():
        spectra = read_muram(source / folder, pattern)
        for kind, grid in g.items():
            t0 = time.time()
            data = resample(spectra, grid)
            np.save(output / OUT_NAMES[kind].format(prefix), data)
            n_nan = sum(np.isnan(d['intensity']).sum() for d in data.values())
            print('%s %s: %d pixels x %d mu, %d NaN, %.0f s -> %s'
                  % (surface, kind, len(grid), len(data), n_nan, time.time() - t0, OUT_NAMES[kind].format(prefix)))
