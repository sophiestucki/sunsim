"""
local_paths: the folders and files that are not part of SunSim (models, spectra, maps, other codes), given once per
computer instead of being written in the notebooks, tests and benchmarks.

    from sunsim import local_path
    folder = local_path('phoenix_models')                     # error with instructions if it is not set
    maps = local_path('sdo_maps', required=False)             # None if it is not set

Every key is looked for, in this order:
1. the environment variable SUNSIM_<KEY> (e.g. SUNSIM_PHOENIX_MODELS=/data/phoenix);
2. the section [paths] of a file local_paths.ini: the file named by the environment variable SUNSIM_PATHS_FILE, or
   local_paths.ini in the current folder or one of its parents (e.g. the root of the repository), or
   ~/.config/sunsim/local_paths.ini. Copy local_paths.example.ini (root of the repository) to local_paths.ini and
   fill in the paths you have; local_paths.ini is not tracked by git.
Paths can start with ~ and can be relative to the folder of the .ini file.

KEYS lists the keys used by the tutorial, the tests and the benchmarks.
"""
import os
from configparser import ConfigParser
from pathlib import Path

KEYS = {
    'phoenix_models':   'folder with the Phoenix models: subfolders Phoenix_mu (SPECINT) and Phoenix (HiRes, with '
                        'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits)',
    'muram_spectra':    'folder with the MURaM (MPS-ATLAS) high resolution spectra av_solar_ssd_200G_*_high_res_grid.npy '
                        'and av_solar_faculae_200G_*_high_res_grid.npy',
    'muram_lowres_spectra': 'folder with the MURaM low resolution spectra (cleaned) '
                        'av_solar_ssd_200G_*_low_res_specint_grid_cleaned.npy and av_solar_faculae_200G_*',
    'muram_spot_spectra': 'folder with the MURaM spectra used for the spots (av_solar_ssd_10_mu_angles_*)',
    'ccf_masks':        'folder with the CCF masks (e.g. HARPS_G20_mps_friendly.mas)',
    'sdo_data':         'folder with SDO/HMI images (hmi.ic_nolimbdark_720s.*continuum.fits, hmi.m_45s.*magnetogram.fits) '
                        'for sunsim.sdo and tutorial/tutorial_sdo.ipynb',
    'sdo_maps':         'folder with 2D maps spot_map_<t>.txt / faculae_map_<t>.txt (e.g. the SDO maps of 2015-2016)',
    'old_starsim':      'folder of the old StarSim package (the folder starsim/ with starsim.py, models/, masks/, '
                        'MPS-ATLAS_spectra/), for the tests against the old version',
    'old_starsim_data': 'an old StarSim package with the data (models/, masks/, MPS-ATLAS_spectra/, SDO_input/), used by '
                        'the tests when the folder of old_starsim does not have them',
    'old_python':       'Python executable of the environment in which the old StarSim runs',
    'old_harps_files':  'folder where the old StarSim reads its HARPS-N files (blaze, wavelengths, plot-data-3.csv)',
    'starsim4':         'folder of StarSim v4 (branch inverse-model), for benchmark_starsim4.ipynb',
    'starsim_david':    'folder of StarSim of D. Baroch, for benchmark_starsim_david.ipynb',
    'sunsim_python':    'Python executable used to run SunSim in the benchmarks (default: the current Python)',
    'sun_mask_data':    'input data of the solar mask notebooks (making_mask.ipynb, masks.ipynb)',
    'harpsn_sun':       'HARPS-N solar telescope csv of the public release (sunsim.observations.load_harpsn)',
    'harpsn_interventions': 'text file with the dates of the HARPS-N interventions, one YYYY-MM-DD per line',
    'sorce_tsi':        'SORCE/TIM total solar irradiance, IDL .sav with time [JD] and tsi (sunsim.observations.load_sorce)',
    'tsis_tsi':         'TSIS-1/TIM total solar irradiance csv, tsis_tsi_24hr.csv (sunsim.observations.load_tsis)',
    'old_sdo_identification': 'identification_utils.py of the previous SDO pipeline (analysis/sdo_old_classification.py)',
    'ref_rv_10_years':  'RV of the 10-year run of the old StarSim (analysis/sdo_old_vs_new_harpsn.py, optional)',
    'ref_days_10_years': 'its days_list.txt (days since 2015-07-18)',
    'sweep_results':    'results of the PIC sweep (PIC/opt_sweep.ipynb): the folder of the sets',
    'supergranulation_results': 'folder of PIC/supergranulation.py (days/, results/), for '
                        'analysis/supergranulation_pic_2021_quality.ipynb',
    'poet_data':        'data of the PoET notebooks (spectra, outputs, figures)',
    'personal_data':    'other input files of the analysis notebooks (e.g. masks made by hand)',
}

_ENV_FILE = 'SUNSIM_PATHS_FILE'
_FILE_NAME = 'local_paths.ini'


def _config_file():
    """The local_paths.ini file to read, or None."""
    if os.environ.get(_ENV_FILE):
        return Path(os.environ[_ENV_FILE]).expanduser()
    here = Path.cwd().resolve()
    for folder in [here, *here.parents, Path(__file__).resolve().parent.parent]:
        if (folder / _FILE_NAME).is_file():
            return folder / _FILE_NAME
    home = Path.home() / '.config' / 'sunsim' / _FILE_NAME
    return home if home.is_file() else None


def local_path(key, required=True, default=None):
    """Path of `key` on this computer (see the module docstring): a pathlib.Path.
    required : if the path is not set (and there is no default), True raises an error that says how to set it,
               False returns None.
    default  : value used when the key is not set (a path, or None)."""
    key = key.lower()
    value = os.environ.get('SUNSIM_' + key.upper())
    origin = 'environment variable SUNSIM_%s' % key.upper()
    base = Path.cwd()
    if not value:
        path = _config_file()
        if path is not None:
            conf = ConfigParser()
            conf.read(path)
            if conf.has_option('paths', key) and conf.get('paths', key).strip():
                value = conf.get('paths', key).strip()
                origin = '%s, [paths] %s' % (path, key)
                base = path.parent
    if not value:
        if default is not None:
            return Path(default)
        if not required:
            return None
        raise FileNotFoundError(
            "the path '%s' is not set on this computer (%s).\nSet it with the environment variable SUNSIM_%s, or in the "
            "section [paths] of a file %s (in the folder of the repository, the current folder or ~/.config/sunsim/): "
            "copy local_paths.example.ini of the repository to %s and fill it in."
            % (key, KEYS.get(key, 'see sunsim.local_paths.KEYS'), key.upper(), _FILE_NAME, _FILE_NAME))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    if required and not path.exists():
        raise FileNotFoundError("the path '%s' given by %s does not exist: %s" % (key, origin, path))
    return path
