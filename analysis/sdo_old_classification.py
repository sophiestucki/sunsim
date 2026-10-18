"""
The previous SDO/HMI identification (simu_rv_opt_fast.py + identification_utils.py, before the fixes of sunsim.sdo), one
day at a time, to compare it with sunsim.sdo.identify_day (analysis/sdo_old_vs_new_harpsn.py).

It runs the functions of the previous identification_utils.py (path 'old_sdo_identification' of local_paths.ini: the
file itself) in the order of simu_rv_opt_fast.py: noise threshold x int(size / 4096) (0 below 4096 px), B / mu with mu
the sine of the heliocentric angle, active pixels |B| > 24 / mu, spots darker than I_th x the mean intensity of the active
pixels, gaussian smoothing (sigma 2 px, cut at 0.2), plage = groups of bright pixels larger than A_th micro-hemispheres
(micro_hemisphere_to_pixels at mu = 0.1); the disc cut with one pixel of margin; both axes flipped.
Returns the spot and facula maps as written by the previous script (1 / 0, NaN outside the disc) and their boolean
masks, in the frame of the maps (read with origin='lower').
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sunsim import local_path                                                    # noqa: E402

_OLD = None


def old_module():
    """The previous identification_utils.py (imported once)."""
    global _OLD
    if _OLD is None:
        path = local_path('old_sdo_identification')
        spec = importlib.util.spec_from_file_location('old_identification_utils', str(path))
        _OLD = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spec.loader.exec_module(_OLD)
    return _OLD


def old_identify(cont_map, los_map, size=4096, noise_th=8.0, I_th=0.89, A_th=20.0):
    """One day with the previous identification (module docstring); cont_map, los_map: sunpy maps of the HMI files.
    Returns a dict 'spot_map', 'facula_map' (float, NaN outside the disc), 'spot', 'facula' (boolean), 'ff_sp', 'ff_fc'
    [% of the disc pixels]."""
    import astropy.units as u
    import sunpy.map
    old = old_module()
    out_shape = (size, size)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):     # the old code prints a lot
        warnings.simplefilter('ignore')
        if size < 4096:
            cont_map = cont_map.resample(out_shape * u.pix)
            los_map = los_map.resample(out_shape * u.pix)
        xg, yg = old.coord_grid(los_map)
        los_map_clean = old.noise_threshold(los_map, threshold=noise_th)
        los_map_clean = old.removing_foreshortening_effect(los_map_clean, xg, yg)
        row = cont_map.data[int(out_shape[0] / 2), :]
        min_p = int(np.argwhere(row >= 0).min() - 1)
        max_p = int(np.argwhere(row >= 0).max() + 1)
        cont_crop = cont_map.data[min_p:max_p, min_p:max_p]
        los_crop = np.abs(los_map_clean.data[min_p:max_p, min_p:max_p])
        cont_map = sunpy.map.sources.HMIMap(cont_crop, cont_map.fits_header)
        los_map = sunpy.map.sources.HMIMap(los_crop, los_map_clean.fits_header)
        xg = xg[min_p:max_p, min_p:max_p]
        yg = yg[min_p:max_p, min_p:max_p]
        active_area = old.active_area_identification(cont_map, los_map, xg, yg, I_th=I_th)
        spot_area, smooth_spot_area, feature_area, smooth_feature_area = old.active_area_smoothing(
            active_area, spot_threshold=0.2, spot_kernel_size=2, feature_threshold=0.2, feature_kernel_size=2)
        area_th = old.micro_hemisphere_to_pixels(los_map_clean, A_th, los_map_clean.scale[0], 0.1).value
        identification, plage_nbr = old.network_identification(feature_area, feature_area, area_th, method='scipy')
    facula_map = np.copy(identification)
    facula_map[np.isnan(facula_map)] = 0
    facula_map[np.isnan(cont_map.data)] = np.nan
    spot_map = np.copy(spot_area)
    spot_map[np.isnan(spot_map)] = 0
    spot_map[np.isnan(cont_map.data)] = np.nan
    facula_map = np.flip(facula_map)
    spot_map = np.flip(spot_map)
    n_disc = max(int(np.sum(np.isfinite(spot_map))), 1)
    spot, facula = spot_map == 1, facula_map == 1
    return {'spot_map': spot_map, 'facula_map': facula_map, 'spot': spot, 'facula': facula,
            'ff_sp': 100.0 * spot.sum() / n_disc, 'ff_fc': 100.0 * facula.sum() / n_disc}
