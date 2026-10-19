"""
supergranulation: line-of-sight velocity of the quiet photosphere from SDO/HMI dopplergrams, for
StarSim(..., quiet_velocity=...), and the HMI images closest in time to given epochs (e.g. every HARPS-N exposure),
downloaded and processed one epoch at a time (the FITS files can be deleted right after).

One epoch (the three images of the same time: continuum, magnetogram, dopplergram)
-------------------------------------------------------------------------------
1. records    query_records(series, jd_start, jd_end) lists the records of a JSOC series (drms, no account needed);
              closest(jd, jd_records) gives the record closest to every time. fetch_record(series, t_rec, segment,
              folder) downloads one image from the JSOC storage (no export request, so no registered email) with the
              keywords of the record, read_record() makes the sunpy map. fetch_epoch() does the three images.
2. identification: sdo.identify_day on the continuum and the magnetogram (spots, plage, network; cell filling
              factors with sdo.cell_filling_factors), as for the daily maps.
3. dopplergram: clean_dopplergram removes from the HMI velocity [m/s, > 0 away from the observer]
                - the velocity of the observer (keywords OBS_VR, OBS_VW, OBS_VN; Haywood et al. 2016, eq. 2-3):
                  v_obs = -(dw OBS_VW + dn OBS_VN + dr OBS_VR) / d, (dw, dn, dr) the vector from SDO to the pixel;
                - the rotation of the Sun: v_rot = x (a0 + a2 sin^2 B + a4 sin^4 B), x the westward distance of the pixel
                  from the central meridian [solar radii] (x = cos B sin L), B the heliographic latitude, fitted on the
                  quiet pixels together with a polynomial in mu (convective blueshift, offsets), which is not removed.
              What is left: the convective blueshift (centre-to-limb), the supergranulation, the meridional flow, the
              p modes left in the 720 s average, an offset.
4. cells      cell_velocities: mean velocity of the quiet pixels of every cell of a SunSim grid (weighted by the
              continuum intensity), in the frame of the maps (origin='lower').
process_epoch() does 2-4 and returns what StarSim needs (filling factors and velocities of the cells).

All the epochs
--------------
remove_ring_profile: the velocity of a cell still has the centre-to-limb convective blueshift seen by HMI, and the
grids of SunSim already have the convective blueshift of their spectra: the mean over all the epochs of the velocity
of every ring is removed (per_epoch=True: the mean of every epoch, which also removes the disc-averaged velocity of
every epoch).
The HMI velocities have a calibration error that follows the velocity of SDO (OBS_VR, +-3 km/s over a day): the
rotation fitted on every dopplergram changes by ~1.5 % and the disc-averaged velocity by a few m/s in phase with it.
The dopplergrams also have a north-south gradient of tens of m/s that follows OBS_VR.
rescale_by_rotation divides every epoch by its rotation (relative to the mean), remove_large_scale removes the north-south
gradient of every epoch, remove_observer_trend removes from the disc-averaged velocity a polynomial of OBS_VR fitted over
all the epochs.
quiet_velocity_field(times, v, N_rings) gives the result to StarSim(quiet_velocity=...) (or
simulate_old_config(quiet_velocity=...)): in every cell, the CCF of the quiet photosphere is shifted by its velocity.

p modes of every exposure (45 s dopplergrams)
---------------------------------------------
The 720 s dopplergrams average the p modes (~5 min) out, and their time is up to 6 min from the exposure. The 45 s
dopplergrams (hmi.V_45s, SERIES_45S) follow them: frame_cell_velocities gives the mean velocity of every cell of one
frame (SDO motion removed, rotation kept), pmode_velocities the mean over every exposure minus the mean over a window
of ref_window [s] centred on it (1800 s: the p modes, the granulation and everything faster than ~30 min; the rotation,
the slow drifts of the HMI zero point and scale, linear over the window, cancel). Added to the velocity of the epoch
(v_sg[epoch of the exposure] + delta), the same p modes as in the exposure.

Needs sunpy and drms (pip install "sunsim[sdo]").
"""
import os
import time as _time
import urllib.request
import warnings
from collections import deque

import numpy as np

from . import sdo

JSOC_URL = 'http://jsoc.stanford.edu'
#kind -> (series, segment) of the three images of an epoch
SERIES = {'continuum': ('hmi.Ic_noLimbDark_720s', 'continuum'),
          'magnetogram': ('hmi.M_45s', 'magnetogram'),
          'dopplergram': ('hmi.V_720s', 'Dopplergram')}
SERIES_45S = ('hmi.V_45s', 'Dopplergram')          # the dopplergrams of the p modes


# =============================================================================================
# times and records
# =============================================================================================
def jd_of_t_rec(values):
    """JD (UTC) of JSOC times 'YYYY.MM.DD_HH:MM:SS[.s]_TAI' (T_REC, T_OBS): an array (NaN for missing values)."""
    from astropy.time import Time
    values = np.atleast_1d(values)
    out = np.full(len(values), np.nan)
    iso, where = [], []
    for k, v in enumerate(values):
        v = str(v)
        if not v.endswith('_TAI') or len(v) < 19:
            continue
        iso.append(v[:-4].replace('.', '-', 2).replace('_', 'T'))
        where.append(k)
    if iso:
        out[where] = Time(iso, format='isot', scale='tai').utc.jd
    return out


def t_rec_of_jd(jd):
    """JSOC time 'YYYY.MM.DD_HH:MM:SS_TAI' of a JD (UTC)."""
    from astropy.time import Time
    return Time(float(jd), format='jd', scale='utc').tai.strftime('%Y.%m.%d_%H:%M:%S') + '_TAI'


def _client():
    try:
        import drms
    except ImportError:
        raise ImportError('the JSOC records need drms: pip install drms (or pip install "sunsim[sdo]")') from None
    return drms.Client()


def query_records(series, jd_start, jd_end, good_only=True, client=None):
    """Records of a JSOC series between two JD (UTC): a pandas DataFrame with T_REC, T_OBS, QUALITY and 'jd' (T_OBS in
    JD UTC), sorted. good_only: only QUALITY = 0 and a valid T_OBS."""
    client = client or _client()
    ds = '%s[%s-%s]' % (series, t_rec_of_jd(jd_start), t_rec_of_jd(jd_end))
    rec = client.query(ds, key='T_REC,T_OBS,QUALITY')
    if len(rec) == 0:
        rec['jd'] = []
        return rec
    rec['jd'] = jd_of_t_rec(rec['T_OBS'].values)
    rec = rec[np.isfinite(rec['jd'])]
    if good_only:
        rec = rec[rec['QUALITY'].astype(np.int64) == 0]
    return rec.sort_values('jd').reset_index(drop=True)


def closest(jd, jd_records):
    """Index of the record closest to every time (jd, jd_records in days) and |difference| [s]."""
    jd = np.atleast_1d(np.asarray(jd, dtype=np.float64))
    r = np.asarray(jd_records, dtype=np.float64)
    if len(r) == 0:
        return np.full(len(jd), -1), np.full(len(jd), np.inf)
    order = np.argsort(r)
    rs = r[order]
    k = np.clip(np.searchsorted(rs, jd), 1, len(rs) - 1) if len(rs) > 1 else np.zeros(len(jd), int)
    if len(rs) > 1:
        k = np.where(np.abs(rs[k - 1] - jd) <= np.abs(rs[k] - jd), k - 1, k)
    return order[k], np.abs(rs[k] - jd) * 86400.0


def fetch_record(series, t_rec, segment, folder, client=None, retries=3, timeout=600):
    """Downloads the image `segment` of the record series[t_rec] into folder (the file of the JSOC storage, without
    an export request: no email needed) and returns (path, keywords): keywords is the dict of all the keywords of the
    record (the file itself has none). read_record(path, keywords) makes the sunpy map."""
    import drms
    client = client or _client()
    ds = '%s[%s]' % (series, t_rec)
    for attempt in range(retries):
        try:
            keys, segs = client.query(ds, key=drms.const.all, seg=segment)
            if len(keys) == 0:
                raise FileNotFoundError('no record %s' % ds)
            path = os.path.join(str(folder), '%s.%s.%s.fits' % (series, t_rec.replace(':', ''), segment))
            with urllib.request.urlopen(JSOC_URL + segs[segment].iloc[0], timeout=timeout) as r, open(path + '.part', 'wb') as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            os.replace(path + '.part', path)
            return path, keys.iloc[0].to_dict()
        except Exception:
            if attempt == retries - 1:
                raise
            _time.sleep(5 * (attempt + 1))


def read_record(path, keywords):
    """sunpy map of an image of fetch_record with the keywords of its record (drms names: DATE__OBS -> DATE-OBS)."""
    from astropy.io import fits
    smap = sdo._sunpy()
    with fits.open(path) as h:
        data = np.asarray(h[-1].data, dtype=np.float32)
    header = fits.Header()
    for k, v in keywords.items():
        name = k.replace('__', '-')
        if len(name) > 8 or name in ('COMMENT', 'HISTORY') or v is None:
            continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            continue
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif not isinstance(v, (int, float, str)):
            v = str(v)
        try:
            header[name] = v
        except ValueError:
            continue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return smap.Map(data, header)


def fetch_epoch(epoch, folder, series=SERIES):
    """The three images of one epoch: epoch = {'continuum': t_rec, 'magnetogram': t_rec, 'dopplergram': t_rec} (the
    T_REC of every kind). Returns {kind: (path, keywords)} (fetch_record); the files are in folder. If one image
    cannot be downloaded, the files already downloaded are deleted and the error is raised."""
    client = _client()
    out = {}
    try:
        for kind in ('continuum', 'magnetogram', 'dopplergram'):
            out[kind] = fetch_record(series[kind][0], epoch[kind], series[kind][1], folder, client=client)
    except Exception:
        for path, _ in out.values():
            if os.path.exists(path):
                os.remove(path)
        raise
    return out


def prefetch(items, fetch, workers=4, ahead=None):
    """Yields (item, fetch(item)) in the order of items, with up to `ahead` (default 2 x workers) items fetched in
    advance by `workers` threads: downloads go on while the previous epoch is processed, and at most `ahead` epochs of
    files are on disk. An exception of fetch is yielded in place of its result."""
    from concurrent.futures import ThreadPoolExecutor
    ahead = max(1, int(ahead or 2 * workers))
    items = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        queue = deque()
        for item in items:
            queue.append((item, pool.submit(fetch, item)))
            if len(queue) >= ahead:
                break
        while queue:
            item, fut = queue.popleft()
            nxt = next(items, None)
            if nxt is not None:
                queue.append((nxt, pool.submit(fetch, nxt)))
            try:
                res = fut.result()
            except Exception as e:      # noqa: BLE001 (given to the caller)
                res = e
            yield item, res


# =============================================================================================
# dopplergram
# =============================================================================================
def disc_coordinates(smap, dtype=np.float32):
    """x (towards solar west) and y (towards solar north) of every pixel in solar radii, from the disc centre in the
    plane of the sky, and mu = sqrt(1 - x^2 - y^2) (NaN outside the disc), from the WCS of the map (offsets from the
    pixel of the disc centre times the plate scale, turned by its rotation matrix: CROTA2 = 180 for HMI)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ny, nx = smap.data.shape
    centre = smap.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=smap.coordinate_frame))
    rs = smap.rsun_obs.to_value(u.arcsec)
    dx = ((np.arange(nx) - centre.x.to_value(u.pix)) * smap.scale.axis1.to_value(u.arcsec / u.pix) / rs).astype(dtype)
    dy = ((np.arange(ny) - centre.y.to_value(u.pix)) * smap.scale.axis2.to_value(u.arcsec / u.pix) / rs).astype(dtype)
    pc = np.asarray(smap.rotation_matrix, dtype=np.float64)
    x = dtype(pc[0, 0]) * dx[None, :] + dtype(pc[0, 1]) * dy[:, None]
    y = dtype(pc[1, 0]) * dx[None, :] + dtype(pc[1, 1]) * dy[:, None]
    with np.errstate(invalid='ignore'):
        mu = np.sqrt(1 - x * x - y * y)
    return x, y, mu


def observer_velocity(smap, x, y, mu):
    """Line-of-sight velocity [m/s, > 0 away from SDO] of every pixel due to the motion of SDO (OBS_VR, OBS_VW, OBS_VN;
    Haywood et al. 2016): -(dw OBS_VW + dn OBS_VN + dr OBS_VR) / d, (dw, dn, dr) the vector from SDO to the pixel
    (west, north, radial from the Sun) and d its length."""
    m = smap.meta
    R = float(m.get('rsun_ref', 696000000.0))
    D = float(m['dsun_obs'])
    vr, vw, vn = (float(m[k]) for k in ('obs_vr', 'obs_vw', 'obs_vn'))
    dw, dn, dr = x * R, y * R, mu * R - D
    d = np.sqrt(dw * dw + dn * dn + dr * dr)
    return -(dw * vw + dn * vn + dr * vr) / d


def clean_dopplergram(v_map, fit_mask=None, mu_min=0.1, n_mu=4, max_fit_pixels=500000):
    """Velocity of the dopplergram v_map [m/s, > 0 away from the observer] without the motion of SDO and the rotation
    of the Sun (module docstring).
    fit_mask : pixels used to fit the rotation (boolean, shape of the map; e.g. the quiet Sun), with mu > mu_min
    n_mu     : degree of the polynomial in mu fitted together with the rotation (not removed)
    max_fit_pixels : the fit uses every k-th of these pixels so that at most this many are used
    Returns a dict: 'v' (residual velocity, float32, NaN outside the disc), 'coeffs' (a0, a2, a4 [m/s] of the rotation),
    'omega' (rotation rate at the equator [deg/day], from a0), 'B0' [deg], 'mu', 'x', 'y' (disc_coordinates), 'n_fit'."""
    v = np.asarray(v_map.data, dtype=np.float32)
    x, y, mu = disc_coordinates(v_map)
    vv = v - observer_velocity(v_map, x, y, mu).astype(np.float32)
    b0 = np.deg2rad(float(v_map.meta.get('crlt_obs', 0.0)))
    with np.errstate(invalid='ignore'):
        sinb2 = (y * np.float32(np.cos(b0)) + mu * np.float32(np.sin(b0))) ** 2
        ok = np.isfinite(vv) & (mu > mu_min)
    if fit_mask is not None:
        ok &= np.asarray(fit_mask, dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 100:
        raise ValueError('only %d pixels to fit the rotation of the dopplergram' % len(idx))
    idx = idx[::max(1, len(idx) // int(max_fit_pixels))]
    xf, sf, mf = (a.ravel()[idx].astype(np.float64) for a in (x, sinb2, mu))
    A = np.column_stack([xf, xf * sf, xf * sf * sf] + [mf ** p for p in range(n_mu + 1)])
    coef = np.linalg.lstsq(A, vv.ravel()[idx].astype(np.float64), rcond=None)[0]
    a0, a2, a4 = (np.float32(c) for c in coef[:3])
    vv -= x * (a0 + sinb2 * (a2 + sinb2 * a4))
    R = float(v_map.meta.get('rsun_ref', 696000000.0))
    omega = float(coef[0]) / (R * np.cos(b0)) * 86400 * 180 / np.pi
    return {'v': vv, 'coeffs': coef[:3], 'omega': omega, 'B0': np.rad2deg(b0), 'mu': mu, 'x': x, 'y': y,
            'n_fit': len(idx)}


def cell_velocities(v, mask, weight, n_rings):
    """Mean velocity of the pixels of mask in every cell of a SunSim grid of n_rings rings, weighted by weight (e.g.
    the continuum intensity); arrays in the frame of the maps of identify_day (origin='lower'). Returns v_cell
    (n_cells,; NaN for a cell without any pixel of mask) and w_cell (the sum of the weights of every cell)."""
    cells, npix = sdo.cell_projection(np.shape(v), n_rings)
    sel = np.asarray(mask, dtype=bool).ravel() & np.isfinite(np.ravel(v)) & np.isfinite(np.ravel(weight))
    sel &= cells >= 0
    c = cells[sel]
    w = np.ravel(weight)[sel].astype(np.float64)
    sw = np.bincount(c, weights=w, minlength=len(npix))
    swv = np.bincount(c, weights=w * np.ravel(v)[sel].astype(np.float64), minlength=len(npix))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(sw > 0, swv / sw, np.nan), sw


def process_epoch(files, size=4096, n_rings=30, thresholds=None, mu_min=0.1, network_velocity=True):
    """Identification and velocities of one epoch from the three images of fetch_epoch (files = {kind: (path,
    keywords)}).
    size        : the images are resampled to size x size pixels (sdo.identify_day)
    n_rings     : SunSim grid of the filling factors and velocities of the cells
    thresholds  : dict of the thresholds of sdo.identify_day (noise_th, B_th, I_th, A_th, mu_area, faculae)
    mu_min      : the rotation is fitted on the quiet pixels with mu > mu_min
    network_velocity : True: the velocity of a cell is the mean over the pixels that SunSim counts as quiet photosphere
                  (neither spot nor facula: with faculae='plage' the network is in it), False: only the pixels that
                  are not active at all (no network)
    Returns a dict: 'ff_sp', 'ff_fc' (cell filling factors, float32), 'v_cell', 'w_cell' (cell_velocities), and the
    numbers of the epoch: 'ff_sp_disc', 'ff_fc_disc', 'ff_nt_disc' [%], 'n_spots', 'n_plages', 'omega' [deg/day],
    'rotation' (a0, a2, a4 [m/s]), 'v_quiet_mean' (mean residual velocity of the quiet pixels [m/s]), 'obs_vr' [m/s],
    'B0' [deg], 'shift_px' (distance between the disc centres of the continuum and the dopplergram [pixels]), and
    'frames' (the arrays of the figures, in the frame of the maps: 'v' residual velocity, 'v_raw', 'quiet')."""
    import astropy.units as u
    cont = read_record(*files['continuum'])
    mag = read_record(*files['magnetogram'])
    dop = read_record(*files['dopplergram'])
    shift = float(np.hypot(cont.reference_pixel.x.to_value(u.pix) - dop.reference_pixel.x.to_value(u.pix),
                           cont.reference_pixel.y.to_value(u.pix) - dop.reference_pixel.y.to_value(u.pix)))
    obs_vr = float(dop.meta.get('obs_vr', np.nan))
    day = sdo.identify_day(cont, mag, size=size, **(thresholds or {}))
    del cont, mag
    if size is not None and int(size) != dop.data.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dop = dop.resample([int(size), int(size)] * u.pix)
    lo, hi = sdo.disc_square(day['cont_map'].data)

    def frame(a):
        return np.flip(np.asarray(a)[lo:hi, lo:hi])

    def unframe(a):
        out = np.zeros(day['cont_map'].data.shape, dtype=bool)
        out[lo:hi, lo:hi] = np.flip(a)
        return out

    still = day['disc'] & ~(day['spot'] | day['plage'] | day['network'])        # not active at all
    quiet = day['disc'] & ~(day['spot'] | day['facula']) if network_velocity else still
    clean = clean_dopplergram(dop, fit_mask=unframe(still), mu_min=mu_min)
    v = frame(clean['v'])
    intensity = frame(day['cont_map'].data)
    v_cell, w_cell = cell_velocities(v, quiet, intensity, n_rings)
    ff_sp, ff_fc = sdo.cell_filling_factors(day, n_rings)
    with np.errstate(invalid='ignore'):
        vq = float(np.nanmean(np.where(quiet, v, np.nan)))
    return {'ff_sp': ff_sp, 'ff_fc': ff_fc, 'v_cell': v_cell.astype(np.float32), 'w_cell': w_cell.astype(np.float32),
            'ff_sp_disc': day['ff_sp'], 'ff_fc_disc': day['ff_fc'], 'ff_nt_disc': day['ff_nt'],
            'n_spots': int(day['n_spots']), 'n_plages': int(day['n_plages']), 'omega': clean['omega'],
            'rotation': clean['coeffs'], 'v_quiet_mean': vq, 'obs_vr': obs_vr, 'B0': clean['B0'], 'shift_px': shift,
            'frames': {'v': v, 'v_raw': frame(np.asarray(dop.data, dtype=np.float32)), 'quiet': quiet,
                       'spot': day['spot'], 'facula': day['facula']}}


# =============================================================================================
# all the epochs
# =============================================================================================
def ring_of_cells(n_rings):
    """Ring index of every cell of a SunSim grid of n_rings rings."""
    from . import nbspectra
    nin = np.asarray(nbspectra.generate_grid_coordinates_nb(int(n_rings))[1], dtype=np.int64)
    return np.repeat(np.arange(int(n_rings)), nin)


def remove_ring_profile(v_cell, w_cell, n_rings, per_epoch=False):
    """Velocities of the cells (n_times, n_cells) without the mean velocity of their ring (weighted by w_cell): the
    centre-to-limb convective blueshift seen by HMI and the offset of the dopplergrams, already in the spectra of
    the grids. per_epoch=False (default): the mean over all the epochs of the ring means (the velocity of the
    supergranulation averaged over the disc stays, with the changes of the zero point of HMI); True: the ring means
    of every epoch (only the structure inside every ring stays).
    Returns v (n_times, n_cells; 0 where v_cell is NaN) and the profile removed ((n_rings,) or (n_times, n_rings))."""
    v = np.asarray(v_cell, dtype=np.float64)
    w = np.where(np.isfinite(v), np.asarray(w_cell, dtype=np.float64), 0.0)
    ring = ring_of_cells(n_rings)
    if v.shape[-1] != len(ring):
        raise ValueError('v_cell has %d cells, the grid of %d rings %d' % (v.shape[-1], n_rings, len(ring)))
    vz = np.where(np.isfinite(v), v, 0.0)
    sw = np.stack([np.bincount(ring, weights=wk, minlength=n_rings) for wk in np.atleast_2d(w)])
    swv = np.stack([np.bincount(ring, weights=wk * vk, minlength=n_rings) for wk, vk in zip(np.atleast_2d(w), np.atleast_2d(vz))])
    if per_epoch:
        with np.errstate(invalid='ignore', divide='ignore'):
            prof = np.where(sw > 0, swv / sw, 0.0)
        out = vz - prof[:, ring]
    else:
        tot = sw.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            prof = np.where(tot > 0, swv.sum(axis=0) / tot, 0.0)
        out = vz - prof[ring]
    out[~np.isfinite(v)] = 0.0
    return out, prof


def rescale_by_rotation(v_cell, a0):
    """Velocities of every epoch divided by a0(t) / <a0>, a0 the equatorial rotation fitted on its dopplergram
    (process_epoch 'rotation'[0]): the rotation of the Sun does not change in hours, so the changes of a0 are changes
    of the velocity scale of HMI (they follow the velocity of SDO, OBS_VR). Returns the velocities and the scales."""
    a0 = np.asarray(a0, dtype=np.float64)
    scale = a0 / np.mean(a0)
    return np.asarray(v_cell, dtype=np.float64) / scale[:, None], scale


LARGE_SCALE_TERMS = {'ns': lambda e, n, mu: n, 'ew': lambda e, n, mu: e, 'ns_ew': lambda e, n, mu: e * n,
                     'ns_mu': lambda e, n, mu: n * mu}


def remove_large_scale(v_cell, w_cell, n_rings, terms=('ns',), mu_degree=3, mu_min=0.1):
    """Large-scale patterns of the velocities of the cells removed at every epoch: for every epoch, a weighted fit of
    the terms (in the plane of the sky, in solar radii: 'ns' = north (gradient north-south), 'ew' = west, 'ns_ew' =
    west x north, 'ns_mu' = north x mu) together with a polynomial in mu of degree mu_degree (not removed: the ring
    profile, remove_ring_profile), on the cells with mu > mu_min; only the terms are removed.
    The HMI dopplergrams have a north-south gradient of tens of m/s that follows the velocity of SDO (OBS_VR), not
    the Sun: it would give an RV wherever the spots and faculae cover the two hemispheres differently.
    Returns v (n_times, n_cells; NaN kept) and the coefficients (n_times, len(terms)) [m/s per solar radius]."""
    from . import nbspectra
    g = nbspectra.generate_grid_coordinates_nb(int(n_rings))
    ring = ring_of_cells(n_rings)
    east_west, north, mu = np.asarray(g[7]), np.asarray(g[8]), np.asarray(g[3])[ring]
    cols = [LARGE_SCALE_TERMS[t](east_west, north, mu) for t in terms]
    v = np.array(v_cell, dtype=np.float64)
    w = np.asarray(w_cell, dtype=np.float64)
    coefs = np.zeros((len(v), len(terms)))
    for k in range(len(v)):
        m = np.isfinite(v[k]) & (mu > mu_min) & (w[k] > 0)
        A = np.column_stack([c[m] for c in cols] + [mu[m] ** p for p in range(int(mu_degree) + 1)])
        sw = np.sqrt(w[k][m])
        c = np.linalg.lstsq(A * sw[:, None], v[k][m] * sw, rcond=None)[0][:len(terms)]
        coefs[k] = c
        v[k] -= sum(ci * col for ci, col in zip(c, cols))
    return v, coefs


def remove_observer_trend(v, weights, obs_vr, degree=2):
    """The disc-averaged velocity of every epoch (mean of v over the cells, weighted by weights, e.g. the quiet
    intensity of the cells times the limb darkening), fitted over all the epochs by a polynomial of OBS_VR of this degree
    and removed from all the cells of the epoch: the zero point of the HMI velocities changes with the velocity of SDO
    (a calibration error), which would otherwise be taken for a velocity of the Sun. What the Sun does in the same way as
    OBS_VR (a daily cycle) is removed too: use a period of several days.
    Returns v (n_times, n_cells), the disc-averaged velocity before and after, and the coefficients (np.polyfit)."""
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    disc = np.sum(w * v, axis=1) / np.sum(w, axis=1)
    obs_vr = np.asarray(obs_vr, dtype=np.float64)
    coef = np.polyfit(obs_vr, disc, int(degree))
    trend = np.polyval(coef, obs_vr)
    return v - trend[:, None], disc, disc - trend, coef


# =============================================================================================
# p modes: 45 s dopplergrams over every exposure
# =============================================================================================
def frame_cell_velocities(files, n_rings=30, mu_min=0.0):
    """Mean velocity of every cell of a SunSim grid in one 45 s dopplergram (files = (path, keywords) of fetch_record):
    the velocity of SDO removed (observer_velocity), the rotation and everything else kept; all the pixels of the disc
    with mu > mu_min, with the same weight (the p modes are everywhere). Same frame as process_epoch.
    Returns a dict: 'v_cell' (n_cells,, float32), 'jd' (T_OBS, UTC), 'obs_vr' [m/s], 'n_pix' (pixels of every cell)."""
    path, keywords = files
    dop = read_record(path, keywords)
    x, y, mu = disc_coordinates(dop)
    v = np.asarray(dop.data, dtype=np.float32) - observer_velocity(dop, x, y, mu).astype(np.float32)
    #the square of the disc from the WCS (the 45 s dopplergrams have values a few % beyond the limb)
    lo, hi = sdo.disc_square(np.where(np.isfinite(mu), np.float32(1), np.float32(np.nan)))
    v, mu = np.flip(v[lo:hi, lo:hi]), np.flip(mu[lo:hi, lo:hi])
    with np.errstate(invalid='ignore'):
        ok = np.isfinite(v) & (mu > mu_min)
    v_cell, n_pix = cell_velocities(v, ok, np.ones(v.shape, dtype=np.float32), n_rings)
    return {'v_cell': v_cell.astype(np.float32), 'n_pix': n_pix.astype(np.float32),
            'jd': float(jd_of_t_rec(keywords['T_OBS'])[0]), 'obs_vr': float(keywords.get('OBS_VR', np.nan))}


def window_weights(t, a, b):
    """Weights (len(a), len(t)) such that weights @ y is the mean over [a, b] of the linear interpolation of the samples
    y at the times t (sorted; a, b in the same units), and the longest step between samples used by every window. A
    window not inside [t[0], t[-1]] has NaN weights."""
    t = np.asarray(t, dtype=np.float64)
    a, b = np.atleast_1d(np.asarray(a, dtype=np.float64)), np.atleast_1d(np.asarray(b, dtype=np.float64))
    w = np.zeros((len(a), len(t)))
    gap = np.zeros(len(a))
    t0, t1 = t[:-1], t[1:]
    h = t1 - t0
    for i, (ai, bi) in enumerate(zip(a, b)):
        if len(t) < 2 or ai < t[0] or bi > t[-1] or bi <= ai:
            w[i] = np.nan
            gap[i] = np.inf
            continue
        p, q = np.maximum(ai, t0), np.minimum(bi, t1)
        seg = np.flatnonzero(q > p)
        p, q, s0, s1, hs = p[seg], q[seg], t0[seg], t1[seg], h[seg]
        np.add.at(w[i], seg, ((s1 - p) ** 2 - (s1 - q) ** 2) / (2 * hs))
        np.add.at(w[i], seg + 1, ((q - s0) ** 2 - (p - s0) ** 2) / (2 * hs))
        w[i] /= bi - ai
        gap[i] = hs.max()
    return w, gap


def pmode_velocities(jd_frames, v_frames, jd_start, jd_end, ref_window=1800.0, max_gap=100.0):
    """Velocity of every cell averaged over every exposure [jd_start, jd_end] (UTC) minus its mean over ref_window [s]
    centred on the exposure, from the frames of frame_cell_velocities (jd_frames (n_frames,), v_frames (n_frames,
    n_cells)): the p modes (and the granulation) seen during the exposure.
    max_gap : [s] an exposure is not done (ok False, delta 0) if a step between the frames of its windows is longer
              (45 s: every frame there) or if its windows are not covered by the frames.
    Returns delta (n_exp, n_cells) [m/s] and ok (n_exp,)."""
    jd_frames = np.asarray(jd_frames, dtype=np.float64)
    order = np.argsort(jd_frames)
    t = (jd_frames[order] - jd_frames[order[0]]) * 86400.0
    v = np.nan_to_num(np.asarray(v_frames, dtype=np.float64)[order])
    a = (np.asarray(jd_start, dtype=np.float64) - jd_frames[order[0]]) * 86400.0
    b = (np.asarray(jd_end, dtype=np.float64) - jd_frames[order[0]]) * 86400.0
    m = (a + b) / 2
    w_exp, g_exp = window_weights(t, a, b)
    w_ref, g_ref = window_weights(t, m - ref_window / 2, m + ref_window / 2)
    ok = (g_exp <= max_gap) & (g_ref <= max_gap)
    delta = np.zeros((len(a), v.shape[1]))
    if np.any(ok):
        delta[ok] = (w_exp[ok] - w_ref[ok]) @ v
    return delta, ok


class quiet_velocity_field(object):
    """Velocities of the quiet photosphere of the cells at several epochs, for StarSim(quiet_velocity=...) (also with
    simulate_in_chunks and simulate_old_config(quiet_velocity=...)).
    times : epochs [days] (n_times,); v : (n_times, n_cells) [m/s] for a grid of N_rings rings."""
    def __init__(self, times, v, N_rings):
        self.times = np.asarray(times, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.N_rings = int(N_rings)
        if self.v.ndim != 2 or self.v.shape[0] != len(self.times):
            raise ValueError('v must be (n_times, n_cells) with one row per epoch, got %s for %d epochs'
                             % (self.v.shape, len(self.times)))

    def velocities(self, times, N_rings, n_cells):
        """The rows of the epochs `times` (n_times, n_cells)."""
        if int(N_rings) != self.N_rings or self.v.shape[1] != int(n_cells):
            raise ValueError('these velocities are for %d rings (%d cells), StarSim has %d rings (%d cells)'
                             % (self.N_rings, self.v.shape[1], N_rings, n_cells))
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        order = np.argsort(self.times)
        pos = np.clip(np.searchsorted(self.times[order], times), 0, len(order) - 1)
        rows = order[pos]
        bad = np.abs(self.times[rows] - times) > 1e-6 * np.maximum(1.0, np.abs(times))
        if np.any(bad):
            raise ValueError('no velocities for the epochs %s' % times[bad][:5])
        return self.v[rows]
