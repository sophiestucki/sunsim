"""
supergranulation: line-of-sight velocity of the quiet photosphere from SDO/HMI 45 s dopplergrams, corrected as in
Haywood et al. (2016, MNRAS 457, 3637, section 3), for StarSim(..., quiet_velocity=...), at given epochs (e.g. every
HARPS-N exposure); the HMI images are downloaded and processed one at a time (the FITS files can be deleted right after).

Records
-------
query_records(series, jd_start, jd_end) lists the records of a JSOC series (drms, no account needed); closest(jd,
jd_records) gives the record closest to every time. fetch_record(series, t_rec, segment, folder) downloads one image
from the JSOC storage (no export request, so no registered email) with the keywords of the record, read_record() makes
the sunpy map. fetch_epoch() downloads the continuum and the magnetogram of an epoch.

Every exposure
--------------
1. identification (identify_epoch): the continuum (hmi.Ic_noLimbDark_720s) and the magnetogram (hmi.M_45s) closest to
   the exposure, sdo.identify_day: the filling factors of the cells and the quiet pixels (neither spot nor facula).
2. every 45 s dopplergram (hmi.V_45s) around the exposure is corrected pixel by pixel (correct_dopplergram):
   - position of the pixel on the Sun (los_geometry; Thompson 2006): the line of sight from SDO (DSUN_OBS) through the
     pixel meets the sphere of radius RSUN_REF at (w, n, r) (heliocentric: west, north, towards SDO); delta = (w, n,
     r - r_sc) is the vector from SDO to the pixel, d its length (eq. 9, 11);
   - motion of SDO (eq. 10): v_sc = -(dw OBS_VW + dn OBS_VN + dr OBS_VR) / d;
   and projected on the cells of the SunSim grid with the quiet pixels (mu > mu_min) of its closest epoch, weighted by
   the continuum intensity (frame_cell_sums: sums of w v and of w of every cell), together with the three images of
   the rotation (rotation_basis, eq. 12-13: v_rot = a1 g1 + a2 g2 + a3 g3, omega(phi) = a1 + a2 sin^2 phi + a3 sin^4 phi,
   phi the heliographic latitude, the axis tilted by B0 (CRLT_OBS) towards SDO, v_rot = delta . (Omega x r) / d).
3. rotation of the Sun: a1, a2, a3 [deg/day, sidereal: the orbital motion of SDO is in OBS_V*] fitted on all the
   frames of the day (fit_rotation; one offset per ring and per frame fitted with them for what is the same east and
   west), and removed from every frame (remove_rotation). Table 1 of Haywood et al. (Snodgrass & Ulrich 1990,
   SNODGRASS_ULRICH) is the rotation of the surface features: ~5 % faster than the plasma seen by the dopplergrams.
   All the velocities are rates of change of the distance to SDO (> 0 away from SDO, as the HMI velocities): the
   receding west limb has v_rot > 0.
4. the corrected frames within window / 2 of the middle of the exposure are averaged (window_average; 720 s: the p
   modes, ~5 min, cancel): the sums over the frames divided give the same as the mean dopplergram projected.
Nothing else is removed: the velocity of a cell still has the centre-to-limb convective blueshift, the gravitational
redshift and the zero point of HMI, the supergranulation and the meridional flow. The HMI velocities left after eq. 10
follow the orbit of SDO (OBS_VR): an east-west gradient (rotation_obs_vr_basis: a1 + a4 OBS_VR in fit_rotation) and the
zero point and shape of the CLV (fit_clv: shifted Legendre polynomials of mu; obs_vr_model: their part that follows
OBS_VR), both optional. The cells without any quiet pixel
(the last ring is all at mu < 0.1) take the mean of their ring, or of the closest ring inside (fill_empty_cells).
quiet_velocity_field(times, v, N_rings) gives the result to StarSim(quiet_velocity=...) (or
simulate_old_config(quiet_velocity=...)): in every cell, the CCF of the quiet photosphere is shifted by its velocity.

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
#kind -> (series, segment) of the two images of the identification of an epoch
SERIES = {'continuum': ('hmi.Ic_noLimbDark_720s', 'continuum'),
          'magnetogram': ('hmi.M_45s', 'magnetogram')}
SERIES_45S = ('hmi.V_45s', 'Dopplergram')          # the dopplergrams
CADENCE_45S = 45.0                                  # [s]
SNODGRASS_ULRICH = (14.713, -2.396, -1.787)         # a1, a2, a3 [deg/day] (Haywood et al. 2016, Table 1: not used)


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
    client = client or _client()
    ds = '%s[%s]' % (series, t_rec)
    for attempt in range(retries):
        try:
            keys, segs = client.query(ds, key='**ALL**', seg=segment)
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
    """The images of the identification of one epoch: epoch = {'continuum': t_rec, 'magnetogram': t_rec} (the T_REC of
    every kind). Returns {kind: (path, keywords)} (fetch_record); the files are in folder. If one image cannot be
    downloaded, the files already downloaded are deleted and the error is raised."""
    client = _client()
    out = {}
    try:
        for kind in ('continuum', 'magnetogram'):
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
# dopplergram: Haywood et al. (2016), section 3.1-3.2
# =============================================================================================
def sun_centre_offsets(smap):
    """Angular offsets [rad] of the pixel columns and rows from the pixel of the disc centre (vectors of nx and ny) and
    the rotation matrix of the map (PCi_j): the offsets on the tangent plane towards solar west and north of pixel
    (i, j) are pc @ (dx[i], dy[j]) (CROTA2 = 180 for HMI)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ny, nx = smap.data.shape
    centre = smap.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=smap.coordinate_frame))
    arcsec = np.pi / 180 / 3600
    dx = (np.arange(nx) - centre.x.to_value(u.pix)) * smap.scale.axis1.to_value(u.arcsec / u.pix) * arcsec
    dy = (np.arange(ny) - centre.y.to_value(u.pix)) * smap.scale.axis2.to_value(u.arcsec / u.pix) * arcsec
    return dx, dy, np.asarray(smap.rotation_matrix, dtype=np.float64)


def los_geometry(tx, ty, dsun, rsun):
    """Lines of sight from the observer through the pixels and their point on the Sun (Thompson 2006).
    tx, ty : offsets of the pixels from the disc centre on the tangent plane (gnomonic, towards solar west and north)
             [rad], arrays of the same shape
    dsun, rsun : distance observer - Sun centre and radius of the Sun [m]
    Heliocentric frame: x west, y north, z from the Sun centre towards the observer (at (0, 0, dsun)).
    Returns u (3 arrays: unit vector from the observer to the pixel, = delta / d of eq. 9 and 11), p (3 arrays: the
    point of the surface seen in the pixel [m]) and mu (NaN off the disc, where the line of sight misses the Sun)."""
    tx, ty = np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)
    cos_rho = 1.0 / np.sqrt(1.0 + tx * tx + ty * ty)             # cos of the angle from the disc centre
    u = (tx * cos_rho, ty * cos_rho, -cos_rho)
    q = rsun * rsun - dsun * dsun * (1.0 - cos_rho * cos_rho)    # R^2 - D^2 sin^2 rho: > 0 on the disc
    with np.errstate(invalid='ignore'):
        s = np.sqrt(q)                                           # NaN off the disc
    d = dsun * cos_rho - s                                       # distance observer - surface (eq. 11)
    p = (d * u[0], d * u[1], dsun - d * cos_rho)
    return u, p, s / rsun


def spacecraft_velocity(u, vw, vn, vr):
    """Eq. 10: velocity [m/s, > 0 away from the observer] of every pixel due to the motion of the observer (vw, vn, vr:
    OBS_VW, OBS_VN, OBS_VR [m/s], west, north and away from the Sun), u of los_geometry (= delta / d)."""
    return -(u[0] * vw + u[1] * vn + u[2] * vr)


def rotation_basis(u, p, b0, rsun):
    """Eq. 12-13 as a sum: the line-of-sight velocity of the solar rotation is a1 g1 + a2 g2 + a3 g3 (alphas in
    deg/day), u and p of los_geometry, b0 the heliographic latitude of the observer [rad] (CRLT_OBS).
    omega(phi) = a1 + a2 sin^2 phi + a3 sin^4 phi, sin phi = (y cos b0 + z sin b0) / R, the velocity of the surface
    Omega x p with Omega = omega (0, cos b0, sin b0), projected on u (the rate of change of the distance, > 0 away from
    the observer): g_k = sin^(2k-2) phi u . ((0, cos b0, sin b0) x p) [m/s per deg/day]. Returns (g1, g2, g3)."""
    cb, sb = np.cos(b0), np.sin(b0)
    sin2 = ((p[1] * cb + p[2] * sb) / rsun) ** 2
    g1 = (u[0] * (cb * p[2] - sb * p[1]) + u[1] * sb * p[0] - u[2] * cb * p[0]) * (np.pi / 180 / 86400)
    g2 = g1 * sin2
    return g1, g2, g2 * sin2


def rotation_velocity(u, p, b0, rsun, alphas=SNODGRASS_ULRICH):
    """Eq. 12-13: line-of-sight velocity [m/s, > 0 away from the observer] of the solar rotation with the coefficients
    alphas (a1, a2, a3) [deg/day] (sidereal: the observer motion relative to the Sun is removed by
    spacecraft_velocity), u and p of los_geometry, b0 [rad] (rotation_basis)."""
    g = rotation_basis(u, p, b0, rsun)
    return alphas[0] * g[0] + alphas[1] * g[1] + alphas[2] * g[2]


def correct_dopplergram(v_map, alphas=None, components=False, basis=False):
    """Velocity of the dopplergram v_map [m/s, > 0 away from SDO] without the motion of SDO (eq. 10) and, if alphas
    (a1, a2, a3) [deg/day] are given, the rotation of the Sun (eq. 13), pixel by pixel (module docstring).
    Returns a dict: 'v' (corrected velocity, float32, NaN off the disc), 'mu' (float32), the keywords used ('obs_vr',
    'obs_vw', 'obs_vn' [m/s], 'B0' [deg], 'dsun' [m]), with components=True 'v_sc' and 'v_rot' (float32; v_rot only
    with alphas) and with basis=True 'g' (the three images of rotation_basis, float32)."""
    m = v_map.meta
    rsun = float(m.get('rsun_ref', 696000000.0))
    dsun = float(m['dsun_obs'])
    vr, vw, vn = (float(m[k]) for k in ('obs_vr', 'obs_vw', 'obs_vn'))
    b0 = np.deg2rad(float(m['crlt_obs']))
    dx, dy, pc = sun_centre_offsets(v_map)
    tx = pc[0, 0] * dx[None, :] + pc[0, 1] * dy[:, None]
    ty = pc[1, 0] * dx[None, :] + pc[1, 1] * dy[:, None]
    u, p, mu = los_geometry(tx, ty, dsun, rsun)
    del tx, ty
    off = ~np.isfinite(mu)
    v_sc = spacecraft_velocity(u, vw, vn, vr)
    v_sc[off] = np.nan
    g = rotation_basis(u, p, b0, rsun) if (alphas is not None or basis) else None
    del u, p
    v = np.asarray(v_map.data, dtype=np.float64) - v_sc
    out = {'mu': mu.astype(np.float32), 'obs_vr': vr, 'obs_vw': vw, 'obs_vn': vn, 'B0': np.rad2deg(b0), 'dsun': dsun}
    if alphas is not None:
        v_rot = alphas[0] * g[0] + alphas[1] * g[1] + alphas[2] * g[2]
        v -= v_rot
        if components:
            out['v_rot'] = np.where(off, np.nan, v_rot).astype(np.float32)
    v[off] = np.nan
    out['v'] = v.astype(np.float32)
    if components:
        out['v_sc'] = v_sc.astype(np.float32)
    if basis:
        out['g'] = tuple(np.where(off, np.nan, gk).astype(np.float32) for gk in g)
    return out


# =============================================================================================
# cells
# =============================================================================================
def cell_sums(v, mask, weight, n_rings):
    """Sums over the pixels of mask in every cell of a SunSim grid of n_rings rings: of weight x v and of weight (e.g.
    the continuum intensity); arrays in the frame of the maps of identify_day (origin='lower'). The mean velocity of a
    cell is swv / sw (cell_velocities). Returns swv, sw (n_cells,, float64)."""
    cells, npix = sdo.cell_projection(np.shape(v), n_rings)
    sel = np.asarray(mask, dtype=bool).ravel() & np.isfinite(np.ravel(v)) & np.isfinite(np.ravel(weight))
    sel &= cells >= 0
    c = cells[sel]
    w = np.ravel(weight)[sel].astype(np.float64)
    sw = np.bincount(c, weights=w, minlength=len(npix))
    swv = np.bincount(c, weights=w * np.ravel(v)[sel].astype(np.float64), minlength=len(npix))
    return swv, sw


def cell_velocities(swv, sw):
    """Mean velocity of every cell from the sums of cell_sums (NaN for a cell without any pixel)."""
    swv, sw = np.asarray(swv, dtype=np.float64), np.asarray(sw, dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(sw > 0, swv / np.where(sw > 0, sw, 1.0), np.nan)


def ring_of_cells(n_rings):
    """Ring index of every cell of a SunSim grid of n_rings rings."""
    from . import nbspectra
    nin = np.asarray(nbspectra.generate_grid_coordinates_nb(int(n_rings))[1], dtype=np.int64)
    return np.repeat(np.arange(int(n_rings)), nin)


def fill_empty_cells(v, w, n_rings):
    """Velocities of the cells (n_times, n_cells) with the cells without data (NaN) set to the mean of their ring
    (weighted by w, e.g. sw of cell_sums), or of the closest ring inside with data (the last ring is all at mu < 0.1).
    Returns v (float64, no NaN unless an epoch has no data at all) and the mask of the cells filled."""
    v = np.array(v, dtype=np.float64, ndmin=2)
    w = np.where(np.isfinite(v), np.asarray(w, dtype=np.float64), 0.0)
    ring = ring_of_cells(n_rings)
    empty = ~np.isfinite(v)
    for k in np.flatnonzero(empty.any(axis=1)):
        sw = np.bincount(ring, weights=w[k], minlength=n_rings)
        swv = np.bincount(ring, weights=w[k] * np.nan_to_num(v[k]), minlength=n_rings)
        mean = np.full(n_rings, np.nan)
        mean[sw > 0] = swv[sw > 0] / sw[sw > 0]
        for i in range(1, n_rings):                           # a ring without data: the ring inside
            if not np.isfinite(mean[i]):
                mean[i] = mean[i - 1]
        v[k, empty[k]] = mean[ring[empty[k]]]
    return v, empty


# =============================================================================================
# one epoch, one frame
# =============================================================================================
def identify_epoch(files, size=4096, n_rings=30, thresholds=None, network_velocity=True):
    """Identification of one epoch from the images of fetch_epoch (files = {kind: (path, keywords)}).
    size        : the images are resampled to size x size pixels (sdo.identify_day)
    n_rings     : SunSim grid of the filling factors
    thresholds  : dict of the thresholds of sdo.identify_day (noise_th, B_th, I_th, A_th, mu_area, faculae)
    network_velocity : True: the quiet pixels are the ones that SunSim counts as quiet photosphere (neither spot nor
                  facula: with faculae='plage' the network is in it), False: only the pixels that are not active at all
    Returns a dict: 'ff_sp', 'ff_fc' (cell filling factors, float32), the numbers of the epoch 'ff_sp_disc',
    'ff_fc_disc', 'ff_nt_disc' [%], 'n_spots', 'n_plages', and 'context', what frame_cell_sums needs: the square of the
    disc in the image ('lo', 'hi'), 'size', 'quiet' (mask) and 'weight' (continuum intensity) in the frame of the maps,
    'crpix' (pixel of the disc centre of the continuum, to check the alignment of the dopplergrams)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    cont = read_record(*files['continuum'])
    mag = read_record(*files['magnetogram'])
    day = sdo.identify_day(cont, mag, size=size, **(thresholds or {}))
    del cont, mag
    cmap = day['cont_map']
    lo, hi = sdo.disc_square(cmap.data)
    c = cmap.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=cmap.coordinate_frame))
    still = day['disc'] & ~(day['spot'] | day['plage'] | day['network'])        # not active at all
    quiet = day['disc'] & ~(day['spot'] | day['facula']) if network_velocity else still
    ff_sp, ff_fc = sdo.cell_filling_factors(day, n_rings)
    return {'ff_sp': ff_sp, 'ff_fc': ff_fc, 'ff_sp_disc': day['ff_sp'], 'ff_fc_disc': day['ff_fc'],
            'ff_nt_disc': day['ff_nt'], 'n_spots': int(day['n_spots']), 'n_plages': int(day['n_plages']),
            'context': {'lo': lo, 'hi': hi, 'size': int(cmap.data.shape[0]), 'quiet': quiet,
                        'weight': np.flip(np.asarray(cmap.data, dtype=np.float32)[lo:hi, lo:hi]),
                        'crpix': (float(c.x.to_value(u.pix)), float(c.y.to_value(u.pix))),
                        'spot': day['spot'], 'facula': day['facula']}}


def frame_cell_sums(files, context, n_rings=30, mu_min=0.1, keep_frame=False):
    """One 45 s dopplergram (files = (path, keywords) of fetch_record) without the motion of SDO (correct_dopplergram,
    eq. 10) projected on the cells with the quiet pixels of the closest epoch (context of identify_epoch), only the
    pixels with mu > mu_min (0.1, Haywood et al. 2016), and the three images of the rotation (rotation_basis) projected
    in the same way: the rotation is fitted later on all the frames of the day (fit_rotation) and removed from the sums
    (remove_rotation). The dopplergram is resampled to the size of the identification if needed and cut to the same
    square of the disc (the HMI images are aligned to ~1 pixel).
    Returns a dict: 'swv', 'sw' (cell_sums), 'swg' (3, n_cells: cell_sums of g1, g2, g3, same pixels and weights),
    'jd' (T_OBS, UTC), the keywords of correct_dopplergram, 'shift_px' (distance between the disc centres of the
    dopplergram and the continuum, at the size of the identification), 'v_quiet_mean' (mean velocity of the quiet
    pixels without the motion of SDO [m/s]) and, with keep_frame, 'v' (that velocity in the frame of the maps, NaN where
    mu <= mu_min)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    path, keywords = files
    dop = read_record(path, keywords)
    c = correct_dopplergram(dop, basis=True)
    with np.errstate(invalid='ignore'):
        far = ~(c['mu'] > mu_min)
    images = [c.pop('v')] + list(c.pop('g'))
    del c['mu']
    size = context['size']
    scale = size / dop.data.shape[0]
    centre = dop.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=dop.coordinate_frame))
    shift = float(np.hypot((centre.x.to_value(u.pix) + 0.5) * scale - 0.5 - context['crpix'][0],
                           (centre.y.to_value(u.pix) + 0.5) * scale - 0.5 - context['crpix'][1]))
    lo, hi = context['lo'], context['hi']
    for k, a in enumerate(images):
        a[far] = np.nan
        if size != dop.data.shape[0]:
            smap = sdo._sunpy()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a = np.asarray(smap.Map(a, dop.meta).resample([size, size] * u.pix).data, dtype=np.float32)
        images[k] = np.flip(a[lo:hi, lo:hi])
    del dop, far
    v = images[0]
    if v.shape != context['quiet'].shape:
        raise ValueError('dopplergram cut to %s, the quiet mask is %s' % (v.shape, context['quiet'].shape))
    sel = context['quiet'] & np.isfinite(v)
    for g in images[1:]:
        sel &= np.isfinite(g)
    swv, sw = cell_sums(v, sel, context['weight'], n_rings)
    swg = np.array([cell_sums(g, sel, context['weight'], n_rings)[0] for g in images[1:]])
    vq = float(np.sum(context['weight'][sel] * v[sel], dtype=np.float64) / np.sum(context['weight'][sel], dtype=np.float64))
    out = dict(c, swv=swv, sw=sw, swg=swg, jd=float(jd_of_t_rec(keywords['T_OBS'])[0]), shift_px=shift, v_quiet_mean=vq)
    if keep_frame:
        out['v'] = v
    return out


# =============================================================================================
# rotation of every day
# =============================================================================================
def fit_rotation(swv, sw, swg, n_rings):
    """Coefficients (a1, a2, a3) [deg/day] of eq. 12 fitted on frames (e.g. all the frames of one day), from the sums of
    frame_cell_sums: the velocity of every cell v = swv / sw against the cell means of the rotation images
    G_k = swg_k / sw (v_rot = a1 G1 + a2 G2 + a3 G3). One offset per ring and per frame is fitted with them and not used
    (everything that is the same east and west: the convective blueshift, the zero point of HMI); least squares weighted
    by sw. swv, sw (n_frames, n_cells), swg (n_frames, 3, n_cells), or with more images (rotation_obs_vr_basis: a4).
    Returns alphas (3,), their formal errors (3,; the supergranulation is not white noise: too small) and the weighted
    RMS of the residuals [m/s] (one coefficient per image of swg)."""
    sw = np.atleast_2d(np.asarray(sw, dtype=np.float64))
    swv = np.atleast_2d(np.asarray(swv, dtype=np.float64))
    swg = np.asarray(swg, dtype=np.float64).reshape(len(sw), -1, sw.shape[1])
    n_f = len(sw)
    ok = sw > 0
    w = np.where(ok, sw, 0.0)
    den = np.where(ok, sw, 1.0)
    idx = (np.arange(n_f)[:, None] * int(n_rings) + ring_of_cells(n_rings)[None, :]).ravel()
    sw_ring = np.bincount(idx, weights=w.ravel(), minlength=n_f * int(n_rings))

    def demeaned(a):
        """a minus its weighted mean in every ring of every frame."""
        s = np.bincount(idx, weights=(w * a).ravel(), minlength=n_f * int(n_rings))
        m = np.where(sw_ring > 0, s / np.where(sw_ring > 0, sw_ring, 1.0), 0.0)
        return (a - m[idx].reshape(a.shape))[ok]

    y = demeaned(np.where(ok, swv / den, 0.0))
    X = np.column_stack([demeaned(np.where(ok, swg[:, k] / den, 0.0)) for k in range(swg.shape[1])])
    q = np.sqrt(w[ok])
    A = X * q[:, None]
    coef = np.linalg.lstsq(A, y * q, rcond=None)[0]
    res = y - X @ coef
    chi2 = np.sum(w[ok] * res ** 2)
    dof = max(len(y) - X.shape[1] - int(np.sum(sw_ring > 0)), 1)
    err = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * chi2 / dof)
    return coef, err, float(np.sqrt(chi2 / np.sum(w[ok])))


def rotation_obs_vr_basis(swg, obs_vr):
    """swg of frame_cell_sums (n_frames, 3, n_cells) with a fourth image g1 x OBS_VR [km/s] of every frame: fitted with
    them (fit_rotation), a1 + a4 OBS_VR. The HMI velocities left after eq. 10 have an east-west gradient that follows
    OBS_VR (~0.1 deg/day per km/s on a1, ~15 m/s at the limb): not the Sun, but the shape of g1."""
    swg = np.asarray(swg, dtype=np.float64)
    vr = np.asarray(obs_vr, dtype=np.float64) / 1000.0
    return np.concatenate([swg, swg[:, :1] * vr[:, None, None]], axis=1)


def north_of_cells(n_rings, size=512):
    """Mean north coordinate [solar radii, > 0 north] of the pixels of every cell, in the frame of the maps of
    identify_day (origin='lower': north up)."""
    cells, npix = sdo.cell_projection((size, size), n_rings)
    y = np.repeat((np.arange(size) + 0.5) / size * 2 - 1, size)
    return np.bincount(cells[cells >= 0], weights=y[cells >= 0], minlength=len(npix)) / np.maximum(npix, 1)


def north_south_basis(swg, sw, obs_vr, n_rings):
    """swg of frame_cell_sums with two more images: the north coordinate of the cells, alone and x OBS_VR [km/s]
    (sums over the quiet pixels: sw x north). Fitted with the rotation (fit_rotation): the HMI velocities left after
    eq. 10 have a north-south gradient (~ -45 m/s per solar radius, +-7 with OBS_VR) that no solar flow gives (the
    rotation is east-west, the meridional flow the same north and south)."""
    swg, sw = np.asarray(swg, dtype=np.float64), np.asarray(sw, dtype=np.float64)
    yn = sw * north_of_cells(n_rings)[None, :]
    vr = np.asarray(obs_vr, dtype=np.float64)[:, None] / 1000.0
    return np.concatenate([swg, yn[:, None], (yn * vr)[:, None]], axis=1)


def remove_rotation(swv, swg, alphas):
    """Sums swv of frame_cell_sums without the rotation of eq. 13 with the coefficients alphas [deg/day] (one per image
    of swg)."""
    swg = np.asarray(swg, dtype=np.float64)
    return np.asarray(swv, dtype=np.float64) - np.tensordot(np.asarray(alphas, dtype=np.float64), swg, axes=([0], [-2]))


# =============================================================================================
# centre-to-limb variation of the quiet cells
# =============================================================================================
def mu_of_cells(n_rings):
    """mu of the centre of the ring of every cell of a SunSim grid of n_rings rings."""
    from . import nbspectra
    return np.asarray(nbspectra.generate_grid_coordinates_nb(int(n_rings))[3], dtype=np.float64)[ring_of_cells(n_rings)]


def clv_basis(n_rings, deg=5):
    """Shifted Legendre polynomials P_n(2 mu - 1), n = 0..deg, of the cells (n_cells, deg + 1): the convective blueshift
    v(mu) = sum c_n P_n(2 mu - 1) of Kashyap & Hanasoge (2021), as in Ervin et al. (2024, eq. 4)."""
    return np.polynomial.legendre.legvander(2.0 * mu_of_cells(n_rings) - 1.0, int(deg))


def fit_clv(v, w, n_rings, deg=5):
    """Coefficients c_n of clv_basis fitted on the cells of every exposure (least squares weighted by w, e.g. sw of
    window_average; the cells with NaN or w = 0 left out). v, w (n_exp, n_cells). Returns c (n_exp, deg + 1)."""
    v = np.array(v, dtype=np.float64, ndmin=2)
    w = np.array(w, dtype=np.float64, ndmin=2)
    P = clv_basis(n_rings, deg)
    c = np.full((len(v), P.shape[1]), np.nan)
    for i in range(len(v)):
        k = np.isfinite(v[i]) & (w[i] > 0)
        q = np.sqrt(w[i][k])
        c[i] = np.linalg.lstsq(P[k] * q[:, None], v[i][k] * q, rcond=None)[0]
    return c


def obs_vr_model(y, obs_vr, day, deg=2, jd=None, harmonics=0):
    """Fit of y (n_exp,) or (n_exp, n) = polynomial of degree deg in OBS_VR [m/s] (no constant) + one offset per day
    (+ with jd [JD UTC] the harmonics 1..harmonics of the orbit of SDO, geosynchronous: 24 h, 12 h, ...; the HMI
    residuals have 24 h and 12 h periods, Ervin et al. 2024, and the same OBS_VR comes twice per orbit), over all the
    exposures: the part of y that follows the orbit of SDO and the zero point of every day; the rest (e.g. the
    supergranulation) is in the residuals. Use many days: a solar signal of the day that follows OBS_VR goes too.
    Returns the fitted values (same shape as y)."""
    y = np.asarray(y, dtype=np.float64)
    vr = np.asarray(obs_vr, dtype=np.float64) / 1000.0
    days, d = np.unique(np.asarray(day), return_inverse=True)
    cols = [vr ** p for p in range(1, int(deg) + 1)]
    if harmonics:
        ph = 2 * np.pi * ((np.asarray(jd, dtype=np.float64) + 0.5) % 1.0)
        cols += [f(k * ph) for k in range(1, int(harmonics) + 1) for f in (np.cos, np.sin)]
    X = np.column_stack(cols + [(d == k).astype(float) for k in range(len(days))])
    return X @ np.linalg.lstsq(X, y, rcond=None)[0]


# =============================================================================================
# all the frames of an exposure
# =============================================================================================
def window_average(jd_frames, swv, sw, jd_mid, window=720.0, min_fill=0.8, cadence=CADENCE_45S):
    """Mean velocity of the quiet pixels of every cell over the frames within window / 2 [s] of the middle of every
    exposure (jd_mid, JD UTC): sum of swv over these frames / sum of sw (frame_cell_sums: as the mean of the corrected
    dopplergrams, projected). jd_frames (n_frames,), swv and sw (n_frames, n_cells).
    min_fill : an exposure needs at least this fraction of the frames of the window (window / cadence), else ok False
    Returns v (n_exp, n_cells; NaN for a cell without quiet pixels), sw (n_exp, n_cells: the sums of the weights),
    n_frames (n_exp,) and ok (n_exp,)."""
    jd_frames = np.asarray(jd_frames, dtype=np.float64)
    swv, sw = np.asarray(swv, dtype=np.float64), np.asarray(sw, dtype=np.float64)
    jd_mid = np.atleast_1d(np.asarray(jd_mid, dtype=np.float64))
    n_cells = swv.shape[1] if swv.ndim == 2 else 0
    v_out, sw_out = np.full((len(jd_mid), n_cells), np.nan), np.zeros((len(jd_mid), n_cells))
    n = np.zeros(len(jd_mid), dtype=np.int64)
    for i, m in enumerate(jd_mid):
        sel = np.abs(jd_frames - m) * 86400.0 <= window / 2
        n[i] = sel.sum()
        if n[i]:
            sw_out[i] = sw[sel].sum(axis=0)
            v_out[i] = cell_velocities(swv[sel].sum(axis=0), sw_out[i])
    ok = n >= min_fill * window / cadence
    return v_out, sw_out, n, ok


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
