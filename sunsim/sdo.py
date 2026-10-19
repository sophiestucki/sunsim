"""
sdo: maps of spots and faculae of the Sun from SDO/HMI images, for StarSim(..., active_regions_mask=...).

From two HMI images of the same epoch, the continuum intensity without limb darkening (hmi.Ic_noLimbDark_720s) and the
line-of-sight magnetogram (hmi.M_45s), every pixel of the disc is classified as quiet Sun, spot, facula (plage) or
network, following Yeo et al. (2013) and Sen et al. (2023):

  1. noise: |B| below noise_th (8 G for 4096 x 4096 images; scaled by N / 4096 for images resampled to N x N pixels, as
     the noise of the mean of k x k pixels decreases as 1/k) is set to 0;
  2. foreshortening: B / mu, mu = cos(heliocentric angle);
  3. active pixels: |B| / mu > B_th / mu (B_th = 24 G);
  4. spots: active pixels darker than I_th times the mean intensity of the quiet Sun (I_th = 0.89); the other active
     pixels are bright features;
  5. plage and network: the connected groups of bright pixels larger than A_th micro-hemispheres (projected at mu = 0.1,
     so that no plage at the limb is missed) are plage, the smaller ones network.
The maps written for SunSim are the spot map (spots) and the facula map (plage; with faculae='plage+network' the network
too), one pair per epoch, cut to the square of the disc. HMI stores the images turned by 180 deg (CROTA2 = 180): the
maps are turned back (both axes flipped), so that row 0 is the south of the disc and the rotation moves the regions
from left to right, the convention of the maps of the old StarSim. Read them with
active_region_mask(folder, origin='lower') (simulate_old_config does it for the 2D maps of an old configuration).

For notebooks
-------------
    from sunsim import sdo
    sdo.download_hmi('2017-07-01', '2017-07-10', folder, email='you@institute')        # optional (JSOC account)
    res = sdo.make_sdo_maps(folder, 'maps/', '2017-07-01', '2017-07-10', size=512)    # writes the maps
    maps = active_region_mask('maps/', origin='lower')                                 # then StarSim, as any 2D maps
    day = sdo.identify_day(*sdo.read_hmi_day(folder, '2017-07-01'), size=512)        # one day, all the masks
    sdo.plot_identification(day)

In a terminal
-------------
    sunsim-sdo-maps --data folder --out maps/ --start 2017-07-01 --end 2017-07-10 --size 512
    python -m sunsim.sdo --help                                                         # the same
With --conf old.conf, the light curve and the RV are then computed from the maps with simulate_old_config (the
configuration of an old StarSim run with sdo_input = 1), as the previous script did with the old StarSim.

Many thresholds (sweep): give several values to any threshold; every combination gets its folder --out/<set>, every day
is read and resampled once for all of them, and with --conf the grids (spectra, CCFs) are computed once for all the sets
and kept in --out/grids_<id>.pkl, so later runs with new values (same --out) only classify and simulate what is new:
    sunsim-sdo-maps --data hmi/ --out sweep/ --start 2015-07-18 --end 2024-01-29 --size 4096 --workers 4 \
                    --I-th 0.85 0.89 0.95 --A-th 20 100 --conf sun.conf
In a notebook: make_sdo_sweep(...) then simulate_sdo_sets(conf, folders, grids_cache=...).

Needs sunpy (and drms to download from JSOC): pip install sunpy drms (or pip install "sunsim[sdo]"). The functions on
arrays (classify_pixels, split_plage_network, count_spots, mu_from_rho) only need NumPy and SciPy.
"""
import argparse
import glob
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np

CONTINUUM_SERIES = 'hmi.Ic_noLimbDark_720s'
MAGNETOGRAM_SERIES = 'hmi.M_45s'
#file names of the JSOC series (the date, then any time of the day)
CONTINUUM_PATTERN = 'hmi.ic_nolimbdark_720s.{date}_*continuum.fits'
MAGNETOGRAM_PATTERN = 'hmi.m_45s.{date}_*magnetogram.fits'
FULL_SIZE = 4096                    # pixels of the HMI images

DEFAULTS = dict(noise_th=8.0, B_th=24.0, I_th=0.89, A_th=20.0, mu_area=0.1)


def _sunpy():
    """sunpy.map (imported only when needed: sunpy is optional)."""
    try:
        import sunpy.map
    except ImportError:
        raise ImportError('the SDO pipeline needs sunpy: pip install sunpy drms (or pip install "sunsim[sdo]")') from None
    return sunpy.map


def _date(value, name='date'):
    """A datetime from 'YYYY-MM-DD', 'YYYYMMDD' or a datetime."""
    if isinstance(value, datetime):
        return value
    for fmt in ('%Y-%m-%d', '%Y%m%d', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            pass
    raise ValueError("%s must be a date 'YYYY-MM-DD' (or 'YYYYMMDD', or a datetime), got %r" % (name, value))


# =============================================================================================
# identification on arrays (NumPy / SciPy only)
# =============================================================================================
def mu_from_rho(rho, radius):
    """mu = cos(heliocentric angle) = sqrt(1 - (rho / radius)^2) for the distances rho from the disc centre (same unit
    as radius); NaN outside the disc."""
    r = np.asarray(rho, dtype=np.float64) / radius
    with np.errstate(invalid='ignore'):
        return np.where(r <= 1.0, np.sqrt(np.clip(1.0 - r * r, 0.0, None)), np.nan)


def noise_threshold_for(size, noise_th=DEFAULTS['noise_th']):
    """Noise threshold [G] of an image of size x size pixels: noise_th (given for the 4096 x 4096 HMI images) x size / 4096
    (the noise of the mean of k x k pixels is 1/k of the noise of one pixel)."""
    return float(noise_th) * float(size) / FULL_SIZE


def classify_pixels(intensity, blos, mu, noise_th=DEFAULTS['noise_th'], B_th=DEFAULTS['B_th'], I_th=DEFAULTS['I_th']):
    """Active, spot and bright pixels of one epoch (Yeo et al. 2013).

    intensity : continuum intensity without limb darkening (2D, NaN outside the disc)
    blos      : line-of-sight magnetic field [G] (same shape)
    mu        : cos(heliocentric angle) of every pixel (NaN outside the disc)
    noise_th  : |B| below it is noise (set to 0) [G], for the resolution of these images (noise_threshold_for)
    B_th      : the pixel is active if |B| / mu > B_th / mu [G] (foreshortening corrected on both sides: mu > 0 cancels,
                so the test is |B| > B_th, done without dividing)
    I_th      : an active pixel is a spot if its intensity < I_th x <I_quiet>, the mean intensity of the quiet Sun
                (the pixels of the disc that are not active); otherwise it is a bright feature
    Returns a dict of boolean arrays: 'disc', 'active', 'spot', 'bright', and 'I_quiet' (the mean quiet intensity).
    Works on the arrays as they are (float32 or float64), without float copies: ~5 boolean images of memory."""
    intensity = np.asarray(intensity)
    blos = np.asarray(blos)
    mu = np.asarray(mu)
    if not intensity.shape == blos.shape == mu.shape or intensity.ndim != 2:
        raise ValueError('intensity, blos and mu must be 2D arrays of the same shape, got %s, %s, %s'
                         % (intensity.shape, blos.shape, mu.shape))
    with np.errstate(invalid='ignore'):
        disc = mu > 0                                    # False outside the disc (NaN)
        disc &= np.isfinite(intensity)
        absb = np.abs(blos)                              # NaN -> no comparison is True
        active = absb >= noise_th                        # 1. noise
        active &= absb > B_th                            # 2-3. |B| / mu > B_th / mu
        active &= disc
        del absb
        quiet = disc & ~active
        if not np.any(quiet):
            raise ValueError('no quiet pixel on the disc: check the images (or the thresholds)')
        I_quiet = float(np.mean(intensity, where=quiet, dtype=np.float64))
        del quiet
        spot = intensity < I_th * I_quiet                # 4. spots
        spot &= active
    return {'disc': disc, 'active': active, 'spot': spot, 'bright': active & ~spot, 'I_quiet': I_quiet}


def split_plage_network(bright, area_th_pixels):
    """Plage and network among the bright pixels: the connected groups (scipy.ndimage.label, 4-connectivity) larger than
    area_th_pixels are plage, the others network. Returns (plage, network) boolean arrays and the number of plages."""
    import scipy.ndimage as nd
    bright = np.asarray(bright, dtype=bool)
    labels, n = nd.label(bright)
    if n == 0:
        return np.zeros_like(bright), np.zeros_like(bright), 0
    sizes = np.bincount(labels.ravel())[1:]                  # pixels of every group
    big = np.concatenate(([False], sizes > area_th_pixels))
    plage = big[labels]
    return plage, bright & ~plage, int(np.sum(sizes > area_th_pixels))


def count_spots(spot, sigma=2.0, threshold=0.2):
    """Number of spots, their centres (row, column) and their areas [pixels]: the spot pixels are smoothed with a gaussian
    of sigma pixels and the groups where the smoothed mask is above threshold are counted (nearby fragments of one spot
    count once)."""
    import scipy.ndimage as nd
    spot = np.asarray(spot, dtype=bool)
    smooth = nd.gaussian_filter(spot.astype(np.float32), sigma=sigma, mode='constant', cval=0.0)
    labels, n = nd.label(smooth >= threshold)
    if n == 0:
        return 0, np.zeros((0, 2)), np.zeros(0, dtype=int)
    centres = np.array(nd.center_of_mass(smooth, labels=labels, index=range(1, n + 1)))
    areas = np.bincount(labels.ravel(), weights=spot.ravel().astype(np.float64))[1:].astype(int)
    return n, centres, areas


def disc_square(intensity):
    """Rows / columns (lo, hi) of the square that contains the disc: the first and last finite pixel of the central
    row (the maps of SunSim are images of the disc, the disc filling the image)."""
    row = np.asarray(intensity)[intensity.shape[0] // 2]
    finite = np.flatnonzero(np.isfinite(row) & (row >= 0))
    if len(finite) == 0:
        raise ValueError('no pixel of the disc on the central row of the image')
    return int(finite.min()), int(finite.max()) + 1


# =============================================================================================
# one epoch with sunpy maps
# =============================================================================================
def find_hmi_files(folder, date):
    """(continuum file, magnetogram file) of one day in folder (first file of the day of each series), or None if one
    of the two is missing. File names of the JSOC exports, e.g. hmi.ic_nolimbdark_720s.20170701_000000_TAI.3.continuum.fits
    and hmi.m_45s.20170701_000045_TAI.2.magnetogram.fits."""
    d = _date(date).strftime('%Y%m%d')
    cont = sorted(glob.glob(os.path.join(str(folder), CONTINUUM_PATTERN.format(date=d))))
    mag = sorted(glob.glob(os.path.join(str(folder), MAGNETOGRAM_PATTERN.format(date=d))))
    if not cont or not mag:
        return None
    return cont[0], mag[0]


def read_hmi_day(folder, date):
    """Continuum and magnetogram maps (sunpy) of one day. FileNotFoundError if one of them is missing."""
    files = find_hmi_files(folder, date)
    if files is None:
        raise FileNotFoundError('no continuum and magnetogram files of %s in %s' % (_date(date).date(), folder))
    smap = _sunpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                  # sunpy: missing metadata of the HMI headers
        return smap.Map(files[0]), smap.Map(files[1])


def mu_of_map(smap, dtype=np.float32):
    """mu = cos(heliocentric angle) of every pixel of a sunpy map (NaN outside the disc).
    The angular distance rho of a pixel from the disc centre comes from its offset from the pixel of the disc centre
    times the plate scale (tan rho in the gnomonic projection of HMI; rho^2 = r^2 (1 - 2 r^2 / 3) to O(r^6)): a few
    vectors and one image of dtype, instead of the sky coordinates of every pixel (~1.5 s and ~1.5 GB for 4096 x 4096
    pixels). Same as the sky coordinates to ~1e-5 near the limb."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ny, nx = smap.data.shape
    centre = smap.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=smap.coordinate_frame))
    arcsec = np.pi / 180 / 3600
    sx = smap.scale.axis1.to_value(u.arcsec / u.pix) * arcsec
    sy = smap.scale.axis2.to_value(u.arcsec / u.pix) * arcsec
    r_sun2 = (smap.rsun_obs.to_value(u.arcsec) * arcsec)**2
    dx2 = ((np.arange(nx) - centre.x.to_value(u.pix)) * sx)**2           # [rad^2], offsets on the tangent plane
    dy2 = ((np.arange(ny) - centre.y.to_value(u.pix)) * sy)**2
    r2 = (dy2[:, None] + dx2[None, :]).astype(dtype)                      # (tan rho)^2
    r2 *= 1 - 2 / 3 * r2                                                  # rho^2 = atan(r)^2 = r^2 (1 - 2 r^2 / 3) + O(r^6)
    one_minus = 1 - r2 / dtype(r_sun2)                                   # 1 - (rho / R)^2
    del r2
    with np.errstate(invalid='ignore'):
        np.sqrt(one_minus, out=one_minus)                 # NaN outside the disc
    return one_minus


def area_threshold_pixels(smap, A_th=DEFAULTS['A_th'], mu=DEFAULTS['mu_area']):
    """Area of A_th micro-hemispheres (1e-6 of the area of the visible hemisphere, 2 pi R^2) projected at mu, in pixels of
    smap. mu = 0.1 (default): the smallest projected area, so that a plage near the limb is not taken for network."""
    import astropy.units as u
    rs = smap.rsun_obs.to_value(u.arcsec)
    pix = abs(smap.scale.axis1.to_value(u.arcsec / u.pix) * smap.scale.axis2.to_value(u.arcsec / u.pix))
    return A_th * 1e-6 * 2 * np.pi * rs**2 * mu / pix


def deprojected_area_uhem(smap, mask):
    """Area on the solar surface of the pixels of mask [micro-hemispheres]: the projected area of every pixel divided by
    its mu, over the area of the visible hemisphere."""
    import astropy.units as u
    mu = mu_of_map(smap)
    pix = abs(smap.scale.axis1.to_value(u.arcsec / u.pix) * smap.scale.axis2.to_value(u.arcsec / u.pix))
    valid = np.asarray(mask, dtype=bool) & np.isfinite(mu) & (mu > 0)
    return float(np.sum(pix / mu[valid]) / (2 * np.pi * smap.rsun_obs.to_value(u.arcsec)**2) * 1e6)


def identify_day(cont_map, los_map, size=512, noise_th=DEFAULTS['noise_th'], B_th=DEFAULTS['B_th'],
                 I_th=DEFAULTS['I_th'], A_th=DEFAULTS['A_th'], mu_area=DEFAULTS['mu_area'], faculae='plage'):
    """Spots, plage and network of one epoch (module docstring), from the sunpy maps of the continuum and of the
    magnetogram.

    size     : the images are resampled to size x size pixels (None or 4096: full resolution); noise_th is scaled to it
    noise_th, B_th, I_th, A_th : thresholds (classify_pixels, split_plage_network) [G, G, -, micro-hemispheres]
    mu_area  : mu at which A_th is projected to pixels (area_threshold_pixels; 0.1 by default, the smallest area: at 512 px
               this is < 1 pixel, so every bright group is plage; mu_area = 1 gives the area at the disc centre)
    faculae  : what the facula map contains: 'plage' (default) or 'plage+network'
    Returns a dict: 'spot_map', 'facula_map' (float32 arrays cut to the disc, row 0 = south of the disc: read them with
    origin='lower'; 1 = covered, 0 = quiet, NaN outside the disc: the maps for SunSim), 'plage', 'network', 'spot',
    'facula', 'disc' (boolean, same frame), 'ff_sp',
    'ff_fc', 'ff_nt' (fractions of the disc [%], projected), 'n_spots', 'n_plages', 'I_quiet', 'noise_th' (the threshold
    used), 'area_th_pixels', 'date', and 'cont_map', 'los_map' (the resampled sunpy maps)."""
    _check_thresholds(noise_th, B_th, I_th, A_th, faculae)
    cont_map, los_map = _resample(cont_map, los_map, size)
    return _identify_resampled(cont_map, los_map, mu_of_map(los_map), noise_th, B_th, I_th, A_th, mu_area, faculae)


def _check_thresholds(noise_th, B_th, I_th, A_th, faculae):
    if faculae not in ('plage', 'plage+network'):
        raise ValueError("faculae must be 'plage' or 'plage+network', got %r" % (faculae,))
    for name, value in (('noise_th', noise_th), ('B_th', B_th), ('I_th', I_th), ('A_th', A_th)):
        if not np.isfinite(value) or value < 0:
            raise ValueError('%s must be a number >= 0, got %r' % (name, value))


def _resample(cont_map, los_map, size):
    """The two maps resampled to size x size pixels (None or their size: unchanged), after checking their orientation."""
    import astropy.units as u
    if size is not None:
        size = int(size)
        if not 16 <= size <= FULL_SIZE:
            raise ValueError('size must be between 16 and %d pixels, got %d' % (FULL_SIZE, size))
    for m, name in ((cont_map, 'cont_map'), (los_map, 'los_map')):
        crota = float(m.meta.get('crota2', 180.0))
        if abs(abs(crota) - 180.0) > 1.0:
            warnings.warn('%s: CROTA2 = %.2f deg, the images are expected upside down (180 deg): the orientation of the '
                          'maps may be wrong' % (name, crota))
    if size is not None and size != cont_map.data.shape[0]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cont_map = cont_map.resample([size, size] * u.pix)
            los_map = los_map.resample([size, size] * u.pix)
    return cont_map, los_map


def _identify_resampled(cont_map, los_map, mu, noise_th, B_th, I_th, A_th, mu_area, faculae):
    """identify_day on maps already resampled, with their mu (mu_of_map): the part that depends on the thresholds
    (make_sdo_sweep reads, resamples and computes mu once per day for all the sets of thresholds)."""
    n = cont_map.data.shape[0]
    noise = noise_threshold_for(n, noise_th)
    c = classify_pixels(cont_map.data, los_map.data, mu, noise_th=noise, B_th=B_th, I_th=I_th)
    area_px = area_threshold_pixels(los_map, A_th, mu_area)
    plage, network, n_plages = split_plage_network(c['bright'], area_px)
    n_spots = count_spots(c['spot'])[0]

    lo, hi = disc_square(cont_map.data)

    def frame(a):
        """Cut to the square of the disc and turned by 180 deg: row 0 = south, east on the left (shown with
        origin='lower': north up, the rotation from left to right)."""
        return np.flip(np.asarray(a)[lo:hi, lo:hi])

    disc = frame(c['disc'])
    fac = plage | network if faculae == 'plage+network' else plage
    maps = {}
    for key, mask in (('spot_map', c['spot']), ('facula_map', fac)):
        m = frame(mask).astype(np.float32)
        m[~disc] = np.nan
        maps[key] = m
    n_disc = max(int(np.sum(disc)), 1)
    return dict(maps, plage=frame(plage), network=frame(network), spot=frame(c['spot']), facula=frame(fac), disc=disc,
                ff_sp=100.0 * np.sum(frame(c['spot'])) / n_disc, ff_fc=100.0 * np.sum(frame(fac)) / n_disc,
                ff_nt=100.0 * np.sum(frame(network)) / n_disc, n_spots=n_spots, n_plages=n_plages,
                I_quiet=c['I_quiet'], noise_th=noise, area_th_pixels=area_px, date=str(cont_map.date),
                cont_map=cont_map, los_map=los_map)


_PROJECTIONS = {}             # (n_rows, n_cols, n_rings) -> (cell of every pixel, pixels of every cell)


def cell_projection(shape, n_rings):
    """Cell of every pixel (flattened, -1 outside the disc) and number of pixels of every cell, for maps of this shape
    in the frame of identify_day (origin='lower') and a SunSim grid of n_rings rings (computed once per shape)."""
    from . import nbspectra
    key = (shape[0], shape[1], int(n_rings))
    if key not in _PROJECTIONS:
        nin = np.asarray(nbspectra.generate_grid_coordinates_nb(int(n_rings))[1], dtype=np.int64)
        cells = nbspectra.pixel_to_cell_nb(shape[0], shape[1], int(n_rings), nin, False, False)     # origin='lower'
        npix = np.bincount(cells[cells >= 0], minlength=int(nin.sum())).astype(np.float64)
        if np.any(npix == 0):
            warnings.warn('%d cells of the grid of %d rings have no pixel in %d x %d maps (taken as quiet): use larger '
                          'maps or fewer rings (max_rings)' % (np.sum(npix == 0), n_rings, shape[0], shape[1]))
        _PROJECTIONS[key] = (cells, npix)
    return _PROJECTIONS[key]


def cell_filling_factors(day, n_rings, spots_over_faculae=True):
    """Spot and facula filling factors of every cell of a SunSim grid of n_rings rings, from one day (identify_day):
    the same projection of the pixels as active_region_mask(folder, origin='lower') (computed once for every image size)
    and the same mean over the pixels of every cell (spots over faculae in every pixel). Returns two float32 arrays
    (n_cells,), what StarSim needs from the maps: the full maps do not have to be written."""
    from . import nbspectra
    cells, npix = cell_projection(day['spot'].shape, n_rings)
    sp = np.ascontiguousarray(day['spot']).view(np.uint8).ravel()
    fc = np.ascontiguousarray(day['facula']).view(np.uint8).ravel()
    ff_sp, ff_fc = nbspectra.maps_to_cells_nb(cells, sp, fc, npix, bool(spots_over_faculae))
    return ff_sp.astype(np.float32), ff_fc.astype(np.float32)


# =============================================================================================
# many epochs: download, maps, simulation
# =============================================================================================
def download_hmi(start, end, folder, email, step=1, window_seconds=30):
    """Downloads from JSOC (sunpy.net.Fido) the continuum and the magnetogram of every step-th day from start to end
    (excluded), the first images after 00:00 TAI (window of window_seconds), into folder. email: an address registered
    at JSOC (http://jsoc.stanford.edu/ajax/register_email.html). Returns the list of the files downloaded."""
    t0, t1 = _date(start, 'start'), _date(end, 'end')
    if not t1 > t0:
        raise ValueError('end (%s) must be after start (%s)' % (t1.date(), t0.date()))
    if not email:
        raise ValueError('a JSOC email address is needed to download (email=...)')
    _sunpy()                                             # clear error if sunpy is missing
    from sunpy.net import Fido, attrs as a
    os.makedirs(folder, exist_ok=True)
    files = []
    day = t0
    while day < t1:
        tr = a.Time(day, day + timedelta(seconds=window_seconds))
        for series in (CONTINUUM_SERIES, MAGNETOGRAM_SERIES):
            if find_hmi_files(folder, day) is not None:
                break                                    # already there
            res = Fido.search(tr, a.jsoc.Series(series), a.jsoc.Notify(email))
            files += list(Fido.fetch(res, path=os.path.join(str(folder), '{file}')))
        day += timedelta(days=step)
    return files


def _hmi_index(folder):
    """{YYYYMMDD: (continuum file, magnetogram file)} of a folder, from one listing (the first file of the day of
    each series): fast for thousands of days."""
    import fnmatch
    names = sorted(os.listdir(str(folder)))
    cont, mag = {}, {}
    for n in names:
        if fnmatch.fnmatch(n, CONTINUUM_PATTERN.format(date='*')):
            cont.setdefault(n.split('.')[2][:8], n)
        elif fnmatch.fnmatch(n, MAGNETOGRAM_PATTERN.format(date='*')):
            mag.setdefault(n.split('.')[2][:8], n)
    return {d: (os.path.join(str(folder), cont[d]), os.path.join(str(folder), mag[d])) for d in cont if d in mag}


THRESHOLD_KEYS = ('noise_th', 'B_th', 'I_th', 'A_th', 'mu_area', 'faculae')


def parameter_sets(noise_th=DEFAULTS['noise_th'], B_th=DEFAULTS['B_th'], I_th=DEFAULTS['I_th'], A_th=DEFAULTS['A_th'],
                   mu_area=DEFAULTS['mu_area'], faculae='plage'):
    """All the combinations of the thresholds (each a value or a list of values): a list of dicts with the keys of
    THRESHOLD_KEYS, in the order of itertools.product."""
    import itertools
    values = []
    for key, v in zip(THRESHOLD_KEYS, (noise_th, B_th, I_th, A_th, mu_area, faculae)):
        v = [v] if isinstance(v, (str, int, float, np.integer, np.floating)) else list(v)
        if not v:
            raise ValueError('%s: give at least one value' % key)
        values.append([x if key == 'faculae' else float(x) for x in v])
    sets = [dict(zip(THRESHOLD_KEYS, combo)) for combo in itertools.product(*values)]
    for p in sets:
        _check_thresholds(p['noise_th'], p['B_th'], p['I_th'], p['A_th'], p['faculae'])
    return sets


def set_label(params):
    """Name of the folder of a set of thresholds in make_sdo_sweep, e.g. noise8_B24_I0.89_A20_mu0.1 (+ '_network' with
    faculae='plage+network')."""
    p = dict(DEFAULTS, faculae='plage')
    p.update(params)
    return ('noise%g_B%g_I%g_A%g_mu%g' % (p['noise_th'], p['B_th'], p['I_th'], p['A_th'], p['mu_area'])
            + ('_network' if p['faculae'] == 'plage+network' else ''))


def _day_done(out_folder, t, date, fmt, n_rings):
    """True if every output of a day is already in out_folder (resume). The files are those of the label stored in
    days/<date>.json (with epoch='index' it can differ from t: the days are numbered again after every run)."""
    import json
    path = os.path.join(out_folder, 'days', date + '.json')
    if not os.path.exists(path):
        return False
    try:
        label = _time_label(json.load(open(path))['t'])
    except (OSError, ValueError, KeyError):
        return False
    if fmt is not None and not os.path.exists(os.path.join(out_folder, 'spot_map_%s.%s' % (label, fmt))):
        return False
    return all(os.path.exists(os.path.join(out_folder, 'cells_N%d' % n, 'ff_%s.npy' % label)) for n in n_rings)


def _process_day(task):
    """One day of make_sdo_maps / make_sdo_sweep (run in a worker process with workers > 1): the images are read,
    resampled and their mu computed once, then for every set of thresholds still to do: identification, maps, cell
    filling factors and the summary row (days/<date>.json of the folder of the set).
    Returns the list of the rows (one per set done), or ('skipped', date, reason)."""
    import json
    data_folder, day, t, files, size, fmt, n_rings, sets = task
    date = day.strftime('%Y-%m-%d')
    try:
        maps = _read_with_retries(data_folder, day, files=files)
    except OSError as e:                                 # unreadable file (e.g. a cloud folder that times out)
        return ('skipped', date, 'images not readable (%s)' % str(e).splitlines()[0][:120])
    cont_map, los_map = _resample(*maps, size)
    del maps
    mu = mu_of_map(los_map)
    label = _time_label(t)
    rows = []
    for out_folder, params in sets:
        res = _identify_resampled(cont_map, los_map, mu, **params)
        for kind, key, mask in (('spot', 'spot_map', 'spot'), ('faculae', 'facula_map', 'facula')):
            path = os.path.join(out_folder, '%s_map_%s' % (kind, label))
            if fmt == 'npy':
                np.save(path + '.npy', res[mask].view(np.uint8))
            elif fmt == 'npz':
                np.savez_compressed(path + '.npz', packed=np.packbits(res[mask]), shape=np.array(res[mask].shape))
            elif fmt == 'txt':
                np.savetxt(path + '.txt', res[key])
        for n in n_rings:
            ff_sp, ff_fc = cell_filling_factors(res, n)
            np.save(os.path.join(out_folder, 'cells_N%d' % n, 'ff_%s.npy' % label), np.stack([ff_sp, ff_fc]))
        row = {'t': t, 'date': date, 'ff_sp': res['ff_sp'], 'ff_fc': res['ff_fc'], 'ff_nt': res['ff_nt'],
               'n_spots': int(res['n_spots']), 'n_plages': int(res['n_plages'])}
        with open(os.path.join(out_folder, 'days', date + '.json'), 'w') as f:
            json.dump(row, f)
        rows.append(row)
        del res
    return rows


def _run_sets(data_folder, sets, start, end, step, size, epoch, fmt, n_rings, workers, resume, download, email,
              verbose):
    """make_sdo_maps for several sets of thresholds, sets = [(out_folder, params)], every day read once.
    Returns the list of the results (dicts of make_sdo_maps), in the order of sets."""
    import json
    t0, t1 = _date(start, 'start'), _date(end, 'end')
    if not t1 > t0:
        raise ValueError('end (%s) must be after start (%s)' % (t1.date(), t0.date()))
    if int(step) < 1:
        raise ValueError('step must be >= 1 day, got %r' % (step,))
    if epoch not in ('days', 'index'):
        raise ValueError("epoch must be 'days' or 'index', got %r" % (epoch,))
    if fmt not in ('npy', 'npz', 'txt', None):
        raise ValueError("fmt must be 'npy', 'npz', 'txt' or None, got %r" % (fmt,))
    n_rings = [] if n_rings is None else sorted({int(n) for n in np.atleast_1d(n_rings)})
    if fmt is None and not n_rings:
        raise ValueError('fmt=None writes no map: give n_rings (the filling factors of the cells), or a format')
    if int(workers) < 1:
        raise ValueError('workers must be >= 1, got %r' % (workers,))
    if size is not None and not 16 <= int(size) <= FULL_SIZE:
        raise ValueError('size must be between 16 and %d pixels, got %r' % (FULL_SIZE, size))
    if not download and not os.path.isdir(str(data_folder)):
        raise FileNotFoundError('the folder of the HMI images does not exist: %s' % data_folder)
    if download:
        download_hmi(t0, t1, data_folder, email, step=step)
    sets = [(str(folder), dict(params)) for folder, params in sets]
    for folder, _ in sets:
        for sub in ['days'] + ['cells_N%d' % n for n in n_rings]:
            os.makedirs(os.path.join(folder, sub), exist_ok=True)

    # the days of the period and their files (one listing of the folder), and the sets still to do for every day
    index = _hmi_index(data_folder)
    days, todo, skipped = [], [], []
    day = t0
    while day < t1:
        files = index.get(day.strftime('%Y%m%d'))
        if files is None:
            skipped.append(day.strftime('%Y-%m-%d'))
        else:
            # epoch='index': a provisional number, different from every final one (0 ... n-1), see _renumber
            t = float((day - t0).days) if epoch == 'days' else 1e6 + len(days)
            days.append((day, t))
            left = [st for st in sets if not (resume and _day_done(st[0], t, day.strftime('%Y-%m-%d'), fmt, n_rings))]
            if left:
                todo.append((str(data_folder), day, t, files, size, fmt, n_rings, left))
        day += timedelta(days=int(step))
    if verbose:
        print('%d days with images, %d to do (%d already done), %d without images; %d set(s) of thresholds; '
              '%d worker(s)' % (len(days), len(todo), len(days) - len(todo), len(skipped), len(sets), int(workers)))

    def report(r, k):
        if isinstance(r, tuple):                         # ('skipped', date, reason)
            skipped.append(r[1])
            if verbose:
                print('%s: %s, skipped' % (r[1], r[2]))
        elif r and any(x['ff_fc'] > 20 or x['ff_sp'] > 5 for x in r):
            print('WARNING %s: ff_sp = %.2f %%, ff_fc = %.2f %% (a set of this day): check the HMI images of this day'
                  % (r[0]['date'], max(x['ff_sp'] for x in r), max(x['ff_fc'] for x in r)))
        if verbose and r and not isinstance(r, tuple):
            x = r[0]
            more = ' (+ %d other sets)' % (len(r) - 1) if len(r) > 1 else ''
            print('[%d/%d] %s (t = %g): ff_sp = %.3f %%, ff_fc = %.3f %%, ff_nt = %.3f %%, %d spots, %d plages%s'
                  % (k, len(todo), x['date'], x['t'], x['ff_sp'], x['ff_fc'], x['ff_nt'], x['n_spots'], x['n_plages'],
                     more))

    if int(workers) == 1:
        for k, tk in enumerate(todo, 1):
            report(_process_day(tk), k)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            for k, r in enumerate(pool.map(_process_day, todo), 1):
                report(r, k)

    results = []
    for folder, params in sets:
        if epoch == 'index':
            _renumber(folder, [d for d, _ in days], fmt, n_rings)
        # the rows of the period (this run and the previous ones), sorted
        rows = []
        for day, t in days:
            path = os.path.join(folder, 'days', day.strftime('%Y-%m-%d') + '.json')
            if os.path.exists(path):
                rows.append(json.load(open(path)))
        rows.sort(key=lambda r: r['t'])
        with open(os.path.join(folder, 'summary.csv'), 'w') as f:
            f.write('t,date,ff_sp,ff_fc,ff_nt,n_spots,n_plages\n')
            for r in rows:
                f.write('%g,%s,%.6f,%.6f,%.6f,%d,%d\n' % (r['t'], r['date'], r['ff_sp'], r['ff_fc'], r['ff_nt'],
                                                         r['n_spots'], r['n_plages']))
        for n in n_rings:                                # all the cells of the period in one file
            ff = [np.load(os.path.join(folder, 'cells_N%d' % n, 'ff_%s.npy' % _time_label(r['t']))) for r in rows]
            np.savez(os.path.join(folder, 'filling_factors_N%d.npz' % n), times=np.array([r['t'] for r in rows]),
                     dates=np.array([r['date'] for r in rows]), N_rings=n,
                     ff_sp=np.array([f[0] for f in ff], dtype=np.float32),
                     ff_fc=np.array([f[1] for f in ff], dtype=np.float32))
        with open(os.path.join(folder, 'parameters.txt'), 'w') as f:
            f.write('data_folder = %s\nstart = %s\nend = %s\nstep = %d\nsize = %s\nnoise_th = %g (4096 px)\nB_th = %g\n'
                    'I_th = %g\nA_th = %g\nmu_area = %g\nfaculae = %s\nepoch = %s\nfmt = %s\nn_rings = %s\nskipped = %s\n'
                    % (data_folder, t0.date(), t1.date(), int(step), size, params['noise_th'], params['B_th'],
                       params['I_th'], params['A_th'], params['mu_area'], params['faculae'], epoch, fmt,
                       ' '.join(map(str, n_rings)) or '-', ', '.join(sorted(set(skipped))) or '-'))
        results.append({'folder': folder, 'params': params, 'times': np.array([r['t'] for r in rows], dtype=float),
                        'dates': [r['date'] for r in rows], 'ff_sp': np.array([r['ff_sp'] for r in rows]),
                        'ff_fc': np.array([r['ff_fc'] for r in rows]), 'ff_nt': np.array([r['ff_nt'] for r in rows]),
                        'n_spots': np.array([r['n_spots'] for r in rows], dtype=int),
                        'n_plages': np.array([r['n_plages'] for r in rows], dtype=int),
                        'skipped': sorted(set(skipped))})
    return results


def _renumber(folder, days, fmt, n_rings):
    """epoch='index': the days of the period with outputs numbered 0, 1, 2, ... in the order of the dates, as the
    previous script (a day whose images could not be read has no number). The index of a day is given before it is
    read, so after a day is skipped the files of the next days are renamed (maps, cells, days/<date>.json), first to
    temporary names so that no file is overwritten by another of this period."""
    import json
    rows = []
    for day in days:
        path = os.path.join(folder, 'days', day.strftime('%Y-%m-%d') + '.json')
        if os.path.exists(path):
            rows.append((path, json.load(open(path))))
    rows.sort(key=lambda r: r[1]['date'])
    moves = [(path, row, float(k)) for k, (path, row) in enumerate(rows) if float(row['t']) != float(k)]

    def files(label):
        out = []
        if fmt is not None:
            out += [os.path.join(folder, '%s_map_%s.%s' % (kind, label, fmt)) for kind in ('spot', 'faculae')]
        out += [os.path.join(folder, 'cells_N%d' % n, 'ff_%s.npy' % label) for n in n_rings]
        return out

    for stage in ('to_tmp', 'to_final'):
        for path, row, k in moves:
            old = files(_time_label(row['t'])) if stage == 'to_tmp' else [f + '.renumber' for f in files(_time_label(row['t']))]
            new = [f + '.renumber' for f in files(_time_label(row['t']))] if stage == 'to_tmp' else files(_time_label(k))
            for a, b in zip(old, new):
                if os.path.exists(a):
                    os.replace(a, b)
    for path, row, k in moves:
        row['t'] = k
        with open(path, 'w') as f:
            json.dump(row, f)
    if fmt is not None:
        import re
        extra = [f for f in os.listdir(folder) if re.fullmatch(r'(spot|faculae)_map_(.+)\.%s' % fmt, f)
                 and not float(re.fullmatch(r'(spot|faculae)_map_(.+)\.%s' % fmt, f).group(2)) < len(rows)]
        if extra:
            warnings.warn('%s has %d maps of epochs that are not days of this period (e.g. %s): '
                          'active_region_mask reads every map of the folder' % (folder, len(extra), sorted(extra)[0]))


def make_sdo_maps(data_folder, out_folder, start, end, step=1, size=512, noise_th=DEFAULTS['noise_th'],
                  B_th=DEFAULTS['B_th'], I_th=DEFAULTS['I_th'], A_th=DEFAULTS['A_th'], mu_area=DEFAULTS['mu_area'],
                  faculae='plage', epoch='days', fmt='npy', n_rings=None, workers=1, resume=True, download=False,
                  email=None, verbose=True):
    """Spot and facula maps of every step-th day from start to end (excluded), from the HMI images of data_folder,
    written in out_folder for active_region_mask(out_folder, origin='lower'). For several values of the thresholds at
    once (every day read once): make_sdo_sweep.

    fmt      : how the maps are written: 'npy' (default: uint8 masks, small and fast to read), 'npz' (bit-packed and
               compressed: a few tens of kB per map even at 4096 px, read by active_region_mask too), 'txt' (float,
               NaN outside the disc, as the old StarSim reads them), or None (no maps: only n_rings).
    n_rings  : None, a number of rings or a list: the spot and facula filling factors of every cell of these SunSim grids
               are computed right after the identification (cell_filling_factors) and written in
               cells_N<n>/ff_<t>.npy, then gathered in filling_factors_N<n>.npz. That is all StarSim needs
               (load_filling_factors), much smaller than the maps at full resolution (e.g. 4096 px, 70 rings: 50 kB
               per day against 2 x 16 MB).
    workers  : number of processes working on different days at the same time (each needs ~1-1.5 GB at 4096 px). In a
               Python script, call make_sdo_maps under `if __name__ == '__main__':` (needed to start processes on macOS
               and Windows; not in a notebook nor with the terminal command).
    resume   : True: the days whose outputs are already in out_folder are not done again (a long run can be stopped
               and started again, or split into periods with the same out_folder).
    epoch    : the {t} of the file names: 'days' (days since start, default: the time unit of SunSim) or 'index'
               (0, 1, 2, ... in the order of the days with images, as the previous script)
    download : True -> download the missing days from JSOC first (download_hmi, needs email)
    Other arguments: identify_day (one value each). The days without both images, or whose images cannot be read (tried
    3 times), are skipped (listed in 'skipped' and in parameters.txt).
    Also writes summary.csv (one row per epoch: t, date, ff_sp, ff_fc, ff_nt [% of the disc], n_spots, n_plages; all the
    days of out_folder in [start, end)) and parameters.txt (the arguments).
    Returns a dict: 'times', 'dates', 'ff_sp', 'ff_fc', 'ff_nt', 'n_spots', 'n_plages' (arrays), 'skipped' (dates),
    'folder' and 'params' (the thresholds)."""
    sets = parameter_sets(noise_th, B_th, I_th, A_th, mu_area, faculae)
    if len(sets) > 1:
        raise ValueError('make_sdo_maps takes one value of every threshold; for several values use make_sdo_sweep')
    return _run_sets(data_folder, [(out_folder, sets[0])], start, end, step, size, epoch, fmt, n_rings, workers,
                     resume, download, email, verbose)[0]


def make_sdo_sweep(data_folder, out_folder, start, end, step=1, size=512, noise_th=DEFAULTS['noise_th'],
                   B_th=DEFAULTS['B_th'], I_th=DEFAULTS['I_th'], A_th=DEFAULTS['A_th'], mu_area=DEFAULTS['mu_area'],
                   faculae='plage', epoch='days', fmt=None, n_rings=None, workers=1, resume=True, download=False,
                   email=None, verbose=True):
    """make_sdo_maps for all the combinations of the thresholds (each a value or a list, parameter_sets), every set in
    its own folder out_folder/<set_label> (e.g. noise8_B24_I0.89_A20_mu0.1). Every day is read, resampled and its mu
    computed once for all the sets (at 4096 px most of the time of a day), then classified for every set.
    fmt is None by default (no maps, only the cell filling factors of n_rings: give n_rings, the rings of the
    simulations); with resume, the sets and days already done are skipped, so new values can be added to a sweep.
    Simulate the sets with simulate_sdo_sets (the grids are computed once for all of them).
    Also writes out_folder/sets.csv (the folder and the thresholds of every set made in out_folder).
    Returns {label: result of make_sdo_maps}."""
    sets = parameter_sets(noise_th, B_th, I_th, A_th, mu_area, faculae)
    labels = [set_label(p) for p in sets]
    if len(set(labels)) != len(labels):
        raise ValueError('the same set of thresholds is given twice: %s' % labels)
    results = _run_sets(data_folder, [(os.path.join(str(out_folder), lb), p) for lb, p in zip(labels, sets)], start,
                        end, step, size, epoch, fmt, n_rings, workers, resume, download, email, verbose)
    # sets.csv: every set ever made in out_folder (a later sweep can add sets)
    path = os.path.join(str(out_folder), 'sets.csv')
    known = {}
    if os.path.exists(path):
        for line in open(path).read().splitlines()[1:]:
            known[line.split(',')[0]] = line
    for lb, p in zip(labels, sets):
        known[lb] = '%s,%g,%g,%g,%g,%g,%s' % (lb, p['noise_th'], p['B_th'], p['I_th'], p['A_th'], p['mu_area'],
                                              p['faculae'])
    with open(path, 'w') as f:
        f.write('label,noise_th,B_th,I_th,A_th,mu_area,faculae\n' + '\n'.join(known[k] for k in sorted(known)) + '\n')
    return dict(zip(labels, results))


def _grids_key(conf_file, observables, reproduce_old_bugs):
    """Identifier of the grids of a configuration: hash of the file without the line of SDO_input_path (the maps may
    change, not the grids), of the observables and of reproduce_old_bugs."""
    import hashlib
    lines = [ln.strip() for ln in open(conf_file).read().splitlines()
             if ln.strip() and not ln.strip().lower().startswith('sdo_input_path')]
    text = '\n'.join(lines) + '|%s|%s' % (','.join(sorted(observables)), bool(reproduce_old_bugs))
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _load_grids(path, key):
    import pickle
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
    except Exception:
        return None
    return d['grids'] if d.get('key') == key else None


def _save_grids(path, key, grids):
    import pickle
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump({'key': key, 'grids': grids}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + '.tmp', path)


def simulate_sdo_sets(conf_file, folders, observables=('lc', 'rv', 'fwhm', 'bis', 'contrast'), grids_cache=None,
                      chunk_size=None, save=True, reproduce_old_bugs=False, resume=True, verbose=True):
    """Simulates the maps of several folders (make_sdo_maps / make_sdo_sweep) with the configuration conf_file
    (simulate_old_config, sdo_input = 1): the grids (spectra, CCFs of the rings: most of the time of a simulation) are
    computed once and used for all the folders.
    folders     : a folder or a list (the SDO_input_path of the configuration is replaced by each of them). A folder
                  with filling_factors_N<n>.npz for the rings of the configuration is simulated from these cell
                  filling factors (load_filling_factors: no map is read), otherwise from its maps.
    grids_cache : a file (.pkl): the grids are read from it if it was made with the same configuration (the file
                  without its SDO_input_path, the observables, reproduce_old_bugs), else computed and written in it.
                  Other runs (e.g. new thresholds, in another call of the terminal command) then do not compute them
                  again. Delete the file if the spectra or masks of the configuration change on disk.
    chunk_size  : epochs simulated together (None: all): the memory is that of chunk_size epochs.
    resume      : True: a folder already simulated with the same configuration and observables (its simulation.json),
                  whose maps did not change since (same summary.csv), is not simulated again: its outputs are read.
    save        : write in every folder lc.txt, rv.txt, fwhm.txt, bis.txt, contrast.txt (time, value), ff.txt (time,
                  ff_sp, ff_fc [%]), ccf.txt (rows: epochs) and ccf_rv.txt, as the terminal command.
    Returns {folder: dict 'time', 'ff_sp', 'ff_fc', 'ff_ph', and the observables ('lc', 'rv', 'fwhm', 'bis',
    'contrast', 'ccf', 'ccf_rv')}."""
    from .old_config import simulate_old_config, read_old_config
    folders = [folders] if isinstance(folders, (str, os.PathLike)) else list(folders)
    observables = [observables] if isinstance(observables, str) else list(observables)
    cfg, _ = read_old_config(conf_file, observables=observables, paths={'sdo_folder': str(folders[0])})
    N = int(cfg['n_grid_rings'])
    key = _grids_key(conf_file, observables, reproduce_old_bugs)
    grids = _load_grids(grids_cache, key)
    if verbose:
        print('grids: %s' % ('read from %s' % grids_cache if grids is not None else 'computed (once for the %d folders)'
                                                                                     % len(folders)))
    out = {}
    for folder in folders:
        folder = str(folder)
        done = _simulated(folder, key, observables) if (resume and save) else None
        if done is not None:
            out[folder] = done
            if verbose:
                print('%s: already simulated (outputs read)' % folder)
            continue
        ff_file = os.path.join(folder, 'filling_factors_N%d.npz' % N)
        if os.path.exists(ff_file):
            mask = load_filling_factors(folder, N)
        else:
            from .active_region_mask import active_region_mask
            mask = None if reproduce_old_bugs else active_region_mask(folder, origin='lower', flip_horizontal=False)
        times = mask.times if mask is not None else None
        if times is None:
            from .active_region_mask import active_region_mask
            times = active_region_mask(folder, origin='lower').times
        step = len(times) if not chunk_size else int(chunk_size)
        parts = []
        for i in range(0, len(times), step):
            r = simulate_old_config(conf_file, times=times[i:i + step], observables=observables,
                                    paths={'sdo_folder': folder}, reproduce_old_bugs=reproduce_old_bugs, verbose=False,
                                    grids=grids, active_regions_mask=mask)
            if grids is None or len(r['grids']) > len(grids):
                grids = r['grids']
                _save_grids(grids_cache, key, grids)
            parts.append({k: r[k] for k in ('time', 'ff_ph', 'ff_sp', 'ff_fc', 'lc', 'rv', 'fwhm', 'bis', 'contrast')
                          if k in r})
            if 'CCF_unshifted' in r:
                parts[-1]['ccf'] = r['CCF_unshifted'][1:]
                ccf_rv = r['CCF_unshifted'][0]
            del r
        res = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if 'ccf' in res:
            res['ccf_rv'] = ccf_rv
        out[folder] = res
        if save:
            for k in ('lc', 'rv', 'fwhm', 'bis', 'contrast'):
                if k in res:
                    np.savetxt(os.path.join(folder, '%s.txt' % k), np.column_stack([res['time'], res[k]]),
                               header='time [d], %s' % k)
            np.savetxt(os.path.join(folder, 'ff.txt'), np.column_stack([res['time'], res['ff_sp'], res['ff_fc']]),
                       header='time [d], ff_sp, ff_fc [% of the disc]')
            if 'ccf' in res:
                np.savetxt(os.path.join(folder, 'ccf.txt'), res['ccf'], header='CCF of every epoch (rows), '
                                                                                 'velocities: ccf_rv.txt')
                np.savetxt(os.path.join(folder, 'ccf_rv.txt'), res['ccf_rv'])
            import json
            with open(os.path.join(folder, 'simulation.json'), 'w') as f:
                json.dump({'conf': os.path.abspath(str(conf_file)), 'grids_key': key, 'observables': sorted(observables),
                           'n_epochs': int(len(res['time'])),
                           'summary': _file_hash(os.path.join(folder, 'summary.csv'))}, f)
        if verbose:
            print('%s: %d epochs simulated%s' % (folder, len(res['time']), ', outputs written' if save else ''))
    return out


def build_grids(conf_file, grids_cache, observables=('lc', 'rv', 'fwhm', 'bis', 'contrast'), reproduce_old_bugs=False,
                verbose=True):
    """Computes the grids of a configuration (sdo_input = 1) and writes them in grids_cache, without any map: the jobs
    that then simulate many sets of thresholds with this configuration (simulate_sdo_sets(..., grids_cache=...), or
    sunsim-sdo-maps --conf ... --grids-cache ...) only read them. Nothing is computed if the file already has them."""
    from .old_config import simulate_old_config, read_old_config
    from .active_region_mask import precomputed_filling_factors
    from . import nbspectra
    observables = [observables] if isinstance(observables, str) else list(observables)
    key = _grids_key(conf_file, observables, reproduce_old_bugs)
    if _load_grids(grids_cache, key) is not None:
        if verbose:
            print('grids already in %s' % grids_cache)
        return grids_cache
    cfg, _ = read_old_config(conf_file, observables=observables, paths={'sdo_folder': os.path.dirname(str(grids_cache))})
    N = int(cfg['n_grid_rings'])
    n_cells = int(np.sum(nbspectra.generate_grid_coordinates_nb(N)[1]))
    quiet = precomputed_filling_factors([0.0], np.zeros((1, n_cells)), np.zeros((1, n_cells)), N)   # one quiet epoch
    r = simulate_old_config(conf_file, times=[0.0], observables=observables, reproduce_old_bugs=reproduce_old_bugs,
                            verbose=False, active_regions_mask=quiet)
    _save_grids(grids_cache, key, r['grids'])
    if verbose:
        print('grids of %s (%d rings, %s) written in %s' % (conf_file, N, ', '.join(observables), grids_cache))
    return grids_cache


def _file_hash(path):
    import hashlib
    return hashlib.sha1(open(path, 'rb').read()).hexdigest() if os.path.exists(path) else None


def _simulated(folder, key, observables):
    """The outputs of simulate_sdo_sets in folder (read from the text files) if they were made with the grids key and
    the observables and the maps did not change since (same summary.csv), else None."""
    import json
    path = os.path.join(folder, 'simulation.json')
    summary = os.path.join(folder, 'summary.csv')
    if not os.path.exists(path):
        return None
    info = json.load(open(path))
    if info.get('grids_key') != key or info.get('observables') != sorted(observables):
        return None
    if info.get('summary') != _file_hash(summary):
        return None
    try:
        ff = np.loadtxt(os.path.join(folder, 'ff.txt'), ndmin=2)
        res = {'time': ff[:, 0], 'ff_sp': ff[:, 1], 'ff_fc': ff[:, 2]}
        for k in ('lc', 'rv', 'fwhm', 'bis', 'contrast'):
            f = os.path.join(folder, '%s.txt' % k)
            if os.path.exists(f):
                res[k] = np.loadtxt(f, ndmin=2)[:, 1]
        if os.path.exists(os.path.join(folder, 'ccf.txt')):
            res['ccf'] = np.loadtxt(os.path.join(folder, 'ccf.txt'), ndmin=2)
            res['ccf_rv'] = np.loadtxt(os.path.join(folder, 'ccf_rv.txt'))
    except (OSError, ValueError):
        return None
    return res if len(res['time']) == info.get('n_epochs') else None


def load_filling_factors(out_folder, n_rings):
    """The cell filling factors written by make_sdo_maps(..., n_rings=n_rings) (filling_factors_N<n>.npz), as an object
    for StarSim(..., active_regions_mask=...): no map is read. Its .times are the epochs, .dates the dates."""
    from .active_region_mask import precomputed_filling_factors
    d = np.load(os.path.join(str(out_folder), 'filling_factors_N%d.npz' % int(n_rings)))
    out = precomputed_filling_factors(d['times'], d['ff_sp'], d['ff_fc'], int(d['N_rings']))
    out.dates = list(d['dates'])
    return out


def _read_with_retries(folder, day, attempts=3, wait=5.0, files=None):
    """read_hmi_day (or the two files given), tried again (attempts times, wait seconds apart) when a file cannot be
    read (OSError: e.g. a folder synchronised from the cloud that times out)."""
    import time
    for k in range(attempts):
        try:
            if files is not None:
                smap = _sunpy()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    return smap.Map(files[0]), smap.Map(files[1])
            return read_hmi_day(folder, day)
        except FileNotFoundError:
            raise
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(wait)


def _time_label(t):
    """{t} of the file names, as a float ('12.0', '12.5'): the names the old StarSim reads (faculae_map_{t}.txt with
    t a float) and active_region_mask reads."""
    return repr(float(t))


# =============================================================================================
# figures
# =============================================================================================
def plot_hmi(cont_map, los_map, figsize=(11, 5)):
    """The continuum intensity and |B| of the magnetogram (logarithmic scale), side by side."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # HMI images are turned by 180 deg: flipped back and shown with origin='lower' (solar north up)
    im0 = axes[0].imshow(np.flip(cont_map.data), cmap='afmhot', vmin=0.0, vmax=1.3, origin='lower')
    im1 = axes[1].imshow(np.flip(np.abs(los_map.data)), cmap='Blues', norm=colors.LogNorm(vmin=9, vmax=1.5e3), origin='lower')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label='continuum intensity')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label='|B| [G]')
    for ax, title in zip(axes, ('continuum intensity', 'LOS magnetogram')):
        ax.set_title('%s, %s' % (title, str(cont_map.date)[:10]), fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes


def plot_identification(day, ax=None, title=None):
    """The classification of one day (identify_day): quiet Sun, spots, plage (faculae), network; solar north up."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    img = np.full(day['disc'].shape, np.nan)
    img[day['disc']] = 0
    img[day['network']] = 3
    img[day['plage']] = 2
    img[day['spot']] = 1
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cmap = ListedColormap(['#f2e6c9', '#2b1d0e', '#e8743b', '#7fb3d5'])
    cmap.set_bad('white')
    im = ax.imshow(img, cmap=cmap, vmin=-0.5, vmax=3.5, origin='lower', interpolation='nearest')      # north up
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046)
    cbar.ax.set_yticklabels(['quiet Sun', 'spot', 'plage (facula)', 'network'])
    ax.set_title(title or '%s: ff_sp = %.2f %%, ff_fc = %.2f %%' % (day['date'][:10], day['ff_sp'], day['ff_fc']), fontsize=10)
    ax.set_axis_off()
    return ax


# =============================================================================================
# terminal command
# =============================================================================================
def _parser():
    p = argparse.ArgumentParser(prog='sunsim-sdo-maps', description=(
        'Spot and facula maps of the Sun from SDO/HMI images (continuum + magnetogram), for SunSim '
        '(active_region_mask). See the docstring of sunsim.sdo.'))
    p.add_argument('--data', help='folder of the HMI images (downloaded there with --download)')
    p.add_argument('--out', help='folder of the maps')
    p.add_argument('--start', help='first day, YYYY-MM-DD')
    p.add_argument('--end', help='last day (excluded), YYYY-MM-DD')
    p.add_argument('--step', type=int, default=1, help='days between two maps (default 1)')
    p.add_argument('--size', type=int, default=512, help='pixels of the maps (default 512; 4096: full resolution)')
    multi = ' (several values: one set of maps per combination, in --out/<set>)'
    p.add_argument('--noise-th', type=float, nargs='+', default=[DEFAULTS['noise_th']],
                   help='noise threshold [G] at 4096 px (default 8)' + multi)
    p.add_argument('--B-th', type=float, nargs='+', default=[DEFAULTS['B_th']],
                   help='active pixels: |B|/mu > B_th/mu [G] (default 24)' + multi)
    p.add_argument('--I-th', type=float, nargs='+', default=[DEFAULTS['I_th']],
                   help='spots: I < I_th <I_quiet> (default 0.89)' + multi)
    p.add_argument('--A-th', type=float, nargs='+', default=[DEFAULTS['A_th']],
                   help='plage: groups larger than A_th micro-hemispheres (default 20)' + multi)
    p.add_argument('--mu-area', type=float, nargs='+', default=[DEFAULTS['mu_area']],
                   help='mu at which A_th is projected (default 0.1)' + multi)
    p.add_argument('--faculae', choices=['plage', 'plage+network'], nargs='+', default=['plage'],
                   help='content of the facula maps' + multi)
    p.add_argument('--epoch', choices=['days', 'index'], default='days', help='{t} of the map files (default: days since start)')
    p.add_argument('--format', choices=['npy', 'npz', 'txt', 'none'], default=None,
                   help='file format of the maps: npy (default with one set), npz (bit-packed, compressed), txt, '
                        'none (no maps, only the cell filling factors: default with several sets)')
    p.add_argument('--n-rings', type=int, nargs='+', default=None,
                   help='also write the filling factors of the cells of SunSim grids of these numbers of rings')
    p.add_argument('--workers', type=int, default=1, help='processes working on different days (default 1)')
    p.add_argument('--no-resume', action='store_true', help='do again the days already done in --out')
    p.add_argument('--download', action='store_true', help='download the missing days from JSOC first')
    p.add_argument('--email', default=None, help='JSOC email address (for --download)')
    p.add_argument('--conf', default=None, help='old StarSim configuration (sdo_input = 1): simulate the maps of every set '
                                                '(simulate_sdo_sets: the grids are computed once) and write lc.txt, '
                                                'rv.txt, fwhm.txt, bis.txt, contrast.txt, ff.txt, ccf.txt, ccf_rv.txt '
                                                'in the folder of every set. The cell filling factors of its number of '
                                                'rings are added to --n-rings')
    p.add_argument('--observables', nargs='+', default=['lc', 'rv', 'fwhm', 'bis', 'contrast'],
                   help='with --conf: lc, rv, fwhm, bis, contrast (default all)')
    p.add_argument('--grids-cache', default=None,
                   help='with --conf: file of the grids, reused by the next runs with the same configuration '
                        '(default --out/grids_<id>.pkl)')
    p.add_argument('--no-grids-cache', action='store_true', help='with --conf: do not read or write the grids file')
    p.add_argument('--chunk-size', type=int, default=None, help='with --conf: epochs simulated together (default all)')
    p.add_argument('--simulate-only', action='store_true',
                   help='with --conf: do not make the maps, simulate the sets already in --out (all the folders '
                        'of --out/sets.csv, or --out itself)')
    p.add_argument('--build-grids', action='store_true',
                   help='with --conf and --grids-cache: only compute the grids of the configuration and write them in '
                        'the file (e.g. one job before many jobs with other thresholds); no --data, --out, --start, --end')
    p.add_argument('--quiet', action='store_true', help='no progress lines')
    return p


def main(argv=None):
    """Terminal command sunsim-sdo-maps (python -m sunsim.sdo): make_sdo_maps (one value per threshold) or make_sdo_sweep
    (several), then with --conf simulate_sdo_sets on every set (the grids computed once, and kept in a file)."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.build_grids:
        if not (args.conf and args.grids_cache):
            parser.error('--build-grids needs --conf and --grids-cache')
        build_grids(args.conf, args.grids_cache, observables=args.observables, verbose=not args.quiet)
        return 0
    needed = ['out'] if args.simulate_only else ['data', 'out', 'start', 'end']
    missing = ['--' + k for k in needed if getattr(args, k) is None]
    if missing:
        parser.error('the following arguments are required: %s' % ', '.join(missing))
    sets = parameter_sets(args.noise_th, args.B_th, args.I_th, args.A_th, args.mu_area, args.faculae)
    sweep = len(sets) > 1
    n_rings = list(args.n_rings or [])
    if args.conf:
        from .old_config import read_old_config
        cfg, _ = read_old_config(args.conf, observables=args.observables, paths={'sdo_folder': args.out})
        if int(cfg['n_grid_rings']) not in n_rings:
            n_rings.append(int(cfg['n_grid_rings']))
    fmt = args.format if args.format is not None else ('none' if sweep and n_rings else 'npy')
    fmt = None if fmt == 'none' else fmt
    kw = dict(step=args.step, size=args.size, epoch=args.epoch, fmt=fmt, n_rings=n_rings or None, workers=args.workers,
              resume=not args.no_resume, download=args.download, email=args.email, verbose=not args.quiet)
    th = dict(noise_th=args.noise_th, B_th=args.B_th, I_th=args.I_th, A_th=args.A_th, mu_area=args.mu_area,
              faculae=args.faculae)
    if args.simulate_only:
        if not args.conf:
            raise SystemExit('--simulate-only needs --conf')
        path = os.path.join(args.out, 'sets.csv')
        folders = ([os.path.join(args.out, ln.split(',')[0]) for ln in open(path).read().splitlines()[1:] if ln]
                   if os.path.exists(path) else [args.out])
    elif sweep:
        res = make_sdo_sweep(args.data, args.out, args.start, args.end, **th, **kw)
        folders = [r['folder'] for r in res.values()]
        print('%d sets of thresholds, %d epochs each, in %s' % (len(res), len(next(iter(res.values()))['times']),
                                                                args.out))
    else:
        th = {k: v[0] for k, v in th.items()}
        res = make_sdo_maps(args.data, args.out, args.start, args.end, **th, **kw)
        folders = [args.out]
        print('%d epochs written in %s (%d days skipped)' % (len(res['times']), args.out, len(res['skipped'])))
    if args.conf:
        cache = None if args.no_grids_cache else (args.grids_cache or os.path.join(
            args.out, 'grids_%s.pkl' % _grids_key(args.conf, args.observables, False)))
        simulate_sdo_sets(args.conf, folders, observables=args.observables, grids_cache=cache,
                          chunk_size=args.chunk_size, resume=not args.no_resume, verbose=not args.quiet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
