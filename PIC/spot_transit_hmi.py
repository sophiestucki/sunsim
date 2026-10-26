#!/usr/bin/env python
"""
Convective shift of one sunspot (umbra and penumbra together, one mask) along its transit of the disc, from SDO/HMI
45 s dopplergrams: the velocity of the spot relative to the quiet Sun around it, against mu. This is what the spot
CCF of SunSim needs (the Dumusque bisector x (a + b mu) of settings_45_dumusque_mu.sh). Two steps:

  python spot_transit_hmi.py run --start 2016-04-08 --end 2016-04-21 --out DIR [--step-hours 6] [--n-frames 16] [...]
      Every step-hours from start to end, one epoch: the 720 s continuum and the 45 s magnetogram closest to it are
      identified (sdo.identify_day through supergranulation.identify_epoch, the thresholds of the sweeps: the spot mask
      is I < I_th <I_quiet> with |B| > B_th, umbra and penumbra together). The largest spot of the disc (optionally in a
      latitude band) is the one followed. Then the n-frames 45 s dopplergrams closest to the epoch (16 x 45 s = 12 min:
      the p modes cancel in the mean) are corrected as in Haywood et al. (2016, section 3), with the rotation fitted:
        1. motion of SDO removed pixel by pixel (eq. 10; supergranulation.correct_dopplergram);
        2. rotation of eq. 12-13, omega(phi) = a1 + a2 sin^2 phi + a3 sin^4 phi, with a1, a2, a3 free: fitted on the
           quiet pixels (neither spot nor plage nor network, mu > mu_min) of all the frames of the epoch
           (supergranulation.frame_cell_sums and fit_rotation: one offset per ring and per frame fitted with them, for
           what is the same east and west, the convective blueshift and the zero point of HMI), then removed pixel by
           pixel (correct_dopplergram(alphas=...)); every dopplergram is then moved by whole pixels onto the disc
           centre of the continuum (the 45 s and 720 s images can be ~6 pixels apart), so that the masks fit it;
      and give per frame (all the means weighted by the continuum intensity, as the RV sees the pixels):
        v_spot  : mean velocity of the spot pixels (the dark umbra counts little, the penumbra most);
        v_ref   : mean velocity of the quiet pixels at the mu of the spot (|mu - mu_spot| < dmu, the whole disc, farther
                  than r_out spot radii from the spot: no moat flow);
        v_quiet : mean velocity of all the quiet pixels with mu > mu_min (the reference of Haywood et al. 2016, eq. 14);
        v_plane : the quiet Sun of the annulus of r_in to r_out spot radii around the spot, a plane a0 + a1 x + a2 y
                  fitted on it and taken at the spot centre (the reference of the first version, for comparison).
      dv = v_spot - v_ref [m/s, > 0: the spot is redshifted, its convective blueshift is suppressed] is then the shift
      of the spot relative to the quiet Sun at the same mu: the suppression of the convective blueshift and the net
      Evershed flow of the penumbra together. The FITS files are deleted after every epoch; an epoch already done
      (DIR/epochs/<T_REC>.npz) is skipped, so the run can be started again.
  python spot_transit_hmi.py collect --out DIR
      Table of all the epochs (DIR/spot_transit.csv), the fit dv = a + b mu + k_orb OBS_VR + k_rot v_rot (OBS_VR: the
      velocity of SDO away from the Sun, +-3 km/s over its 24 h orbit; v_rot: the rotation velocity of the spot, +-2
      km/s east to west: the HMI velocities of the spot have an error that follows the whole Doppler shift of its line,
      much larger than in the quiet Sun, Schuck et al. 2016), dv - k_orb OBS_VR - k_rot v_rot, a and b as a Dumusque
      scaling, the comparison with Palumbo et al. (2024) and the disc-integrated RV of the spot (dumusque_mu_scaling = a / <p>, b / <p>, <p> the mean of the
      Dumusque bisector over the CCF heights) and the figure DIR/spot_transit.png.

A transit of one large, isolated, round spot is the simple case to start with, e.g. NOAA 12529 (2016-04-08 to
2016-04-21, ~N10). Other spots on the disc are not a problem as long as the one followed stays the largest (or give
--lat-min / --lat-max).

Needs sunsim with the sdo extras (sunpy, drms) and internet access to jsoc.stanford.edu.
"""
import argparse
import glob
import os
import re
import shutil
import tempfile
import time

import numpy as np

EPOCH_INFO = ('jd', 'mu', 'x', 'y', 'radius_deg', 'area_uhem', 'contrast', 'n_spot', 'n_ref', 'n_spots')
R_SUN = 696000000.0                     # [m] (RSUN_REF)
# limb darkening of the HMI continuum (6173 A), I(mu) / I(1): polynomial (highest power first) fitted on
# hmi.Ic_720s / hmi.Ic_noLimbDark_720s of 2016.04.14_00:00:00_TAI (max error 0.004 for mu > 0.05)
HMI_LD = np.poly1d([1.15333, -3.62938, 4.52343, -2.96468, 1.6294, 0.29151])
# Palumbo, Saar & Haywood (2024, ApJ 973, 11; arXiv:2404.16747), Appendix B: HMI 2014-2018, every 4 h (the 12 h and
# 24 h systematics of the orbit average out), intensity-weighted, mu bins of 0.1. Table 1: quiet Sun velocity
# [m/s] (v_grav and a constant convective offset removed); Table 2: dv_conv = v_feature - v_quiet at the same mu
# [m/s]. Umbra: I < 0.45 <I_quiet>, penumbra 0.45 - 0.89 (umbra + penumbra = the spot mask of sdo with I_th = 0.89)
PALUMBO_MU = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
PALUMBO = {'quiet':     np.array([213.95, 95.06, 20.31, -21.66, -42.22, -49.15, -44.72, -25.22, 8.57]),
           'network':   np.array([-53.76, -22.74, 9.67, 37.01, 59.81, 82.34, 105.17, 124.36, 129.50]),
           'plage':     np.array([-82.47, -56.14, -16.70, 19.24, 49.58, 80.78, 111.50, 145.31, 160.66]),
           'umbra':     np.array([-161.60, -65.30, 44.70, 110.51, 159.09, 189.38, 215.56, 228.84, 225.24]),
           'penumbra':  np.array([-216.29, -128.93, -31.76, 47.32, 115.11, 174.19, 224.75, 255.76, 256.74])}


def _key(t_rec):
    """File name of a record: the digits of its T_REC (YYYYMMDDhhmmss)."""
    return re.sub(r'[^0-9]', '', str(t_rec))[:14]


def _log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _frame(a, context):
    """An image of the HMI size cut to the square of the disc and turned as the maps of identify_day (row 0 = south,
    east on the left)."""
    if a.shape[0] != context['size']:
        raise ValueError('image of %d pixels, the identification has %d: run with --size 4096' % (a.shape[0], context['size']))
    return np.flip(np.asarray(a)[context['lo']:context['hi'], context['lo']:context['hi']])


def _aligned(a, smap, context):
    """_frame of an image of smap (a dopplergram) moved by the whole pixels between its disc centre and the one of the
    continuum of the identification (context['crpix']): the 45 s and 720 s images can be several pixels apart."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    c = smap.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=smap.coordinate_frame))
    dx = int(round(context['crpix'][0] - c.x.to_value(u.pix)))
    dy = int(round(context['crpix'][1] - c.y.to_value(u.pix)))
    return _frame(np.roll(a, (dy, dx), axis=(0, 1)), context)


def disc_geometry(smap, context):
    """mu and the heliocentric coordinates (x west, y north, z towards SDO) [solar radii] of the point of the Sun seen
    in every pixel (supergranulation.los_geometry), in the frame of the maps. float32, NaN off the disc."""
    from sunsim import supergranulation as sg
    m = smap.meta
    rsun, dsun = float(m.get('rsun_ref', 696000000.0)), float(m['dsun_obs'])
    dx, dy, pc = sg.sun_centre_offsets(smap)
    tx = pc[0, 0] * dx[None, :] + pc[0, 1] * dy[:, None]
    ty = pc[1, 0] * dx[None, :] + pc[1, 1] * dy[:, None]
    _, p, mu = sg.los_geometry(tx, ty, dsun, rsun)
    del tx, ty
    return {'mu': _aligned(mu.astype(np.float32), smap, context),
            'xyz': np.stack([_aligned((pk / rsun).astype(np.float32), smap, context) for pk in p])}


def follow_spot(spot, geo, lat_range=None, sigma=2.0, threshold=0.2):
    """The spot followed: the largest group of spot pixels (fragments closer than ~sigma pixels count as one, as
    sdo.count_spots), with its centre in the latitude band lat_range [deg] if given (latitude from y, without B0).
    Returns the boolean mask of its pixels (None if there is no spot) and the number of groups on the disc."""
    import scipy.ndimage as nd
    smooth = nd.gaussian_filter(spot.astype(np.float32), sigma=sigma, mode='constant', cval=0.0)
    labels, n = nd.label(smooth >= threshold)
    if n == 0:
        return None, 0
    labels = np.where(spot, labels, 0)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    for k in np.argsort(areas)[::-1]:
        mask = labels == k + 1
        if lat_range is None:
            return mask, n
        lat = np.rad2deg(np.arcsin(np.clip(np.nanmean(geo['xyz'][1][mask]), -1, 1)))
        if lat_range[0] <= lat <= lat_range[1]:
            return mask, n
    return None, n


def spot_and_references(mask, geo, weight, quiet, r_in, r_out, mu_min, dmu):
    """Centre of the spot (unit vector, mean of the positions of its pixels weighted by their intensity), its radius
    [deg] (the radius of a disc of its area on the sphere), its area [micro-hemispheres] and the quiet pixels of the
    three references (mu > mu_min): 'ref' at the mu of the spot (|mu - mu_spot| < dmu, farther than r_out radii from
    it), 'quiet' all of them, 'plane' the annulus between r_in and r_out radii."""
    xyz, mu = geo['xyz'], geo['mu']
    w = weight[mask].astype(np.float64)
    c = np.array([np.sum(w * xyz[k][mask]) for k in range(3)])
    c /= np.linalg.norm(c)
    # area on the sphere: every pixel covers (pixel / R)^2 / mu of the surface; the pixel side from the disc in pixels
    pix = 1.0 / np.sqrt(np.sum(np.isfinite(mu)) / np.pi)            # pixel side [solar radii]
    area = np.sum(pix * pix / mu[mask])                               # [R^2]
    radius = np.rad2deg(np.arccos(1 - area / (2 * np.pi)))           # spherical cap of that area
    with np.errstate(invalid='ignore'):
        dist = np.rad2deg(np.arccos(np.clip(c[0] * xyz[0] + c[1] * xyz[1] + c[2] * xyz[2], -1, 1)))
        quiet = quiet & (mu > mu_min)
        refs = {'ref': quiet & (np.abs(mu - c[2]) < dmu) & (dist > r_out * radius),
                'quiet': quiet,
                'plane': quiet & (dist >= r_in * radius) & (dist <= r_out * radius)}
    return c, radius, 1e6 * area / (2 * np.pi), refs


def _wmean(v, sel, weight):
    """Mean of v over the pixels of sel weighted by weight (NaN pixels left out)."""
    ok = sel & np.isfinite(v)
    w = weight[ok].astype(np.float64)
    return float(np.sum(w * v[ok]) / np.sum(w))


def frame_velocities(v, mask, refs, geo, weight, centre):
    """v_spot, v_ref, v_quiet and v_plane (spot_and_references) of one corrected dopplergram v (frame of the maps)
    [m/s]."""
    out = {'v_spot': _wmean(v, mask, weight), 'v_ref': _wmean(v, refs['ref'], weight),
           'v_quiet': _wmean(v, refs['quiet'], weight)}
    ok = refs['plane'] & np.isfinite(v)
    w = weight[ok].astype(np.float64)
    X = np.column_stack([np.ones(w.size), geo['xyz'][0][ok] - centre[0], geo['xyz'][1][ok] - centre[1]])
    q = np.sqrt(w)
    out['v_plane'] = float(np.linalg.lstsq(X * q[:, None], v[ok] * q, rcond=None)[0][0])
    return out


# =============================================================================================
# run: every epoch of the transit
# =============================================================================================
def run(a):
    from astropy.time import Time
    from sunsim import supergranulation as sg

    folder = os.path.join(a.out, 'epochs')
    os.makedirs(folder, exist_ok=True)
    jd0, jd1 = Time(a.start).jd, Time(a.end).jd + 1
    targets = np.arange(jd0, jd1, a.step_hours / 24)
    rec = {k: sg.query_records(sg.SERIES[k][0], jd0 - 0.05, jd1 + 0.05) for k in sg.SERIES}
    r45 = sg.query_records(sg.SERIES_45S[0], jd0 - 0.05, jd1 + 0.05)
    _log('%d epochs to do, %d continua, %d magnetograms, %d dopplergrams'
         % (len(targets), len(rec['continuum']), len(rec['magnetogram']), len(r45)))
    thresholds = {k: v for k, v in (('noise_th', a.noise_th), ('B_th', a.B_th), ('I_th', a.I_th)) if v is not None}

    for jd in targets:
        k, dt = sg.closest(jd, rec['continuum']['jd'].values)
        if dt[0] > a.max_dt:
            _log('%s: no continuum within %.0f s, skipped' % (Time(jd, format='jd').iso, a.max_dt))
            continue
        tr = rec['continuum']['T_REC'].values[k[0]]
        out = os.path.join(folder, _key(tr) + '.npz')
        if os.path.exists(out):
            continue
        jd_c = rec['continuum']['jd'].values[k[0]]
        km, _ = sg.closest(jd_c, rec['magnetogram']['jd'].values)
        epoch = {'continuum': tr, 'magnetogram': rec['magnetogram']['T_REC'].values[km[0]]}
        near = np.argsort(np.abs(r45['jd'].values - jd_c))[:a.n_frames]
        tmp = tempfile.mkdtemp(prefix='spot_', dir=a.tmp)
        try:
            ident = sg.identify_epoch(sg.fetch_epoch(epoch, tmp), size=4096, thresholds=thresholds,
                                      network_velocity=False)
            ctx = ident['context']
            for f in glob.glob(os.path.join(tmp, '*.fits')):
                os.remove(f)
            # pass 1: the quiet pixels of every frame, the rotation fitted on all the frames of the epoch
            frames, sums = [], []
            for t_rec in r45['T_REC'].values[near]:
                files = sg.fetch_record(sg.SERIES_45S[0], t_rec, sg.SERIES_45S[1], tmp)
                frames.append(files)
                sums.append(sg.frame_cell_sums(files, ctx, n_rings=a.n_rings, mu_min=a.mu_min))
            if len(frames) < 2:
                raise RuntimeError('%d frames' % len(frames))
            alphas, alpha_err, rot_rms = sg.fit_rotation([s['swv'] for s in sums], [s['sw'] for s in sums],
                                                         [s['swg'] for s in sums], a.n_rings)
            # pass 2: every frame without the motion of SDO and this rotation, pixel by pixel
            geo, mask = None, None
            vel = {k: [] for k in ('v_spot', 'v_ref', 'v_quiet', 'v_plane')}
            jd_f = []
            for path, keys in frames:
                dop = sg.read_record(path, keys)
                if geo is None:
                    geo = disc_geometry(dop, ctx)
                    mask, n_groups = follow_spot(ctx['spot'], geo, a.lat_range)
                    if mask is None:
                        break
                    centre, radius, area, refs = spot_and_references(mask, geo, ctx['weight'], ctx['quiet'], a.r_in,
                                                                     a.r_out, a.mu_min, a.dmu)
                v = _aligned(sg.correct_dopplergram(dop, alphas=alphas)['v'], dop, ctx)
                for k, x in frame_velocities(v, mask, refs, geo, ctx['weight'], centre).items():
                    vel[k].append(x)
                jd_f.append(float(sg.jd_of_t_rec(keys['T_OBS'])[0]))
                del dop, v
            if mask is None:
                _log('%s: no spot followed' % tr)
                np.savez(out, jd=jd_c, n_spots=0)
                continue
            mu_c = float(centre[2])
            info = dict(jd=jd_c, mu=mu_c, x=float(centre[0]), y=float(centre[1]), radius_deg=radius, area_uhem=area,
                        contrast=float(np.mean(ctx['weight'][mask]) / np.mean(ctx['weight'][ctx['quiet']])),
                        n_spot=int(mask.sum()), n_ref=int(refs['ref'].sum()), n_spots=n_groups)
            np.savez(out, jd_frames=np.array(jd_f), alphas=alphas, alpha_err=alpha_err, rot_rms=rot_rms,
                     shift_px=np.array([s['shift_px'] for s in sums]),
                     obs_vr=float(np.mean([s['obs_vr'] for s in sums])),
                     **{k: np.array(x) for k, x in vel.items()}, **info)
            _log('%s: mu = %.3f, x = %+.3f, area %.0f uhem, %d frames, a = %.3f %.3f %.3f deg/d, dv = %+.1f m/s'
                 % (tr, mu_c, centre[0], area, len(jd_f), *alphas,
                    np.mean(np.array(vel['v_spot']) - np.array(vel['v_ref']))))
        except Exception as e:
            _log('%s: failed (%s: %s)' % (tr, type(e).__name__, e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


# =============================================================================================
# collect: table, fit, figure
# =============================================================================================
def collect(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sunsim.ccf_grid import dumusque_bisector

    rows, jd_frames = [], []
    for path in sorted(glob.glob(os.path.join(a.out, 'epochs', '*.npz'))):
        with np.load(path) as z:
            if 'v_spot' not in z.files or len(z['v_spot']) == 0:
                continue
            if 'alphas' not in z.files:
                continue                              # the first version (no rotation fitted)
            dv = z['v_spot'] - z['v_ref']
            err = dv.std(ddof=1) / np.sqrt(len(dv)) if len(dv) > 1 else np.nan
            rows.append([float(z[k]) for k in EPOCH_INFO] +
                        [dv.mean(), err, np.mean(z['v_spot'] - z['v_quiet']), np.mean(z['v_spot'] - z['v_plane'])] +
                        [z[k].mean() for k in ('v_spot', 'v_ref', 'v_quiet', 'v_plane')] + list(z['alphas']) +
                        [float(z['rot_rms']), len(dv), float(z['obs_vr']) if 'obs_vr' in z.files else np.nan])
            jd_frames.append(z['jd_frames'])
    if not rows:
        print('no epoch with a spot in %s/epochs' % a.out)
        return 1
    cols = list(EPOCH_INFO) + ['dv', 'dv_err', 'dv_quiet_disc', 'dv_plane', 'v_spot', 'v_ref', 'v_quiet', 'v_plane',
                               'a1', 'a2', 'a3', 'rot_rms', 'n_frames', 'obs_vr']
    t = np.array(rows)
    t = t[np.argsort(t[:, 0])]
    # one transit: the spot moves from east to west; once it has set at the west limb, the largest spot is another
    # one rising at the east limb (x jumps back): those epochs are left out
    jump = np.flatnonzero(np.diff(t[:, EPOCH_INFO.index('x')]) < -0.5)
    if len(jump):
        print('%d epochs after the spot set at the west limb left out (another spot)' % (len(t) - jump[0] - 1))
        t = t[:jump[0] + 1]
        jd_frames = [jd_frames[i] for i in np.argsort([j.mean() for j in jd_frames])][:jump[0] + 1]
    missing = ~np.isfinite(t[:, -1])
    if missing.any():                                 # epochs of a run that did not keep it: from the JSOC keywords
        t[missing, -1] = _obs_vr([j for j, m in zip(jd_frames, missing) if m])
    T = {c: t[:, i] for i, c in enumerate(cols)}

    # rotation velocity of the spot centre (fitted plasma rotation; > 0 west, receding; B0 neglected)
    s2 = np.clip(T['y'], -1, 1) ** 2                                      # sin^2 latitude
    omega = np.deg2rad(T['a1'] + T['a2'] * s2 + T['a3'] * s2 ** 2) / 86400
    T['v_rot'] = omega * R_SUN * T['x']

    # dv = a + b mu + k_orb OBS_VR + k_rot v_rot: what is symmetric east-west (mu) and the error of the HMI velocities
    # of the spot, which follows the whole Doppler shift of its line at HMI: the orbit of SDO (OBS_VR, +-3 km/s over
    # 24 h) and the rotation of the Sun (v_rot, +-2 km/s east to west). Much larger in the spot than in the quiet Sun
    # (Schuck et al. 2016); k_orb and k_rot come out alike (AR 12529: -117 and -154 m/s per km/s): the HMI velocity
    # scale of the spot pixels is ~13 % short, and the rotation fitted on the quiet Sun leaves k_rot v_rot in the spot.
    use = T['mu'] > a.mu_min
    print('%d epochs (%d with mu > %.2f)' % (len(t), use.sum(), a.mu_min))
    coef = None
    T['dv_corr'] = T['dv'].copy()
    if use.sum() > 4:
        vr, vrot = T['obs_vr'] / 1000, T['v_rot'] / 1000                # [km/s]
        A = np.column_stack([np.ones(use.sum()), T['mu'][use], vr[use], vrot[use]])
        coef, *_ = np.linalg.lstsq(A, T['dv'][use], rcond=None)
        rms = np.std(T['dv'][use] - A @ coef)
        T['dv_corr'] = T['dv'] - coef[2] * vr - coef[3] * vrot
        p_mean = np.polyint(dumusque_bisector(True)(1.0))
        p_mean = (p_mean(1) - p_mean(0)) * 1000                         # mean over the heights [m/s]
        print('dv = %.1f + %.1f mu + %.1f OBS_VR + %.1f v_rot  [m/s, velocities in km/s], rms %.1f m/s'
              % (coef[0], coef[1], coef[2], coef[3], rms))
        print('Dumusque bisector: mean %.1f m/s -> dumusque_mu_scaling = %.3f, %.3f' % (p_mean, coef[0] / p_mean,
                                                                                        coef[1] / p_mean))
    else:
        print('no fit: it needs more than 4 epochs')
    # the quiet Sun at the mu of the spot, the same 24 h term removed (k_q), for the comparison with Palumbo et al.
    T['v_ref_corr'] = T['v_ref'].copy()
    if use.sum() > 4:
        A = np.column_stack([T['mu'][use] ** k for k in range(4)] + [vr[use]])
        cq, *_ = np.linalg.lstsq(A, T['v_ref'][use], rcond=None)
        T['v_ref_corr'] = T['v_ref'] - cq[-1] * vr
        print('quiet Sun at the mu of the spot: %.1f m/s per km/s of OBS_VR' % cq[-1])

    # the spot in the disc-integrated RV (Haywood et al. 2016, eq. 11 and 14-16): fraction of the light of the disc
    # in the spot f (its projected area, its contrast, the limb darkening), convective term f dv and photometric term
    # (the light missing, (1 - contrast), at the rotation velocity of the spot)
    mu_g = np.linspace(0, 1, 2001)
    ld_disc = np.trapezoid(2 * HMI_LD(mu_g) * mu_g, mu_g)                 # mean intensity of the disc / I(1)
    f_geom = 2e-6 * T['area_uhem'] * T['mu']                              # projected area / disc
    T['f_spot'] = f_geom * T['contrast'] * HMI_LD(T['mu']) / ld_disc
    T['drv_conv'] = T['f_spot'] * T['dv_corr']
    T['drv_phot'] = -f_geom * (1 - T['contrast']) * HMI_LD(T['mu']) / ld_disc * T['v_rot']
    extra = ['dv_corr', 'v_ref_corr', 'f_spot', 'v_rot', 'drv_conv', 'drv_phot']
    t = np.column_stack([t] + [T[k] for k in extra])
    np.savetxt(os.path.join(a.out, 'spot_transit.csv'), t, delimiter=',', header=','.join(cols + extra),
               comments='', fmt='%.6f')
    print('disc-integrated RV of the spot: convective %+.2f to %+.2f m/s, photometric %+.2f to %+.2f m/s'
          % (T['drv_conv'].min(), T['drv_conv'].max(), T['drv_phot'].min(), T['drv_phot'].max()))
    palumbo_figure(a, T)

    t_d = T['jd'] - T['jd'].min()
    west = T['x'] > 0
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    ax[0, 0].errorbar(t_d, T['dv'], T['dv_err'], fmt='o', ms=3, color='0.6', label='dv')
    ax[0, 0].plot(t_d, T['dv_corr'], 'ko', ms=3, label='dv - k_orb OBS_VR - k_rot v_rot')
    ax[0, 0].set_xlabel('days from %s' % _iso(T['jd'].min()))
    ax[0, 0].set_ylabel('v_spot - v_quiet(mu) [m/s]')
    ax[0, 0].legend(fontsize=8)
    for sel, colour, label in ((~west, 'C0', 'east'), (west, 'C3', 'west')):
        ax[0, 1].errorbar(T['mu'][sel], T['dv_corr'][sel], T['dv_err'][sel], fmt='o', ms=3, color=colour, label=label)
    if coef is not None:
        mu = np.linspace(max(a.mu_min, T['mu'].min()), 1, 50)
        ax[0, 1].plot(mu, coef[0] + coef[1] * mu, 'k-', label='a + b mu')
    ax[0, 1].set_xlabel('mu')
    ax[0, 1].set_ylabel('dv - k_orb OBS_VR - k_rot v_rot [m/s]')
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(t_d, T['area_uhem'], 'o-', ms=3)
    ax[1, 0].set_xlabel('days')
    ax[1, 0].set_ylabel('area [uhem]')
    ax2 = ax[1, 0].twinx()
    ax2.plot(t_d, T['a1'], 'x', ms=3, color='C1')
    ax2.set_ylabel('a1 fitted [deg/day]', color='C1')
    if coef is not None:
        res = T['dv'] - coef[0] - coef[1] * T['mu']
        vtot = (T['obs_vr'] + T['v_rot']) / 1000
        for sel, colour, label in ((~west, 'C0', 'east'), (west, 'C3', 'west')):
            ax[1, 1].plot(vtot[sel], res[sel], 'o', ms=3, color=colour, label=label)
        v = np.linspace(vtot.min(), vtot.max(), 2)
        ax[1, 1].plot(v, coef[2] * v, 'k-', label='k_orb = %.0f m/s per km/s' % coef[2])
        ax[1, 1].plot(v, coef[3] * v, 'k--', label='k_rot = %.0f m/s per km/s' % coef[3])
        ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_xlabel('OBS_VR + v_rot: Doppler shift of the spot at HMI [km/s]')
    ax[1, 1].set_ylabel('dv - (a + b mu) [m/s]')
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, 'spot_transit.png'), dpi=130)
    print('written: %s, %s' % (os.path.join(a.out, 'spot_transit.csv'), os.path.join(a.out, 'spot_transit.png')))
    return 0


def palumbo_figure(a, T):
    """The spot against Palumbo et al. (2024) in their mu bins (table printed), the quiet Sun at the mu of the spot
    against their quiet Sun (one constant removed: the zero points differ), and the disc-integrated RV of the spot."""
    import matplotlib.pyplot as plt
    edges = np.append(PALUMBO_MU - 0.05, 1.0)
    k = np.digitize(T['mu'], edges) - 1
    binned = {q: np.array([np.mean(T[q][k == i]) if np.any(k == i) else np.nan for i in range(len(PALUMBO_MU))])
              for q in ('dv_corr', 'v_ref_corr')}
    n = np.bincount(k[(k >= 0) & (k < len(PALUMBO_MU))], minlength=len(PALUMBO_MU))
    off = np.nanmean(binned['v_ref_corr'] - PALUMBO['quiet'])
    print('\n  mu   n | spot (this transit) | Palumbo: umbra penumbra   plage network | quiet: this - %.0f  Palumbo'
          % off)
    for i, mu in enumerate(PALUMBO_MU):
        print('%5.2f %3d | %19.1f | %15.1f %8.1f %7.1f %7.1f | %17.1f %8.1f'
              % (mu, n[i], binned['dv_corr'][i], PALUMBO['umbra'][i], PALUMBO['penumbra'][i], PALUMBO['plage'][i],
                 PALUMBO['network'][i], binned['v_ref_corr'][i] - off, PALUMBO['quiet'][i]))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    styles = {'umbra': ('C3', '--'), 'penumbra': ('C1', '--'), 'plage': ('C2', '-'), 'network': ('C9', '-')}
    for q, (c, ls) in styles.items():
        ax[0].plot(PALUMBO_MU, PALUMBO[q], ls, color=c, marker='.', label='Palumbo+24 %s' % q)
    west = T['x'] > 0
    ax[0].plot(T['mu'][~west], T['dv_corr'][~west], 'o', ms=3, color='0.55', mfc='none', label='spot, east')
    ax[0].plot(T['mu'][west], T['dv_corr'][west], 's', ms=3, color='0.55', mfc='none', label='spot, west')
    ax[0].plot(PALUMBO_MU, binned['dv_corr'], 'ko-', lw=2, label='spot (umbra + penumbra), binned')
    ax[0].axhline(0, color='0.8', lw=0.8)
    ax[0].set_xlabel('mu')
    ax[0].set_ylabel('dv_conv = v - v_quiet(mu) [m/s]')
    ax[0].legend(fontsize=7)
    ax[1].plot(PALUMBO_MU, PALUMBO['quiet'], 'C0.-', label='Palumbo+24 quiet Sun')
    ax[1].plot(T['mu'], T['v_ref_corr'] - off, 'k.', ms=4, label='quiet Sun at the spot mu - %.0f m/s' % off)
    ax[1].set_xlabel('mu')
    ax[1].set_ylabel('v_quiet [m/s]')
    ax[1].legend(fontsize=8)
    t_d = T['jd'] - T['jd'].min()
    ax[2].plot(t_d, T['drv_conv'], 'C3o-', ms=3, label='convective: f dv')
    ax[2].plot(t_d, T['drv_phot'], 'C0o-', ms=3, label='photometric')
    ax[2].plot(t_d, T['drv_conv'] + T['drv_phot'], 'k-', label='total')
    ax[2].axhline(0, color='0.8', lw=0.8)
    ax[2].set_xlabel('days from %s' % _iso(T['jd'].min()))
    ax[2].set_ylabel('disc-integrated RV of the spot [m/s]')
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, 'spot_transit_palumbo.png'), dpi=130)


def _obs_vr(jd_frames):
    """Mean OBS_VR [m/s] (velocity of SDO away from the Sun) of the frames of every epoch (list of arrays of their T_OBS,
    JD UTC), from the keywords of hmi.V_45s (one JSOC query)."""
    from sunsim import supergranulation as sg
    lo = min(j.min() for j in jd_frames) - 1e-3
    hi = max(j.max() for j in jd_frames) + 1e-3
    k = sg._client().query('%s[%s-%s]' % (sg.SERIES_45S[0], sg.t_rec_of_jd(lo), sg.t_rec_of_jd(hi)),
                           key='T_OBS,OBS_VR')
    jd = sg.jd_of_t_rec(k['T_OBS'].values)
    vr = k['OBS_VR'].values.astype(np.float64)
    out = []
    for j in jd_frames:
        idx, _ = sg.closest(j, jd)
        out.append(vr[idx].mean())
    return np.array(out)


def _iso(jd):
    from astropy.time import Time
    return Time(jd, format='jd').iso[:16]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='step', required=True)
    r = sub.add_parser('run', help='every epoch of the transit')
    r.add_argument('--start', required=True, help='first day (YYYY-MM-DD)')
    r.add_argument('--end', required=True, help='last day (YYYY-MM-DD, included)')
    r.add_argument('--out', required=True)
    r.add_argument('--step-hours', type=float, default=6.0, help='time between epochs [h] (6)')
    r.add_argument('--n-frames', type=int, default=16, help='45 s dopplergrams averaged per epoch (16 = 12 min)')
    r.add_argument('--max-dt', type=float, default=3600.0, help='largest |continuum - epoch| [s] (3600)')
    r.add_argument('--noise-th', type=float, default=None, help='[G] (sdo default)')
    r.add_argument('--B-th', type=float, default=None, help='[G] (sdo default)')
    r.add_argument('--I-th', type=float, default=None, help='spot: I < I_th <I_quiet> (sdo default 0.89)')
    r.add_argument('--lat-min', type=float, default=None, help='latitude band of the spot followed [deg]')
    r.add_argument('--lat-max', type=float, default=None)
    r.add_argument('--r-in', type=float, default=3.0, help='reference annulus, inner radius [spot radii] (3)')
    r.add_argument('--r-out', type=float, default=5.0, help='reference annulus, outer radius [spot radii] (5)')
    r.add_argument('--mu-min', type=float, default=0.1, help='quiet pixels with mu > mu_min (0.1, Haywood et al.)')
    r.add_argument('--dmu', type=float, default=0.02, help='reference at the mu of the spot: |mu - mu_spot| < dmu')
    r.add_argument('--n-rings', type=int, default=30, help='rings of the cells of the rotation fit (30)')
    r.add_argument('--tmp', default=None, help='folder of the FITS files while an epoch is done')
    c = sub.add_parser('collect', help='table, fit and figure')
    c.add_argument('--out', required=True)
    c.add_argument('--mu-min', type=float, default=0.2, help='epochs of the fit: spot centre at mu > mu_min (0.2)')
    a = p.parse_args(argv)
    if a.step == 'run':
        a.lat_range = None if a.lat_min is None and a.lat_max is None else \
            (-90.0 if a.lat_min is None else a.lat_min, 90.0 if a.lat_max is None else a.lat_max)
        return run(a)
    return collect(a)


if __name__ == '__main__':
    raise SystemExit(main())
