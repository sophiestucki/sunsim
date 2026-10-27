"""
uninformed_masks: synthetic ("uninformed") spot and plage masks of the Sun or of a Sun-like star, from a statistical
model of active regions fitted on SDO masks (see uninformed_masks_model.ipynb).

The model
---------
* Active regions emerge as a Poisson process: a constant rate, or a rate that follows activity cycles (shape of
  Hathaway, Wilson & Reichmann 1994, consecutive cycles overlap). Each region belongs to one cycle; its latitude is
  drawn around the butterfly-diagram latitude of that cycle (exponential drift, Hathaway 2011), or, with probability
  p_nest, next to a recent spotted region (nesting, Isik et al. 2018).
* A region has a lognormal peak area, grows linearly for growth_days and decays linearly at D = D0 (A_peak/100)^p.
  Its spots are a circular leader and a few circular followers behind it; pores are one-day dots.
* Plage is a flux density in units of the plage threshold: a gaussian per region (k_fac x peak area) that spreads by
  surface diffusion (eta) and decays exponentially (tau_fac), plage-only regions, a background in the activity belts,
  and two lognormal fields (static texture and daily specks) that break it into fragments. A pixel is plage where
  density x mu^mu_power >= 1; patches below the plage/network area threshold of sunsim.sdo (A_th) are removed.
* Regions rotate differentially: omega(lat) = omega_eq (1 - shear sin^2 lat) [deg/day].

Areas are in micro-hemispheres (MSH, 1e-6 of the visible hemisphere), so the parameters do not depend on the radius of
the star (except eta, in km^2/s, through radius_km).

Use
---
    import uninformed_masks as um
    P = um.params(um.SUN_2015)                                  # a preset, then change what you want
    star = um.SUN                                               # or um.Star(inclination=60, omega_eq=30, ...)
    res = um.simulate(P, star, times=np.arange(0, 90), size=512, year0=2015.67, seed=1)
    um.show(res, 10); um.plot_filling_factors(res)
    um.write_maps(res, 'maps/')    # then SunSim: active_region_mask('maps/', origin='lower')

The maps follow the convention of sunsim.sdo: the image is cut to the disc, row 0 is the south (bottom) of the disc,
the rotation moves the regions from left to right.
"""
from dataclasses import dataclass, replace
import os

import numpy as np
import scipy.ndimage as nd

DEG = np.pi / 180
A_DET = 3.0              # MSH, regions below this size are pores (as in the extraction of the notebook)
A_TH = 20.0              # MSH, plage / network threshold of sunsim.sdo, projected at MU_AREA
MU_AREA = 0.1


# =============================================================================================
# parameters
# =============================================================================================
DEFAULTS = dict(
    # --- emergence in time ---
    rate=None,             # regions / day over the whole star; None: rate_max x cycle shape
    rate_max=3.0,          # regions / day at the maximum of a cycle (if rate is None)
    cycle_period=11.0,     # years
    cycle_start=2008.96,   # decimal year of the start (minimum) of one cycle
    cycle_width=4.0,       # years, b of Hathaway et al. (1994) for an 11-year cycle (scaled with cycle_period)
    # --- latitudes ---
    lat_mean=None,         # deg, fixed |latitude| of emergence; None: butterfly (lat_start, lat_tau)
    lat_start=28.0,        # deg, |latitude| at the start of a cycle (Hathaway 2011)
    lat_tau=7.5,           # years, e-folding time of the drift towards the equator (Hathaway 2011: 90 months)
    lat_sigma=5.0,         # deg, spread around the mean |latitude|
    # --- sizes and evolution ---
    logA_mean=np.log(30),  # ln MSH, lognormal peak area
    logA_sigma=1.0,
    A_min=3.0,             # MSH, smaller regions are not drawn
    growth_days=2.0,       # linear growth to the peak
    D0=10.0,               # MSH / day, decay rate at A_peak = 100 MSH: D = D0 (A_peak / 100)^decay_p
    decay_p=0.5,           # 0: Gnevyshev-Waldmeier (same D for all); 0.5: Hathaway & Choudhary (2008)
    # --- nesting ---
    p_nest=0.3,            # probability to emerge next to a spotted region of the last nest_days
    nest_sigma=3.0,        # deg, offset of a nested emergence
    nest_days=27.0,
    # --- spots of a region ---
    leader_frac=0.7,       # area fraction of the leader spot
    follower_scale=60.0,   # one follower per follower_scale MSH (Poisson, at most 30)
    sep_deg=3.0,           # deg, extent of the followers at 100 MSH, grows as sqrt(A)
    pore_rate=5.0,         # pores / day over the star (one day, < A_DET, no faculae)
    pore_near=0.15,        # fraction of the pores born next to a region
    pore_rate_min=None,    # pores / day at zero activity (rate None): pore_rate_min + (pore_rate - pore_rate_min) x
                           # activity, activity = cycle rate / rate_max; None: pore_rate at every activity
    # --- plage of the regions ---
    k_fac=20.0,            # facular flux per MSH of peak spot area [plage threshold x MSH]
    spotless_rate=0.0,     # plage-only regions / day (same area law, no spots)
    spotless_rate_min=None,  # plage-only regions / day at zero activity (as pore_rate_min)
    sigma0_deg=2.0,        # deg, initial width of the facular flux
    eta=250.0,             # km^2 / s, surface diffusion
    tau_fac=30.0,          # days, decay of the facular flux after the peak
    texture=1.0,           # amplitude of the static lognormal texture (0: smooth plage)
    # --- background plage ---
    bg_level=0.22,         # background flux density in the belts [plage threshold]
    bg_lat=15.0,           # deg, latitude of the background belts
    bg_lat_sigma=18.0,     # deg, width of the belts
    bg_floor=0.5,          # background outside the belts, fraction of bg_level
    bg_level_min=None,     # background flux density at zero activity (as pore_rate_min; simulate)
    speck_sigma=1.2,       # amplitude of the daily small-scale lognormal field (0: none)
    speck_scale_deg=0.10,  # deg, size of the specks
    speck_corr=0.0,        # correlation of the specks from one day to the next
    mu_power=0.3,          # plage detection towards the limb: density x mu^mu_power
)

# Fit on the 88 daily sunsim.sdo masks of 2015-09-01 to 2015-11-30 (uninformed_masks_model.ipynb, after the closed loop
# and the calibrations). With these values the synthetic Sun has the latitudes, sizes, decay, nesting, pores and plage
# of SDO, but about 2.5 times less spot area (the closed loop is noisy on 3 months): fit a longer stretch for better values.
SUN_2015 = dict(DEFAULTS, rate=2.12, lat_mean=13.9, lat_sigma=5.92, logA_mean=3.26, logA_sigma=1.595, A_min=3.04,
                growth_days=1.0, D0=26.2, decay_p=0.851, p_nest=0.80, leader_frac=0.589, follower_scale=15.4,
                pore_rate=12.1, pore_near=0.135, k_fac=20.0, spotless_rate=1.0, bg_level=0.22, speck_sigma=1.1,
                speck_scale_deg=0.10)


def params(base=DEFAULTS, **changes):
    """A copy of a parameter set with some values changed; unknown names raise an error."""
    bad = set(changes) - set(DEFAULTS)
    if bad:
        raise KeyError('unknown parameters: %s' % ', '.join(sorted(bad)))
    out = dict(DEFAULTS)
    out.update(base)
    out.update(changes)
    return out


def cycle_params(P, year):
    """The same parameters with a cycle-driven rate and butterfly latitudes, the rate scaled so that it equals the
    rate of P (constant) at the decimal year `year` (e.g. SUN_2015 at 2015.67)."""
    Q = params(P, rate=None, lat_mean=None)
    Q['rate_max'] = P['rate'] / max(cycle_rate(year, Q) / Q['rate_max'], 1e-3)
    return Q


@dataclass
class Star:
    """Geometry and rotation of the star.
    inclination : deg, angle between the rotation axis and the line of sight (90 = equator on); None = the Sun seen
                  from the Earth at the dates of the epochs (B0 and L0 from sunpy)
    omega_eq    : deg / day, sidereal rotation rate of the spots at the equator
    shear       : omega(lat) = omega_eq (1 - shear sin^2 lat)
    omega_frame : deg / day, rotation of the frame of the longitudes (None: omega_eq); Carrington for the Sun
    radius_km   : stellar radius [km] (for the diffusion of the plage)
    """
    inclination: float = None
    omega_eq: float = 14.38
    shear: float = 2.77 / 14.38
    omega_frame: float = None
    radius_km: float = 6.957e5

    @property
    def frame(self):
        return self.omega_eq if self.omega_frame is None else self.omega_frame

    @property
    def rotation_period(self):
        return 360.0 / self.omega_eq


# The Sun seen from the Earth (dates), spot rotation of Newton & Nunn (1951), Carrington frame
SUN = Star(inclination=None, omega_eq=14.38, shear=2.77 / 14.38, omega_frame=360 / 25.38, radius_km=6.957e5)


# =============================================================================================
# geometry
# =============================================================================================
def unit_vector(lat, lon):
    lat = np.asarray(lat) * DEG
    lon = np.asarray(lon) * DEG
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def disc_geometry(n, B0, L0):
    """Latitude, longitude [deg], mu, unit vectors and deprojected area [MSH] of every pixel of an n x n image of the
    disc (row 0 = south, columns to the west), for a sub-observer latitude B0 and central longitude L0."""
    c = (n - 1) / 2
    R = n / 2
    y, x = (np.mgrid[:n, :n] - c) / R
    z2 = 1 - x**2 - y**2
    inside = z2 > 0
    z = np.sqrt(np.where(inside, z2, np.nan))
    b0 = B0 * DEG
    lat = np.arcsin(np.clip(y * np.cos(b0) + z * np.sin(b0), -1, 1)) / DEG
    lon = (np.arctan2(x, z * np.cos(b0) - y * np.sin(b0)) / DEG + L0) % 360
    g = dict(n=n, R=R, lat=lat, lon=lon, mu=z, inside=inside, B0=B0, L0=L0)
    g['msh'] = 1e6 / (2 * np.pi * R**2 * z)
    g['vec'] = unit_vector(lat, lon).astype(np.float32)
    return g


def observer(t, star, date=None, L0_start=0.0):
    """Sub-observer latitude B0 and central longitude L0 [deg] at t [days]."""
    if star.inclination is None:
        from astropy.time import Time
        import sunpy.coordinates.sun as sun
        tt = Time(date)
        return sun.B0(tt).deg, sun.L0(tt).deg
    return 90.0 - star.inclination, (L0_start - star.frame * t) % 360


def omega(lat, star):
    return star.omega_eq * (1 - star.shear * np.sin(np.asarray(lat) * DEG)**2)


def drift(lat, dt, star):
    """Change of longitude [deg] in the frame of the star of a feature at lat after dt days."""
    return (omega(lat, star) - star.frame) * dt


# =============================================================================================
# activity cycles
# =============================================================================================
def _shape(x, b, c=0.71):
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore'):
        return np.where(x > 0, x**3 / (np.exp(x**2 / b**2) - c), 0.0)


def _cycle_terms(year, P):
    """Rate (relative to the maximum) and mean |latitude| of the current and the previous cycle at `year`."""
    per = P['cycle_period']
    b = P['cycle_width'] * per / 11.0
    tt = np.linspace(0.01, 2 * per, 2000)
    peak = np.max(_shape(tt, b))
    k = np.floor((year - P['cycle_start']) / per)
    out = []
    for kk in (k, k - 1):
        x = year - (P['cycle_start'] + kk * per)
        out.append((float(_shape(x, b)) / peak, P['lat_start'] * np.exp(-x / P['lat_tau'])))
    return out


def cycle_rate(year, P):
    """Emergence rate [regions / day] of the cycles at `year` (sum of the overlapping cycles)."""
    if P['rate'] is not None:
        return P['rate']
    return P['rate_max'] * sum(r for r, _ in _cycle_terms(year, P))


def activity(year, P):
    """Rate of the cycles at `year` relative to rate_max (None with a constant rate)."""
    if P['rate'] is not None:
        return None
    return sum(r for r, _ in _cycle_terms(year, P))


def scaled(P, key, year):
    """P[key] at `year`: P[key + '_min'] + (P[key] - P[key + '_min']) x activity, or P[key] if there is no minimum
    value or the rate is constant."""
    vmin = P.get(key + '_min')
    act = activity(year, P)
    if vmin is None or act is None:
        return P[key]
    return max(vmin + (P[key] - vmin) * act, 0.0)


# =============================================================================================
# regions
# =============================================================================================
def area_at(reg, t):
    """Spot area [MSH] of a region at t: linear growth, then linear decay at D = D0 (A_peak / 100)^p."""
    dt = t - reg['t0']
    if dt < 0:
        return 0.0
    if dt < reg['tg']:
        return reg['A'] * dt / reg['tg']
    return max(reg['A'] - reg['D'] * (dt - reg['tg']), 0.0)


def flux_at(reg, t):
    """Facular flux [plage threshold x MSH] of a region at t: grows with the spots, then decays exponentially."""
    dt = t - reg['t0']
    if dt < 0:
        return 0.0
    if dt < reg['tg']:
        return reg['k'] * reg['A'] * dt / reg['tg']
    return reg['k'] * reg['A'] * np.exp(-(dt - reg['tg']) / reg['tau'])


def generate_regions(P, star, t_start, t_end, year0=2015.0, spin_up=150.0, seed=1):
    """All regions (and pores) that emerge between t_start - spin_up and t_end [days]. year0 = decimal year of t = 0
    (used for the cycles). Returns a list of dicts: t0, lat, lon (frame of the star), A (peak area, MSH), cycle, ..."""
    rng = np.random.default_rng(seed)
    regs, recent = [], []                 # recent: spotted regions that can still be parents of a nest or of a pore
    keep = max(P['nest_days'], 1.0)
    for day in np.arange(np.floor(t_start - spin_up), t_end + 1):
        yr = year0 + day / 365.25
        recent = [q for q in recent if q['t0'] > day - keep or area_at(q, day) > 0]
        if P['rate'] is not None:
            terms = [(P['rate'], P['lat_mean'] if P['lat_mean'] is not None else P['lat_start'])]
        else:
            terms = [(P['rate_max'] * r, lat) for r, lat in _cycle_terms(yr, P)]
            if P['lat_mean'] is not None:
                terms = [(r, P['lat_mean']) for r, _ in terms]
        spotless = scaled(P, 'spotless_rate', yr)
        for cyc, (rate, lat_mean) in enumerate(terms):
            for kind, r in (('spots', rate), ('plage', spotless * (rate / max(sum(x for x, _ in terms), 1e-9)))):
                for _ in range(rng.poisson(r)):
                    A = np.exp(rng.normal(P['logA_mean'], P['logA_sigma']))
                    if A < P['A_min']:
                        continue
                    t0 = day + rng.uniform()
                    parents = [q for q in recent if t0 - P['nest_days'] < q['t0'] < t0]
                    if parents and rng.uniform() < P['p_nest']:
                        w = np.array([q['A'] for q in parents])
                        q = parents[rng.choice(len(parents), p=w / w.sum())]
                        lat = q['lat'] + rng.normal(0, P['nest_sigma'])
                        lon = q['lon'] + drift(q['lat'], t0 - q['t0'], star) + rng.normal(0, P['nest_sigma'])
                    else:
                        lat = rng.choice([-1, 1]) * abs(rng.normal(lat_mean, P['lat_sigma']))
                        lon = rng.uniform(0, 360)
                    n_f = min(rng.poisson(A / P['follower_scale']), 30) if kind == 'spots' else 0
                    off = np.column_stack([-rng.uniform(0.2, 1.0, n_f), rng.normal(0, 0.3, n_f)])
                    reg = dict(t0=t0, lat=float(np.clip(lat, -89, 89)), lon=lon % 360, A=A, tg=P['growth_days'],
                               D=P['D0'] * (A / 100)**P['decay_p'], k=P['k_fac'], tau=P['tau_fac'], n_f=n_f,
                               spots=(kind == 'spots'), off=off, cycle=cyc, pore=False)
                    regs.append(reg)
                    if reg['spots']:
                        recent.append(reg)
        lat_belt = terms[0][1] if P['lat_mean'] is None else P['lat_mean']
        for _ in range(rng.poisson(scaled(P, 'pore_rate', yr))):
            t0 = day + rng.uniform()
            alive = [q for q in recent if q['t0'] < t0 and area_at(q, t0) > 0]
            if alive and rng.uniform() < P['pore_near']:
                q = alive[rng.integers(len(alive))]
                lat = q['lat'] + rng.normal(0, 10 / 3)
                lon = q['lon'] + drift(q['lat'], t0 - q['t0'], star) + rng.normal(0, 10 / 3)
            else:
                lat = rng.choice([-1, 1]) * abs(rng.normal(lat_belt, P['lat_sigma']))
                lon = rng.uniform(0, 360)
            A = rng.uniform(0.7, A_DET)
            regs.append(dict(t0=t0, lat=float(np.clip(lat, -89, 89)), lon=lon % 360, A=A, tg=0.01, D=A, k=0.0,
                             tau=1.0, n_f=0, spots=True, off=np.zeros((0, 2)), cycle=0, pore=True))
    for r in regs:                          # last day with something visible (spots, or flux above 2% of the threshold)
        t_spot = r['t0'] + r['tg'] + (r['A'] / r['D'] if r['spots'] else 0.0)
        t_flux = r['t0'] + r['tg'] + r['tau'] * np.log(max(r['k'] * r['A'] * 1e-6 / (P['sigma0_deg'] * DEG)**2 / 0.02,
                                                           1.0)) if r['k'] > 0 else r['t0']
        r['t_end'] = max(t_spot, t_flux)
    return regs


# =============================================================================================
# rendering
# =============================================================================================
_FIELDS = {}


def _field(key, shape_deg_step, smooth_deg, seed):
    """Random field (zero mean, unit variance) on a latitude-longitude grid, cached."""
    if key not in _FIELDS:
        if len(_FIELDS) > 10:
            _FIELDS.pop(next(iter(_FIELDS)))
        step = shape_deg_step
        rng = np.random.default_rng(seed)
        f = nd.gaussian_filter(rng.normal(size=(int(180 / step), int(360 / step))).astype(np.float32),
                               smooth_deg / step, mode='wrap')
        _FIELDS[key] = ((f - f.mean()) / f.std()).astype(np.float32), step
    return _FIELDS[key]


def _speck_field(t, P, seed):
    k = int(np.floor(t + 0.5))
    sc = P['speck_scale_deg']
    if P['speck_corr'] <= 0:
        return _field(('speck', seed, k, sc), 0.1, sc, [seed, 7919, k + 100000])
    w = P['speck_corr']**np.arange(6)
    f = sum(wj * _field(('speck', seed, k - j, sc), 0.1, sc, [seed, 7919, k - j + 100000])[0] for j, wj in enumerate(w))
    return f / np.sqrt(np.sum(w**2)), 0.1


def _sample(field, lat, lon):
    F, step = field
    i = np.clip(((lat + 90) / step).astype(int), 0, F.shape[0] - 1)
    j = np.clip(((lon % 360) / step).astype(int), 0, F.shape[1] - 1)
    return F[i, j]


def render(regs, t, n, P, star, B0, L0, seed=1):
    """Spot mask, plage mask (booleans) and plage flux density [threshold units] of an n x n image of the disc at t."""
    g = disc_geometry(n, B0, L0)
    ins = g['inside']
    V = g['vec'][ins]
    spot = np.zeros(V.shape[0], bool)
    rho = np.zeros(V.shape[0], np.float32)
    obs = unit_vector(B0, L0)
    for r in regs:
        if r['t0'] > t or t > r['t_end']:
            continue
        lon = r['lon'] + drift(r['lat'], t - r['t0'], star)
        vc = unit_vector(r['lat'], lon).astype(np.float32)
        if vc @ obs < -0.3:
            continue
        F = flux_at(r, t)
        if F > 0:
            sig = np.sqrt((P['sigma0_deg'] * DEG)**2 + 2 * P['eta'] * (t - r['t0']) * 86400 / star.radius_km**2)
            peak = F * 1e-6 / sig**2
            if peak > 0.02:
                cosd = V @ vc
                near = cosd > np.cos(min(5 * sig, np.pi))
                d2 = np.arccos(np.clip(cosd[near], -1, 1))**2
                rho[near] += peak * np.exp(-d2 / (2 * sig**2))
        A = area_at(r, t) if r['spots'] else 0.0
        if A <= 0:
            continue
        sep = P['sep_deg'] * np.sqrt(A / 100)
        parts = [(0.0, 0.0, A * (P['leader_frac'] if r['n_f'] else 1.0))]
        for k in range(r['n_f']):
            parts.append((sep * r['off'][k, 0], sep * r['off'][k, 1], A * (1 - P['leader_frac']) / r['n_f']))
        for dlon, dlat, a in parts:
            vs = unit_vector(r['lat'] + dlat, lon + dlon / max(np.cos(r['lat'] * DEG), 0.2)).astype(np.float32)
            spot |= (V @ vs) > np.cos(np.arccos(1 - a * 1e-6))
    lat, lon = g['lat'][ins], g['lon'][ins]
    if P['bg_level'] > 0:
        belt = np.exp(-0.5 * ((np.abs(lat) - P['bg_lat']) / P['bg_lat_sigma'])**2)
        rho += (P['bg_level'] * (P['bg_floor'] + (1 - P['bg_floor']) * belt)).astype(np.float32)
    if P['texture'] > 0:
        s = P['texture']
        rho *= np.exp(s * _sample(_field(('texture', seed), 0.25, 0.5, seed), lat, lon) - s**2 / 2)
    if P['speck_sigma'] > 0:
        s = P['speck_sigma']
        rho *= np.exp(s * _sample(_speck_field(t, P, seed), lat, lon) - s**2 / 2)
    if P['mu_power'] != 0:
        rho *= g['mu'][ins].astype(np.float32)**P['mu_power']
    out_s = np.zeros((n, n), bool)
    out_f = np.zeros((n, n), bool)
    out_r = np.zeros((n, n), np.float32)
    out_s[ins] = spot
    out_f[ins] = (rho >= 1) & ~spot
    out_r[ins] = rho
    lab, nl = nd.label(out_f)                            # plage / network cut of sunsim.sdo (4-connectivity)
    if nl:
        big = np.concatenate(([False], np.bincount(lab.ravel())[1:] > A_TH * 1e-6 * 2 * np.pi * (n / 2)**2 * MU_AREA))
        out_f = big[lab]
    return out_s, out_f, out_r, g


def simulate(P, star=SUN, times=np.arange(0, 30), size=512, year0=2015.0, seed=1, dates=None, keep_density=False,
             verbose=False):
    """Synthetic masks at the epochs `times` [days from year0].
    For the Sun (star.inclination None) the dates of the epochs give B0 and L0: `dates` (ISO strings) or computed
    from year0 + times. Returns a dict: times, dates, spot (list of n x n bool), facula (list), density (if
    keep_density), ff_sp / ff_fc (filling factors of the disc, projected), ffs_sp / ffs_fc (of the surface, mu > 0.3),
    regions (all the regions), params, star."""
    from astropy.time import Time
    times = np.asarray(times, dtype=float)
    if dates is None:
        dates = [Time(year0 + t / 365.25, format='decimalyear').isot for t in times]
    regs = generate_regions(P, star, times.min(), times.max(), year0=year0, seed=seed)
    out = dict(times=times, dates=list(dates), spot=[], facula=[], density=[], ff_sp=[], ff_fc=[], ffs_sp=[],
               ffs_fc=[], regions=regs, params=dict(P), star=star, size=size)
    for i, (t, d) in enumerate(zip(times, dates)):
        B0, L0 = observer(t, star, d)
        Pt = dict(P, bg_level=scaled(P, 'bg_level', year0 + t / 365.25))
        s, f, rho, g = render(regs, t, size, Pt, star, B0, L0, seed=seed)
        out['spot'].append(s)
        out['facula'].append(f)
        if keep_density:
            out['density'].append(rho)
        npx = g['inside'].sum()
        out['ff_sp'].append(s.sum() / npx)
        out['ff_fc'].append(f.sum() / npx)
        w = np.where(g['inside'] & (g['mu'] > 0.3), g['msh'], 0.0)
        out['ffs_sp'].append((w * s).sum() / w.sum())
        out['ffs_fc'].append((w * f).sum() / w.sum())
        if verbose and i % 10 == 0:
            print('epoch %d / %d' % (i + 1, len(times)))
    for k in ('ff_sp', 'ff_fc', 'ffs_sp', 'ffs_fc'):
        out[k] = np.array(out[k])
    return out


def _tag(t):
    """Epoch in a file name: 12 for 12.0, 12.5 otherwise."""
    return '%d' % t if float(t).is_integer() else '%g' % t


def write_maps(res, folder, fmt='npy'):
    """Write the masks as sunsim.sdo does (spot_map_{t}.npy, faculae_map_{t}.npy, uint8, row 0 = south) with a
    summary.csv. Read them in SunSim with active_region_mask(folder, origin='lower')."""
    os.makedirs(folder, exist_ok=True)
    rows = ['t,date,ff_sp,ff_fc']
    for t, d, s, f, a, b in zip(res['times'], res['dates'], res['spot'], res['facula'], res['ff_sp'], res['ff_fc']):
        if fmt == 'npz':
            np.savez_compressed(os.path.join(folder, 'spot_map_%s.npz' % _tag(t)), packed=np.packbits(s), shape=s.shape)
            np.savez_compressed(os.path.join(folder, 'faculae_map_%s.npz' % _tag(t)), packed=np.packbits(f), shape=f.shape)
        else:
            np.save(os.path.join(folder, 'spot_map_%s.npy' % _tag(t)), s.astype(np.uint8))
            np.save(os.path.join(folder, 'faculae_map_%s.npy' % _tag(t)), f.astype(np.uint8))
        rows.append('%s,%s,%.6g,%.6g' % (_tag(t), d[:19], a, b))
    with open(os.path.join(folder, 'summary.csv'), 'w') as fh:
        fh.write('\n'.join(rows) + '\n')
    with open(os.path.join(folder, 'parameters.txt'), 'w') as fh:
        for k, v in res['params'].items():
            fh.write('%s = %s\n' % (k, v))
        fh.write('star = %s\n' % (res['star'],))
    return folder


# =============================================================================================
# plots
# =============================================================================================
def mask_image(spot, facula):
    """0 quiet, 1 plage, 2 spot, NaN outside the disc."""
    img = np.zeros(spot.shape)
    img[facula] = 1
    img[spot] = 2
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n]
    img[np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) > n / 2] = np.nan
    return img


def show(res, epochs=(0,), ax=None, title=None):
    """Masks of some epochs (quiet orange, plage light, spots dark)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#e3a857', '#fff4cf', '#2b1d12'])
    epochs = np.atleast_1d(epochs)
    if ax is None:
        fig, ax = plt.subplots(1, len(epochs), figsize=(4.5 * len(epochs), 4.8), squeeze=False)
        ax = ax[0]
    for a, k in zip(np.atleast_1d(ax), epochs):
        a.imshow(mask_image(res['spot'][k], res['facula'][k]), origin='lower', cmap=cmap, vmin=-0.5, vmax=2.5,
                 interpolation='nearest')
        a.set_title(title or 't = %.0f d (%s)' % (res['times'][k], res['dates'][k][:10]), fontsize=10)
        a.axis('off')
    return ax


def plot_filling_factors(res, ax=None, label=None, surface=False):
    """Spot and plage filling factors against time [%] (of the disc, or of the surface with mu > 0.3)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    sp, fc = (res['ffs_sp'], res['ffs_fc']) if surface else (res['ff_sp'], res['ff_fc'])
    ax[0].plot(res['times'], 100 * sp, label=label)
    ax[1].plot(res['times'], 100 * fc, label=label)
    ax[0].set_ylabel('spot filling factor [%]')
    ax[1].set_ylabel('plage filling factor [%]')
    for a in ax:
        a.set_xlabel('t [day]')
    return ax


def plot_butterfly(res, ax=None, year0=None):
    """Latitude of the regions against their emergence time (size ~ sqrt(peak area))."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    r = [q for q in res['regions'] if q['spots'] and not q['pore']]
    t = np.array([q['t0'] for q in r])
    x = t / 365.25 + year0 if year0 is not None else t
    ax.scatter(x, [q['lat'] for q in r], s=np.sqrt([q['A'] for q in r]), c='k', alpha=0.3, lw=0)
    ax.set_xlabel('year' if year0 is not None else 't [day]')
    ax.set_ylabel('latitude [deg]')
    return ax
