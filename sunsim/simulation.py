"""
main: the StarSim class, which simulates the time series of a star with active regions (spots, faculae) and
transiting planets.

How a simulation works
----------------------
1. The visible disc is cut in N_rings concentric rings and every ring in cells (nbspectra.generate_grid_coordinates_nb).
2. For each kind of surface (quiet photosphere, spot, facula) a grid gives the signal of every cell if it were
   entirely of that kind: flux_grid (photometry: one value per ring; spectroscopy: one spectrum per ring without
   rotation, per cell with rotation) or CCF_grid
   (one CCF per ring, placed here on the velocity of every cell).
3. At every epoch the position of the regions and of the planets gives the filling factors of every cell
   (fraction quiet / spot / facula / planet), and the signal of the star is
        signal(t) = sum over the cells of ff_quiet*grid_quiet + ff_spot*grid_spot + ff_facula*grid_facula
   (the planets are dark). The geometry of all the epochs is computed first with NumPy, then all the epochs are done in
   one parallel Numba call (nbspectra.timeseries_nb).
4. In ccf mode, the RV, FWHM, contrast and BIS of every CCF are measured (compute_ccf_params).

Example
-------
    ss = StarSim(time, N_rings=20, inclination=0, rotation_period=25.4, mode=['photometry'],
                 active_regions_map=[active_region('sp', 10, 90, 180)],
                 flux_grid_qp=g_quiet.grid, flux_grid_sp=g_spot.grid, flux_grid_fc=g_facula.grid)
    ss.generate_timeseries()
    ss.flux_var, ss.ff_sp
"""
#only the libraries needed to simulate: matplotlib (plot_test) is imported where it is used
import os

import numpy as np
import warnings
import time as _time

from . import nbspectra
from . import _checks


from .rotation import differential_rotation_law

#Solar differential rotation: coefficients (B, C) [deg/day] of Omega(lat) = Omega_eq + B sin^2(lat) + C sin^4(lat).
#The profile used in StarSim (with the sin^2 term only, the solar value is B = -2.66 deg/day, Poljancic Beljan et al. 2017);
#the same as rotation.solar_differential_rotation(-2.66).
SOLAR_DIFFERENTIAL_ROTATION = (-1.698, -2.346)


class StarSim(object): 
    """
    Simulation of a star with active regions and planets: light curve, spectra and/or CCFs at the epochs `time`.

    Star
      time               : epochs [days] (array)
      N_rings            : number of concentric rings of the grid of the disc (the grids given must have the same)
      inclination        : angle between the rotation axis and the plane of the sky [rad]: 0 = equator-on, pi/2 = pole-on
      rotation_period    : rotation period at the equator [days]
      differential_rotation : law of the differential rotation (module rotation): 0 / None = rigid rotation; coefficients
                           (c1, c2, ...) [deg/day] of Omega(lat) = Omega_eq + c1 sin^2(lat) + c2 sin^4(lat) + ... (one
                           number = c1 only; SOLAR_DIFFERENTIAL_ROTATION or rotation.solar_differential_rotation(B) for
                           the Sun); or any function law(latitude [deg]) -> Omega(lat) - Omega_eq [deg/day] (vectorised).
                           It moves the regions and changes the rotation velocity of the cells (ccf mode).
      stellar_radius     : [solar radii], for the rotation velocity
    Surface
      active_regions_map : list of active_region.active_region (spots 'sp' and faculae 'fc'); None = no region
      active_regions_mask: active_region_mask.active_region_mask object: the spots and faculae come from 2D maps of the disc, one pair of
                           maps per epoch (map mode; every epoch of `time` must have its maps, e.g. time = maps.times).
                           Not together with active_regions_map.
      planet             : planet.planet object, a list of them, or None (no planet)
      planet_coverage    : 'area' (default): the fraction of every cell covered by a planet is the exact projected area
                           of the planet disc inside the cell, right for any size of planet. 'ramp': the formula of
                           the old StarSim (a linear ramp in the distance between the centres of the cell and of the
                           planet), only to reproduce old results: planets smaller than a cell are much too deep with it
                           (depth x8 for 0.01 Rstar with 15 rings, x2 for 0.03 Rstar, +9 % for 0.1 Rstar).
    Grids (the three kinds of surface: qp = quiet photosphere, sp = spot, fc = facula)
      flux_grid_qp, flux_grid_sp, flux_grid_fc : flux_grid(..., mode='photometry').grid (one value per ring), for
                           'photometry'
      spec_grid_qp, spec_grid_sp, spec_grid_fc : flux_grid(..., mode='spectroscopy').grid, for 'spectroscopy': one
                           spectrum per ring (N_rings, n_wv) without rotation, or per cell (n_cells, n_wv) with rotation.
                           With 'spectroscopy' alone the spectroscopy grids can also be given as flux_grid_xx (as in
                           the previous versions).
      ccf_grid_qp, ccf_grid_sp, ccf_grid_fc    : CCF_grid objects (after built_grid()), for 'ccf'
    mode                 : list with any of 'photometry', 'spectroscopy', 'ccf': all the modes listed are computed in the
                           same simulation (the geometry and the filling factors of the cells are computed once).
    total_flux_qp        : deprecated, ignored (a warning is given if it is set): the flux of the quiet star is the sum of
                           the quiet grid (flux_grid(...).total_brightness). Kept in the signature so that the arguments
                           after it can still be given by position; it will be removed.
    ccf_cells_max_memory : ccf mode [MB]. The CCF of every cell (its ring CCF shifted by its rotation velocity) is
                           computed once for the 3 surfaces and kept (3 arrays n_cells x n_rv) when they take at most
                           this memory (default 512 MB: up to ~240 rings with 145 velocities; 500 rings = 2.2 GB).
                           Above it, the CCF of a cell is computed only when a region or a planet touches it, at every
                           epoch: the same values (bit for bit), without these arrays, as fast for ~100 epochs and
                           ~1.5x slower for thousands of epochs. None = always keep them, 0 = never. Map mode
                           (active_regions_mask) always keeps them.
    spec_cells_max_memory : spectroscopy with stellar_rotation [MB], the same for the spectra: when the three
                           spec_grid_xx are flux_grid objects (not their .grid) and the spectra of all the cells
                           (3 x n_cells x n_wavelengths x 8 bytes) would take more than this memory, they are not built:
                           the spectrum of a cell (spectrum of its ring Doppler shifted by the velocity of the cell) is
                           computed only when a region or a planet touches it. Same values, bit for bit. None = always
                           build them, 0 = never. Map mode always builds them.
    ccf_on_the_fly, spec_on_the_fly : 'auto' (default: decided by ccf_cells_max_memory / spec_cells_max_memory), False
                           (never: the CCFs / spectra of all the cells are built and kept, whatever their memory) or True
                           (always, except in map mode). The CCF time series of the disc (ccf_var) is computed in all the
                           cases; this only changes how the CCF of a cell is obtained (same values, bit for bit).
    quiet_velocity       : ccf mode with 2D maps (active_regions_mask) only: a line-of-sight velocity [m/s] of the quiet
                           photosphere of every cell at every epoch, added to its rotation velocity (e.g. the
                           supergranulation measured on SDO/HMI dopplergrams, sunsim.supergranulation): an array
                           (n_times, n_cells) in the order of `time`, or an object with a method
                           velocities(times, N_rings, n_cells) -> (n_times, n_cells) (supergranulation.quiet_velocity_field,
                           which also works with simulate_in_chunks). The quiet part of every cell, ff_quiet x its CCF, is
                           shifted by this velocity; the spots and the faculae are not (nbspectra.quiet_velocity_ccf_nb).

    After generate_timeseries():
      flux_var (photometry), spec_var (spectroscopy), ccf_var and ccf_rv (ccf) and, in ccf mode, rv_var, fwhm_var,
      contrast_var, bis_var, xbis_var, ybis_var and rv_kepler;
      ff_quiet, ff_sp, ff_fc, ff_pl: fraction of the disc covered by each kind of surface [% of the projected area];
      results() returns all these time series in a dict, save_results(path) saves them (with the CCF time series);
      regions_pos, regions_vec, planet_pos, visible: geometry of the regions and planets at every epoch.
    Other attributes: vsini (equatorial, m/s), rvel (line-of-sight rotation velocity of every cell, m/s),
    rotation_rate (Omega(lat)/Omega_eq of every cell), amu, parea, Ngrid_in_ring, vec_grid (grid of the disc).
    """
    def __init__(self, time, N_rings, inclination, rotation_period, differential_rotation=0, stellar_radius=1, active_regions_map=None, active_regions_mask=None, total_flux_qp=None, flux_grid_qp=None, flux_grid_sp=None, flux_grid_fc=None, spec_grid_qp=None, spec_grid_sp=None, spec_grid_fc=None, ccf_grid_qp=None, ccf_grid_sp=None, ccf_grid_fc=None, planet=None, mode=['photometry'], ccf_cells_max_memory=512, planet_coverage='area', spec_cells_max_memory=512, ccf_on_the_fly='auto', spec_on_the_fly='auto', quiet_velocity=None):

        #checks of the arguments (clear errors here rather than in the Numba kernels)
        time = _checks.array1d('time', time)
        N_rings = _checks.number('N_rings', N_rings, integer=True, low=1)
        inclination = _checks.inclination('inclination', inclination)
        rotation_period = _checks.number('rotation_period', rotation_period, positive=True)
        stellar_radius = _checks.number('stellar_radius', stellar_radius, positive=True)
        if ccf_cells_max_memory is not None:
            ccf_cells_max_memory = _checks.number('ccf_cells_max_memory', ccf_cells_max_memory, nonnegative=True,
                                                  allow_inf=True)
        if active_regions_map is not None and not isinstance(active_regions_map, (list, tuple)):
            active_regions_map = [active_regions_map]         # a single region
        for r in active_regions_map or []:
            if not all(hasattr(r, a) for a in ('type', 'radius', 'colatitude', 'longitude', 'reference_time')):
                raise TypeError("active_regions_map must be a list of active_region.active_region objects, got an "
                                "element %r" % (r,))

        self.obs_times = time
        if total_flux_qp is not None:
            warnings.warn('StarSim(total_flux_qp=...) is not used and will be removed: the flux of the quiet star is the '
                          'sum of the quiet grid (flux_grid(...).total_brightness)', FutureWarning, stacklevel=2)
        self.flux_grid_qp = flux_grid_qp
        self.flux_grid_sp = flux_grid_sp
        self.flux_grid_fc = flux_grid_fc
        self.spec_grid_qp = spec_grid_qp
        self.spec_grid_sp = spec_grid_sp
        self.spec_grid_fc = spec_grid_fc

        self.ccf_grid_qp = ccf_grid_qp
        self.ccf_grid_sp = ccf_grid_sp
        self.ccf_grid_fc = ccf_grid_fc
        self.inclination = inclination
        self.rotation_period = rotation_period
        #differential rotation: law(latitude [deg]) -> Omega(lat) - Omega_eq [deg/day] (module rotation), from
        #coefficients (c1, c2, ...) of c1 sin^2(lat) + c2 sin^4(lat) + ..., 0 / None (rigid) or any function
        self.differential_rotation = 0.0 if differential_rotation is None else differential_rotation
        self._differential_law = differential_rotation_law(differential_rotation)
        self.active_regions_map = [] if active_regions_map is None else list(active_regions_map)
        #2D maps of the spots and faculae (active_region_mask.active_region_mask): if given, the regions come from the maps (map mode)
        if active_regions_mask is not None:
            if not hasattr(active_regions_mask, 'filling_factors'):
                raise TypeError("active_regions_mask must be a active_region_mask.active_region_mask object, got %r" % (active_regions_mask,))
            if self.active_regions_map:
                raise ValueError("give either active_regions_map (circular regions) or active_regions_mask (2D maps), not both")
        self.active_regions_mask = active_regions_mask

        #transiting planets: a planet.planet object, a list of them, or None (no planet)
        planets = [] if planet is None else (list(planet) if isinstance(planet, (list, tuple)) else [planet])
        for pl in planets:
            if not hasattr(pl, 'positions'):
                raise TypeError("planet must be a planet.planet object, a list of them, or None; got %r" % (pl,))
        self.planet = planet
        self.planets = planets
        self.simulate_planet = len(planets) > 0

        if isinstance(mode, str):                          # a single mode given as a string still works
            mode = [mode]
        mode = list(mode)
        valid = ('photometry', 'spectroscopy', 'ccf')
        if not mode or any(md not in valid for md in mode):
            raise ValueError("mode must be a list with some of %s, got %r" % (valid, mode))
        self.mode = mode
        if planet_coverage not in ('area', 'ramp'):
            raise ValueError("planet_coverage must be 'area' (exact) or 'ramp' (old StarSim), got %r" % (planet_coverage,))
        self.planet_coverage = planet_coverage
        self.ccf_cells_max_memory = ccf_cells_max_memory
        for name, value in (('ccf_on_the_fly', ccf_on_the_fly), ('spec_on_the_fly', spec_on_the_fly)):
            if not (value == 'auto' or isinstance(value, (bool, np.bool_))):
                raise ValueError("%s must be 'auto', True or False, got %r" % (name, value))
        self.ccf_on_the_fly = ccf_on_the_fly
        self.spec_on_the_fly = spec_on_the_fly
        if spec_cells_max_memory is not None:
            spec_cells_max_memory = _checks.number('spec_cells_max_memory', spec_cells_max_memory, nonnegative=True)
        self.spec_cells_max_memory = spec_cells_max_memory
        if quiet_velocity is not None:
            if active_regions_mask is None:
                raise ValueError('quiet_velocity needs the 2D maps (active_regions_mask): the filling factors of the cells')
            if 'ccf' not in mode:
                raise ValueError("quiet_velocity only changes the CCFs: it needs mode 'ccf', got %r" % (mode,))
            if planets:
                raise NotImplementedError('quiet_velocity with planets is not implemented')
            if not hasattr(quiet_velocity, 'velocities'):
                quiet_velocity = np.asarray(quiet_velocity, dtype=np.float64)
                if quiet_velocity.ndim != 2 or quiet_velocity.shape[0] != len(time):
                    raise ValueError('quiet_velocity must be an array (n_times, n_cells) with one row per epoch of time '
                                     '(%d), got shape %s' % (len(time), quiet_velocity.shape))
        self.quiet_velocity = quiet_velocity
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea = nbspectra.generate_grid_coordinates_nb(N_rings)

        self.N_rings = N_rings
        self.Ngrid_in_ring = Ngrid_in_ring
        self.amu = amu
        self.parea = parea
        self.rs = rs

        self.vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian

        #array copies, for the vectorised parts (the lists above are kept for generate_ff)
        self._Nin_arr = np.asarray(Ngrid_in_ring, dtype=np.int64)
        self._parea_arr = np.asarray(parea, dtype=np.float64)

        #rotation velocity of every cell [m/s], for the Doppler shift of the CCF (same formulas as flux_grid / old StarSim)
        self.stellar_radius = stellar_radius
        #theta: colatitude of every cell with respect to the rotation axis (0 at the visible pole), phi: its longitude
        #measured from the line of sight. The line-of-sight velocity of a rigid rotation is vsini * sin(theta) * sin(phi).
        theta = np.arccos(zs*np.cos(-inclination)-xs*np.sin(-inclination))
        phi = np.arctan2(ys, xs*np.cos(-inclination)+zs*np.sin(-inclination))
        vsini = 1000*2*np.pi*(stellar_radius*696342)*np.cos(inclination)/(rotation_period*86400)   #at the equator
        self.vsini = vsini
        #with differential rotation every cell rotates at Omega(lat)/Omega_eq times the equatorial rate (sin(lat) = cos(theta))
        self.rotation_rate = nbspectra.rotation_rate_relative(np.cos(theta), rotation_period, self.differential_rotation)
        self.rvel = np.ascontiguousarray(vsini*self.rotation_rate*np.sin(theta)*np.sin(phi), dtype=np.float64)

    def _regions_arrays(self):
        """Parameters of the active regions as arrays (degrees), and their type: 1 = spot, 2 = facula."""
        regs = self.active_regions_map
        colat = np.array([r.colatitude for r in regs], dtype=np.float64)
        longi = np.array([r.longitude for r in regs], dtype=np.float64)
        size = np.array([r.size for r in regs], dtype=np.float64)
        ref = np.array([r.reference_time for r in regs], dtype=np.float64)
        typ = np.array([1 if r.type == 'sp' else 2 for r in regs], dtype=np.int64)
        return colat, longi, size, ref, typ

    def compute_regions_geometry(self, times):
        """Geometry of all the regions at all the epochs, in one NumPy pass.
        Returns spot_pos (n_times, n_regions, 3): colatitude, longitude and radius [rad];
                vec_spot (n_times, n_regions, 3): centre of the regions in cartesian coordinates (x towards the observer);
                visible  (n_times, n_regions) bool: some part of the region is on the visible hemisphere.
        The radius is given by the evolution law of every region (active_region.radius); a region with radius 0
        is not there (not visible): outside its lifetime, or when its evolution law gives a radius <= 0."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))[:, None]
        colat, longi, size, ref, _ = self._regions_arrays()
        dt = t - ref
        #rotation rate relative to the equator [deg/day] at the latitude of every region (law of the differential rotation)
        domega = np.asarray(self._differential_law(90.0 - colat), dtype=np.float64)
        #longitude with rotation and differential rotation, between 0 and 360
        pht = longi + dt/self.rotation_period%1*360 + dt*domega
        theta = np.broadcast_to(np.deg2rad(colat), dt.shape)
        phi = np.deg2rad(pht%360)
        #radius of every region at every epoch (evolution law, or constant size); <= 0 -> 0, the region is not there
        if self.active_regions_map:
            rad = np.deg2rad(np.stack([r.radius(t[:, 0]) for r in self.active_regions_map], axis=1))
        else:
            rad = np.zeros(dt.shape)
        spot_pos = np.stack([theta, phi, rad], axis=-1)

        #centre of every region in the frame of the grid (x towards the observer, z along the projected rotation axis):
        #position (theta, phi) on the rotating star, rotated by the inclination of the axis
        ci, si = np.cos(self.inclination), np.sin(self.inclination)
        st, ct = np.sin(theta), np.cos(theta)
        vec_spot = np.stack([ci*st*np.cos(phi)+si*ct, st*np.sin(phi), ct*ci-si*st*np.cos(phi)], axis=-1)

        #visible if some part of the region is on the visible hemisphere: angular distance of its centre from the disc
        #centre (arccos(x)) minus its radius below 90 degrees
        visible = (np.arccos(np.clip(vec_spot[..., 0], -1.0, 1.0)) - rad <= np.pi/2) & (rad > 0)
        return spot_pos, vec_spot, visible

    def compute_regions_position(self, t):
        """Colatitude, longitude and radius [rad] of every region at time t, array (n_regions, 3)."""
        return self.compute_regions_geometry([t])[0][0]

    def compute_ccf_params(self, rv=None, ccf=None, plot_test=False):
        '''
        Compute the parameters of the CCFs and their bisector span (10-40% bottom minus 60-90% top).
        By default the CCFs are self.ccf_var on the velocity grid self.ccf_rv.
        Returns rvs, contrast, fwhm, BIS, raw_xbis, raw_ybis: rv, contrast and fwhm come from a gaussian fit, BIS is the
        bisector span, raw_xbis / raw_ybis are the bisectors (velocity and height) of every CCF.
        The CCFs are not modified. All the CCFs are done in one parallel Numba call (nbspectra.ccf_params_nb: bisector
        and Levenberg-Marquardt gaussian fit with analytic Jacobian); identical CCFs (e.g. the epochs without any
        visible region) are computed only once. The gaussian is fitted on the part of the CCF between the wings of
        the first CCF.
        '''
        rv = np.ascontiguousarray(self.ccf_rv if rv is None else rv, dtype=np.float64)
        ccf = np.ascontiguousarray(np.atleast_2d(self.ccf_var if ccf is None else ccf), dtype=np.float64)

        #wings of the first CCF: the gaussian of every CCF is fitted between them
        c0 = ccf[0] - ccf[0].min() + 0.000001
        cutleft, cutright, _, _ = nbspectra.speed_bisector_nb(rv, c0/c0.max(), True)

        #identical CCFs are computed once (hash of every row: linear, much faster than np.unique on the rows)
        index = {}
        inverse = np.empty(len(ccf), dtype=np.int64)
        first = []
        for i, row in enumerate(ccf):
            k = index.setdefault(row.tobytes(), len(first))
            if k == len(first):
                first.append(i)
            inverse[i] = k
        res = nbspectra.ccf_params_nb(rv, np.ascontiguousarray(ccf[first]), int(cutleft), int(cutright), float(self.vsini))
        rvs, contrast, fwhm, BIS, xbis, ybis, failed = (a[inverse] for a in res)

        n_failed = int(np.sum(failed))
        if n_failed:
            warnings.warn("the gaussian fit failed for %d of %d CCFs: they have the placeholder values rv=100000 m/s, contrast=1" % (n_failed, len(ccf)))
        if plot_test:
            import matplotlib.pyplot as plt
            for xb, yb in zip(xbis, ybis):
                plt.plot(xb, 1-yb, 'b')
                plt.show(block=True)
        return rvs, contrast, fwhm, BIS, list(xbis), list(ybis)

    def _flux_grids(self, ndim, mode_name):
        """Quiet / spot / facula flux grids as float arrays: (N_rings,) for photometry (one value per ring),
        (N_rings, n_wv) or (n_cells, n_wv) for spectroscopy (one spectrum per ring or per cell).
        Returns the three grids and per_ring (True: one row per ring). Spectroscopy grids of the two kinds together:
        the ones per ring are repeated for the cells of their ring.
        Photometry: flux_grid_xx. Spectroscopy: spec_grid_xx, or flux_grid_xx when no spec_grid is given and the
        photometry is not computed (previous versions)."""
        n_cells = len(self.vec_grid)
        allowed = (self.N_rings,) if ndim == 1 else (self.N_rings, n_cells)
        what = 'rings' if ndim == 1 else 'rings or %d cells' % n_cells
        name = 'flux_grid'
        grids = (self.flux_grid_qp, self.flux_grid_sp, self.flux_grid_fc)
        spec = (self.spec_grid_qp, self.spec_grid_sp, self.spec_grid_fc)
        if ndim == 2 and (any(g is not None for g in spec) or 'photometry' in self.mode):
            name, grids = 'spec_grid', spec
        out = []
        for key, g in zip(('qp', 'sp', 'fc'), grids):
            if g is None:
                raise ValueError("mode '%s' needs %s_qp, %s_sp and %s_fc (%s_%s is missing)" % (mode_name, name, name, name, name, key))
            if hasattr(g, 'built_grid'):                       # a flux_grid object instead of its .grid
                if getattr(g, 'grid', None) is None:
                    raise RuntimeError("%s_%s has no grid yet: run its built_grid() first" % (name, key))
                g = g.grid
            try:
                g = np.ascontiguousarray(g, dtype=np.float64)
            except (TypeError, ValueError):
                raise TypeError("%s_%s must be an array (flux_grid(...).grid), got %s" % (name, key, type(g).__name__)) from None
            _checks.nan_warning('%s_%s' % (name, key), g)
            if g.ndim != ndim or g.shape[0] not in allowed:
                raise ValueError("mode '%s' needs %dD grids with %d %s, %s_%s has shape %s (photometry: flux_grid_xx of a "
                                 "flux_grid in mode 'photometry'; spectroscopy: spec_grid_xx of a flux_grid in mode "
                                 "'spectroscopy')" % (mode_name, ndim, self.N_rings, what, name, key, g.shape))
            out.append(g)
        if ndim == 1:
            return out + [True]
        if len({g.shape[1] for g in out}) > 1:
            raise ValueError("the spectroscopy grids have different numbers of wavelengths: %s (with and without "
                             "stellar_rotation, the wavelengths of the rotating grids are shortened)" % [g.shape for g in out])
        rows = {g.shape[0] for g in out}
        if len(rows) > 1:                                  # rings and cells mixed: all per cell
            out = [np.repeat(g, self._Nin_arr, axis=0) if g.shape[0] == self.N_rings else g for g in out]
        per_ring = out[0].shape[0] == self.N_rings
        return out + [per_ring]

    def _spec_on_the_fly(self):
        """The three spectroscopy flux_grid objects when the spectra of the cells are computed on the fly (see
        spec_cells_max_memory), else None."""
        if self.active_regions_mask is not None or self.spec_on_the_fly is False:
            return None
        if self.spec_on_the_fly == 'auto' and self.spec_cells_max_memory is None:
            return None
        grids = (self.spec_grid_qp, self.spec_grid_sp, self.spec_grid_fc)
        if all(g is None for g in grids) and 'photometry' not in self.mode:
            grids = (self.flux_grid_qp, self.flux_grid_sp, self.flux_grid_fc)
        if any(getattr(g, 'ring_spectra', None) is None for g in grids):
            return None
        g0 = grids[0]
        for g in grids[1:]:
            if not (np.array_equal(g.wv_original, g0.wv_original) and np.array_equal(g.wv, g0.wv)
                    and np.array_equal(g.vrel, g0.vrel) and g.N_rings == g0.N_rings):
                return None
        if g0.N_rings != self.N_rings or len(g0.vrel) != len(self.vec_grid):
            return None
        if self.spec_on_the_fly == 'auto' and 3 * len(self.vec_grid) * len(g0.wv) * 8 / 1e6 <= self.spec_cells_max_memory:
            return None
        return grids

    def _ccf_rings(self):
        """Checks the three CCF grids and returns the velocity grid rv, the velocity axis of every ring (3, N, n_rv) and
        the CCF of every ring weighted by the area of one of its cells (3, N, n_rv), for the quiet photosphere, the
        spots and the faculae."""
        grids = (('qp', self.ccf_grid_qp), ('sp', self.ccf_grid_sp), ('fc', self.ccf_grid_fc))
        for key, g in grids:
            if g is None:
                raise ValueError("mode 'ccf' needs ccf_grid_qp, ccf_grid_sp and ccf_grid_fc (ccf_grid_%s is missing)" % key)
            if not all(hasattr(g, a) for a in ('ccf_rings', 'rvs_ring', 'rvs', 'N_rings')):
                raise TypeError("ccf_grid_%s must be a CCF_grid object (after built_grid()), got %s" % (key, type(g).__name__))
            if g.ccf_rings is None or g.rvs_ring is None:
                raise RuntimeError("ccf_grid_%s has no CCFs yet: run its built_grid() first" % key)
            _checks.nan_warning('the CCFs of ccf_grid_%s' % key, g.ccf_rings)
            if g.N_rings != self.N_rings:
                raise ValueError("ccf_grid_%s has %d rings but StarSim has %d" % (key, g.N_rings, self.N_rings))
        rv = np.ascontiguousarray(self.ccf_grid_qp.rvs, dtype=np.float64)
        weight = self._parea_arr / (4 * np.pi)                      #brightness of one cell, as in the old code
        for key, g in grids:
            if not np.array_equal(rv, g.rvs):
                raise ValueError("the three CCF grids must have the same velocity grid")
        ccf_ring = np.ascontiguousarray([np.asarray(g.ccf_rings, dtype=np.float64) * weight[:, None] for _, g in grids])
        rvs_ring = np.ascontiguousarray([np.asarray(g.rvs_ring, dtype=np.float64) for _, g in grids])
        self.ccf_rv = rv
        return rv, rvs_ring, ccf_ring

    def _quiet_velocities(self):
        """quiet_velocity as a float64 array (n_times, n_cells), NaN set to 0."""
        n_cells = len(self.vec_grid)
        qv = self.quiet_velocity
        v = qv.velocities(self.obs_times, self.N_rings, n_cells) if hasattr(qv, 'velocities') else qv
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (len(self.obs_times), n_cells):
            raise ValueError('quiet_velocity: %d epochs x %d cells expected (grid of %d rings), got %s'
                             % (len(self.obs_times), n_cells, self.N_rings, v.shape))
        return np.ascontiguousarray(np.nan_to_num(v, nan=0.0))

    def ccf_cells_on_the_fly(self):
        """True if the CCF of the cells is computed for the touched cells at every epoch instead of being kept for all
        the cells (ccf_cells_max_memory; map mode always keeps them)."""
        if self.active_regions_mask is not None or self.ccf_on_the_fly is False:
            return False
        if self.ccf_on_the_fly is True:
            return True
        if self.ccf_cells_max_memory is None:
            return False
        if getattr(self.ccf_grid_qp, 'rvs', None) is None:
            self._ccf_rings()                       #raises the error that explains what is missing
        n_rv = len(self.ccf_grid_qp.rvs)
        return 3 * len(self.vec_grid) * n_rv * 8 / 1e6 > self.ccf_cells_max_memory

    def _build_ccf_cells(self):
        """CCF of every cell for the quiet photosphere, the spots and the faculae: array (n_cells, n_rv) each.
        The CCF of every ring (ccf_grid_xx.ccf_rings) is placed on its velocity axis (ccf_grid_xx.rvs_ring), shifted by the
        rotation velocity of the cell (Doppler shift) and weighted by the area of the cell, as in the old
        compute_immaculate_sphere_rv."""
        rv, rvs_ring, ccf_ring = self._ccf_rings()
        n_cells = int(self._Nin_arr.sum())
        return [nbspectra.loop_compute_immaculate_nb(self.N_rings, self._Nin_arr, np.zeros([n_cells, len(rv)]),
                                                     self.rvel, rv, rvs_ring[s], ccf_ring[s]) for s in range(3)]

    def generate_timeseries(self, ccf_params=True, verbose=True):
        '''Loop for all the epochs and assign, for every observable in self.mode, the signal of the grid elements:
        'photometry'   -> self.flux_var  (n_times,)
        'spectroscopy' -> self.spec_var  (n_times, n_wavelengths)
        'ccf'          -> self.ccf_var   (n_times, n_rv), velocities in self.ccf_rv, and (if ccf_params) the parameters of the CCFs:
                          self.rv_var, self.contrast_var, self.fwhm_var, self.bis_var (n_times,) and the bisectors
                          self.xbis_var, self.ybis_var (n_times, 50)  [see compute_ccf_params]
                          With planets, the Keplerian RV of the star (self.rv_kepler, sum of planet.keplerian_rv) is added to rv_var.
        The geometry (filling factors of every cell) is computed once per epoch and shared by all the modes.
        The geometry of the regions and of the planets at all the epochs is computed first, vectorised
        (compute_regions_geometry, planet.positions), and stored: self.regions_pos, self.regions_vec
        (n_times, n_regions, 3), self.planet_pos (n_times, n_planets, 3), self.visible (n_times, n_regions+n_planets;
        regions first, then the planets in the order given). self.ff_pl is the fraction of the disc covered by all the planets.
        Then the whole loop over the epochs is a single Numba call, nbspectra.timeseries_nb (epochs in parallel).
        Map mode (active_regions_mask): the spot / facula filling factors of the cells come from the 2D maps of every
        epoch (active_region_mask.filling_factors) and the epochs are done by nbspectra.timeseries_maps_nb; regions_pos and
        regions_vec are then empty and visible only has the planets. With verbose, the progress of the reading of the maps
        is printed (date, filling factors and progress of every epoch).
        The time spent in every step [s] is stored in self.timing: 'grids' (preparation of the grids, CCFs of the
        cells), 'planets' (positions), 'geometry' (regions) or 'maps' (reading and projecting the 2D maps),
        'timeseries' (the Numba loop over the epochs), 'quiet_velocity' (with quiet_velocity), 'ccf_params' (RV, FWHM, ... of the CCFs) and 'total'.
        '''
        timing = {}
        t_start = t_step = _time.perf_counter()

        def lap(name):
            """Time since the previous step, stored in timing[name]."""
            nonlocal t_step
            now = _time.perf_counter()
            timing[name] = now - t_step
            t_step = now

        n_times = len(self.obs_times)
        n_cells = len(self.vec_grid)
        empty1, empty2 = np.zeros(0), np.zeros((n_cells, 0))

        #Every observable is signal(epoch) = sum over cells of ff_quiet*grid_quiet + ff_sp*grid_sp + ff_fc*grid_fc.
        #Grids are prepared once (unused ones are empty); the total of the quiet grid is the signal when no region is visible.
        ph = (empty1, empty1, empty1, np.zeros(1))
        sp = (empty2, empty2, empty2, empty1, False)
        cc = (empty2, empty2, empty2, empty1)
        #ccf on the fly (large grids): no CCF grid of the cells, the rings are given to the kernel
        empty3 = np.zeros((3, 0, 0))
        lazy = (False, empty1, self.rvel, empty3, empty3)
        lazy_sp = (False, empty1, empty1, empty3, empty1)
        if 'photometry' in self.mode:
            gq, gs, gf, _ = self._flux_grids(1, 'photometry')   #per ring
            ph = (gq, gs, gf, np.array([np.dot(self._Nin_arr, gq)]))
        if 'spectroscopy' in self.mode:
            #one spectrum per ring (no rotation) or per cell (rotation)
            lazy_grids = None if self.active_regions_mask is not None else self._spec_on_the_fly()
            if lazy_grids is not None:
                #spectra of the touched cells computed at every epoch from the spectra of the rings
                g0 = lazy_grids[0]
                n_wv = len(g0.wv)
                e2 = np.zeros((0, n_wv))
                sp = (e2, e2, e2, np.ascontiguousarray(g0.disc_spectrum, dtype=np.float64), False)
                lazy_sp = (True, np.ascontiguousarray(g0.wv_original, dtype=np.float64),
                           np.ascontiguousarray(g0.wv, dtype=np.float64),
                           np.ascontiguousarray([g.ring_spectra for g in lazy_grids], dtype=np.float64),
                           np.ascontiguousarray(g0.vrel, dtype=np.float64))
            else:
                gq, gs, gf, sp_per_ring = self._flux_grids(2, 'spectroscopy')
                quiet_sp = self._Nin_arr.astype(np.float64) @ gq if sp_per_ring else gq.sum(axis=0)
                sp = (gq, gs, gf, quiet_sp, sp_per_ring)
        if 'ccf' in self.mode:
            if self.ccf_cells_on_the_fly():
                rv, rvs_ring, ccf_ring = self._ccf_rings()
                quiet = nbspectra._quiet_ccf_total_nb(self.N_rings, self._Nin_arr, self.rvel, rv, rvs_ring[0], ccf_ring[0])
                cc = (empty2, empty2, empty2, quiet)
                lazy = (True, rv, self.rvel, rvs_ring, ccf_ring)
            else:
                gq, gs, gf = self._build_ccf_cells()
                cc = (gq, gs, gf, gq.sum(axis=0))
        lap('grids')

        #position of the planets at all the epochs, vectorised. A planet can be visible when the distance of its
        #centre from the disc centre (rho) is below 1 + its radius.
        if self.planets:
            planet_pos = np.stack([pl.positions(self.obs_times) for pl in self.planets], axis=1)   #(n_times, n_planets, 3)
        else:
            planet_pos = np.zeros((n_times, 0, 3))
        planet_vis = planet_pos[:, :, 0] - planet_pos[:, :, 2] < 1
        self.planet_pos = planet_pos
        lap('planets')
        grid_args = (self.N_rings, self._Nin_arr, self._parea_arr, np.asarray(self.amu, dtype=np.float64),
                     np.ascontiguousarray(self.vec_grid, dtype=np.float64))

        if self.active_regions_mask is not None:
            #MAP MODE: spot and facula filling factors of every cell from the 2D maps (one epoch read at a time),
            #then all the epochs in one Numba call (planets on top)
            ff_sp_maps, ff_fc_maps = self.active_regions_mask.filling_factors(self.obs_times, self.N_rings, self._Nin_arr,
                                                                              np.repeat(self._parea_arr, self._Nin_arr),
                                                                              verbose=verbose)
            lap('maps')
            self.regions_pos, self.regions_vec = np.zeros((n_times, 0, 3)), np.zeros((n_times, 0, 3))
            self.visible = planet_vis
            flux, spec, ccf, filling = nbspectra.timeseries_maps_nb(
                *grid_args, ff_sp_maps, ff_fc_maps, bool(self.simulate_planet), np.ascontiguousarray(planet_pos),
                np.ascontiguousarray(planet_vis), *ph, *sp, *cc, self.planet_coverage == 'area')
            lap('timeseries')
            if self.quiet_velocity is not None:
                #quiet photosphere of every cell moving on top of the rotation (supergranulation): its CCF shifted
                v = self._quiet_velocities()
                rv, rvs_ring, ccf_ring = self._ccf_rings()
                ccf = ccf + nbspectra.quiet_velocity_ccf_nb(self.N_rings, self._Nin_arr, ff_sp_maps, ff_fc_maps, v,
                                                            self.rvel, rv, rvs_ring[0], ccf_ring[0], cc[0])
                lap('quiet_velocity')
        else:
            #CIRCULAR REGIONS: geometry of all the regions at all the epochs, vectorised (stored for plots / checks)
            spot_pos, vec_spot, visible = self.compute_regions_geometry(self.obs_times)
            vis = np.concatenate([visible, planet_vis], axis=1).astype(np.float64)
            self.regions_pos, self.regions_vec, self.visible = spot_pos, vec_spot, vis.astype(bool)
            lap('geometry')

            #the whole loop over the epochs is one Numba call (epochs in parallel)
            typ = self._regions_arrays()[4]
            flux, spec, ccf, filling = nbspectra.timeseries_nb(
                *grid_args, np.ascontiguousarray(spot_pos), np.ascontiguousarray(vec_spot), vis, typ,
                bool(self.simulate_planet), np.ascontiguousarray(planet_pos), *ph, *sp, *cc, *lazy,
                self.planet_coverage == 'area', *lazy_sp)
            lap('timeseries')
        filling_ph, filling_sp, filling_fc, filling_pl = filling

        if 'photometry' in self.mode:
            self.flux_var = flux
        if 'spectroscopy' in self.mode:
            self.spec_var = spec
        if 'ccf' in self.mode:
            self.ccf_var = ccf
        self.ff_quiet = filling_ph
        self.ff_sp = filling_sp
        self.ff_fc = filling_fc
        self.ff_pl = filling_pl

        #parameters of the CCFs last, so that everything above is already stored if this step fails
        if 'ccf' in self.mode and ccf_params:
            rvs, contrast, fwhm, bis, xbis, ybis = self.compute_ccf_params()
            #Keplerian RV of the star: sum of the planets
            self.rv_kepler = np.zeros(n_times)
            for pl in self.planets:
                self.rv_kepler = self.rv_kepler + pl.keplerian_rv(self.obs_times)
            rvs = rvs + self.rv_kepler
            self.rv_var, self.contrast_var, self.fwhm_var, self.bis_var = rvs, contrast, fwhm, bis
            self.xbis_var, self.ybis_var = np.array(xbis), np.array(ybis)
            lap('ccf_params')
        timing['total'] = _time.perf_counter() - t_start
        self.timing = timing

    def results(self):
        """The time series of generate_timeseries in a dict (only those computed): 'time', 'ff_quiet', 'ff_sp', 'ff_fc',
        'ff_pl' [%], 'flux' (photometry), 'spec' (spectroscopy, n_times x n_wavelengths), 'ccf' (n_times x n_rv) and
        'ccf_rv' [m/s], 'rv', 'fwhm', 'bis', 'contrast', 'rv_kepler' (ccf)."""
        if getattr(self, 'ff_quiet', None) is None:
            raise RuntimeError('no results yet: run generate_timeseries() first')
        out = {'time': np.asarray(self.obs_times)}
        for key, attr in (('ff_quiet', 'ff_quiet'), ('ff_sp', 'ff_sp'), ('ff_fc', 'ff_fc'), ('ff_pl', 'ff_pl'),
                          ('flux', 'flux_var'), ('spec', 'spec_var'), ('ccf', 'ccf_var'), ('ccf_rv', 'ccf_rv'),
                          ('rv', 'rv_var'), ('fwhm', 'fwhm_var'), ('bis', 'bis_var'), ('contrast', 'contrast_var'),
                          ('rv_kepler', 'rv_kepler')):
            value = getattr(self, attr, None)
            if value is not None:
                out[key] = np.asarray(value)
        return out

    def save_results(self, path, fmt='npz', ccf=True, spec=True):
        """Saves the time series of results().
        fmt='npz' : one file path(.npz) with all the arrays (np.load(path) gives them back).
        fmt='txt' : a folder path with timeseries.txt (one row per epoch: time, ff_quiet, ff_sp, ff_fc, ff_pl and flux,
                    rv, fwhm, bis, contrast when computed; the names in the header), ccf.txt (one row per epoch) and
                    ccf_rv.txt (the velocities), spec.npy (spectra).
        ccf, spec : False -> the CCF / the spectra are not saved (the other time series are).
        Returns the path written."""
        res = self.results()
        if not ccf:
            res.pop('ccf', None)
            res.pop('ccf_rv', None)
        if not spec:
            res.pop('spec', None)
        path = os.fspath(path)
        if fmt == 'npz':
            if not path.endswith('.npz'):
                path += '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez(path, **res)
        elif fmt == 'txt':
            os.makedirs(path, exist_ok=True)
            cols = [k for k in ('time', 'ff_quiet', 'ff_sp', 'ff_fc', 'ff_pl', 'flux', 'rv', 'fwhm', 'bis', 'contrast',
                                'rv_kepler') if k in res]
            np.savetxt(os.path.join(path, 'timeseries.txt'), np.column_stack([res[k] for k in cols]),
                       header=' '.join(cols))
            if 'ccf' in res:
                np.savetxt(os.path.join(path, 'ccf.txt'), res['ccf'], header='CCF of every epoch (rows), velocities: ccf_rv.txt')
                np.savetxt(os.path.join(path, 'ccf_rv.txt'), res['ccf_rv'], header='velocities [m/s]')
            if 'spec' in res:
                np.save(os.path.join(path, 'spec.npy'), res['spec'])
        else:
            raise ValueError("fmt must be 'npz' or 'txt', got %r" % (fmt,))
        return path


def simulate_in_chunks(time, chunk_size=500, ccf_params=True, verbose=False, **starsim_kwargs):
    """StarSim on consecutive chunks of chunk_size epochs (every epoch is independent), the outputs concatenated: the
    memory of one simulation is that of chunk_size epochs (with 2D maps or precomputed filling factors, the filling
    factors of the cells of the chunk; the CCFs and spectra of the chunk). starsim_kwargs: the arguments of StarSim
    except `time` (N_rings, inclination, rotation_period, grids, active_regions_mask or active_regions_map, mode, ...).
    Returns a dict with the time series of the modes computed: 'time', 'ff_quiet', 'ff_sp', 'ff_fc', 'ff_pl', and
    'flux_var' (photometry), 'spec_var' (spectroscopy), 'ccf_var', 'ccf_rv' and, with ccf_params, 'rv_var', 'fwhm_var',
    'bis_var', 'contrast_var', 'rv_kepler' (ccf)."""
    time = _checks.array1d('time', time)
    chunk_size = int(_checks.number('chunk_size', chunk_size, integer=True, low=1))
    keys = ['ff_quiet', 'ff_sp', 'ff_fc', 'ff_pl', 'flux_var', 'spec_var', 'ccf_var', 'rv_var', 'fwhm_var', 'bis_var',
            'contrast_var', 'rv_kepler']
    parts = {k: [] for k in keys}
    out = {'time': time}
    for i in range(0, len(time), chunk_size):
        ss = StarSim(time[i:i + chunk_size], **starsim_kwargs)
        ss.generate_timeseries(ccf_params=ccf_params, verbose=verbose)
        for k in keys:
            if getattr(ss, k, None) is not None:
                parts[k].append(np.asarray(getattr(ss, k)))
        if getattr(ss, 'ccf_rv', None) is not None:
            out['ccf_rv'] = ss.ccf_rv
        del ss
    for k, v in parts.items():
        if v:
            out[k] = np.concatenate(v, axis=0)
    return out
