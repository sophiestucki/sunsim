"""
main: the StarSim class, which simulates the time series of a star with active regions (spots, faculae) and
transiting planets.

How a simulation works
----------------------
1. The visible disc is cut in N_rings concentric rings and every ring in cells (nbspectra.generate_grid_coordinates_nb).
2. For each kind of surface (quiet photosphere, spot, facula) a grid gives the signal of every cell if it were
   entirely of that kind: flux_grid (photometry: one value per ring; spectroscopy: one spectrum per cell) or CCF_grid
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
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import sys
import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
import collections
from astropy.io import fits
import pandas as pd
from scipy import interpolate
from scipy import optimize
import math as m
import warnings

import nbspectra


#Solar differential rotation: coefficients (B, C) [deg/day] of Omega(lat) = Omega_eq + B sin^2(lat) + C sin^4(lat).
#The profile used in StarSim (with the sin^2 term only, the solar value is B = -2.66 deg/day, Poljancic Beljan et al. 2017).
SOLAR_DIFFERENTIAL_ROTATION = (-1.698, -2.346)


class StarSim(object): 
    """
    Simulation of a star with active regions and planets: light curve, spectra and/or CCFs at the epochs `time`.

    Star
      time               : epochs [days] (array)
      N_rings            : number of concentric rings of the grid of the disc (the grids given must have the same)
      inclination        : angle between the rotation axis and the plane of the sky [rad]: 0 = equator-on, pi/2 = pole-on
      rotation_period    : rotation period at the equator [days]
      differential_rotation : coefficients (c1, c2, ...) [deg/day] of Omega(lat) = Omega_eq + c1 sin^2(lat) + c2 sin^4(lat)
                           + ...; 0 / None = rigid rotation, one number = c1 only, SOLAR_DIFFERENTIAL_ROTATION for the
                           Sun. It moves the regions and changes the rotation velocity of the cells (ccf mode).
      stellar_radius     : [solar radii], for the rotation velocity
    Surface
      active_regions_map : list of active_region.active_region (spots 'sp' and faculae 'fc'); None = no region
      active_regions_mask: active_region_mask.active_region_mask object: the spots and faculae come from 2D maps of the disc, one pair of
                           maps per epoch (map mode; every epoch of `time` must have its maps, e.g. time = maps.times).
                           Not together with active_regions_map.
      planet             : planet.planet object, a list of them, or None (no planet)
    Grids (the three kinds of surface: qp = quiet photosphere, sp = spot, fc = facula)
      flux_grid_qp, flux_grid_sp, flux_grid_fc : flux_grid(...).grid, for 'photometry' (one value per ring, mode
                           'photometry' of flux_grid) or 'spectroscopy' (one spectrum per cell, mode 'spectroscopy')
      ccf_grid_qp, ccf_grid_sp, ccf_grid_fc    : CCF_grid objects (after built_grid()), for 'ccf'
    mode                 : list with any of 'photometry', 'spectroscopy', 'ccf' (the geometry is shared). Photometry and
                           spectroscopy need different flux grids, so they cannot be run together.
    total_flux_qp        : not used (kept for compatibility)

    After generate_timeseries():
      flux_var (photometry), spec_var (spectroscopy), ccf_var and ccf_rv (ccf) and, in ccf mode, rv_var, fwhm_var,
      contrast_var, bis_var, xbis_var, ybis_var and rv_kepler;
      ff_quiet, ff_sp, ff_fc, ff_pl: fraction of the disc covered by each kind of surface [% of the projected area];
      regions_pos, regions_vec, planet_pos, visible: geometry of the regions and planets at every epoch.
    Other attributes: vsini (equatorial, m/s), rvel (line-of-sight rotation velocity of every cell, m/s),
    rotation_rate (Omega(lat)/Omega_eq of every cell), amu, parea, Ngrid_in_ring, vec_grid (grid of the disc).
    """
    def __init__(self, time, N_rings, inclination, rotation_period, differential_rotation=0, stellar_radius=1, active_regions_map=None, active_regions_mask=None, total_flux_qp=None, flux_grid_qp=None, flux_grid_sp=None, flux_grid_fc=None, ccf_grid_qp=None, ccf_grid_sp=None, ccf_grid_fc=None, planet=None, mode=['photometry']):

        self.obs_times = time
        self.total_flux_qp = total_flux_qp
        self.flux_grid_qp = flux_grid_qp
        self.flux_grid_sp = flux_grid_sp
        self.flux_grid_fc = flux_grid_fc

        self.ccf_grid_qp = ccf_grid_qp
        self.ccf_grid_sp = ccf_grid_sp
        self.ccf_grid_fc = ccf_grid_fc
        self.inclination = inclination
        self.rotation_period = rotation_period
        #differential rotation: coefficients (c1, c2, ...) [deg/day] of
        #Omega(lat) = Omega_eq + c1 sin^2(lat) + c2 sin^4(lat) + ...   (Omega_eq = 360/rotation_period deg/day)
        #0 or None = rigid rotation; a single number = c1 only; SOLAR_DIFFERENTIAL_ROTATION for the Sun
        self.differential_rotation = nbspectra.differential_rotation_coeffs(differential_rotation)
        self.active_regions_map = [] if active_regions_map is None else active_regions_map
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
        colat = np.array([r.latitude for r in regs], dtype=np.float64)
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
        sl = np.sin(np.deg2rad(90 - colat))                     #sin(latitude)
        dt = t - ref
        #rotation rate relative to the equator [deg/day]: c1 sin^2(lat) + c2 sin^4(lat) + ...
        domega = np.polynomial.polynomial.polyval(sl**2, (0.0,) + self.differential_rotation)
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
            for xb, yb in zip(xbis, ybis):
                plt.plot(xb, 1-yb, 'b')
                plt.show(block=True)
        return rvs, contrast, fwhm, BIS, list(xbis), list(ybis)

    def _flux_grids(self, ndim, mode_name):
        """Quiet / spot / facula flux grids as float arrays: (N_rings,) for photometry (one value per ring),
        (n_cells, n_wv) for spectroscopy."""
        n_expected = self.N_rings if ndim == 1 else len(self.vec_grid)
        what = 'rings' if ndim == 1 else 'cells'
        out = []
        for key, g in (('qp', self.flux_grid_qp), ('sp', self.flux_grid_sp), ('fc', self.flux_grid_fc)):
            if g is None:
                raise ValueError("mode '%s' needs flux_grid_qp, flux_grid_sp and flux_grid_fc (flux_grid_%s is missing)" % (mode_name, key))
            g = np.ascontiguousarray(g, dtype=np.float64)
            if g.ndim != ndim or g.shape[0] != n_expected:
                raise ValueError("mode '%s' needs %dD flux grids with %d %s, flux_grid_%s has shape %s (photometry and "
                                 "spectroscopy together need two different sets of grids)" % (mode_name, ndim, n_expected, what, key, g.shape))
            out.append(g)
        return out

    def _build_ccf_cells(self):
        """CCF of every cell for the quiet photosphere, the spots and the faculae: array (n_cells, n_rv) each.
        The CCF of every ring (ccf_grid_xx.ccf_rings) is placed on its velocity axis (ccf_grid_xx.rvs_ring), shifted by the
        rotation velocity of the cell (Doppler shift) and weighted by the area of the cell, as in the old
        compute_immaculate_sphere_rv."""
        grids = (('qp', self.ccf_grid_qp), ('sp', self.ccf_grid_sp), ('fc', self.ccf_grid_fc))
        for key, g in grids:
            if g is None:
                raise ValueError("mode 'ccf' needs ccf_grid_qp, ccf_grid_sp and ccf_grid_fc (ccf_grid_%s is missing)" % key)
            if g.ccf_rings is None or g.rvs_ring is None:
                raise RuntimeError("ccf_grid_%s has no CCFs yet: run its built_grid() first" % key)
            if g.N_rings != self.N_rings:
                raise ValueError("ccf_grid_%s has %d rings but StarSim has %d" % (key, g.N_rings, self.N_rings))
        rv = np.ascontiguousarray(self.ccf_grid_qp.rvs, dtype=np.float64)
        weight = self._parea_arr / (4 * np.pi)                      #brightness of one cell, as in the old code
        n_cells = int(self._Nin_arr.sum())
        cells = []
        for key, g in grids:
            if not np.array_equal(rv, g.rvs):
                raise ValueError("the three CCF grids must have the same velocity grid")
            ccf_ring = np.ascontiguousarray(np.asarray(g.ccf_rings, dtype=np.float64) * weight[:, None])
            rvs_ring = np.ascontiguousarray(g.rvs_ring, dtype=np.float64)
            cells.append(nbspectra.loop_compute_immaculate_nb(self.N_rings, self._Nin_arr, np.zeros([n_cells, len(rv)]),
                                                              self.rvel, rv, rvs_ring, ccf_ring))
        self.ccf_rv = rv
        return cells

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
        '''
        n_times = len(self.obs_times)
        n_cells = len(self.vec_grid)
        empty1, empty2 = np.zeros(0), np.zeros((n_cells, 0))

        #Every observable is signal(epoch) = sum over cells of ff_quiet*grid_quiet + ff_sp*grid_sp + ff_fc*grid_fc.
        #Grids are prepared once (unused ones are empty); the total of the quiet grid is the signal when no region is visible.
        ph = (empty1, empty1, empty1, np.zeros(1))
        sp = (empty2, empty2, empty2, empty1)
        cc = (empty2, empty2, empty2, empty1)
        if 'photometry' in self.mode:
            gq, gs, gf = self._flux_grids(1, 'photometry')   #per ring
            ph = (gq, gs, gf, np.array([np.dot(self._Nin_arr, gq)]))
        if 'spectroscopy' in self.mode:
            gq, gs, gf = self._flux_grids(2, 'spectroscopy')
            sp = (gq, gs, gf, gq.sum(axis=0))
        if 'ccf' in self.mode:
            gq, gs, gf = self._build_ccf_cells()
            cc = (gq, gs, gf, gq.sum(axis=0))

        #position of the planets at all the epochs, vectorised. A planet can be visible when the distance of its
        #centre from the disc centre (rho) is below 1 + its radius.
        if self.planets:
            planet_pos = np.stack([pl.positions(self.obs_times) for pl in self.planets], axis=1)   #(n_times, n_planets, 3)
        else:
            planet_pos = np.zeros((n_times, 0, 3))
        planet_vis = planet_pos[:, :, 0] - planet_pos[:, :, 2] < 1
        self.planet_pos = planet_pos
        grid_args = (self.N_rings, self._Nin_arr, self._parea_arr, np.asarray(self.amu, dtype=np.float64),
                     np.ascontiguousarray(self.vec_grid, dtype=np.float64))

        if self.active_regions_mask is not None:
            #MAP MODE: spot and facula filling factors of every cell from the 2D maps (one epoch read at a time),
            #then all the epochs in one Numba call (planets on top)
            ff_sp_maps, ff_fc_maps = self.active_regions_mask.filling_factors(self.obs_times, self.N_rings, self._Nin_arr,
                                                                              np.repeat(self._parea_arr, self._Nin_arr),
                                                                              verbose=verbose)
            self.regions_pos, self.regions_vec = np.zeros((n_times, 0, 3)), np.zeros((n_times, 0, 3))
            self.visible = planet_vis
            flux, spec, ccf, filling = nbspectra.timeseries_maps_nb(
                *grid_args, ff_sp_maps, ff_fc_maps, bool(self.simulate_planet), np.ascontiguousarray(planet_pos),
                np.ascontiguousarray(planet_vis), *ph, *sp, *cc)
        else:
            #CIRCULAR REGIONS: geometry of all the regions at all the epochs, vectorised (stored for plots / checks)
            spot_pos, vec_spot, visible = self.compute_regions_geometry(self.obs_times)
            vis = np.concatenate([visible, planet_vis], axis=1).astype(np.float64)
            self.regions_pos, self.regions_vec, self.visible = spot_pos, vec_spot, vis.astype(bool)

            #the whole loop over the epochs is one Numba call (epochs in parallel)
            typ = self._regions_arrays()[4]
            flux, spec, ccf, filling = nbspectra.timeseries_nb(
                *grid_args, np.ascontiguousarray(spot_pos), np.ascontiguousarray(vec_spot), vis, typ,
                bool(self.simulate_planet), np.ascontiguousarray(planet_pos), *ph, *sp, *cc)
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