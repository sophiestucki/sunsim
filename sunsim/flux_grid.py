"""
flux_grid: flux (photometry) or spectrum (spectroscopy) of the cells of the stellar disc for one kind of surface
(quiet photosphere, spot or facula), from its specific intensity spectra at several mu.

The grid is given to StarSim (flux_grid_qp / flux_grid_sp / flux_grid_fc = flux_grid(...).grid), which combines the
three kinds of surface with the filling factors of every cell at every epoch.
"""
import os
import warnings
import numpy as np
from . import nbspectra
from . import _checks

# np.trapz was removed in recent NumPy versions
_trapz = getattr(np, 'trapezoid', None) or np.trapz


class flux_grid(object):
    """
    Flux or spectrum of the cells of the disc for one kind of surface.

    N_rings            : number of rings of the grid of the disc (the same as in StarSim)
    flux               : specific intensity spectra {mu: {'wav': wavelengths [A], 'intensity': intensity}} at >= 2 mu
                         (all on the same wavelength grid; the keys can be numbers or strings)
    typ                : label of the surface ('quiet', 'sp', 'fc', ...), only stored
    filter_path        : file with two columns (wavelength [A], transmission) multiplied to the spectra, or None for a
                         box between wavelength_lower_limit and wavelength_upper_limit
    wavelength_lower_limit, wavelength_upper_limit : [A] range of the box filter (filter_path=None, or file not found)
    stellar_radius     : [solar radii], rotation_period: [days] at the equator, inclination: [rad] (0 = equator-on),
                         differential_rotation: law of the differential rotation, as in StarSim (coefficients [deg/day]
                         or a function of the latitude, module rotation) -- only used for the Doppler shifts
                         of mode='spectroscopy' with stellar_rotation=True
    stellar_rotation   : spectroscopy only: True -> the spectrum of every cell is Doppler shifted by its rotation velocity
    mode               : 'photometry'   -> grid (N_rings,): flux of ONE cell of each ring (all the cells of a ring are
                                           equal), integrated over the wavelengths with the filter
                         'spectroscopy' -> spectrum times the filter. Without stellar_rotation all the cells of a ring
                                           are equal: grid (N_rings, n_wavelengths), the spectrum of ONE cell of each
                                           ring. With stellar_rotation every cell has its own Doppler shift:
                                           grid (n_cells, n_wavelengths), the spectrum of every cell.

    After built_grid(): grid, per_ring (True if grid has one row per ring, False if one row per cell),
    total_brightness (flux or integrated spectrum of the whole quiet disc), disc_spectrum (spectroscopy: spectrum of the
    whole quiet disc), cell_grid (spectroscopy: one row per cell, also without rotation) and wv (wavelengths of the
    spectroscopy grid; with stellar_rotation it is shortened at both ends by the largest Doppler shift).
    StarSim takes both kinds of spectroscopy grids (one row per ring or per cell).
    The flux of a cell is I(mu of its ring) * projected area of the cell / (4 pi) * filter, with I interpolated linearly
    in mu between the tabulated angles and extrapolated linearly to 0 at the limb (I proportional to mu) below the smallest.
    """

    def __init__(self, N_rings, flux, typ, filter_path, wavelength_lower_limit, wavelength_upper_limit, stellar_radius=1, rotation_period=25, inclination=0, stellar_rotation=False, mode='photometry', differential_rotation=0):

        #checks of the arguments
        N_rings = _checks.number('N_rings', N_rings, integer=True, low=1)
        _checks.spectra('flux', flux, min_mu=2)
        if mode not in ('photometry', 'spectroscopy'):
            raise ValueError("mode must be 'photometry' or 'spectroscopy', got %r" % (mode,))
        wavelength_lower_limit = _checks.number('wavelength_lower_limit', wavelength_lower_limit)
        wavelength_upper_limit = _checks.number('wavelength_upper_limit', wavelength_upper_limit)
        if not wavelength_lower_limit < wavelength_upper_limit:
            raise ValueError('wavelength_lower_limit (%g) must be < wavelength_upper_limit (%g)'
                             % (wavelength_lower_limit, wavelength_upper_limit))
        if filter_path is not None and not isinstance(filter_path, (str, bytes, os.PathLike)):
            raise TypeError('filter_path must be the path of a filter file, or None, got %r' % (filter_path,))
        stellar_radius = _checks.number('stellar_radius', stellar_radius, positive=True)
        rotation_period = _checks.number('rotation_period', rotation_period, positive=True)
        inclination = _checks.inclination('inclination', inclination)

        self.N_rings = N_rings
        self.high_res_flux = flux
        self.type = typ
        self.filter_path = filter_path
        self.wavelength_lower_limit = wavelength_lower_limit
        self.wavelength_upper_limit = wavelength_upper_limit
        self.stellar_radius = stellar_radius
        self.rotation_period = rotation_period
        self.inclination = inclination
        self.stellar_rotation = stellar_rotation
        #differential rotation, as in StarSim: coefficients (c1, c2, ...) [deg/day] of Omega(lat) = Omega_eq + c1 sin^2(lat)
        #+ c2 sin^4(lat) + ..., or any law(latitude [deg]) -> Omega(lat) - Omega_eq (module rotation)
        from .rotation import differential_rotation_law
        differential_rotation_law(differential_rotation)          # checks it
        self.differential_rotation = 0.0 if differential_rotation is None else differential_rotation
        self.mode = mode


        # Grid geometry -- computed only once
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea = nbspectra.generate_grid_coordinates_nb(N_rings)

        self.Ngrids = Ngrids
        self.Ngrid_in_ring = np.asarray(Ngrid_in_ring, dtype=np.int64)
        self.amu = np.asarray(amu, dtype=np.float64)
        self.parea = np.asarray(parea, dtype=np.float64)
        #the per-cell geometry (xs, ys, zs, rotation_rate, vrel: one value per cell, ~2(2N-1)^2/pi cells) is only needed
        #by the spectroscopy with rotation: computed the first time it is used (cell_geometry), not kept in photometry
        self._geometry = None

        # Starting pixel index of each ring
        self.ring_start = np.concatenate(([0], np.cumsum(self.Ngrid_in_ring[:-1]))).astype(np.int64)

        # Spectral quantities -- built only once
        # IMPORTANT: sort the mu keys (ascending). The limb extrapolation below assumes
        # that index 0 is the smallest mu, and limb_lim is the smallest mu.
        keys = sorted(self.high_res_flux.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two mu angles to build the limb-darkening grid")
        self._acd = np.asarray(keys, dtype=np.float64)
        self._keys = keys

        #grid of the cells: with stellar_rotation it is only built when it is used (property grid); the spectrum of every
        #ring and the velocity of every cell are kept instead (StarSim can then compute the touched cells on the fly)
        self._grid = None
        self._ring_spectra = None
        self._wv_original = None
        self._disc = None

        #spectra (n_mu, n_wavelengths): stacked the first time they are needed (spectroscopy); photometry only needs
        #one integral per mu and does not make this copy. They are read when the grid is built (built_grid).
        self._flux_i = None

        #wavelengths: not copied if they cannot be modified (read-only arrays of spectrum_utils)
        wv = np.asarray(self.high_res_flux[keys[0]]['wav'], dtype=np.float64)
        self._wv = wv.copy() if wv.flags.writeable else wv


    def cell_geometry(self):
        """Per-cell geometry, computed once on first use: dict with xs, ys, zs (cartesian coordinates of the cells),
        rotation_rate (Omega(lat)/Omega_eq of every cell) and vrel (line-of-sight rotation velocity [m/s])."""
        if self._geometry is None:
            _, _, _, _, _, _, xs, ys, zs, _, _ = nbspectra.generate_grid_coordinates_nb(self.N_rings)
            xs, ys, zs = (np.asarray(a, dtype=np.float64) for a in (xs, ys, zs))
            theta = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination))
            phi = np.arctan2(ys, xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))
            #with differential rotation every cell rotates at Omega(lat)/Omega_eq times the equatorial rate (sin(lat) = cos(theta))
            rotation_rate = nbspectra.rotation_rate_relative(np.cos(theta), self.rotation_period, self.differential_rotation)
            self._geometry = {'xs': xs, 'ys': ys, 'zs': zs, 'rotation_rate': rotation_rate,
                              'vrel': self.vsini*rotation_rate*np.sin(theta)*np.sin(phi)}
        return self._geometry

    @property
    def xs(self):
        """x coordinate of every cell (towards the observer)."""
        return self.cell_geometry()['xs']

    @property
    def ys(self):
        """y coordinate of every cell."""
        return self.cell_geometry()['ys']

    @property
    def zs(self):
        """z coordinate of every cell."""
        return self.cell_geometry()['zs']

    @property
    def rotation_rate(self):
        """Omega(lat)/Omega_eq of every cell."""
        return self.cell_geometry()['rotation_rate']

    @property
    def vsini(self):
        """Projected rotation velocity at the equator [m/s]: 2 pi R cos(inclination) / P (inclination 0 = equator-on)."""
        return 1000*2*np.pi*(self.stellar_radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400)


    @property
    def vrel(self):
        """Line-of-sight rotation velocity of every cell [m/s] (with the differential rotation, if any)."""
        return self.cell_geometry()['vrel']


    @property
    def acd(self):
        """Tabulated mu angles of the spectra, ascending."""
        return self._acd


    @property
    def limb_lim(self):
        """Smallest tabulated mu (closest to the limb): below it the intensity is extrapolated linearly to 0."""
        return self._acd[0]


    @property
    def wv(self):
        """Wavelength grid [A] (of the spectroscopy grid after built_grid())."""
        return self._wv


    @wv.setter
    def wv(self, value):
        """Set the wavelength grid (used by built_grid with stellar_rotation)."""
        self._wv = np.asarray(value, dtype=np.float64)


    @property
    def flux_i(self):
        """Specific intensity spectra at the tabulated mu (n_mu, n_wavelengths), in the order of acd."""
        if self._flux_i is None:
            self._flux_i = np.asarray([self.high_res_flux[key]['intensity'] for key in self._keys], dtype=np.float64)
        return self._flux_i


    @property
    def f_filt(self):
        """Transmission of the filter as a function of the wavelength [A] (0 outside the filter).
        From filter_path, or a box of transmission 1 between wavelength_lower_limit and wavelength_upper_limit."""

        if self.filter_path is None:
            wv = np.array([self.wavelength_lower_limit, self.wavelength_upper_limit])
            filt = np.array([1., 1.])

        else:
            try:
                wv, filt = np.loadtxt(self.filter_path, unpack=True)

            except (OSError, ValueError):
                warnings.warn("could not read the filter %s: a box filter between the wavelength limits is used"
                              % self.filter_path, stacklevel=2)
                wv = np.array([self.wavelength_lower_limit, self.wavelength_upper_limit])
                filt = np.array([1., 1.])

        #linear interpolation, 0 outside the filter (as scipy's interp1d(..., bounds_error=False, fill_value=0), without
        #importing scipy)
        order = np.argsort(wv)
        wv, filt = np.asarray(wv, dtype=np.float64)[order], np.asarray(filt, dtype=np.float64)[order]
        return lambda x: np.interp(x, wv, filt, left=0.0, right=0.0)



    @property
    def grid(self):
        """The grid (see the class docstring). With stellar_rotation the spectra of the cells (n_cells, n_wavelengths)
        are built the first time grid is used (doppler_shift_grid_nb) and kept; StarSim does not need them when it gets
        the flux_grid object itself (spec_grid_xx=flux_grid, see StarSim(spec_cells_max_memory=...))."""
        if self._grid is None and self._ring_spectra is not None:
            self._grid = nbspectra.doppler_shift_grid_nb(self._wv_original, self._wv, self._ring_spectra, self.vrel,
                                                         self.Ngrid_in_ring, self.ring_start)
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    @property
    def ring_spectra(self):
        """Spectroscopy with rotation: spectrum of one cell of every ring before its Doppler shift (N_rings,
        n_wavelengths of wv_original), and wv_original, the wavelengths before the cut at the largest shifts."""
        return self._ring_spectra

    @property
    def wv_original(self):
        return self._wv_original

    @property
    def per_ring(self):
        """True if grid has one row per ring (photometry, spectroscopy without rotation), False if one per cell."""
        return self.mode == 'photometry' or not self.stellar_rotation

    @property
    def disc_spectrum(self):
        """Spectroscopy: spectrum of the whole quiet disc (sum over all the cells), on the wavelengths wv."""
        if self.mode != 'spectroscopy':
            raise AttributeError("disc_spectrum is only defined in mode='spectroscopy' (photometry: total_brightness)")
        if self.per_ring:
            return self.Ngrid_in_ring.astype(np.float64) @ self.grid
        if self._disc is not None:
            return self._disc                    # computed without the grid of the cells (doppler_disc_total_nb)
        return self.grid.sum(axis=0)

    @property
    def cell_grid(self):
        """Spectroscopy: spectrum of every cell (n_cells, n_wavelengths); without rotation every ring row is repeated
        for the cells of the ring (a new array of n_cells rows)."""
        if self.mode != 'spectroscopy':
            raise AttributeError("cell_grid is only defined in mode='spectroscopy'")
        if self.per_ring:
            return np.repeat(self.grid, self.Ngrid_in_ring, axis=0)
        return self.grid

    def _ring_intensity(self, mu):
        """Specific intensity at angle mu: linear interpolation between tabulated angles,
        and linear extrapolation to the limb (I proportional to mu) below the smallest tabulated mu."""
        acd, I = self._acd, self.flux_i
        if mu < acd[0]:
            return mu/acd[0]*I[0]
        j = int(np.clip(np.searchsorted(acd, mu), 1, len(acd)-1))
        w = min((mu-acd[j-1])/(acd[j]-acd[j-1]), 1.0)
        return I[j-1]+w*(I[j]-I[j-1])

    def _ring_weights(self):
        """Interpolation of _ring_intensity for all the rings at once: index j and weights (a, b) such that
        I(mu of ring i) = a[i]*I[j[i]-1] + b[i]*I[j[i]] (below the smallest mu: j = 1, a = mu/acd[0], b = 0)."""
        acd, mu = self._acd, self.amu
        j = np.clip(np.searchsorted(acd, mu), 1, len(acd)-1)
        b = np.minimum((mu-acd[j-1])/(acd[j]-acd[j-1]), 1.0)
        a = 1.0-b
        limb = mu < acd[0]
        j[limb], a[limb], b[limb] = 1, mu[limb]/acd[0], 0.0
        return j, a, b

    def built_grid(self):
        """Build the grid: self.grid and self.total_brightness (see the class docstring).
        Call it once: with stellar_rotation it replaces self.wv by the shortened wavelength grid."""

        # Original common wavelength grid
        wv_original = self._wv
        self._ring_spectra, self._disc = None, None

        # Evaluate filter only once
        filt = np.asarray(self.f_filt(wv_original), dtype=np.float64)

        if self.mode == 'photometry':
            # one value per ring (all cells of a ring have the same flux). The interpolation in mu and the integral over
            # the wavelengths are both linear, so the spectrum of every tabulated mu is integrated once with the filter
            # (one matrix-vector product) and these integrals are interpolated in mu for every ring.
            dw = np.diff(wv_original)
            trapz_weights = np.zeros(len(wv_original))
            trapz_weights[:-1] += dw/2
            trapz_weights[1:] += dw/2
            fw = filt*trapz_weights
            # integral of I(mu) x filter, for every tabulated mu
            flux_mu = np.array([np.dot(np.asarray(self.high_res_flux[key]['intensity'], dtype=np.float64), fw)
                                for key in self._keys])
            j, a, b = self._ring_weights()
            #flux of one cell of ring i: integrated intensity x projected area of the cell / 4 pi
            self.grid = (a*flux_mu[j-1] + b*flux_mu[j])*self.parea/(4*np.pi)
            self.total_brightness = np.dot(self.Ngrid_in_ring, self.grid)
            return

        # Compute spectrum for each ring only once
        flp = np.zeros((self.N_rings, len(wv_original)), dtype=np.float64)
        for i in range(self.N_rings):

            dlp = self._ring_intensity(self.amu[i])

            #spectrum of one cell of ring i: intensity x projected area of the cell / 4 pi x filter
            flp[i, :] = dlp*self.parea[i]/(4*np.pi)*filt


        if self.mode == 'spectroscopy':
            # -----------------------------------------------------
            # ROTATING STAR
            # -----------------------------------------------------

            if self.stellar_rotation:

                c = 2.99792458e8

                #keep only the wavelengths covered by the spectra of all the cells after their Doppler shift
                min_wv = wv_original[0]*(1+np.max(self.vrel)/c)
                max_wv = wv_original[-1]*(1+np.min(self.vrel)/c)

                new_wv = wv_original[(wv_original > min_wv) & (wv_original < max_wv)]



                #the spectra of the cells are not built here (property grid): the spectrum of every ring and the
                #velocity of every cell are kept, and the spectrum of the quiet disc is the sum of the shifted spectra
                #of the cells, added in their order (the values of np.sum(grid, axis=0))
                self._ring_spectra = flp
                self._wv_original = wv_original
                self.wv = new_wv
                self._grid = None
                self._disc = nbspectra.doppler_disc_total_nb(wv_original, new_wv, flp, self.vrel, self.Ngrid_in_ring,
                                                             self.ring_start)
                self.total_brightness = _trapz(self._disc, self.wv)

            # -----------------------------------------------------
            # NON-ROTATING STAR
            # -----------------------------------------------------

            else:

                #no Doppler shift: all the cells of a ring have the same spectrum, kept once per ring
                self.grid = flp

                self.total_brightness = _trapz(self.disc_spectrum, self.wv)
