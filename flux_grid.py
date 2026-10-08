"""
flux_grid: flux (photometry) or spectrum (spectroscopy) of the cells of the stellar disc for one kind of surface
(quiet photosphere, spot or facula), from its specific intensity spectra at several mu.

The grid is given to StarSim (flux_grid_qp / flux_grid_sp / flux_grid_fc = flux_grid(...).grid), which combines the
three kinds of surface with the filling factors of every cell at every epoch.
"""
import numpy as np
from scipy import interpolate
import nbspectra

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
                         differential_rotation: coefficients [deg/day] as in StarSim -- only used for the Doppler shifts
                         of mode='spectroscopy' with stellar_rotation=True
    stellar_rotation   : spectroscopy only: True -> the spectrum of every cell is Doppler shifted by its rotation velocity
    mode               : 'photometry'   -> grid (N_rings,): flux of ONE cell of each ring (all the cells of a ring are
                                           equal), integrated over the wavelengths with the filter
                         'spectroscopy' -> grid (n_cells, n_wavelengths): spectrum of every cell, times the filter

    After built_grid(): grid, total_brightness (flux or integrated spectrum of the whole quiet disc) and wv (wavelengths of
    the spectroscopy grid; with stellar_rotation it is shortened at both ends by the largest Doppler shift).
    The flux of a cell is I(mu of its ring) * projected area of the cell / (4 pi) * filter, with I interpolated linearly
    in mu between the tabulated angles and extrapolated linearly to 0 at the limb (I proportional to mu) below the smallest.
    """

    def __init__(self, N_rings, flux, typ, filter_path, wavelength_lower_limit, wavelength_upper_limit, stellar_radius=1, rotation_period=25, inclination=90, stellar_rotation=False, mode='photometry', differential_rotation=0):

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
        #differential rotation (c1, c2, ...) [deg/day], as in StarSim: Omega(lat) = Omega_eq + c1 sin^2(lat) + c2 sin^4(lat) + ...
        self.differential_rotation = nbspectra.differential_rotation_coeffs(differential_rotation)
        self.mode = mode


        # Grid geometry -- computed only once
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea = nbspectra.generate_grid_coordinates_nb(N_rings)

        self.Ngrids = Ngrids
        self.Ngrid_in_ring = np.asarray(Ngrid_in_ring, dtype=np.int64)
        self.amu = np.asarray(amu, dtype=np.float64)
        self.parea = np.asarray(parea, dtype=np.float64)

        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.zs = np.asarray(zs, dtype=np.float64)

        # Starting pixel index of each ring
        self.ring_start = np.concatenate(([0], np.cumsum(self.Ngrid_in_ring[:-1]))).astype(np.int64)

        # Spectral quantities -- built only once
        # IMPORTANT: sort the mu keys (ascending). The limb extrapolation below assumes
        # that index 0 is the smallest mu, and limb_lim is the smallest mu.
        keys = sorted(self.high_res_flux.keys())
        if len(keys) < 2:
            raise ValueError("Need at least two mu angles to build the limb-darkening grid")
        self._acd = np.asarray(keys, dtype=np.float64)

        self._flux_i = np.asarray([self.high_res_flux[key]['intensity'] for key in keys], dtype=np.float64)

        self._wv = np.asarray(self.high_res_flux[keys[0]]['wav'], dtype=np.float64).copy()

        # Rotational velocity of every surface element -- computed only once
        theta = np.arccos(self.zs*np.cos(-self.inclination)-self.xs*np.sin(-self.inclination))
        phi = np.arctan2(self.ys, self.xs*np.cos(-self.inclination)+self.zs*np.sin(-self.inclination))

        #with differential rotation every cell rotates at Omega(lat)/Omega_eq times the equatorial rate (sin(lat) = cos(theta))
        self.rotation_rate = nbspectra.rotation_rate_relative(np.cos(theta), self.rotation_period, self.differential_rotation)
        self._vrel = self.vsini*self.rotation_rate*np.sin(theta)*np.sin(phi)


    @property
    def vsini(self):
        """Projected rotation velocity at the equator [m/s]: 2 pi R cos(inclination) / P (inclination 0 = equator-on)."""
        return 1000*2*np.pi*(self.stellar_radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400)


    @property
    def vrel(self):
        """Line-of-sight rotation velocity of every cell [m/s] (with the differential rotation, if any)."""
        return self._vrel


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
                print(f"Could not load filter {self.filter_path}. Using wavelength range instead.")
                wv = np.array([self.wavelength_lower_limit, self.wavelength_upper_limit])
                filt = np.array([1., 1.])

        return interpolate.interp1d(wv, filt, bounds_error=False, fill_value=0)



    def _ring_intensity(self, mu):
        """Specific intensity at angle mu: linear interpolation between tabulated angles,
        and linear extrapolation to the limb (I proportional to mu) below the smallest tabulated mu."""
        acd, I = self._acd, self._flux_i
        if mu < acd[0]:
            return mu/acd[0]*I[0]
        j = int(np.clip(np.searchsorted(acd, mu), 1, len(acd)-1))
        w = min((mu-acd[j-1])/(acd[j]-acd[j-1]), 1.0)
        return I[j-1]+w*(I[j]-I[j-1])

    def built_grid(self):
        """Build the grid: self.grid and self.total_brightness (see the class docstring).
        Call it once: with stellar_rotation it replaces self.wv by the shortened wavelength grid."""

        # Original common wavelength grid
        wv_original = self._wv

        # Evaluate filter only once
        filt = np.asarray(self.f_filt(wv_original), dtype=np.float64)

        # Compute spectrum for each ring only once
        flp = np.zeros((self.N_rings, len(wv_original)), dtype=np.float64)
        if self.mode == 'photometry':
            # one value per ring (all cells of a ring have the same flux)
            sflp = np.zeros(self.N_rings, dtype=np.float64)

        for i in range(self.N_rings):

            dlp = self._ring_intensity(self.amu[i])

            #spectrum of one cell of ring i: intensity x projected area of the cell / 4 pi x filter
            flp[i, :] = dlp*self.parea[i]/(4*np.pi)*filt
            if self.mode == 'photometry':
                sflp[i] = _trapz(flp[i,:], self.wv)


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



                # Numba performs all Doppler interpolation
                final_flp = nbspectra.doppler_shift_grid_nb(wv_original, new_wv, flp, self.vrel, self.Ngrid_in_ring, self.ring_start)

                self.wv = new_wv

                self.grid = final_flp

                self.total_brightness = _trapz(np.sum(final_flp, axis=0), self.wv)

            # -----------------------------------------------------
            # NON-ROTATING STAR
            # -----------------------------------------------------

            else:

                #no Doppler shift: every cell has the spectrum of its ring
                final_flp = np.empty((self.Ngrids, len(wv_original)), dtype=np.float64)

                for i in range(self.N_rings):
                    start = self.ring_start[i]
                    end = start+self.Ngrid_in_ring[i]
                    final_flp[start:end, :] = flp[i, :]

                self.grid = final_flp

                self.total_brightness = _trapz(np.sum(final_flp, axis=0), self.wv)
        elif self.mode == 'photometry':
                # grid has shape (N_rings,): flux of one cell of each ring
                self.grid = sflp
                self.total_brightness = np.dot(self.Ngrid_in_ring, sflp)
