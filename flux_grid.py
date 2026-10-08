import numpy as np
from scipy import interpolate
import nbspectra

# np.trapz was removed in recent NumPy versions
_trapz = getattr(np, 'trapezoid', None) or np.trapz


class flux_grid(object):
    """
    Flux grid class
    """

    def __init__(self, N_rings, flux, typ, filter_path, wavelength_lower_limit, wavelength_upper_limit, stellar_radius=1, rotation_period=25, inclination=90, stellar_rotation=False, mode='photometry'):

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

        self._vrel = self.vsini*np.sin(theta)*np.sin(phi)


    @property
    def vsini(self):
        return 1000*2*np.pi*(self.stellar_radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400)


    @property
    def vrel(self):
        return self._vrel


    @property
    def acd(self):
        return self._acd


    @property
    def limb_lim(self):
        # smallest tabulated mu (closest to the limb)
        return self._acd[0]


    @property
    def wv(self):
        return self._wv


    @wv.setter
    def wv(self, value):
        self._wv = np.asarray(value, dtype=np.float64)


    @property
    def flux_i(self):
        return self._flux_i


    @property
    def f_filt(self):

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

            flp[i, :] = dlp*self.parea[i]/(4*np.pi)*filt
            if self.mode == 'photometry':
                sflp[i] = _trapz(flp[i,:], self.wv)


        if self.mode == 'spectroscopy':
            # -----------------------------------------------------
            # ROTATING STAR
            # -----------------------------------------------------

            if self.stellar_rotation:

                c = 2.99792458e8

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
