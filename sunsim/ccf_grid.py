"""
CCF_grid: CCF of the rings of the stellar disc, in two independent steps.

    step 1  compute_ccf()    CCF of every ring in the rest frame, with or without instrument
    step 2  rv_treatment()   velocity axis of every ring (bisector removal / bisector added)

The numerical work is done by nbspectra.cross_correlation_mask (phoenix_resolution is passed straight to it).

How StarSim uses it (mode 'ccf'): ccf_grid_qp / ccf_grid_sp / ccf_grid_fc are CCF_grid objects of the quiet
photosphere, the spots and the faculae (after built_grid()). For every cell, StarSim takes the CCF of its ring
(ccf_rings), places it on the velocity axis of the ring (rvs_ring) shifted by the rotation velocity of the cell, and
weights it by the area of the cell (StarSim._build_ccf_cells). The CCFs here are therefore in the rest frame of the
surface, without rotation.

The CCFs are normalised (nbspectra.cross_correlation_mask), so they do not know how bright the ring is: the limb
darkening and the contrast of the spots / faculae are given by mu_ratio (brightness of every ring relative to the
quiet disc centre).
"""
import warnings
import numpy as np

from . import _checks
from . import nbspectra



# =============================================================================================
# helpers
# =============================================================================================
def ring_intensity(mu, acd, fln, limb_lim, limb_extrapolation='linear'):
    """Spectrum of a ring at projected angle mu (acd ascending, fln = spectra at acd).
    Linear interpolation between the tabulated angles; below limb_lim the spectrum is extrapolated
    towards the limb ('constant' or 'linear')."""
    if mu > limb_lim:
        acd_low = np.max(acd[acd < mu])                 # tabulated angles below and above mu
        acd_upp = np.min(acd[acd >= mu])
        idx_low = np.where(acd == acd_low)[0][0]
        idx_upp = np.where(acd == acd_upp)[0][0]
        return fln[idx_low] + (fln[idx_upp] - fln[idx_low]) * (mu - acd_low) / (acd_upp - acd_low)

    if limb_extrapolation == 'constant':
        return fln[0]
    if limb_extrapolation == 'linear':
        if limb_lim == acd.min():
            return mu / limb_lim * fln[0]
        acd_low = np.max(acd[acd < limb_lim])
        acd_upp = np.min(acd[acd >= limb_lim])
        idx_low = np.where(acd == acd_low)[0][0]
        idx_upp = np.where(acd == acd_upp)[0][0]
        dlp_lim = fln[idx_low] + (fln[idx_upp] - fln[idx_low]) * (limb_lim - acd_low) / (acd_upp - acd_low)
        return mu / limb_lim * dlp_lim
    raise ValueError("limb_extrapolation must be 'constant' or 'linear', got %r" % (limb_extrapolation,))


def ring_ccf(mu, acd, ccfs, limb_lim, limb_extrapolation='linear'):
    """CCF of a ring at projected angle mu from the CCFs of the spectra at the tabulated angles (acd ascending,
    ccfs = CCFs at acd). Linear interpolation between the tabulated angles. Below limb_lim:
      'constant' -> CCF at the smallest tabulated angle (the CCF of the spectrum used by ring_intensity);
      'linear'   -> CCF at limb_lim: ring_intensity only rescales the spectrum there (mu/limb_lim), which does not
                    change the normalised CCF (the brightness of the ring is given by mu_ratio)."""
    if mu > limb_lim:
        return ring_intensity(mu, acd, ccfs, limb_lim, limb_extrapolation)
    if limb_extrapolation == 'constant':
        return ccfs[0]
    if limb_extrapolation == 'linear':
        if limb_lim == acd.min():
            return ccfs[0]
        acd_low = np.max(acd[acd < limb_lim])           # tabulated angles below and above limb_lim
        acd_upp = np.min(acd[acd >= limb_lim])
        idx_low = np.where(acd == acd_low)[0][0]
        idx_upp = np.where(acd == acd_upp)[0][0]
        return ccfs[idx_low] + (ccfs[idx_upp] - ccfs[idx_low]) * (limb_lim - acd_low) / (acd_upp - acd_low)
    raise ValueError("limb_extrapolation must be 'constant' or 'linear', got %r" % (limb_extrapolation,))


def _cubic_spline_not_a_knot(x, y):
    """Cubic spline through (x, y) (x strictly increasing, >= 4 points) with not-a-knot end conditions, as
    scipy.interpolate.interp1d(x, y, kind='cubic') (equal to it to the rounding errors). Returns f(xp), with f = y[0]
    below x[0] and y[-1] above x[-1]. Written with numpy so that scipy.interpolate (~0.3 s to import) is not needed.
    Second derivatives M of the spline: continuity of the first derivative at the inner points, and of the third
    derivative at x[1] and x[-2] (not-a-knot)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, :3] = h[1], -(h[0]+h[1]), h[0]
    A[-1, -3:] = h[-1], -(h[-2]+h[-1]), h[-2]
    for i in range(1, n-1):
        A[i, i-1:i+2] = h[i-1], 2*(h[i-1]+h[i]), h[i]
        b[i] = 6*((y[i+1]-y[i])/h[i]-(y[i]-y[i-1])/h[i-1])
    M = np.linalg.solve(A, b)

    def f(xp):
        xp = np.asarray(xp, dtype=np.float64)
        i = np.clip(np.searchsorted(x, xp, side='right')-1, 0, n-2)
        hi, a, c = h[i], x[i+1]-xp, xp-x[i]
        out = (M[i]*a**3+M[i+1]*c**3)/(6*hi)+(y[i]/hi-M[i]*hi/6)*a+(y[i+1]/hi-M[i+1]*hi/6)*c
        return np.where(xp < x[0], y[0], np.where(xp > x[-1], y[-1], out))
    return f


def bin_edges(centres):
    """Edges of the pixels of an order (halfway between the centres, the outer ones extrapolated), always computed in
    float64, also for float32 wavelengths (as in many instrument files): float32 edges are off by ~2e-4 A at 5000 A,
    1.6 % of a HARPS pixel, which changes the resampled flux by up to 0.6 % (specutils >= 2 computes them in the
    precision of the wavelengths; specutils 1.x, used by the old StarSim, in float64)."""
    return nbspectra.bin_edges_nb(np.asarray(centres, dtype=np.float64))


def bisector_fit(rv, ccf, kind_interp='linear', integrated_bis=False):
    """Function rv = f(ccf height) interpolating the bisector of the CCF (constant beyond its ends).
    'linear' and 'cubic' are computed with numpy; the other kinds of scipy.interpolate.interp1d with scipy."""
    xnew, ynew, xbis, ybis = nbspectra.speed_bisector_nb(rv, ccf, integrated_bis)
    if kind_interp == 'linear':
        return lambda h: np.interp(h, ybis, xbis, left=xbis[0], right=xbis[-1])
    if kind_interp == 'cubic':
        return _cubic_spline_not_a_knot(ybis, xbis)
    from scipy import interpolate      #imported here: ~0.3 s, only for the other kinds
    return interpolate.interp1d(ybis, xbis, kind=kind_interp, fill_value=(xbis[0], xbis[-1]), bounds_error=False)


def dumusque_bisector(normalization):
    """Bisector added by rv_treatment, the same for every mu (Dumusque 2014): returns f(mu) -> p(ccf), p in km/s.
    Any other bisector can be given to rv_treatment as a function f(mu) -> p(ccf) (p in km/s), e.g. a mu-dependent one."""
    p = np.poly1d(np.array([-1.51773453,  3.52774949, -3.18794328,  1.22541774,  -0.22479665+0.4]) if normalization else np.array([ 69.72233642, -111.07077255,   66.13347937,  -19.34395212,    2.85797555, 0.16536442]))
    return lambda mu: p


# =============================================================================================
# the grid
# =============================================================================================
class CCF_grid(object):
    """
    CCF of every ring of the disc for one kind of surface (quiet photosphere, spot or facula).

    N_rings       : number of rings of the grid of the disc (the same as in StarSim)
    high_res_flux : {mu: {'wav': wavelengths [A], 'intensity': flux}}   (any key order, it is sorted)
    RVs           : velocity grid of the CCF [m/s]
    wvm, fm       : mask lines [A] and weights
    convective_blueshift : factor that modulates the bisector added in step 2 (rv_treatment)
    phoenix_resolution   : True if the spectra are on the Phoenix wavelength grid (fast index in
                    nbspectra.cross_correlation_mask).  Only used for the whole-spectrum CCF: after resampling on the
                    orders the grid is not the Phoenix one, so the order CCFs always use the generic branch.
    mu_ratio      : None -> CCF of every ring from its own spectrum (needs >= 2 mu angles).
                    array (N_rings) -> brightness of every ring: the CCF of the ring is multiplied by mu_ratio[i].
                      * high_res_flux with >= 2 mu angles: CCF of every ring at its own mu (see interpolation),
                        times mu_ratio[i]. The bisector of every ring is kept.
                      * high_res_flux with ONE spectrum, at mu = 1.0 (it is checked): CCF of the disc centre times
                        mu_ratio[i] for every ring; its bisector is removed in rv_treatment.
    normalization : False -> -ccf/min(ccf) + 1; True -> scaled to [0, 1] (ccf_norm of
                    nbspectra.cross_correlation_mask); 'none' -> no normalisation: the raw CCF (minus the weighted sum of
                    the fluxes), proportional to the spectrum. With 'none' the CCF of every ring keeps the brightness of
                    its spectrum (no mu_ratio needed with spectra at several mu), and the CCFs of the orders are added
                    with the flux of every order.
    instrument    : None -> CCF of the whole spectrum.
                    'HARPS-N', 'HARPS', 'EXPRESS', 'NEID' or a resolving power R (a number) -> every spectrum is
                    degraded to the resolution of the instrument (spectrum_utils.add_resol), then:
                      * without blaze_function: CCF of the whole degraded spectrum (as StarSim v4 with an instrument);
                      * with blaze_function: resampled on the wavelength grid of every order, multiplied by
                        blaze*efficiency of the order, CCF of every order, orders summed.
    blaze_function: None, or (wvb, blaze): wavelength grid and blaze of every order, [n_orders][n_pix]
    instrumental_efficiency : None | callable eff(wavelength) | (wavelengths, efficiency) table.
                    It is evaluated on the wavelengths of EACH order (so it can cover a much wider range than one
                    order) and multiplied by the blaze of that order.  A table is interpolated (flat outside it);
                    a callable, e.g. a polynomial fit, is only meaningful inside its fit range.
                    The CCF of every order is normalised (normalization) before the orders are added, so every order
                    counts the same: a factor common to a whole order (its mean efficiency, a flattening of the
                    spectrum) disappears, and the blaze and the efficiency only change the weights of the lines inside
                    an order.
    limb_lim, limb_extrapolation : limb treatment of the ring spectra (limb_lim default: smallest tabulated mu)
    interpolation : how the CCF of a ring is obtained when several mu are given:
                    'ccf' (default) -> the CCF of the spectrum at every tabulated mu is computed once (self.ccf_mu),
                                       and the CCF of every ring is interpolated in mu between them (ring_ccf);
                    'spectrum'      -> the spectrum of every ring is interpolated in mu (ring_intensity) and its CCF
                                       is computed (one CCF per ring).
                    In both cases the Doppler shift of the rotation is added later, by StarSim, cell by cell.
    """

    def __init__(self, N_rings, high_res_flux, RVs, wvm, fm, convective_blueshift=0, phoenix_resolution=False,
                 mu_ratio=None, normalization=False, instrument=None, blaze_function=None,
                 instrumental_efficiency=None, limb_lim=None, limb_extrapolation='linear', interpolation='ccf'):
        if instrument is not None:
            from .spectrum_utils import INSTRUMENT_RESOLUTION
            if isinstance(instrument, str):
                if instrument not in INSTRUMENT_RESOLUTION:
                    raise ValueError('unknown instrument %r: give one of %s, or the resolving power R as a number'
                                     % (instrument, sorted(INSTRUMENT_RESOLUTION)))
            else:
                instrument = _checks.number('instrument (resolving power)', instrument, positive=True)
        if blaze_function is not None and instrument is None:
            raise ValueError('blaze_function needs the instrument (its resolution): instrument=\'HARPS-N\', ... or R')
        if instrumental_efficiency is not None and blaze_function is None:
            raise ValueError('instrumental_efficiency needs blaze_function (the efficiency multiplies the blaze of '
                             'every order)')
        #checks of the arguments (the CCF kernels do not check their indices)
        N_rings = _checks.number('N_rings', N_rings, integer=True, low=1)
        _checks.spectra('high_res_flux', high_res_flux, min_mu=1)
        RVs = _checks.array1d('RVs', RVs, min_size=3, increasing=True)
        wvm = _checks.array1d('wvm', wvm)
        fm = _checks.array1d('fm', fm)
        if len(wvm) != len(fm):
            raise ValueError('wvm (%d lines) and fm (%d weights) must have the same length' % (len(wvm), len(fm)))
        convective_blueshift = _checks.number('convective_blueshift', convective_blueshift)
        if mu_ratio is not None:
            mu_ratio = _checks.array1d('mu_ratio', mu_ratio)
            if len(mu_ratio) != N_rings:
                raise ValueError('mu_ratio must have one value per ring (%d), got %d' % (N_rings, len(mu_ratio)))
            if np.any(mu_ratio < 0):
                raise ValueError('mu_ratio (brightness of the rings) must be >= 0')
        if limb_extrapolation not in ('constant', 'linear'):
            raise ValueError("limb_extrapolation must be 'constant' or 'linear', got %r" % (limb_extrapolation,))
        if limb_lim is not None:
            limb_lim = _checks.number('limb_lim', limb_lim, low=0.0, high=1.0)
        wv_check = np.asarray(next(iter(high_res_flux.values()))['wav'], dtype=np.float64)
        if phoenix_resolution and blaze_function is None and not _checks.is_phoenix_grid(wv_check):
            raise ValueError('phoenix_resolution=True needs spectra on the wavelength grid of the Phoenix HiRes models '
                             '(the CCF finds the pixels from its steps); these wavelengths are not: use '
                             'phoenix_resolution=False')
        if blaze_function is None:                     # CCF of the whole spectrum
            _checks.mask_in_spectrum(wvm, RVs, wv_check)
        self.N_rings = N_rings
        self.high_res_flux = high_res_flux
        self.rvs = np.asarray(RVs, dtype=np.float64)
        self.wvm = np.asarray(wvm, dtype=np.float64)
        self.fm = np.asarray(fm, dtype=np.float64)
        self.convective_blueshift = convective_blueshift
        self.phoenix_resolution = phoenix_resolution
        self.mu_ratio = mu_ratio
        if normalization not in (False, True, 'none'):
            raise ValueError("normalization must be False, True or 'none', got %r" % (normalization,))
        self.normalization = normalization
        self._ccf_norm = 2 if normalization == 'none' else bool(normalization)     # code of cross_correlation_mask
        self.instrument = instrument
        self.blaze_function = blaze_function
        self.instrumental_efficiency = instrumental_efficiency
        self.limb_extrapolation = limb_extrapolation
        if interpolation not in ('ccf', 'spectrum'):
            raise ValueError("interpolation must be 'ccf' or 'spectrum', got %r" % (interpolation,))
        self.interpolation = interpolation
        self.ccf_mu = None             # CCF of the spectrum at every tabulated mu (interpolation='ccf')

        # spectral quantities -- built only once.  The mu keys are sorted: index 0 = smallest mu, last = disc centre
        keys = sorted(self.high_res_flux.keys())
        if mu_ratio is None and len(keys) < 2:
            raise ValueError("Need at least two mu angles (or a single spectrum at mu=1.0 together with mu_ratio)")
        self._acd = np.asarray(keys, dtype=np.float64)
        self._check_centre_spectrum()
        self._flux_i = np.asarray([self.high_res_flux[key]['intensity'] for key in keys], dtype=np.float64)
        wv = np.asarray(self.high_res_flux[keys[0]]['wav'], dtype=np.float64)
        self._wv = wv.copy() if wv.flags.writeable else wv     # read-only arrays (spectrum_utils) are not copied
        self.limb_lim = self._acd[0] if limb_lim is None else float(limb_lim)

        # projected angle of every ring
        self.amu = np.asarray(nbspectra.generate_grid_coordinates_nb(N_rings)[3], dtype=np.float64)

        self._orders = None            # per-order data, built the first time it is needed
        self._orders_in_sum = 1        # number of orders summed in the CCF (instrument), set by _ccf_by_order
        self._tools = None             # add_resol, imported the first time an instrument is used
        self._ccf_mode = None          # 'rings' (every ring from its spectrum) or 'centre' (disc centre only), set by step 1
        self.ccf_centre = None         # CCF of the disc-centre spectrum ('centre' mode only)
        self.ccf_shape = None          # CCF of every ring before the mu_ratio scaling (its bisector is the ring's one)
        self.ccf_rings = None          # result of step 1
        self.rvs_ring = None           # result of step 2

    # ---------------------------------------------------------------- properties
    @property
    def wv(self):
        """Wavelength grid of the spectra [A]."""
        return self._wv

    @property
    def flux_i(self):
        """Spectra at the tabulated mu (n_mu, n_wavelengths), in the order of acd."""
        return self._flux_i

    @property
    def acd(self):
        """Tabulated mu angles of the spectra, ascending (index 0 = closest to the limb)."""
        return self._acd

    # ---------------------------------------------------------------- step 1: general CCF
    def _check_centre_spectrum(self):
        """With mu_ratio and a single spectrum, the rings share the disc-centre one: it must be at mu = 1.0."""
        if self.mu_ratio is None or len(self._acd) >= 2:
            return
        if not np.isclose(self._acd[0], 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("mu_ratio with a single spectrum: it must be the disc centre, at mu=1.0; it is at "
                             "mu=%s" % np.round(self._acd, 4))

    def ring_spectrum(self, i):
        """Spectrum of ring i, interpolated / extrapolated in mu (needs >= 2 mu angles)."""
        if len(self._acd) < 2:
            raise RuntimeError("only the disc-centre spectrum is given: the rings do not have their own spectrum")
        return ring_intensity(self.amu[i], self._acd, self._flux_i, self.limb_lim, self.limb_extrapolation)

    def compute_ccf(self):
        """Step 1. CCF of every ring in the rest frame (not weighted by area): array (N_rings, len(RVs)).
        Whole spectrum if instrument is None, by order otherwise.
        >= 2 mu angles : every ring has its own CCF ('rings' mode), times mu_ratio[i] if mu_ratio is given:
                         interpolated in mu between the CCFs of the tabulated spectra (interpolation='ccf'), or the
                         CCF of the spectrum interpolated at the mu of the ring (interpolation='spectrum').
        one spectrum   : CCF of the disc-centre spectrum times mu_ratio[i] ('centre' mode; the disc-centre CCF alone
                         is kept in self.ccf_centre).
        The CCF of every ring before the mu_ratio scaling is kept in self.ccf_shape."""
        ratio = None
        if self.mu_ratio is not None:
            ratio = np.asarray(self.mu_ratio, dtype=np.float64)
            if ratio.shape != (self.N_rings,):
                raise ValueError("mu_ratio must have one value per ring (%d), got shape %s"
                                 % (self.N_rings, ratio.shape))
        if len(self._acd) >= 2:
            if self.interpolation == 'ccf':
                #one CCF per tabulated mu, then interpolation in mu (before the Doppler shifts of the rotation)
                self.ccf_mu = np.array([self._ccf_of_spectrum(f) for f in self._flux_i])
                shape = np.array([ring_ccf(self.amu[i], self._acd, self.ccf_mu, self.limb_lim, self.limb_extrapolation)
                                  for i in range(self.N_rings)])
            else:
                shape = np.array([self._ccf_of_spectrum(self.ring_spectrum(i)) for i in range(self.N_rings)])
            self._ccf_mode, self.ccf_centre = 'rings', None
        else:
            self._check_centre_spectrum()
            self.ccf_centre = self._ccf_of_spectrum(self._flux_i[0])
            shape = np.tile(self.ccf_centre, (self.N_rings, 1))
            self._ccf_mode = 'centre'
        self.ccf_shape = shape
        self.ccf_rings = shape if ratio is None else ratio[:, None] * shape
        return self.ccf_rings

    def _ccf_of_spectrum(self, flux):
        """CCF of one spectrum on self.rvs: whole spectrum (no instrument, or an instrument without blaze: the spectrum
        degraded to its resolution first) or summed over the orders of the instrument."""
        if self.blaze_function is not None:
            return self._ccf_by_order(flux)
        flux = np.asarray(flux, dtype=np.float64)
        if self.instrument is not None:
            flux = np.asarray(self._get_tools()(self._wv, flux, self.instrument), dtype=np.float64)
        return nbspectra.cross_correlation_mask(self.rvs, self._wv, flux, self.wvm, self.fm, self.phoenix_resolution,
                                                self._ccf_norm)

    def _get_orders(self):
        """Wavelength grid, blaze*efficiency and mask lines of every order (built once)."""
        if self._orders is not None:
            return self._orders
        wvb, blaze = self.blaze_function
        eff = self.instrumental_efficiency
        orders = []
        for j in range(len(blaze)):
            wj = np.asarray(wvb[j], dtype=np.float64)
            bj = np.asarray(blaze[j], dtype=np.float64)
            if len(wj) != len(bj):
                raise ValueError("order %d: wavelength and blaze have different lengths" % j)
            if not wj[0] < self._wv[-1]:                 # order starts beyond the end of the spectrum
                continue
            if eff is None:
                ej = 1.0
            elif callable(eff):
                ej = np.asarray(eff(wj), dtype=np.float64)
            else:
                wl, ef = (np.asarray(x, dtype=np.float64) for x in eff)
                srt = np.argsort(wl)
                wl, ef = wl[srt], ef[srt]
                if wj[0] < wl[0] or wj[-1] > wl[-1]:
                    warnings.warn("order %d (%.1f-%.1f A) is partly outside the efficiency table (%.1f-%.1f A): "
                                  "the efficiency is kept constant outside" % (j, wj[0], wj[-1], wl[0], wl[-1]))
                ej = np.interp(wj, wl, ef)
            mask = (self.wvm >= wj[0]) & (self.wvm <= wj[-1])
            orders.append({'wav': wj, 'edges': bin_edges(wvb[j]), 'b': bj * ej, 'wvm': self.wvm[mask],
                           'fm': self.fm[mask]})
        self._orders = orders
        return orders

    def _get_tools(self):
        """add_resol (imported only when an instrument is used: scipy and astropy.convolution)."""
        if self._tools is None:
            from . import spectrum_utils
            self._tools = spectrum_utils.add_resol
        return self._tools

    def _ccf_by_order(self, flux):
        """Degrade -> resample on every order -> multiply by blaze*efficiency -> CCF of the order -> sum of the orders.
        The resampling conserves the flux (nbspectra.flux_conserving_resample_nb: the numbers of
        specutils.FluxConservingResampler, which the old StarSim used); the parts of an order beyond the spectrum are
        NaN and their CCF is left out of the sum (nansum)."""
        add_resol = self._get_tools()
        convolved_flux = np.asarray(add_resol(self._wv, flux, self.instrument), dtype=np.float64)
        edges = nbspectra.bin_edges_nb(self._wv)
        orders = self._get_orders()
        ccf_orders = np.zeros([len(orders), len(self.rvs)])
        for k, o in enumerate(orders):
            flux_order = o['b'] * nbspectra.flux_conserving_resample_nb(edges, convolved_flux, o['edges'])
            # the order grid is not the Phoenix grid: generic branch (phoenix_resolution=False)
            ccf_orders[k, :] = nbspectra.cross_correlation_mask(
                self.rvs, o['wav'], np.ascontiguousarray(flux_order, dtype=np.float64), o['wvm'], o['fm'],
                False, self._ccf_norm)
        self._orders_in_sum = int(np.sum(np.all(np.isfinite(ccf_orders), axis=1)))    # orders without NaN
        return np.nansum(ccf_orders, axis=0)

    # ---------------------------------------------------------------- step 2: RV treatment
    def rv_treatment(self, bisector='dumusque', kind_interp='cubic'):
        """Step 2. Velocity axis on which the CCF of every ring is placed: array (N_rings, len(RVs)).

        'centre' mode (one spectrum, all rings are the disc-centre CCF):
            the bisector of that CCF is removed,  rv - bis(ccf),  and then a bisector is added.
        'rings' mode (every ring has its own spectrum, so its own bisector; with or without mu_ratio):
            nothing is removed, a bisector is added.
        Added bisector:  + p_mu(ccf) * 1000 * convective_blueshift    (p in km/s, ccf = CCF of the ring before the
        mu_ratio scaling, self.ccf_shape, so that the amplitude scaling of the ring does not enter). With an
        instrument the CCF is the sum of the CCFs of the orders (heights up to ~the number of orders): the bisector is
        evaluated on their mean (sum / number of orders in it), whose heights are those of one CCF, as the polynomial
        expects.
        bisector : 'dumusque' -> the same polynomial for every mu;
                   function f(mu) -> p(ccf)  -> any other, e.g. mu-dependent (such as the old cifist_coeff_interpolate);
                   None -> no bisector added.
        kind_interp : interpolation of the CCF bisector that is removed.
        Uses the CCFs of step 1, so it can be re-run with other options without recomputing them."""
        if self.ccf_rings is None:
            raise RuntimeError("run compute_ccf() first")
        rv = self.rvs
        rvs_ring = np.tile(rv, (len(self.ccf_rings), 1))
        shape = self.ccf_shape                       # CCF each ring's bisector refers to (without the mu_ratio scaling)

        if self._ccf_mode == 'centre':
            # scaling a CCF does not change its bisector, so the one of the disc centre is removed from every ring
            rvs_ring -= bisector_fit(rv, self.ccf_centre, kind_interp)(self.ccf_centre)[None, :]

        if bisector is not None and self.convective_blueshift != 0:
            if isinstance(bisector, str):
                if bisector != 'dumusque':
                    raise ValueError("bisector must be 'dumusque', None or a function f(mu) -> p(ccf)")
                bisector = dumusque_bisector(self._ccf_norm == 1)      # True (also numpy True) -> the [0, 1] polynomial
            heights = shape if self.blaze_function is None else shape / max(self._orders_in_sum, 1)
            for i in range(len(rvs_ring)):
                rvs_ring[i, :] += bisector(self.amu[i])(heights[i]) * 1000 * self.convective_blueshift
        self.rvs_ring = rvs_ring
        return rvs_ring

    # ---------------------------------------------------------------- both steps
    def built_grid(self, **rv_options):
        """Step 1 then step 2 (rv_options go to rv_treatment). Sets ccf_rings and rvs_ring."""
        self.compute_ccf()
        self.rv_treatment(**rv_options)