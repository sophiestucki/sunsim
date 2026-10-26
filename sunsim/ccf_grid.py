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


def _mu_factor(mu_scaling):
    """Factor of the bisector at mu: None -> 1; (a, b) -> a + b mu (linear); (c0, c1, ..., cn) -> c0 + c1 mu + ... +
    cn mu^n (custom polynomial, increasing powers); a function f(mu) -> any law."""
    if mu_scaling is None:
        return None
    if callable(mu_scaling):
        return mu_scaling
    coeffs = np.asarray(mu_scaling, dtype=np.float64)
    if coeffs.ndim != 1 or len(coeffs) == 0 or not np.all(np.isfinite(coeffs)):
        raise ValueError('mu_scaling must be None, a function f(mu) or finite coefficients (c0, c1, ...) of '
                         'c0 + c1 mu + ..., got %r' % (mu_scaling,))
    return np.polynomial.Polynomial(coeffs)


def dumusque_bisector(normalization=True, mu_scaling=None):
    """Bisector of Dumusque (2014), added by rv_treatment: returns f(mu) -> p(ccf), p in km/s (np.poly1d).
    normalization : True for CCFs in [0, 1] (CCF_grid(normalization=True)), False for the raw CCF heights.
    mu_scaling    : None -> the same bisector at every mu; (a, b) -> the bisector at mu is multiplied by a + b mu
                    (e.g. (0, 1): proportional to mu, zero at the limb); custom: (c0, c1, ..., cn) -> multiplied by
                    c0 + c1 mu + ... + cn mu^n (e.g. the mu law of the faculae, (1.571, -4.512, 17.404, -13.463)); or
                    any function g(mu) -> factor.
    Any other bisector can be given to rv_treatment as a function f(mu) -> p(ccf) (p in km/s)."""
    p = np.poly1d(np.array([-1.51773453,  3.52774949, -3.18794328,  1.22541774,  -0.22479665+0.4]) if normalization else np.array([ 69.72233642, -111.07077255,   66.13347937,  -19.34395212,    2.85797555, 0.16536442]))
    factor = _mu_factor(mu_scaling)
    if factor is None:
        return lambda mu: p
    return lambda mu: p * float(factor(mu))


#CIFIST bisectors (from 3D hydrodynamic CIFIST model atmospheres, as in the old StarSim): polynomials at mu = 1.0, 0.9,
#..., 0.1 (coefficients, highest power first)
_CIFIST_MU = np.arange(1, 0.0, -0.1)
_CIFIST_COEFFS = np.array([
    [-3.51974861, 11.1702017, -13.22368296, 6.67694456, -0.63201573, -0.44695616, -0.36838495],
    [-4.05903967, 13.21901003, -16.47215949, 9.51023171, -2.13104764, -0.05153799, -0.36973749],
    [-3.92153131, 12.76694663, -15.96958217, 9.39599116, -2.34394028, 0.12546611, -0.42092905],
    [-3.81892968, 12.62209118, -16.06973368, 9.71487198, -2.61439945, 0.25356088, -0.43310756],
    [-5.37213406, 17.6604689, -22.52477323, 13.91461247, -4.13186181, 0.60271171, -0.46427559],
    [-6.35351933, 20.92046705, -26.83933359, 16.86220487, -5.28285592, 0.90643187, -0.47696283],
    [-7.67270144, 25.60866105, -33.4381214, 21.58855269, -7.1527039, 1.35990694, -0.48001707],
    [-9.24152009, 31.09337903, -41.07410957, 27.04196984, -9.32910982, 1.89291407, -0.455407],
    [-11.62006536, 39.30962189, -52.38161244, 34.98243089, -12.40650704, 2.57940618, -0.37337442],
    [-14.14768805, 47.9566719, -64.20294114, 43.23156971, -15.57423374, 3.13318175, -0.14451226]])


def _cifist_poly(mu):
    """CIFIST bisector at mu: the polynomials of the two tabulated mu around it interpolated linearly (extrapolated
    below mu = 0.1), as cifist_coeff_interpolate of the old StarSim."""
    amv, cxm = _CIFIST_MU, _CIFIST_COEFFS
    if mu <= 0.1:
        c = cxm[9] + (cxm[8] - cxm[9]) * (mu - amv[9]) / (amv[8] - amv[9])
    else:
        low = np.max(amv[amv < mu])
        i_low, i_upp = np.where(amv == low)[0][0], np.where(amv == np.min(amv[amv >= mu]))[0][0]
        c = cxm[i_low] + (cxm[i_upp] - cxm[i_low]) * (mu - amv[i_low]) / (amv[i_upp] - amv[i_low])
    return np.poly1d(c)


def cifist_bisector(mu_scaling=None):
    """CIFIST bisector (3D hydrodynamic models of the solar photosphere, tabulated at mu = 0.1, ..., 1), added by
    rv_treatment: returns f(mu) -> p(ccf), p in km/s (np.poly1d), the bisector of the mu of every ring (CCF heights in
    [0, 1]).
    mu_scaling : None -> the CIFIST bisectors as they are; (a, b) -> the bisector at mu is also multiplied by a + b mu;
                 (c0, c1, ..., cn) -> by c0 + c1 mu + ... + cn mu^n; or any function g(mu) -> factor."""
    factor = _mu_factor(mu_scaling)
    if factor is None:
        return _cifist_poly
    return lambda mu: _cifist_poly(mu) * float(factor(mu))


# =============================================================================================
# the grid
# =============================================================================================
class CCF_grid(object):
    """
    CCF of every ring of the disc for one kind of surface (quiet photosphere, spot or facula).

    N_rings       : number of rings of the grid of the disc (the same as in StarSim)
    high_res_flux : {mu: {'wav': wavelengths [A], 'intensity': flux}}   (any key order, it is sorted)
    RVs           : velocity grid of the CCF [m/s]
    wvm, fm       : mask lines [A] and weights (spectrum_utils.mask_weights: air wavelengths by default)
    medium        : 'air' (default) or 'vacuum': the wavelengths in which the CCF is computed. The spectra, the mask
                    and the orders of the instrument are all converted to this medium (spectrum_utils.vacuum2air /
                    air2vacuum) when they are given in the other one. A mask and a spectrum in different media would be
                    shifted by ~80 km/s. The choice of the medium itself only changes how the pixels sample the lines:
                    the RV, FWHM and BIS variations differ by ~1e-4 of their amplitude between air and vacuum (and
                    their absolute values by < 1 m/s in RV).
    spectrum_medium : medium of the wavelengths of high_res_flux: 'vacuum' (default: Phoenix and MURaM spectra) or 'air'
    mask_medium   : medium of wvm: 'air' (default: the output of mask_weights, and HARPS-like masks) or 'vacuum'
    orders_medium : medium of the wavelengths of blaze_function: 'air' (default: HARPS / HARPS-N pipelines) or 'vacuum'
    convective_blueshift : factor that modulates the bisector added in step 2 (rv_treatment)
    phoenix_resolution   : True if the spectra are on the Phoenix wavelength grid (fast index in
                    nbspectra.cross_correlation_mask).  Only used for the whole-spectrum CCF: after resampling on the
                    orders the grid is not the Phoenix one, so the order CCFs always use the generic branch. When the
                    spectra are converted to another medium (air) the grid is not the Phoenix one either: the generic
                    branch is used (the same CCF, a bit slower).
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
                    An instrument.Instrument object (e.g. Instrument.harps()) -> its resolution, orders, efficiency
                    and orders_medium (blaze_function, instrumental_efficiency and orders_medium must then be left out).
                    'HARPS-N', 'HARPS', 'EXPRESS', 'NEID' or a resolving power R (a number) -> the same as
                    Instrument(instrument, blaze_function, instrumental_efficiency, orders_medium). Every spectrum is
                    degraded to the resolution of the instrument (spectrum_utils.add_resol), then:
                      * without orders: CCF of the whole degraded spectrum (as StarSim v4 with an instrument);
                      * with orders: resampled on the wavelength grid of every order, multiplied by
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
                 instrumental_efficiency=None, limb_lim=None, limb_extrapolation='linear', interpolation='ccf',
                 medium='air', spectrum_medium='vacuum', mask_medium='air', orders_medium='air'):
        for name, value in (('medium', medium), ('spectrum_medium', spectrum_medium), ('mask_medium', mask_medium),
                            ('orders_medium', orders_medium)):
            if value not in ('air', 'vacuum'):
                raise ValueError("%s must be 'air' or 'vacuum', got %r" % (name, value))
        from .spectrum_utils import vacuum2air, air2vacuum
        to_medium = lambda w, given: np.asarray(w, dtype=np.float64) if given == medium else \
            np.asarray((vacuum2air if medium == 'air' else air2vacuum)(np.asarray(w, dtype=np.float64)))
        from .instrument import Instrument
        if isinstance(instrument, Instrument):
            if blaze_function is not None or instrumental_efficiency is not None:
                raise ValueError('instrument is an Instrument object: its orders and efficiency are used, leave '
                                 'blaze_function and instrumental_efficiency out')
            inst = instrument
            orders_medium = inst.orders_medium
            blaze_function = (inst.wave_orders, inst.blaze_orders) if inst.has_orders else None
            instrumental_efficiency = inst.efficiency
        else:
            if blaze_function is not None and instrument is None:
                raise ValueError('blaze_function needs the instrument (its resolution): instrument=\'HARPS-N\', ... or R')
            if instrumental_efficiency is not None and blaze_function is None:
                raise ValueError('instrumental_efficiency needs blaze_function (the efficiency multiplies the blaze of '
                                 'every order)')
            inst = None if instrument is None else Instrument(instrument, blaze_function, instrumental_efficiency,
                                                              orders_medium)
            if inst is not None:
                instrument = inst.resolution
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
        wvm = to_medium(wvm, mask_medium)
        wv_check = np.asarray(next(iter(high_res_flux.values()))['wav'], dtype=np.float64)
        if phoenix_resolution and blaze_function is None and not _checks.is_phoenix_grid(wv_check):
            raise ValueError('phoenix_resolution=True needs spectra on the wavelength grid of the Phoenix HiRes models '
                             '(the CCF finds the pixels from its steps); these wavelengths are not: use '
                             'phoenix_resolution=False')
        if blaze_function is None:                     # CCF of the whole spectrum
            _checks.mask_in_spectrum(wvm, RVs, to_medium(wv_check, spectrum_medium))
        self.N_rings = N_rings
        self.high_res_flux = high_res_flux
        self.rvs = np.asarray(RVs, dtype=np.float64)
        self.wvm = np.asarray(wvm, dtype=np.float64)
        self.fm = np.asarray(fm, dtype=np.float64)
        self.convective_blueshift = convective_blueshift
        self.medium, self.spectrum_medium, self.mask_medium, self.orders_medium = medium, spectrum_medium, mask_medium, orders_medium
        self._to_medium = to_medium
        #the Phoenix grid is in vacuum: converted to air it is not the Phoenix grid anymore (generic branch of the CCF)
        self.phoenix_resolution = bool(phoenix_resolution) and spectrum_medium == medium
        self.mu_ratio = mu_ratio
        if normalization not in (False, True, 'none'):
            raise ValueError("normalization must be False, True or 'none', got %r" % (normalization,))
        self.normalization = normalization
        self._ccf_norm = 2 if normalization == 'none' else bool(normalization)     # code of cross_correlation_mask
        self.instrument = instrument
        self._instrument = inst          # instrument.Instrument (None without instrument): resolution, orders, efficiency
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
        if spectrum_medium != medium:
            self._wv = self._to_medium(wv, spectrum_medium)      # wavelengths of the spectra in the medium of the CCF
        else:
            self._wv = wv.copy() if wv.flags.writeable else wv     # read-only arrays (spectrum_utils) are not copied
        self.limb_lim = self._acd[0] if limb_lim is None else float(limb_lim)

        # projected angle of every ring
        self.amu = np.asarray(nbspectra.generate_grid_coordinates_nb(N_rings)[3], dtype=np.float64)

        self._orders = None            # per-order data, built the first time it is needed
        self._orders_in_sum = 1        # number of orders summed in the CCF (instrument), set by _ccf_by_order
        self._ccf_mode = None          # 'rings' (every ring from its spectrum) or 'centre' (disc centre only), set by step 1
        self.ccf_centre = None         # CCF of the disc-centre spectrum ('centre' mode only)
        self.ccf_shape = None          # CCF of every ring before the mu_ratio scaling (its bisector is the ring's one)
        self.ccf_rings = None          # result of step 1
        self.rvs_ring = None           # result of step 2

    # ---------------------------------------------------------------- properties
    @property
    def wv(self):
        """Wavelength grid of the spectra [A], in the medium of the CCF (medium)."""
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
        if self._instrument is not None:
            flux = self._instrument.degrade(self._wv, flux)
        return nbspectra.cross_correlation_mask(self.rvs, self._wv, flux, self.wvm, self.fm, self.phoenix_resolution,
                                                self._ccf_norm)

    def _get_orders(self):
        """Wavelength grid, blaze*efficiency (Instrument.order_table, in the medium of the CCF) and mask lines of every
        order that starts below the end of the spectra (built once)."""
        if self._orders is not None:
            return self._orders
        orders = []
        for o in self._instrument.order_table(self.medium, self._wv[-1]):
            wj = o['wav']
            mask = (self.wvm >= wj[0]) & (self.wvm <= wj[-1])
            orders.append({'wav': wj, 'edges': o['edges'], 'b': o['b'], 'wvm': self.wvm[mask], 'fm': self.fm[mask]})
        self._orders = orders
        return orders

    def _ccf_by_order(self, flux):
        """Degrade -> resample on every order -> multiply by blaze*efficiency -> CCF of the order -> sum of the orders.
        The resampling conserves the flux (nbspectra.flux_conserving_resample_nb: the numbers of
        specutils.FluxConservingResampler, which the old StarSim used); the parts of an order beyond the spectrum are
        NaN and their CCF is left out of the sum (nansum)."""
        convolved_flux = self._instrument.degrade(self._wv, flux)
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
        bisector : 'dumusque' -> dumusque_bisector(), the same polynomial for every mu;
                   'cifist'   -> cifist_bisector(), the bisector of the mu of every ring;
                   function f(mu) -> p(ccf)  -> any other, e.g. dumusque_bisector(mu_scaling=(a, b)) or
                   cifist_bisector(mu_scaling=...) (linear mu dependence), or your own;
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
                if bisector == 'dumusque':
                    bisector = dumusque_bisector(self._ccf_norm == 1)      # True (also numpy True) -> the [0, 1] polynomial
                elif bisector == 'cifist':
                    bisector = cifist_bisector()
                else:
                    raise ValueError("bisector must be 'dumusque', 'cifist', None or a function f(mu) -> p(ccf)")
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