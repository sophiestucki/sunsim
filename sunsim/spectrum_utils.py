"""
spectrum_utils: tools to prepare the spectra and masks given to flux_grid and CCF_grid.

* general            : black_body, vacuum2air / air2vacuum (the CCFs are computed in air by default: CCF_grid(medium=))
* faculae            : limb brightening of facular spectra that have none (e.g. Phoenix): facula_dT,
                       limb_brightening_factor, add_limb_brightening, FACULA_DT_LAWS
* limb darkening     : limb_darkened_spectra (spectra at many mu from one spectrum at mu = 1 and a parametric or custom
                       law, LIMB_DARKENING_LAWS)
* continuum          : flatten_spectra (every mu divided by the continuum of the disc-centre spectrum)
* Phoenix models     : interpolate_phoenix_mu (SPECINT, specific intensity at many mu, low resolution) and
                       interpolate_phoenix (HiRes, disc-integrated, high resolution), interpolated in temperature, logg
                       and metallicity; helpers phoenix_grid, phoenix_weights, phoenix_continuum
* instruments        : add_resol (degrade a spectrum to the resolution of an instrument)
* CCF masks          : mask_weights

All the spectra are returned in the format of SunSim, {mu: {'wav': wavelengths [A], 'intensity': intensity}}.
"""
import os
import numpy as np
from . import _checks
from pathlib import Path
from functools import lru_cache
from astropy.io import fits
#matplotlib (plots), scipy.interpolate and astropy.convolution (add_resol) are imported where they are used:
#importing them here costs ~2 s and ~110 MB in every session







########################################################################################
########################################################################################
#                                GENERAL FUNCTIONS                                     #
########################################################################################
########################################################################################

def black_body(wv,T):
    """Planck function B_lambda(T) [erg/s/cm^2/cm/sr] at the wavelengths wv [A] and temperature T [K]."""
    c = 2.99792458e10 #speed of light in cm/s
    k = 1.380658e-16  #boltzmann constant
    h = 6.6260755e-27 #planck
    w=wv*1e-8 #Angstrom to cm
    bb=2*h*c**2*w**(-5)*(np.exp(h*c/k/T/w)-1)**(-1)
    return bb

def vacuum2air(wv): #wv in angstroms
	"""Vacuum wavelengths [A] -> air wavelengths [A] (refractive index of the air, two-term dispersion formula)."""
	wv=wv*1e-4 #A to micrometer
	a=0
	b1=5.792105e-2
	b2=1.67917e-3
	c1=238.0185
	c2=57.362

	n=1+a+b1/(c1-(1/wv**2))+b2/(c2-(1/wv**2))

	w=(wv/n)*1e4 #to Angstroms
	return w

def air2vacuum(wv): #wv in angstroms
    """Air wavelengths [A] -> vacuum wavelengths [A] (same refractive index as vacuum2air, evaluated at the air
    wavelength: an approximation of the inverse)."""
    wv=wv*1e-4 #A to micrometer
    a=0
    b1=5.792105e-2
    b2=1.67917e-3
    c1=238.0185
    c2=57.362

    n=1+a+b1/(c1-(1/wv**2))+b2/(c2-(1/wv**2))

    w=(wv*n)*1e4 #to Angstroms
    return w


########################################################################################
#                       LIMB BRIGHTENING OF THE FACULAE                                #
########################################################################################

#Temperature excess of the faculae with respect to the photosphere as a function of mu, dT(mu) = c0 + c1*mu + c2*mu^2 [K]
FACULA_DT_LAWS = {
    'meunier': (250.9, -407.7, 190.9),     #Meunier et al. (2010): dT = 250.9 K at the limb, 34.1 K at the disc centre
    'starsim': (155.0, -290.0, 128.0),     #law used by default in StarSim (meunier=0): 155 K at the limb, -7 K at the centre
}


def facula_dT(mu, law='starsim', scale=1.0):
    """Temperature excess of a facula with respect to the photosphere [K] at the projected angles mu.
    law   : name of a law of FACULA_DT_LAWS ('meunier', 'starsim'), polynomial coefficients (c0, c1, c2, ...) in
            increasing powers of mu, or a function dT(mu) [K]
    scale : the law is multiplied by this factor (e.g. facula_T_contrast / 250.9 to rescale the Meunier law)"""
    mu = np.asarray(mu, dtype=np.float64)
    if callable(law):
        dT = law(mu)
    else:
        coeffs = FACULA_DT_LAWS[law] if isinstance(law, str) else law
        dT = np.polynomial.polynomial.polyval(mu, np.asarray(coeffs, dtype=np.float64))
    return scale * np.asarray(dT, dtype=np.float64)


def limb_brightening_factor(mu, T_photosphere, law='starsim', scale=1.0):
    """Bolometric limb brightening of the faculae, ((T_photosphere + dT(mu)) / T_photosphere)^4, at the angles mu
    (dT from facula_dT). It does not depend on the wavelength."""
    return ((T_photosphere + facula_dT(mu, law, scale)) / T_photosphere)**4


def add_limb_brightening(flux, T_photosphere, law='starsim', scale=1.0):
    """Facula spectra with the bolometric limb brightening: the spectrum at every mu of a {mu: {'wav', 'intensity'}}
    dictionary is multiplied by limb_brightening_factor(mu). Returns a new dictionary (flux is not modified).
    For spectra without their own facular centre-to-limb variation, e.g. Phoenix spectra at T_photosphere +
    facula_T_contrast (in the old StarSim this factor was applied to the rings when computing the time series)."""
    out = {}
    for mu, d in flux.items():
        new = dict(d)
        new['intensity'] = np.asarray(d['intensity'], dtype=np.float64) * limb_brightening_factor(float(mu), T_photosphere, law, scale)
        out[mu] = new
    return out


########################################################################################
#                       LIMB DARKENING LAWS, CONTINUUM FLATTENING                       #
########################################################################################

#Parametric limb-darkening laws: I(mu) / I(1) as a function of mu and of the coefficients. The coefficients can be
#numbers or arrays over the wavelengths (wavelength-dependent laws).
LIMB_DARKENING_LAWS = {
    'uniform':     lambda mu: np.ones_like(mu),                                          # no limb darkening
    'linear':      lambda mu, u: 1 - u * (1 - mu),
    'quadratic':   lambda mu, a, b: 1 - a * (1 - mu) - b * (1 - mu)**2,
    'square-root': lambda mu, c, d: 1 - c * (1 - mu) - d * (1 - np.sqrt(mu)),
    'logarithmic': lambda mu, e, f: 1 - e * (1 - mu) - f * mu * np.log(mu),
    'power-2':     lambda mu, c, alpha: 1 - c * (1 - mu**alpha),
    'nonlinear':   lambda mu, a1, a2, a3, a4: 1 - sum(a * (1 - mu**(k / 2)) for k, a in enumerate((a1, a2, a3, a4), 1)),
}
#mu of the spectra built by limb_darkened_spectra (dense near the limb, where the laws change fast)
DEFAULT_LD_MU = np.round(np.r_[0.01, 0.025, 0.05, 0.075, np.linspace(0.1, 1.0, 19)], 6)


def limb_darkened_spectra(wavelength, intensity, law='quadratic', coefficients=(), mu=None):
    """Spectra at several mu built from ONE spectrum at mu = 1 (disc centre) and a limb-darkening law, in the format of
    flux_grid / CCF_grid: {mu: {'wav': wavelength, 'intensity': I(1, wavelength) * law(mu, *coefficients)}}.

    wavelength, intensity : the spectrum at mu = 1 (wavelengths [A]; intensity in any unit)
    law          : a name of LIMB_DARKENING_LAWS, I(mu)/I(1) =
                       'uniform'      1
                       'linear'       1 - u (1 - mu)                                    coefficients (u,)
                       'quadratic'    1 - a (1 - mu) - b (1 - mu)^2                     (a, b)
                       'square-root'  1 - c (1 - mu) - d (1 - sqrt(mu))                 (c, d)
                       'logarithmic'  1 - e (1 - mu) - f mu ln(mu)                      (e, f)
                       'power-2'      1 - c (1 - mu^alpha)                              (c, alpha)
                       'nonlinear'    1 - sum_k a_k (1 - mu^(k/2)), k = 1..4 (Claret)   (a1, a2, a3, a4)
                   or any function law(mu, *coefficients) -> I(mu)/I(1) (a custom law; mu is a number)
    coefficients : the coefficients of the law; each one a number or an array over the wavelengths (wavelength-dependent
                   limb darkening, e.g. interpolated from tables)
    mu           : the mu of the output spectra (default DEFAULT_LD_MU: 23 values from 0.01 to 1, denser near the
                   limb); 1 is always included. flux_grid / CCF_grid interpolate between them.
    The wavelength array is shared by all the mu (not copied)."""
    wav = np.asarray(wavelength, dtype=np.float64)
    i1 = np.asarray(intensity, dtype=np.float64)
    if wav.ndim != 1 or i1.shape != wav.shape:
        raise ValueError('wavelength and intensity must be 1D arrays of the same length, got %s and %s' % (wav.shape, i1.shape))
    if callable(law):
        func = law
    elif law in LIMB_DARKENING_LAWS:
        func = LIMB_DARKENING_LAWS[law]
    else:
        raise ValueError('law must be one of %s or a function law(mu, *coefficients), got %r' % (list(LIMB_DARKENING_LAWS), law))
    if np.ndim(coefficients) == 0 or (isinstance(coefficients, np.ndarray) and coefficients.ndim == 1
                                      and len(coefficients) == len(wav) and len(wav) > 4):
        coefficients = [coefficients]               # one coefficient (a number, or an array over the wavelengths)
    coeffs = []
    for c in coefficients:
        c = np.asarray(c, dtype=np.float64)
        if c.ndim not in (0, 1) or (c.ndim == 1 and len(c) != len(wav)):
            raise ValueError('every coefficient must be a number or an array with one value per wavelength (%d)' % len(wav))
        coeffs.append(c)
    mus = np.unique(np.r_[DEFAULT_LD_MU if mu is None else np.atleast_1d(np.asarray(mu, dtype=np.float64)), 1.0])
    if np.any(mus <= 0) or np.any(mus > 1):
        raise ValueError('the mu must be in (0, 1], got %s' % mus)
    out = {}
    for m in mus:
        factor = np.asarray(func(float(m), *coeffs), dtype=np.float64)
        if factor.ndim not in (0, 1) or (factor.ndim == 1 and len(factor) != len(wav)) or not np.all(np.isfinite(factor)):
            raise ValueError('the law must give a finite number, or one value per wavelength, at mu = %g' % m)
        out[float(m)] = {'wav': wav, 'intensity': i1 * factor}
    return out


def flatten_spectra(spectra, nbins=20, deg=6, wavelength_lower_limit=None, wavelength_upper_limit=None):
    """Spectra at several mu flattened with the continuum of the disc-centre spectrum: every spectrum is divided by the
    SAME continuum, the one of the spectrum at mu = 1 (or the largest mu), so that the lines are normalised while the
    limb darkening (the ratio between the mu) is kept. Dividing every mu by its own continuum would remove it.

    spectra : {mu: {'wav', 'intensity'}} (e.g. interpolate_phoenix_mu, limb_darkened_spectra, MURaM spectra)
    nbins, deg : the continuum is phoenix_continuum: a polynomial of degree deg through the maximum of the flux in
                 nbins bins between wavelength_lower_limit and wavelength_upper_limit (default: the range of the
                 disc-centre spectrum)
    Returns (flattened spectra, continuum): a new dictionary in the same format, and the continuum as
    {'wav': wavelengths of the disc-centre spectrum, 'continuum': values}. Spectra on other wavelengths than the
    disc-centre one get the continuum interpolated on their wavelengths."""
    if not isinstance(spectra, dict) or not spectra:
        raise TypeError('spectra must be a dictionary {mu: {"wav": ..., "intensity": ...}}')
    mu_c = max(spectra, key=float)
    wv = np.asarray(spectra[mu_c]['wav'], dtype=np.float64)
    flux = np.asarray(spectra[mu_c]['intensity'], dtype=np.float64)
    lower = wv.min() if wavelength_lower_limit is None else float(wavelength_lower_limit)
    upper = wv.max() if wavelength_upper_limit is None else float(wavelength_upper_limit)
    continuum, _, _ = phoenix_continuum(wv, flux, lower, upper, nbins=nbins, deg=deg)
    out = {}
    for mu, d in spectra.items():
        w = np.asarray(d['wav'], dtype=np.float64)
        c = continuum if (w.shape == wv.shape and np.array_equal(w, wv)) else np.interp(w, wv, continuum)
        new = dict(d)
        new['intensity'] = np.asarray(d['intensity'], dtype=np.float64) / c
        out[mu] = new
    return out, {'wav': wv, 'continuum': continuum}


    ########################################################################################
########################################################################################
#                                PHOTOMETRY FUNCTIONS                                  #
########################################################################################
########################################################################################


#The Phoenix files read are kept in memory (cut at the wavelength range, as float64, read-only), so that the next
#simulations with the same models do not read them again. A file is read again if it was modified (its mtime is in the
#key). PHOENIX_CACHE_SIZE files of each kind at most (SPECINT 3700-9500 A: ~3.6 MB per file; HiRes mask range: ~3 MB).
PHOENIX_CACHE_SIZE = 16


def _read_only(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@lru_cache(maxsize=PHOENIX_CACHE_SIZE)
def _read_specint(model, mtime, lower, upper):
    """mu, wavelengths and intensity (n_mu, n_wavelengths) of one SPECINT file, cut at lower < wavelength < upper."""
    with fits.open(model) as hdul:
        mu = _read_only(np.asarray(hdul[1].data, dtype=np.float64).ravel())
        wavelength = 500.0 + np.arange(hdul[0].data.shape[1])          #wavelength in A
        idx_wv = (wavelength > lower) & (wavelength < upper)
        intensity = _read_only(hdul[0].data[:, idx_wv])                  #cut before the conversion to float64
    return mu, _read_only(wavelength[idx_wv]), intensity


@lru_cache(maxsize=4)
def _read_hires_wavelengths(wave_file, mtime, lower, upper):
    """Mask of the HiRes wavelengths in lower < wavelength < upper, and these wavelengths."""
    with fits.open(wave_file) as hdul:
        wavelength = hdul[0].data
        idx_wv = (wavelength > lower) & (wavelength < upper)
        idx_wv.setflags(write=False)
        return idx_wv, _read_only(wavelength[idx_wv])


@lru_cache(maxsize=PHOENIX_CACHE_SIZE)
def _read_hires(model, mtime, wave_file, wave_mtime, lower, upper):
    """Flux of one HiRes file, cut at lower < wavelength < upper."""
    idx_wv, _ = _read_hires_wavelengths(wave_file, wave_mtime, lower, upper)
    with fits.open(model) as hdul:
        return _read_only(hdul[0].data[idx_wv])


@lru_cache(maxsize=8)
def _read_mask(pathmask, mtime):
    """Columns of a CCF mask file (np.loadtxt(..., unpack=True)), read-only."""
    d = np.loadtxt(pathmask, unpack=True)
    d.setflags(write=False)
    return d


def clear_phoenix_cache():
    """Forget the Phoenix files and CCF masks kept in memory (to free the memory)."""
    for f in (_read_specint, _read_hires_wavelengths, _read_hires, _read_mask):
        f.cache_clear()


def _weighted_sum(arrays_weights):
    """Sum of w * array (a new array)."""
    total = None
    for a, w in arrays_weights:
        total = w * a if total is None else total + w * a
    return total


def interpolate_phoenix_mu(path, temp, logg, metallicity=0.0, wavelength_lower_limit=3000.0, wavelength_upper_limit=10000.0,
                           overhead=1.0, mu_angles=None):
    """Phoenix specific intensity spectra (SPECINT models, low resolution, at several mu) interpolated (linearly) at the
    temperature, logg and metallicity [Fe/H] desired, cut at the wavelength range (+- overhead [A]). For photometry and
    low resolution spectroscopy.

    path        : folder with the Phoenix SPECINT models lte*.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits (its subfolders are
                  searched too). Download them from PHOENIX_SPECINT_URL
    temp [K], logg [cgs], metallicity [Fe/H] : must be inside the grid of available models (no extrapolation)

    Returns the spectra in the format used by flux_grid / CCF_grid, {mu: {'wav': wavelengths [A], 'intensity': intensity}}
    ('wav' is read-only and shared by all the mu and by the next calls with the same range: copy it to modify it),
    one key for every mu of the models (the mu are read from the second extension of the files). The intensity is the
    Phoenix specific intensity [erg/s/cm^2/cm/sr]. The wavelength grid of the SPECINT models is 1 A from 500 A.
    Below the smallest mu, flux_grid extrapolates the intensity linearly to 0 at mu = 0 (as the zero row at mu = 0 that
    the previous version of this function added).
    The SPECINT models do not all tabulate the same mu (e.g. 5700 and 5800 K, ~1 % apart at the limb): as in StarSim, the
    rows of the models are added without interpolation in mu (row k of every model, mu ascending), and the result is
    given at the mu of the first model of the interpolation (lowest T, logg and [Fe/H]).
    mu_angles   : None, or the spectra of another surface ({mu: ...}, e.g. the photosphere) or its mu: the rows are given
                  at these mu instead of their own (row k at the k-th mu, ascending), as the old StarSim codes, which
                  used the angles of one surface for all of them (StarSim v4: the photosphere; StarSim (D. Baroch): the
                  spot). Same number of mu needed.
    """
    _wavelength_range(wavelength_lower_limit, wavelength_upper_limit, overhead)
    weights = phoenix_weights(Path(path), temp, logg, metallicity, 'SPECINT')
    lower, upper = float(wavelength_lower_limit - overhead), float(wavelength_upper_limit + overhead)

    amu = None
    models = []
    for model, w in weights.items():       #lowest T, logg and [Fe/H] first
        mu_model, wv, data = _read_specint(str(model), os.stat(model).st_mtime_ns, lower, upper)
        order = np.argsort(mu_model)
        if not np.array_equal(order, np.arange(len(order))):
            data = data[order]
        if amu is None:
            amu = np.asarray(mu_model)[order]
        elif len(mu_model) != len(amu):
            raise ValueError('The Phoenix SPECINT models used for the interpolation have a different number of mu angles '
                             '(%s: %d, %d in the first model)' % (model.name, len(mu_model), len(amu)))
        models.append((data, w))
    intensity = _weighted_sum(models)       #(n_mu, n_wavelengths), a new array (the cached data are not modified)

    if mu_angles is not None:
        other = np.sort(np.asarray(list(mu_angles) if isinstance(mu_angles, dict) else mu_angles, dtype=np.float64).ravel())
        if len(other) != len(amu):
            raise ValueError('mu_angles has %d angles, the Phoenix SPECINT models %d' % (len(other), len(amu)))
        amu = other

    #the wavelengths are the read-only array kept in memory (shared, not copied); the intensities are new arrays
    return {float(mu): {'wav': wv, 'intensity': intensity[i]} for i, mu in enumerate(amu)}


########################################################################################
########################################################################################
#                                SPECTROSCOPY FUNCTIONS                                #
########################################################################################
########################################################################################

PHOENIX_URL = 'http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'
PHOENIX_SPECINT_URL = 'https://phoenix.astro.physik.uni-goettingen.de/?page_id=73'
#end of the file names of the two kinds of Phoenix models
PHOENIX_SUFFIX = {'HiRes': 'PHOENIX-ACES-AGSS-COND-2011-HiRes.fits',        #disc-integrated flux, high resolution
                  'SPECINT': 'PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'}    #specific intensity at several mu, low resolution


def phoenix_grid(path, kind='HiRes'):
    """Phoenix models of one kind ('HiRes' or 'SPECINT') available in path (searched in its subfolders too, e.g.
    Z-0.0/, Z-0.5/, Z+0.5/). File names: lte{T:05d}-{logg:.2f}{[Fe/H]:+.1f}.<PHOENIX_SUFFIX[kind]> (solar is written -0.0).
    Only models without alpha enhancement are used.
    Returns {(T, logg, [Fe/H]): file path}."""
    import re
    suffix = PHOENIX_SUFFIX[kind]
    pattern = re.compile(r'^lte(\d{5})-(\d+\.\d{2})([+-]\d+\.\d)\.' + re.escape(suffix) + '$')
    grid = {}
    for f in Path(path).rglob('lte*' + suffix):
        match = pattern.match(f.name)
        if match:
            T, logg, feh = (float(v) for v in match.groups())
            grid[(T, logg, feh + 0.0)] = f      # + 0.0 turns -0.0 into 0.0
    return grid


def _wavelength_range(lower, upper, overhead):
    """Checks the wavelength limits [A] of the Phoenix readers."""
    lower = _checks.number('wavelength_lower_limit', lower, positive=True)
    upper = _checks.number('wavelength_upper_limit', upper, positive=True)
    _checks.number('overhead', overhead, nonnegative=True)
    if not lower < upper:
        raise ValueError('wavelength_lower_limit (%g) must be < wavelength_upper_limit (%g) [A]' % (lower, upper))


def phoenix_weights(path, temp, logg, metallicity, kind='HiRes'):
    """Models and weights of the (tri)linear interpolation in T, logg and [Fe/H] between the closest Phoenix models.
    Returns {file path: weight} (only the models with weight > 0; the weights add up to 1).
    Raises an error if the parameters are outside the grid (no extrapolation) or if a model needed is missing."""
    temp = _checks.number('temp', temp, positive=True)
    logg = _checks.number('logg', logg)
    metallicity = _checks.number('metallicity', metallicity)
    url = PHOENIX_URL if kind == 'HiRes' else PHOENIX_SPECINT_URL
    grid = phoenix_grid(path, kind)
    if not grid:
        raise FileNotFoundError('No Phoenix %s models (lte*.%s) in %s. Download them from %s' % (kind, PHOENIX_SUFFIX[kind], path, url))

    #models immediately below and above the desired value of every parameter
    brackets = []
    for name, value, axis in (('T', temp, 0), ('logg', logg, 1), ('[Fe/H]', metallicity, 2)):
        available = np.unique([key[axis] for key in grid])
        if value < available.min() or value > available.max():
            raise ValueError('The desired %s (%s) is outside the grid of Phoenix %s models (%s to %s), extrapolation is not '
                             'supported. Download the models covering it from %s' % (name, value, kind, available.min(), available.max(), url))
        low, upp = available[available <= value].max(), available[available >= value].min()
        weight = 0.0 if upp == low else (value - low) / (upp - low)     #avoid dividing by 0
        brackets.append(((low, 1.0 - weight), (upp, weight)))

    #the 8 (or less) corners of the interpolation must exist
    weights = {}
    for T, wT in brackets[0]:
        for g, wg in brackets[1]:
            for z, wz in brackets[2]:
                w = wT * wg * wz
                if w == 0.0:
                    continue
                if (T, g, z) not in grid:
                    raise FileNotFoundError('The file lte{:05d}-{:.2f}{:+.1f}.{} required for the interpolation does not exist. '
                                            'Download it from {} and add it to {}'.format(
                                                int(T), g, z if z != 0 else -0.0, PHOENIX_SUFFIX[kind], url, path))
                weights[grid[(T, g, z)]] = weights.get(grid[(T, g, z)], 0.0) + w
    return weights


def phoenix_continuum(wv, flux, lower, upper, nbins=20, deg=6):
    """Continuum of a Phoenix spectrum: 6th degree polynomial fitted to the maximum of the flux in each of nbins bins.
    20 bins work for all reasonable parameters: with more bins the maxima fall on absorption lines, with less the fit degrades.
    Returns the continuum at wv and the points of the fit (x_bin, y_bin)."""
    edges = np.linspace(lower, upper, nbins)
    #wv is sorted: the bin [edges[i], edges[i+1]) is wv[start[i]:start[i+1]]
    start = np.searchsorted(wv, edges, side='left')
    k = np.array([a + np.argmax(flux[a:b]) for a, b in zip(start[:-1], start[1:]) if b > a], dtype=np.int64)
    x_bin, y_bin = wv[k], flux[k]
    #Polynomial.fit rescales the wavelengths to [-1, 1]: same polynomial as np.polyfit, without its conditioning problems
    poly = np.polynomial.Polynomial.fit(x_bin, y_bin, min(deg, len(x_bin) - 1))
    #poly(wv) in one compiled pass (numpy needs ~8 passes over the ~10^5-10^6 wavelengths): the same operations
    from . import nbspectra
    off, scl = poly.mapparms()
    continuum = nbspectra.polyval_mapped_nb(np.ascontiguousarray(wv, dtype=np.float64), float(off), float(scl),
                                            np.ascontiguousarray(poly.coef, dtype=np.float64))
    return continuum, x_bin, y_bin


def interpolate_phoenix(path, temp, logg, metallicity=0.0, wavelength_lower_limit=3000.0, wavelength_upper_limit=10000.0,
                        normalize=False, overhead=1.0, plot=False, keep_continuum=False):
    """Phoenix HiRes spectrum interpolated (linearly) at the temperature, logg and metallicity [Fe/H] desired,
    cut at the wavelength range (+- overhead [A], to allow for Doppler shifts without losing information).

    path        : folder with the Phoenix HiRes models (and its subfolders) and WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
                  (download them from PHOENIX_URL)
    temp [K], logg [cgs], metallicity [Fe/H] : must be inside the grid of available models (no extrapolation)
    normalize   : True -> the flux is divided by its continuum (phoenix_continuum); False -> Phoenix flux [erg/s/cm^2/cm]
    plot        : plot the spectrum and its continuum (only with normalize=True)
    keep_continuum : with normalize=True, also return 'flux' (not normalized) and 'continuum' (two more arrays as
                  long as the spectrum: off by default to save memory)

    Returns the spectrum in the format used by flux_grid / CCF_grid, {mu: {'wav': wavelengths [A], 'intensity': flux}}
    ('wav' is read-only and shared by the next calls with the same range: copy it to modify it),
    with the single key mu = 1.0: the HiRes models are disc-integrated, they have no mu dependence. Use it with
    CCF_grid(..., mu_ratio=...), which accepts one spectrum at mu = 1.0. flux_grid needs spectra at several mu
    (interpolate_phoenix_mu, with the SPECINT models). The wavelengths are in vacuum.
    With normalize=True and keep_continuum=True, the dictionary also has 'flux' (not normalized) and 'continuum'.
    """
    _wavelength_range(wavelength_lower_limit, wavelength_upper_limit, overhead)
    path = Path(path)
    weights = phoenix_weights(path, temp, logg, metallicity, 'HiRes')

    #Phoenix wavelengths, cut at the desired range
    wave_file = next(path.rglob('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'), None)
    if wave_file is None:
        raise FileNotFoundError('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits not found in %s. Download it from %s' % (path, PHOENIX_URL))
    lower, upper = float(wavelength_lower_limit - overhead), float(wavelength_upper_limit + overhead)
    wave_key = (str(wave_file), os.stat(wave_file).st_mtime_ns)
    wv = _read_hires_wavelengths(*wave_key, lower, upper)[1]      #read-only array kept in memory (not copied)

    #trilinear interpolation: weighted sum of the corner models (a new array: the cached fluxes are not modified)
    flux = _weighted_sum((_read_hires(str(model), os.stat(model).st_mtime_ns, *wave_key, lower, upper), w)
                         for model, w in weights.items())

    spectrum = {'wav': wv, 'intensity': flux}
    if normalize:
        continuum, x_bin, y_bin = phoenix_continuum(wv, flux, wavelength_lower_limit - overhead, wavelength_upper_limit + overhead)
        if keep_continuum or plot:
            spectrum = {'wav': wv, 'intensity': flux / continuum}
            if keep_continuum:
                spectrum.update(flux=flux, continuum=continuum)
        else:
            flux /= continuum           #in place: flux is a new array, no third copy of the spectrum
            spectrum = {'wav': wv, 'intensity': flux}
        if plot:      #to check the normalisation
            import matplotlib.pyplot as plt
            plt.plot(wv, flux)
            plt.plot(x_bin, y_bin, 'ok')
            plt.plot(wv, continuum)
            plt.xlabel('wavelength [$\\AA$]')
            plt.show()
            plt.close()

    return {1.0: spectrum}


#resolving power of the instruments known by add_resol (a number can be given instead of a name)
INSTRUMENT_RESOLUTION = {'EXPRESS': 137000.0, 'HARPS': 115000.0, 'HARPS-N': 118000.0, 'NEID': 120000.0}


def vlambda(inlamb, vstep):
    """Wavelength grid of constant velocity step (logarithmic) covering inlamb, centred on the geometric mean of its
    ends, for the convolution of add_resol (resampler of the SteParSyn broadener, as in the old StarSim)."""
    xw2 = max(inlamb) - 0.1
    xw1 = min(inlamb) + 0.1
    iw1 = np.where(inlamb > xw1)[0]
    iw2 = np.where(inlamb > xw2)[0]
    w1 = inlamb[iw1[0] + 1]
    w2 = inlamb[iw2[0] - 1]
    npix = np.int64((iw2[0] - 1) - (iw1[0] + 1) * vstep)
    wmid = np.sqrt(w1 * w2)
    dwave = (np.log10(w2) - np.log10(w1)) / (npix - 1.)
    vwavel = np.log10(wmid) - (dwave * np.arange(np.int64(npix / 2.)))
    vwaveu = np.log10(wmid) + (dwave * np.arange(np.int64(npix / 2.)))
    vwavel = np.sort(10. ** vwavel[1:len(vwavel) - 1])
    vwaveu = 10. ** vwaveu[1:len(vwaveu) - 1]
    return np.concatenate((vwavel, [wmid], vwaveu))


def add_resol(wavelength, flux, instrument):
    """Spectrum (wavelength [A], flux) degraded to the resolution of an instrument: convolved with a gaussian of FWHM
    c/R in velocity on a grid of constant velocity step (vlambda), then resampled on the original wavelengths.
    instrument: a name of INSTRUMENT_RESOLUTION (EXPRESS R = 137000, HARPS 115000, HARPS-N 118000, NEID 120000) or the
    resolving power R itself. Based on the SteParSyn broadener (Tabernero et al. 2022, two-clause BSD licence); the
    same numbers as the old StarSim."""
    from scipy import interpolate
    from astropy.convolution import convolve_fft
    if isinstance(instrument, str):
        if instrument not in INSTRUMENT_RESOLUTION:
            raise ValueError('unknown instrument %r: give one of %s, or the resolving power R as a number'
                             % (instrument, sorted(INSTRUMENT_RESOLUTION)))
        resolution = INSTRUMENT_RESOLUTION[instrument]
    else:
        resolution = _checks.number('instrument (resolving power)', instrument, positive=True)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    vstep = 1
    vlight = 2.99792458e5
    vwave = vlambda(wavelength, vstep)
    xw2 = max(wavelength) - 0.1
    xw1 = min(wavelength) + 0.1
    iw1 = np.where(wavelength > xw1)[0]
    iw2 = np.where(wavelength > xw2)[0]
    w1 = wavelength[iw1[0] + 1]
    w2 = wavelength[iw2[0] - 1]
    tck = interpolate.splrep(wavelength, flux, k=3, s=0)
    vflux = interpolate.splev(vwave, tck, der=0)
    wmid = np.sqrt(w1 * w2)

    #gaussian kernel of FWHM c/R
    vibr = (vlight / resolution) / (2. * np.sqrt(2. * np.log(2.)))
    sigma = vibr * wmid / vlight
    nx1 = len(vwave)
    dx1 = (vwave[nx1 - 1] - vwave[0]) / float(nx1 - 1)
    xk = (np.arange(nx1) - nx1 / 2) * dx1
    zk = xk / sigma
    yk = 1. / np.sqrt(2. * np.pi) / sigma * np.exp(-(zk ** 2.) / 2.)
    outflux = convolve_fft(vflux, yk / np.sum(yk), boundary='fill', fill_value=1.)

    #flux resampled back to the original sampling
    tck2 = interpolate.splrep(vwave, outflux, k=3, s=0)
    return interpolate.splev(wavelength, tck2, der=0)

def mask_weights(pathmask, mask_medium=None, medium='air'):
    """Lines of a CCF mask from a text file: wavelengths wvm [A] and weights fm.
    Two columns (wavelength, weight), or three (start, end, weight, as the HARPS masks: the centre of every line).
    mask_medium : medium of the wavelengths of the file: 'air' or 'vacuum'. None (default): the usual convention of
                  StarSim, three columns = air (HARPS masks), two columns = vacuum. Say it explicitly if your mask
                  follows another convention.
    medium      : medium of the wavelengths returned: 'air' (default, the medium in which CCF_grid computes the CCFs by
                  default) or 'vacuum'; converted with vacuum2air / air2vacuum when it is not the one of the file.
    FileNotFoundError if the file is missing, ValueError if it has another format.
    The file is read once and kept in memory (read again if it is modified); the arrays returned are new copies."""
    for name, value in (('mask_medium', mask_medium), ('medium', medium)):
        if value not in (None, 'air', 'vacuum') or (name == 'medium' and value is None):
            raise ValueError("%s must be 'air' or 'vacuum'%s, got %r" % (name, ' (or None)' if name == 'mask_medium' else '', value))
    try:
        d = _read_mask(str(pathmask), os.stat(pathmask).st_mtime_ns)
    except OSError:
        raise FileNotFoundError('mask file not found: %s' % pathmask) from None
    except ValueError as e:
        raise ValueError('mask file %s could not be read as columns of numbers (%s)' % (pathmask, e)) from None
    if d.ndim == 1:
        raise ValueError('mask file %s: a single column or a single line; give two (wavelength, weight) or three '
                         '(wv1, wv2, weight) columns' % pathmask)
    if len(d) == 2:
        wvm = np.atleast_1d(np.array(d[0], dtype=np.float64))
        fm = np.atleast_1d(np.array(d[1], dtype=np.float64))
        given = 'vacuum' if mask_medium is None else mask_medium
    elif len(d) == 3:
        wvm = np.atleast_1d(np.asarray((d[0] + d[1]) / 2, dtype=np.float64))
        fm = np.atleast_1d(np.array(d[2], dtype=np.float64))
        given = 'air' if mask_medium is None else mask_medium
    else:
        raise ValueError('mask file %s has %d columns: two (wavelength, weight) or three (wv1, wv2, weight) are '
                         'expected' % (pathmask, len(d)))
    if given != medium:
        wvm = np.atleast_1d(vacuum2air(wvm) if medium == 'air' else air2vacuum(wvm))
    return wvm, fm