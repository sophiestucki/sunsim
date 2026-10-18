"""
_checks: checks of the arguments of the public classes and functions of SunSim (StarSim, flux_grid, CCF_grid,
active_region, planet, spectrum_utils). A wrong argument gives a clear error when the object is created, instead of a
crash or a wrong result later (in a Numba function, where the error is hard to find).
The checks are cheap (no copy of large arrays) and do not change any result.
"""
import numbers
import warnings

import numpy as np

C = 2.99792458e8          # speed of light [m/s]


def number(name, value, positive=False, nonnegative=False, low=None, high=None, integer=False, allow_inf=False):
    """value as a float (an int with integer=True), checked: a real number, finite (unless allow_inf), > 0
    (positive), >= 0 (nonnegative), within [low, high]."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Real, np.number)):
        arr = np.asarray(value)
        if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
            value = arr.item()
        else:
            raise TypeError('%s must be a number, got %r' % (name, value))
    if integer:
        if float(value) != int(value):
            raise TypeError('%s must be an integer, got %r' % (name, value))
        value = int(value)
    else:
        value = float(value)
    if np.isnan(value) or (np.isinf(value) and not allow_inf):
        raise ValueError('%s must be finite, got %r' % (name, value))
    if positive and not value > 0:
        raise ValueError('%s must be > 0, got %r' % (name, value))
    if nonnegative and not value >= 0:
        raise ValueError('%s must be >= 0, got %r' % (name, value))
    if low is not None and value < low:
        raise ValueError('%s must be >= %g, got %r' % (name, low, value))
    if high is not None and value > high:
        raise ValueError('%s must be <= %g, got %r' % (name, high, value))
    return value


def array1d(name, value, min_size=1, increasing=False, strictly=True):
    """value as a 1D float array (a number gives an array of one value), finite, at least min_size values,
    increasing if asked."""
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError):
        raise TypeError('%s must be a 1D array of numbers, got %r' % (name, type(value).__name__)) from None
    if arr.ndim != 1:
        raise ValueError('%s must be a 1D array, got shape %s' % (name, arr.shape))
    if arr.size < min_size:
        raise ValueError('%s needs at least %d value(s), got %d' % (name, min_size, arr.size))
    if not np.all(np.isfinite(arr)):
        raise ValueError('%s contains NaN or infinite values' % name)
    if increasing and arr.size > 1:
        d = np.diff(arr)
        if (strictly and not np.all(d > 0)) or (not strictly and not np.all(d >= 0)):
            raise ValueError('%s must be %sincreasing' % (name, 'strictly ' if strictly else ''))
    return arr


def inclination(name, value):
    """Inclination of the rotation axis [rad], SunSim convention: 0 = equator-on, +-pi/2 = pole-on. A value beyond
    pi/2 is almost surely in degrees or in the usual convention (90 = equator-on)."""
    value = number(name, value)
    if abs(value) > np.pi / 2 + 1e-9:
        raise ValueError('%s = %g: it is in radians with 0 = equator-on and pi/2 = pole-on (|%s| <= pi/2). For an '
                         'inclination i in degrees in the usual convention (90 = equator-on), give np.deg2rad(90 - i)'
                         % (name, value, name))
    return value


def spectra(name, flux, min_mu=1):
    """Spectra {mu: {'wav': wavelengths, 'intensity': values}}: a dict, at least min_mu angles, numeric mu in [0, 1],
    the same number of wavelengths and intensities, the same increasing wavelength grid for every mu (only the
    first, the last and the length are compared, no copy). Returns the angles as floats, ascending."""
    if not isinstance(flux, dict):
        raise TypeError("%s must be a dict {mu: {'wav': wavelengths, 'intensity': values}}, got %s"
                        % (name, type(flux).__name__))
    if len(flux) < min_mu:
        raise ValueError('%s needs at least %d angle(s) mu, got %d' % (name, min_mu, len(flux)))
    mus = []
    ref = None
    for key, d in flux.items():
        try:
            mu = float(key)
        except (TypeError, ValueError):
            raise TypeError('%s: the keys must be the angles mu (numbers, or strings of numbers), got %r'
                            % (name, key)) from None
        if not 0 <= mu <= 1:
            raise ValueError('%s: mu = cos(angle) must be in [0, 1], got %r' % (name, key))
        if not isinstance(d, dict) or 'wav' not in d or 'intensity' not in d:
            raise TypeError("%s[%r] must be a dict with the keys 'wav' and 'intensity'" % (name, key))
        wav, inten = np.asarray(d['wav']), np.asarray(d['intensity'])
        if wav.ndim != 1 or inten.ndim != 1 or len(wav) != len(inten) or len(wav) < 2:
            raise ValueError("%s[%r]: 'wav' and 'intensity' must be 1D arrays of the same length (>= 2), got %s and %s"
                             % (name, key, wav.shape, inten.shape))
        if not wav[1] > wav[0]:
            raise ValueError("%s[%r]: the wavelengths must be increasing" % (name, key))
        sig = (len(wav), float(wav[0]), float(wav[-1]))
        if ref is None:
            ref = sig
        elif sig != ref:
            raise ValueError('%s: all the angles must have the same wavelength grid (mu = %r has %d points %.4f-%.4f, '
                             'the first one %d points %.4f-%.4f)' % ((name, key) + sig + ref))
        mus.append(mu)
    if len(set(mus)) != len(mus):
        raise ValueError('%s: the same mu appears twice' % name)
    return np.sort(np.array(mus))


def nan_warning(name, arr):
    """Warn if an array (a grid) contains NaN or infinite values: the outputs that use it would be NaN."""
    arr = np.asarray(arr)
    if arr.size and not np.all(np.isfinite(arr)):
        warnings.warn('%s contains NaN or infinite values: the outputs that use it will be NaN' % name, stacklevel=3)


def phoenix_step(wv):
    """Wavelength step of the Phoenix HiRes grid at the wavelengths wv [A]."""
    return np.select([wv < 3000, wv < 5000, wv < 10000, wv < 15000], [0.1, 0.006, 0.01, 0.02], 0.03)


def is_phoenix_grid(wv):
    """True if the wavelengths follow the steps of the Phoenix HiRes grid (checked on a sample of the points)."""
    wv = np.asarray(wv, dtype=np.float64)
    if len(wv) < 3:
        return False
    idx = np.unique(np.linspace(0, len(wv) - 2, min(len(wv) - 1, 2000)).astype(np.int64))
    step = wv[idx + 1] - wv[idx]
    expected = phoenix_step(wv[idx])
    ok = np.abs(step - expected) <= 0.05 * expected
    return np.mean(ok) > 0.99           # a few points at the changes of step


def mask_in_spectrum(wvm, rvs, wv, what='the spectrum'):
    """The mask lines, Doppler shifted by the smallest and the largest velocity, must stay inside the wavelengths
    (with one pixel on each side): otherwise the CCF reads outside the spectrum."""
    wvm = np.asarray(wvm, dtype=np.float64)
    lo = wvm.min() * (1 + np.min(rvs) / C)
    hi = wvm.max() * (1 + np.max(rvs) / C)
    if lo <= wv[1] or hi >= wv[-2]:
        raise ValueError('the mask lines shifted by the velocity grid (%.3f-%.3f A) go beyond %s (%.3f-%.3f A): cut the '
                         'mask, reduce the velocity range, or give a wider spectrum' % (lo, hi, what, wv[0], wv[-1]))
