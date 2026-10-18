"""
rotation: differential rotation laws, for StarSim(..., differential_rotation=...) and flux_grid(..., differential_rotation=...).

A law gives the angular velocity at every latitude relative to the equator:
    law(latitude) = Omega(lat) - Omega_eq   [deg/day],   latitude in degrees (-90 south pole, 0 equator, 90 north pole),
with Omega_eq = 360 / rotation_period deg/day. StarSim uses it to move the regions (longitude at t =
longitude + (Omega_eq + law(lat)) (t - reference_time)) and for the rotation velocity of the cells (Doppler shifts).

differential_rotation can be
    0 or None                         rigid rotation
    a number c1 or a sequence (c1, c2, ...)   the polynomial c1 sin^2(lat) + c2 sin^4(lat) + ... [deg/day]
    any function law(latitude [deg]) -> deg/day, vectorised (it receives NumPy arrays)
Laws given here:
    solar_differential_rotation(B)    the law of the old StarSim, B sin^2-scaled solar profile (the default B = -2.66 deg/day
                                      gives SOLAR_DIFFERENTIAL_ROTATION = (-1.698, -2.346))
    polynomial_differential_rotation(c1, c2, ...)
"""
import numpy as np

SOLAR_B = -2.66          # deg/day: B of the solar law of StarSim (Poljancic Beljan et al. 2017)


def polynomial_differential_rotation(*coefficients):
    """Law Omega(lat) - Omega_eq = c1 sin^2(lat) + c2 sin^4(lat) + ... [deg/day]. The coefficients are kept in
    law.coefficients."""
    coeffs = tuple(float(c) for c in coefficients)
    if not all(np.isfinite(coeffs)):
        raise ValueError('the coefficients of the differential rotation must be finite [deg/day], got %r' % (coeffs,))

    def law(latitude):
        s2 = np.sin(np.deg2rad(np.asarray(latitude, dtype=np.float64)))**2
        return np.polynomial.polynomial.polyval(s2, (0.0,) + coeffs)
    law.coefficients = coeffs
    law.__doc__ = 'Omega(lat) - Omega_eq = ' + ' + '.join('%g sin^%d(lat)' % (c, 2 * (k + 1)) for k, c in enumerate(coeffs))
    return law


def solar_differential_rotation(B=SOLAR_B):
    """The differential rotation of the old StarSim: Omega(lat) - Omega_eq = (B / 2.66) (1.698 sin^2(lat) +
    2.346 sin^4(lat)) deg/day: the solar profile scaled by B. B = -2.66 gives SOLAR_DIFFERENTIAL_ROTATION =
    (-1.698, -2.346); B = 0 a rigid rotation."""
    return polynomial_differential_rotation(1.698 * B / 2.66, 2.346 * B / 2.66)


def differential_rotation_law(differential_rotation):
    """Function law(latitude [deg]) -> Omega(lat) - Omega_eq [deg/day] from any accepted form of differential_rotation
    (see the module docstring)."""
    if differential_rotation is None:
        return polynomial_differential_rotation()
    if callable(differential_rotation):
        law = differential_rotation
        test = np.asarray(law(np.linspace(-90.0, 90.0, 181)), dtype=np.float64)
        if test.shape != (181,) or not np.all(np.isfinite(test)):
            raise ValueError('differential_rotation(latitude) must return finite values [deg/day] for an array of '
                             'latitudes (one value per latitude), got shape %s' % (test.shape,))
        return law
    coeffs = np.atleast_1d(np.asarray(differential_rotation, dtype=np.float64))
    if coeffs.ndim != 1:
        raise ValueError('differential_rotation must be 0, a number, a sequence of coefficients or a function of the '
                         'latitude, got %r' % (differential_rotation,))
    return polynomial_differential_rotation(*coeffs)
