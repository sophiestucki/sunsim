import numpy as np


class active_region(object):
    """
    Active region (spot or facula). Spots and faculae are independent objects: any position,
    size and lifetime, in any order. Where they overlap, the spot is on top of the facula.

    typ            : 'sp' (spot) or 'fc' (facula)
    size           : angular radius [deg] (the radius when there is no evolution law; it is also passed to the law)
    latitude       : COLATITUDE [deg]  (90 = equator) -- kept under this name for compatibility
    longitude      : longitude at reference_time [deg]
    appearance_time, lifetime : region exists for appearance_time <= t <= appearance_time + lifetime
                     (not applied yet, except through the evolution law: dt below is t - appearance_time)
    reference_time : time at which `longitude` is defined
    evolution      : None -> constant radius `size`.
                     function evolution(dt, size) -> radius [deg], with dt = t - appearance_time [days] (an array of times).
                     Any law can be used (linear growth and decay, exponential, gaussian, from a table, ...).
                     A radius <= 0 (or NaN) means that the region is not there at that time.
    """
    def __init__(self, typ, size, latitude, longitude, appearance_time=0, lifetime=1000,
                 reference_time=0, evolution=None):
        if typ not in ('sp', 'fc'):
            raise ValueError("typ must be 'sp' or 'fc'")
        if evolution is not None and not callable(evolution):
            raise TypeError("evolution must be None or a function evolution(dt, size) -> radius [deg]")
        self.type = typ
        self.size = size
        self.latitude = latitude
        self.longitude = longitude
        self.appearance_time = appearance_time
        self.lifetime = lifetime
        self.reference_time = reference_time
        self.evolution = evolution

    def radius(self, t):
        """Angular radius [deg] at the times t (array): `size`, or the evolution law. Negative and NaN radii are set to 0
        (no region)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if self.evolution is None:
            return np.full(t.shape, float(self.size))
        dt = t - self.appearance_time
        try:                                    #vectorised law (numpy)
            r = np.asarray(self.evolution(dt, self.size), dtype=np.float64)
            r = np.broadcast_to(r, t.shape).copy()
        except (TypeError, ValueError):         #law written for one time (math, if/else, ...): one call per time
            r = np.array([float(self.evolution(x, self.size)) for x in dt])
        r[~np.isfinite(r) | (r < 0)] = 0.0
        return r
