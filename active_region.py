"""
active_region: a circular spot or facula on the surface of the star, for StarSim(..., active_regions_map=[...]).

A region has a position on the rotating star (colatitude, longitude at reference_time), an angular radius that can
change with time (evolution law), and a lifetime. StarSim moves it with the rotation (and the differential rotation)
and computes, at every epoch, the fraction of every cell of the disc that it covers.
"""
import numpy as np


class active_region(object):
    """
    Active region (spot or facula). Spots and faculae are independent objects: any position,
    size and lifetime, in any order. Where they overlap, the spot is on top of the facula.

    typ            : 'sp' (spot) or 'fc' (facula)
    size           : angular radius [deg] (the radius when there is no evolution law; it is also passed to the law)
    latitude       : COLATITUDE [deg]  (90 = equator) -- kept under this name for compatibility
    longitude      : longitude at reference_time [deg]
    appearance_time, lifetime : [days] the region exists for appearance_time <= t <= appearance_time + lifetime,
                     it is absent (radius 0) at the other times. Default: always present (-inf, inf).
    reference_time : time at which `longitude` is defined
    evolution      : None -> constant radius `size` during the life of the region.
                     function evolution(dt, size) -> radius [deg], with dt = t - appearance_time [days] (an array of
                     times); it needs a finite appearance_time. Any law can be used (linear growth and decay,
                     exponential, gaussian, from a table, ...). A radius <= 0 (or NaN) means that the region is not
                     there at that time; outside its lifetime the region is absent whatever the law gives.
    """
    def __init__(self, typ, size, latitude, longitude, appearance_time=-np.inf, lifetime=np.inf,
                 reference_time=0, evolution=None):
        if typ not in ('sp', 'fc'):
            raise ValueError("typ must be 'sp' or 'fc'")
        if evolution is not None and not callable(evolution):
            raise TypeError("evolution must be None or a function evolution(dt, size) -> radius [deg]")
        if evolution is not None and not np.isfinite(appearance_time):
            raise ValueError("an evolution law needs a finite appearance_time (the origin of dt = t - appearance_time)")
        if not lifetime >= 0:
            raise ValueError("lifetime must be >= 0 (np.inf = forever)")
        if np.isinf(appearance_time) and np.isfinite(lifetime):
            raise ValueError("a finite lifetime needs a finite appearance_time")
        self.type = typ
        self.size = size
        self.latitude = latitude
        self.longitude = longitude
        self.appearance_time = appearance_time
        self.lifetime = lifetime
        self.reference_time = reference_time
        self.evolution = evolution

    @property
    def disappearance_time(self):
        """End of the life of the region [days] (appearance_time + lifetime)."""
        if np.isinf(self.lifetime):
            return np.inf
        return self.appearance_time + self.lifetime

    def alive(self, t):
        """True at the times t (array) when the region exists: appearance_time <= t <= appearance_time + lifetime."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return (t >= self.appearance_time) & (t <= self.disappearance_time)

    def radius(self, t):
        """Angular radius [deg] at the times t (array): `size`, or the evolution law, during the life of the region,
        and 0 (no region) outside it. Negative and NaN radii are set to 0."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        alive = self.alive(t)
        if self.evolution is None:
            return np.where(alive, float(self.size), 0.0)
        dt = t - self.appearance_time
        try:                                    #vectorised law (numpy)
            r = np.asarray(self.evolution(dt, self.size), dtype=np.float64)
            r = np.broadcast_to(r, t.shape).copy()
        except (TypeError, ValueError):         #law written for one time (math, if/else, ...): one call per time
            r = np.array([float(self.evolution(x, self.size)) for x in dt])
        r[~np.isfinite(r) | (r < 0) | ~alive] = 0.0
        return r
