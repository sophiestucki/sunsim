class active_region(object):
    """
    Active region (spot or facula). Spots and faculae are independent objects: any position,
    size and lifetime, in any order. Where they overlap, the spot is on top of the facula.

    typ            : 'sp' (spot) or 'fc' (facula)
    size           : angular radius [deg]
    latitude       : COLATITUDE [deg]  (90 = equator) -- kept under this name for compatibility
    longitude      : longitude at reference_time [deg]
    appearance_time, lifetime : region exists for appearance_time <= t <= appearance_time + lifetime
    reference_time : time at which `longitude` is defined
    """
    def __init__(self, typ, size, latitude, longitude, appearance_time=0, lifetime=1000,
                 reference_time=0):
        if typ not in ('sp', 'fc'):
            raise ValueError("typ must be 'sp' or 'fc'")
        self.type = typ
        self.size = size
        self.latitude = latitude
        self.longitude = longitude
        self.appearance_time = appearance_time
        self.lifetime = lifetime
        self.reference_time = reference_time