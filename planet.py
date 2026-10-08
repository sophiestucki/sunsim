import numpy as np

import nbspectra


class planet(object):
    """
    Transiting planet: a dark disc in front of the star, on a circular or eccentric orbit.
    Give it to StarSim(..., planet=planet(...)), or several: StarSim(..., planet=[planet(...), planet(...)]).

    planet_period           : orbital period [days]
    planet_transit_t0       : time of the centre of the transit [days]
    planet_radius           : radius [stellar radii]
    planet_semi_major_axis  : semi-major axis [stellar radii]
    planet_impact_param     : impact parameter b [stellar radii] (0 = the planet crosses the disc centre)
    planet_esinw, planet_ecosw : e*sin(omega) and e*cos(omega) (0, 0 = circular orbit)
    planet_spin_orbit_angle : projected spin-orbit angle lambda [rad]: angle, on the sky, between the projected rotation
                              axis of the star and the projected normal of the orbit (equivalently between the stellar
                              equator and the path of the planet on the disc). 0 = aligned prograde orbit, pi/2 = polar
                              orbit, pi = retrograde orbit. Each planet has its own lambda. The true (3D) obliquity psi
                              follows from lambda, the orbital inclination and the stellar inclination (obliquity()).
    planet_semi_amplitude   : RV semi-amplitude K of the star induced by the planet [m/s] (ccf mode: the Keplerian RV
                              is added to rv_var). K = 203.244*(Mp*sini/Mjup)*((Ms+Mp)/Msun)^(-2/3)*(P/1day)^(-1/3)
    """
    def __init__(self, planet_period=None, planet_transit_t0=0, planet_radius=None, planet_semi_major_axis=None, planet_impact_param=0, planet_esinw=0, planet_ecosw=0, planet_spin_orbit_angle=0, planet_semi_amplitude=0):
        if None in (planet_period, planet_radius, planet_semi_major_axis):
            raise ValueError("a planet needs planet_period, planet_radius and planet_semi_major_axis")
        self.planet_period = planet_period
        self.planet_transit_t0 = planet_transit_t0
        self.planet_radius = planet_radius
        self.planet_semi_major_axis = planet_semi_major_axis
        self.planet_impact_param = planet_impact_param
        self.planet_esinw = planet_esinw
        self.planet_ecosw = planet_ecosw
        self.planet_spin_orbit_angle = planet_spin_orbit_angle
        self.planet_semi_amplitude = planet_semi_amplitude

    # ---------------------------------------------------------------- orbit
    @staticmethod
    def ecc_omega(esinw, ecosw):
        """Eccentricity and argument of periastron [rad] from e*sin(omega) and e*cos(omega)."""
        if(esinw==0 and ecosw==0):
            return 0.0, 0.0
        return np.sqrt(esinw*esinw+ecosw*ecosw), np.arctan2(esinw,ecosw)

    @staticmethod
    def true_anomaly(x,period,ecc,tperi):
        #sin and cos of the true anomaly at the times x (Newton's method, nbspectra.true_anomaly_nb)
        return nbspectra.true_anomaly_nb(np.atleast_1d(np.asarray(x, dtype=np.float64)), float(period), float(ecc), float(tperi))

    @staticmethod
    def Ttrans_2_Tperi(T0, P, e, w):
        #time of periastron from the time of transit
        return nbspectra.ttrans_2_tperi_nb(float(T0), float(P), float(e), float(w))

    @staticmethod
    def keplerian_orbit(x,params):
        #RV of the star induced by the planet [m/s]; params = [period, K, esinw, ecosw, t_transit]
        period=params[0]
        t_trans=params[4]
        krv=params[1]
        esinw=params[2]
        ecosw=params[3]

        ecc, omega = planet.ecc_omega(esinw, ecosw)

        t_peri = planet.Ttrans_2_Tperi(t_trans, period, ecc, omega)
        sinf,cosf=planet.true_anomaly(x,period,ecc,t_peri)
        cosftrueomega=cosf*np.cos(omega)-sinf*np.sin(omega)
        y= krv*(ecc*np.cos(omega)+cosftrueomega)

        return y

    def keplerian_rv(self, times):
        """RV of the star induced by this planet [m/s] at the times (keplerian_orbit)."""
        return self.keplerian_orbit(times, [self.planet_period, self.planet_semi_amplitude, self.planet_esinw,
                                            self.planet_ecosw, self.planet_transit_t0])

    def orbital_inclination(self):
        """Inclination of the orbit [rad] (pi/2 = edge-on), from the impact parameter: cos i = b/a * (1+e sin w)/(1-e^2)."""
        ecc, _ = self.ecc_omega(self.planet_esinw, self.planet_ecosw)
        return np.arccos((self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2))

    def obliquity(self, stellar_inclination):
        """True (3D) spin-orbit angle psi [rad], between the rotation axis of the star and the normal of the orbit:
        cos psi = cos i* cos i_p + sin i* sin i_p cos lambda.
        stellar_inclination : the `inclination` of StarSim [rad], 0 = equator-on (it is pi/2 - i*, with i* the usual
        angle between the rotation axis and the line of sight)."""
        i_star = np.pi/2 - stellar_inclination
        i_p = self.orbital_inclination()
        cospsi = np.cos(i_star)*np.cos(i_p) + np.sin(i_star)*np.sin(i_p)*np.cos(self.planet_spin_orbit_angle)
        return np.arccos(np.clip(cospsi, -1.0, 1.0))

    # ---------------------------------------------------------------- position on the disc
    def positions(self, times):
        """Position of the planet at all the epochs, in one NumPy pass: array (n_times, 3) with rho, theta (polar
        coordinates on the disc) and the radius (in Rstar). Behind the star (secondary transit) rho = 1+2*radius."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        ecc, omega = self.ecc_omega(self.planet_esinw, self.planet_ecosw)

        t_peri = self.Ttrans_2_Tperi(self.planet_transit_t0,self.planet_period, ecc, omega)
        sinf,cosf=self.true_anomaly(t,self.planet_period,ecc,t_peri)

        cosftrueomega=cosf*np.cos(omega+np.pi/2)-sinf*np.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
        sinftrueomega=cosf*np.sin(omega+np.pi/2)+sinf*np.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

        cosi = (self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

        lam = self.planet_spin_orbit_angle
        rpl=self.planet_semi_major_axis*(1-ecc**2)/(1+ecc*cosf)
        xpl=rpl*(-np.cos(lam)*sinftrueomega-np.sin(lam)*cosftrueomega*cosi)
        ypl=rpl*(np.sin(lam)*sinftrueomega-np.cos(lam)*cosftrueomega*cosi)

        pos = np.empty((len(t), 3))
        pos[:, 0] = np.sqrt(ypl**2+xpl**2)
        pos[:, 1] = np.arctan2(ypl,xpl)
        pos[:, 2] = self.planet_radius
        behind = cosftrueomega > 0.0                 #avoid secondary transits
        pos[behind, 0] = 1+self.planet_radius*2
        pos[behind, 1] = 0.0
        return pos

    def position(self, t):
        """rho, theta (polar coordinates on the disc) and radius (in Rstar) of the planet at time t."""
        return self.positions([t])[0]
