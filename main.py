import numpy as np
from pathlib import Path
from multiprocessing import Pool
import sys
import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
import collections
from astropy.io import fits
import pandas as pd
from scipy import interpolate
from scipy import optimize
import math as m
import warnings

import nbspectra


class StarSim(object): 
    """
    Simulation class
    """
    def __init__(self, time, N_rings, inclination, rotation_period, differential_rotation=0, stellar_radius=1, active_regions_map=None, active_regions_mask=None, total_flux_qp=None, flux_grid_qp=None, flux_grid_sp=None, flux_grid_fc=None, ccf_grid_qp=None, ccf_grid_sp=None, ccf_grid_fc=None, simulate_planet=False, mode=['photometry']):

        self.obs_times = time
        self.total_flux_qp = total_flux_qp
        self.flux_grid_qp = flux_grid_qp
        self.flux_grid_sp = flux_grid_sp
        self.flux_grid_fc = flux_grid_fc

        self.ccf_grid_qp = ccf_grid_qp
        self.ccf_grid_sp = ccf_grid_sp
        self.ccf_grid_fc = ccf_grid_fc
        self.inclination = inclination
        self.rotation_period = rotation_period
        self.differential_rotation = differential_rotation
        self.active_regions_map = active_regions_map
        self.active_regions_mask = active_regions_mask
        self.simulate_planet = simulate_planet

        if isinstance(mode, str):                          # a single mode given as a string still works
            mode = [mode]
        mode = list(mode)
        valid = ('photometry', 'spectroscopy', 'ccf')
        if not mode or any(md not in valid for md in mode):
            raise ValueError("mode must be a list with some of %s, got %r" % (valid, mode))
        self.mode = mode
        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea = nbspectra.generate_grid_coordinates_nb(N_rings)

        self.N_rings = N_rings
        self.Ngrid_in_ring = Ngrid_in_ring
        self.amu = amu
        self.parea = parea
        self.rs = rs

        self.vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian

        #array copies, for the vectorised parts (the lists above are kept for generate_ff)
        self._Nin_arr = np.asarray(Ngrid_in_ring, dtype=np.int64)
        self._parea_arr = np.asarray(parea, dtype=np.float64)

        #rotation velocity of every cell [m/s], for the Doppler shift of the CCF (same formulas as flux_grid / old StarSim)
        self.stellar_radius = stellar_radius
        theta = np.arccos(zs*np.cos(-inclination)-xs*np.sin(-inclination))
        phi = np.arctan2(ys, xs*np.cos(-inclination)+zs*np.sin(-inclination))
        vsini = 1000*2*np.pi*(stellar_radius*696342)*np.cos(inclination)/(rotation_period*86400)
        self.vsini = vsini
        self.rvel = np.ascontiguousarray(vsini*np.sin(theta)*np.sin(phi), dtype=np.float64)

    def compute_regions_position(self, t):
        pos=np.zeros([len(self.active_regions_map),3])

        for i in range(len(self.active_regions_map)):
            tini = self.active_regions_map[i].appearance_time #time of spot apparence
            dur = self.active_regions_map[i].lifetime #duration of the spot
            tfin = tini + dur #final time of spot
            colat = self.active_regions_map[i].latitude #colatitude
            lat = 90 - colat #latitude
            longi = self.active_regions_map[i].longitude #longitude
            rad = self.active_regions_map[i].size #coefficients for the evolution od the radius. Depends on the desired law.
            #TODO add generalized evoltuion law

            #update longitude adding diff rotation
            pht = longi + (t-self.active_regions_map[i].reference_time)/self.rotation_period%1*360 + (t-self.active_regions_map[i].reference_time)*self.differential_rotation/(2.66)*(1.698*np.sin(np.deg2rad(lat))**2+2.346*np.sin(np.deg2rad(lat))**4)
            phsr = pht%360 #make the phase between 0 and 360. 
            
            pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad)])
            #return position and radii of spots at t in radians.

        return pos

    def true_anomaly(x,period,ecc,tperi):
        sinf=[]
        cosf=[]
        for i in range(len(x)):
            fmean=2.0*np.pi*(x[i]-tperi)/period
            #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
            fecc=fmean
            diff=1.0
            while(diff>1.0E-6):
                fecc_0=fecc
                fecc=fecc_0-(fecc_0-ecc*np.sin(fecc_0)-fmean)/(1.0-ecc*np.cos(fecc_0))
                diff=np.abs(fecc-fecc_0)
            sinf.append(np.sqrt(1.0-ecc*ecc)*np.sin(fecc)/(1.0-ecc*np.cos(fecc)))
            cosf.append((np.cos(fecc)-ecc)/(1.0-ecc*np.cos(fecc)))
        return np.array(sinf),np.array(cosf)


    def Ttrans_2_Tperi(T0, P, e, w):

        f = np.pi/2 - w
        E = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-e)/(1+e)))  # eccentric anomaly
        Tp = T0 - P/(2*np.pi) * (E - e*np.sin(E))      # time of periastron

        return Tp



    def compute_planet_pos(self,t):
        
        if(self.planet_esinw==0 and self.planet_ecosw==0):
            ecc=0
            omega=0
        else:
            ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
            omega=np.arctan2(self.planet_esinw,self.planet_ecosw)

        t_peri = Ttrans_2_Tperi(self.planet_transit_t0,self.planet_period, ecc, omega)
        sinf,cosf=true_anomaly([t],self.planet_period,ecc,t_peri)


        cosftrueomega=cosf*np.cos(omega+np.pi/2)-sinf*np.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
        sinftrueomega=cosf*np.sin(omega+np.pi/2)+sinf*np.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

        if cosftrueomega>0.0: return np.array([1+self.planet_radius*2, 0.0, self.planet_radius]) #avoid secondary transits

        cosi = (self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

        rpl=self.planet_semi_major_axis*(1-ecc**2)/(1+ecc*cosf)
        xpl=rpl*(-np.cos(self.planet_spin_orbit_angle)*sinftrueomega-np.sin(self.planet_spin_orbit_angle)*cosftrueomega*cosi)
        ypl=rpl*(np.sin(self.planet_spin_orbit_angle)*sinftrueomega-np.cos(self.planet_spin_orbit_angle)*cosftrueomega*cosi)

        rhopl=np.sqrt(ypl**2+xpl**2)
        thpl=np.arctan2(ypl,xpl)

        pos=np.array([float(rhopl), float(thpl), self.planet_radius]) #rho, theta, and radii (in Rstar) of the planet
        return pos

    def compute_ccf_params(self, rv=None, ccf=None, plot_test=False):
        '''
        Compute the parameters of the CCFs and their bisector span (10-40% bottom minus 60-90% top).
        By default the CCFs are self.ccf_var on the velocity grid self.ccf_rv.
        Returns rvs, contrast, fwhm, BIS, raw_xbis, raw_ybis: rv, contrast and fwhm come from a gaussian fit, BIS is the
        bisector span, raw_xbis / raw_ybis are the bisectors (velocity and height) of every CCF.
        The CCFs are not modified.
        '''
        rv = np.ascontiguousarray(self.ccf_rv if rv is None else rv, dtype=np.float64)
        ccf = np.atleast_2d(self.ccf_var if ccf is None else ccf)
        rvs=np.zeros(len(ccf)) #initialize
        fwhm=np.zeros(len(ccf))
        contrast=np.zeros(len(ccf))
        BIS=np.zeros(len(ccf))
        #bisector F/F_c
        raw_xbis = []
        raw_ybis = []
        done = {} #identical CCFs (e.g. the epochs without any visible region) have identical parameters
        n_failed = 0

        for i in range(len(ccf)): #loop for each ccf
            c = np.ascontiguousarray(ccf[i] - ccf[i].min() + 0.000001, dtype=np.float64) #shifted to 0.0 (on a copy, ccf is not changed)
            key = c.tobytes()
            if key in done:
                rvs[i], contrast[i], fwhm[i], BIS[i], xbis, ybis = done[key]
                raw_xbis.append(xbis)
                raw_ybis.append(ybis)
                continue

            #Compute bisector and remove wings
            cutleft0,cutright0,xbis,ybis=nbspectra.speed_bisector_nb(rv,c/c.max(),integrated_bis=True)

            raw_xbis.append(xbis)
            raw_ybis.append(ybis)
            BIS[i]=np.mean(xbis[np.array(ybis>=0.1) & np.array(ybis<=0.4)])-np.mean(xbis[np.array(ybis<=0.9) & np.array(ybis>=0.6)])
            if i==0:
                cutleft,cutright=cutleft0,cutright0
            try:
                popt,_=optimize.curve_fit(nbspectra.gaussian2, rv[cutleft:cutright], c[cutleft:cutright],p0=[np.max(c[cutleft:cutright]),rv[cutleft:cutright][np.argmax(c[cutleft:cutright])]+100,1.5*self.vsini+1000,0.000001]) #fit a gaussian
            except Exception:
                popt=[1.0,100000.0,1,100000.0]
                n_failed += 1
            contrast[i]=popt[0] #amplitude
            rvs[i]=popt[1] #mean
            fwhm[i]=2*m.sqrt(2*np.log(2))*np.abs(popt[2]) #fwhm relation to std
            done[key] = (rvs[i], contrast[i], fwhm[i], BIS[i], xbis, ybis)

            if plot_test: 
                plt.plot(xbis,1-ybis,'b')
                plt.show(block=True)

        if n_failed:
            warnings.warn("the gaussian fit failed for %d of %d CCFs: they have the placeholder values rv=100000 m/s, contrast=1" % (n_failed, len(ccf)))
        return rvs, contrast, fwhm, BIS, raw_xbis, raw_ybis

    def _flux_grids(self, ndim, mode_name):
        """Quiet / spot / facula flux grids as float arrays: (n_cells,) for photometry, (n_cells, n_wv) for spectroscopy."""
        out = []
        for key, g in (('qp', self.flux_grid_qp), ('sp', self.flux_grid_sp), ('fc', self.flux_grid_fc)):
            if g is None:
                raise ValueError("mode '%s' needs flux_grid_qp, flux_grid_sp and flux_grid_fc (flux_grid_%s is missing)" % (mode_name, key))
            g = np.ascontiguousarray(g, dtype=np.float64)
            if g.ndim != ndim or g.shape[0] != len(self.vec_grid):
                raise ValueError("mode '%s' needs %dD flux grids with %d cells, flux_grid_%s has shape %s (photometry and "
                                 "spectroscopy together need two different sets of grids)" % (mode_name, ndim, len(self.vec_grid), key, g.shape))
            out.append(g)
        return out

    def _build_ccf_cells(self):
        """CCF of every cell for the quiet photosphere, the spots and the faculae: array (n_cells, n_rv) each.
        The CCF of every ring (ccf_grid_xx.ccf_rings) is placed on its velocity axis (ccf_grid_xx.rvs_ring), shifted by the
        rotation velocity of the cell (Doppler shift) and weighted by the area of the cell, as in the old
        compute_immaculate_sphere_rv."""
        grids = (('qp', self.ccf_grid_qp), ('sp', self.ccf_grid_sp), ('fc', self.ccf_grid_fc))
        for key, g in grids:
            if g is None:
                raise ValueError("mode 'ccf' needs ccf_grid_qp, ccf_grid_sp and ccf_grid_fc (ccf_grid_%s is missing)" % key)
            if g.ccf_rings is None or g.rvs_ring is None:
                raise RuntimeError("ccf_grid_%s has no CCFs yet: run its built_grid() first" % key)
            if g.N_rings != self.N_rings:
                raise ValueError("ccf_grid_%s has %d rings but StarSim has %d" % (key, g.N_rings, self.N_rings))
        rv = np.ascontiguousarray(self.ccf_grid_qp.rvs, dtype=np.float64)
        weight = self._parea_arr / (4 * np.pi)                      #brightness of one cell, as in the old code
        n_cells = int(self._Nin_arr.sum())
        cells = []
        for key, g in grids:
            if not np.array_equal(rv, g.rvs):
                raise ValueError("the three CCF grids must have the same velocity grid")
            ccf_ring = np.ascontiguousarray(np.asarray(g.ccf_rings, dtype=np.float64) * weight[:, None])
            rvs_ring = np.ascontiguousarray(g.rvs_ring, dtype=np.float64)
            cells.append(nbspectra.loop_compute_immaculate_nb(self.N_rings, self._Nin_arr, np.zeros([n_cells, len(rv)]),
                                                              self.rvel, rv, rvs_ring, ccf_ring))
        self.ccf_rv = rv
        return cells

    def generate_timeseries(self, ccf_params=True):
        '''Loop for all the epochs and assign, for every observable in self.mode, the signal of the grid elements:
        'photometry'   -> self.flux_var  (n_times,)
        'spectroscopy' -> self.spec_var  (n_times, n_wavelengths)
        'ccf'          -> self.ccf_var   (n_times, n_rv), velocities in self.ccf_rv, and (if ccf_params) the parameters of the CCFs:
                          self.rv_var, self.contrast_var, self.fwhm_var, self.bis_var (n_times,) and the bisectors
                          self.xbis_var, self.ybis_var (n_times, 50)  [see compute_ccf_params]
        The geometry of the regions (filling factors of every cell) is computed once per epoch for all of them.
        '''
        simulate_planet=self.simulate_planet
        N = self.N_rings #Number of concentric rings
        n_times = len(self.obs_times)
        area_tot = np.dot(self.Ngrid_in_ring,self.parea) #total projected area
        active_region_types = [region.type for region in self.active_regions_map]

        #Every observable is signal(epoch) = sum over cells of ff_quiet*grid_quiet + ff_sp*grid_sp + ff_fc*grid_fc.
        #Grids are prepared once; the total of the quiet grid is the signal when no region is visible.
        signals = {}
        if 'photometry' in self.mode:
            gq, gs, gf = self._flux_grids(1, 'photometry')
            signals['photometry'] = (np.zeros(n_times), gq, gs, gf, gq.sum(axis=0))
        if 'spectroscopy' in self.mode:
            gq, gs, gf = self._flux_grids(2, 'spectroscopy')
            signals['spectroscopy'] = (np.zeros([n_times, gq.shape[1]]), gq, gs, gf, gq.sum(axis=0))
        if 'ccf' in self.mode:
            gq, gs, gf = self._build_ccf_cells()
            signals['ccf'] = (np.zeros([n_times, gq.shape[1]]), gq, gs, gf, gq.sum(axis=0))

        filling_sp=np.zeros(n_times)
        filling_ph=np.zeros(n_times)
        filling_pl=np.zeros(n_times)
        filling_fc=np.zeros(n_times)

        sys.stdout.write(" ")
        for k,t in enumerate(self.obs_times):

            if simulate_planet:
                planet_pos=compute_planet_pos(self,t)#compute the planet position at current time. In polar coordinates!! 
            else:
                planet_pos = [2.0,0.0,0.0]


            if len(self.active_regions_map)==0:
                spot_pos=np.array([np.array([m.pi/2,-m.pi,0.0,0.0])])
            else:
                spot_pos=self.compute_regions_position(t) #compute the position of all spots at the current time. Returns theta and phi of each spot.      

            vec_spot=np.zeros([len(self.active_regions_map),3])
            xspot = np.cos(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])+np.sin(self.inclination)*np.cos(spot_pos[:,0])
            yspot = np.sin(spot_pos[:,0])*np.sin(spot_pos[:,1])
            zspot = np.cos(spot_pos[:,0])*np.cos(self.inclination)-np.sin(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])
            vec_spot[:,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian

            #COMPUTE IF ANY SPOT IS VISIBLE
            vis=np.zeros(len(vec_spot)+1)
            for i in range(len(vec_spot)):
                dist = m.acos(np.dot(vec_spot[i],np.array([1,0,0])))
                
                if (dist-spot_pos[i,2])<= (np.pi/2):
                    vis[i]=1.0
            
            if (planet_pos[0]-planet_pos[2]<1):
                vis[-1]=1.0
    
            if (np.sum(vis)==0.0):
                #nothing visible: the star is the quiet photosphere
                filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k] = area_tot, 0.0, 0.0, 0.0
                for out, gq, gs, gf, quiet_total in signals.values():
                    out[k] = quiet_total

            else:
                ff_quiet, ff_sp, ff_fc, ff_p, filling_ph[k], filling_sp[k],  filling_fc[k],  filling_pl[k] = nbspectra.generate_ff(N,self.Ngrid_in_ring,self.parea,self.amu,spot_pos,self.vec_grid,vec_spot,self.simulate_planet,planet_pos,vis, active_region_types)
                ff_quiet, ff_sp, ff_fc = np.asarray(ff_quiet), np.asarray(ff_sp), np.asarray(ff_fc)
                for out, gq, gs, gf, quiet_total in signals.values():   #one matrix product per grid instead of a loop over cells
                    out[k] = ff_quiet @ gq + ff_sp @ gs + ff_fc @ gf

            filling_ph[k]=100*filling_ph[k]/area_tot
            filling_sp[k]=100*filling_sp[k]/area_tot
            filling_fc[k]=100*filling_fc[k]/area_tot
            filling_pl[k]=100*filling_pl[k]/area_tot
            
           
            sys.stdout.write("\rDate {0}. ff_ph={1:.3f}%. ff_sp={2:.3f}%. ff_fc={3:.3f}%. ff_pl={4:.3f}%. [{5}/{6}]%".format(t,filling_ph[k],filling_sp[k],filling_fc[k],filling_pl[k],k+1,n_times))

        if 'photometry' in signals:
            self.flux_var = signals['photometry'][0]
        if 'spectroscopy' in signals:
            self.spec_var = signals['spectroscopy'][0]
        if 'ccf' in signals:
            self.ccf_var = signals['ccf'][0]
        self.ff_quiet = filling_ph
        self.ff_sp = filling_sp
        self.ff_fc = filling_fc

        #parameters of the CCFs last, so that everything above is already stored if this step fails
        if 'ccf' in signals and ccf_params:
            rvs, contrast, fwhm, bis, xbis, ybis = self.compute_ccf_params()
            self.rv_var, self.contrast_var, self.fwhm_var, self.bis_var = rvs, contrast, fwhm, bis
            self.xbis_var, self.ybis_var = np.array(xbis), np.array(ybis)