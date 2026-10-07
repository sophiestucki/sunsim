import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import optimize
import numpy as np
from pathlib import Path
from scipy import interpolate
import scipy.integrate as integrate
from scipy.ndimage import median_filter


import sys
import math as m
import  nbspectra
import pickle
import os
import shutil
from configparser import ConfigParser
from astropy.convolution import convolve_fft 
# from raccoon import ccf

from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from astropy import units as u







########################################################################################
########################################################################################
#                                GENERAL FUNCTIONS                                     #
########################################################################################
########################################################################################

def black_body(wv,T):
    #Computes the BB flux with temperature T at wavelengths wv(in nanometers)
    c = 2.99792458e10 #speed of light in cm/s
    k = 1.380658e-16  #boltzmann constant
    h = 6.6260755e-27 #planck
    w=wv*1e-8 #Angstrom to cm
    bb=2*h*c**2*w**(-5)*(np.exp(h*c/k/T/w)-1)**(-1)
    return bb

def vacuum2air(wv): #wv in angstroms
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
########################################################################################
#                                PHOTOMETRY FUNCTIONS                                  #
########################################################################################
########################################################################################


def interpolate_Phoenix_mu_lc(self,temp,grav):
    """Cut and interpolate phoenix models at the desired wavelengths, temperatures, logg and metalicity(not yet). For spectroscopy.
    Inputs
    temp: temperature of the model; 
    grav: logg of the model
    Returns
    creates a temporal file with the interpolated spectra at the temp and grav desired, for each surface element.
    """
    #Demanar tambe la resolucio i ficarho aqui.

    import warnings
    warnings.filterwarnings("ignore")

    path = self.path / 'models' / 'Phoenix_mu' #path relatve to working directory 
    files = [x.name for x in path.glob('lte*fits') if x.is_file()]
    list_temp=np.unique([float(t[3:8]) for t in files])
    list_grav=np.unique([float(t[9:13]) for t in files])

    #check if the parameters are inside the grid of models
    if grav<np.min(list_grav) or grav>np.max(list_grav):
        sys.exit('Error in the interpolation of Phoenix_mu models. The desired logg ({}) is outside the grid of models, extrapolation is not supported. Please download the \
        Phoenix intensity models covering the desired logg from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73'.format(grav))

    if temp<np.min(list_temp) or temp>np.max(list_temp):
        print(temp, list_temp)
        sys.exit('Error in the interpolation of Phoenix_mu models. The desired T ({}) is outside the grid of models, extrapolation is not supported. Please download the \
        Phoenix intensity models covering the desired T from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73'.format(temp))
        


    lowT=list_temp[list_temp<=temp].max() #find the model with the temperature immediately below the desired temperature
    uppT=list_temp[list_temp>=temp].min() #find the model with the temperature immediately above the desired temperature
    lowg=list_grav[list_grav<=grav].max() #find the model with the logg immediately below the desired logg
    uppg=list_grav[list_grav>=grav].min() #find the model with the logg immediately above the desired logg

    #load the flux of the four phoenix model
    name_lowTlowg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'.format(int(lowT),lowg)
    name_lowTuppg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'.format(int(lowT),uppg)
    name_uppTlowg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'.format(int(uppT),lowg)
    name_uppTuppg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'.format(int(uppT),uppg)


    #Check if the files exist in the folder
    if name_lowTlowg not in files:
        sys.exit('The file '+name_lowTlowg+' required for the interpolation does not exist. Please download it from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73 and add it to your path: '+str(path))
    if name_lowTuppg not in files:
        sys.exit('The file '+name_lowTuppg+' required for the interpolation does not exist. Please download it from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73 and add it to your path: '+path)
    if name_uppTlowg not in files:
        sys.exit('The file '+name_uppTlowg+' required for the interpolation does not exist. Please download it from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73 and add it to your path: '+path)
    if name_uppTuppg not in files:
        sys.exit('The file '+name_uppTuppg+' required for the interpolation does not exist. Please download it from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73 and add it to your path: '+path)
 
    wavelength=np.arange(500,26000) #wavelength in A
    idx_wv=np.array(wavelength>self.wavelength_lower_limit) & np.array(wavelength<self.wavelength_upper_limit)

    #read flux files and cut at the desired wavelengths
    with fits.open(path / name_lowTlowg) as hdul:
        amu = hdul[1].data
        amu = np.append(amu[::-1],0.0)
        flux_lowTlowg=hdul[0].data[:,idx_wv]
    with fits.open(path / name_lowTuppg) as hdul:
        flux_lowTuppg=hdul[0].data[:,idx_wv]
    with fits.open(path / name_uppTlowg) as hdul:
        flux_uppTlowg=hdul[0].data[:,idx_wv]
    with fits.open(path / name_uppTuppg) as hdul:
        flux_uppTuppg=hdul[0].data[:,idx_wv]

    #interpolate in temperature for the two gravities
    if uppT==lowT: #to avoid nans
        flux_lowg = flux_lowTlowg 
        flux_uppg = flux_lowTuppg
    else:
        flux_lowg = flux_lowTlowg + ( (temp - lowT) / (uppT - lowT) ) * (flux_uppTlowg - flux_lowTlowg)
        flux_uppg = flux_lowTuppg + ( (temp - lowT) / (uppT - lowT) ) * (flux_uppTuppg - flux_lowTuppg)
    #interpolate in log g
    if uppg==lowg: #to avoid dividing by 0
        flux = flux_lowg
    else:
        flux = flux_lowg + ( (grav - lowg) / (uppg - lowg) ) * (flux_uppg - flux_lowg)



    angle0 = flux[0]*0.0 #LD of 90 deg, to avoid dividing by 0? (not sure, ask Kike)

    flux_joint = np.vstack([flux[::-1],angle0]) #add LD coeffs at 0 and 1 proj angles
    # flpk=flux_joint[0]*np.pi*np.sin(np.cos(amu[0]))**2#Add all fluxes of all angles multiplied by their areas to compute the integrated flux
    # for i in range(1,len(amu)):
    #     flpk=flpk+flux_joint[i]*(np.sin(np.cos(amu[i]))**2-np.sin(np.cos(amu[i-1]))**2)*np.pi



    return amu, wavelength[idx_wv], flux_joint


def add_resol(wavelength, flux, instrument):
#"""
  # This function is mostly based on the SteParSyn broadener (Tabernero et al. 2022) 
  # SteParsyn is under the two-clause BSD licence, I added a disclaimer to take this into account
  # Input is wavelength in A, and flux in any unit. If input is in RV you should convert from RV to wavelength by assuming a central lambda (i.e. 6705.1 A).
        vstep=1
        vlight=2.99792458e5
        vwave=vlambda(wavelength,vstep)
        xw2=max(wavelength)-0.1
        xw1=min(wavelength)+0.1
        iw1=np.where(wavelength > xw1)
        iw2=np.where(wavelength > xw2)
        iw1=iw1[0]
        iw2=iw2[0]
        w1=wavelength[iw1[0]+1]
        w2=wavelength[iw2[0]-1]
        tck=interpolate.splrep(wavelength,flux,k=3, s=0)
        vflux=interpolate.splev(vwave,tck,der=0)
        wmid=np.sqrt(w1*w2)

        x1=vwave
        y1=vflux

        if instrument == 'EXPRESS':
          Resolution = 137000.
          kop ='g'
        elif instrument == 'HARPS':
          Resolution = 115000.
          kop = 'g'
        elif instrument == 'HARPS-N':
          Resolution = 118000.
          kop = 'g'
        elif instrument == 'NEID':
          Resolution = 120000. 
          kop = 'g'
      
        if kop == 'g': 
            vibr=(vlight/Resolution)/(2.*np.sqrt(2.*np.log(2.)))
            sigma=vibr*wmid/vlight
            nx1=len(x1)
            dx1=(x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            a1=0.
            a2=sigma
            zk=(xk-a1)/a2
            a0=1./np.sqrt(2.*np.pi)/sigma
            yk=a0*np.exp(-(zk**2.)/2.)

        nfact = np.sum(yk)
        outflux = convolve_fft(y1, yk/nfact, boundary='fill',fill_value=1.)  
  
        #flux in resampled back to he original sampling
        tck2 = interpolate.splrep(vwave,outflux,k=3, s=0)
        convolved_flux = interpolate.splev(wavelength, tck2, der=0)

        return convolved_flux

def mask_weights(pathmask):
    try:
        d = np.loadtxt(pathmask,unpack=True)
        if len(d) == 2:
            wvm = d[0]
            fm = d[1]
        elif len(d) == 3:
            wvm = np.atleast_1d(air2vacuum((d[0]+d[1])/2)) #HARPS mask ar in air, not vacuum
            fm = np.atleast_1d(d[2])
        else:
            sys.exit('Mask format not valid. Must have two (wv and weight) or three columns (wv1 wv2 weight).')

    except:
        sys.exit('Mask file not found. Save it inside the masks folder.')
    return wvm, fm