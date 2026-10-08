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
########################################################################################
#                                PHOTOMETRY FUNCTIONS                                  #
########################################################################################
########################################################################################


def interpolate_phoenix_mu(path, temp, logg, metallicity=0.0, wavelength_lower_limit=3000.0, wavelength_upper_limit=10000.0,
                           overhead=1.0):
    """Phoenix specific intensity spectra (SPECINT models, low resolution, at several mu) interpolated (linearly) at the
    temperature, logg and metallicity [Fe/H] desired, cut at the wavelength range (+- overhead [A]). For photometry and
    low resolution spectroscopy.

    path        : folder with the Phoenix SPECINT models lte*.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits (its subfolders are
                  searched too). Download them from PHOENIX_SPECINT_URL
    temp [K], logg [cgs], metallicity [Fe/H] : must be inside the grid of available models (no extrapolation)

    Returns the spectra in the format used by flux_grid / CCF_grid, {mu: {'wav': wavelengths [A], 'intensity': intensity}},
    one key for every mu of the models (the mu are read from the second extension of the files). The intensity is the
    Phoenix specific intensity [erg/s/cm^2/cm/sr]. The wavelength grid of the SPECINT models is 1 A from 500 A.
    Below the smallest mu, flux_grid extrapolates the intensity linearly to 0 at mu = 0 (as the zero row at mu = 0 that
    the previous version of this function added).
    """
    weights = phoenix_weights(Path(path), temp, logg, metallicity, 'SPECINT')

    amu = None
    intensity = None
    for model, w in weights.items():
        with fits.open(model) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)     #(n_mu, n_wavelengths)
            mu_model = np.asarray(hdul[1].data, dtype=np.float64).ravel()
        if amu is None:
            amu = mu_model
            wavelength = 500.0 + np.arange(data.shape[1])          #wavelength in A
            idx_wv = (wavelength > wavelength_lower_limit - overhead) & (wavelength < wavelength_upper_limit + overhead)
            intensity = np.zeros((len(amu), idx_wv.sum()))
        elif not np.array_equal(mu_model, amu):
            raise ValueError('The Phoenix SPECINT models used for the interpolation have different mu angles (%s)' % model.name)
        intensity += w * data[:, idx_wv]

    wv = wavelength[idx_wv]
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


def phoenix_weights(path, temp, logg, metallicity, kind='HiRes'):
    """Models and weights of the (tri)linear interpolation in T, logg and [Fe/H] between the closest Phoenix models.
    Returns {file path: weight} (only the models with weight > 0; the weights add up to 1).
    Raises an error if the parameters are outside the grid (no extrapolation) or if a model needed is missing."""
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
    x_bin, y_bin = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = np.flatnonzero((wv >= a) & (wv < b))
        if len(sel):
            k = sel[np.argmax(flux[sel])]
            x_bin.append(wv[k])
            y_bin.append(flux[k])
    x_bin, y_bin = np.array(x_bin), np.array(y_bin)
    #Polynomial.fit rescales the wavelengths to [-1, 1]: same polynomial as np.polyfit, without its conditioning problems
    poly = np.polynomial.Polynomial.fit(x_bin, y_bin, min(deg, len(x_bin) - 1))
    return poly(wv), x_bin, y_bin


def interpolate_phoenix(path, temp, logg, metallicity=0.0, wavelength_lower_limit=3000.0, wavelength_upper_limit=10000.0,
                        normalize=False, overhead=1.0, plot=False):
    """Phoenix HiRes spectrum interpolated (linearly) at the temperature, logg and metallicity [Fe/H] desired,
    cut at the wavelength range (+- overhead [A], to allow for Doppler shifts without losing information).

    path        : folder with the Phoenix HiRes models (and its subfolders) and WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
                  (download them from PHOENIX_URL)
    temp [K], logg [cgs], metallicity [Fe/H] : must be inside the grid of available models (no extrapolation)
    normalize   : True -> the flux is divided by its continuum (phoenix_continuum); False -> Phoenix flux [erg/s/cm^2/cm]
    plot        : plot the spectrum and its continuum (only with normalize=True)

    Returns the spectrum in the format used by flux_grid / CCF_grid, {mu: {'wav': wavelengths [A], 'intensity': flux}},
    with the single key mu = 1.0: the HiRes models are disc-integrated, they have no mu dependence. Use it with
    CCF_grid(..., mu_ratio=...), which accepts one spectrum at mu = 1.0. flux_grid needs spectra at several mu
    (interpolate_phoenix_mu, with the SPECINT models). The wavelengths are in vacuum.
    With normalize=True, the dictionary also has 'flux' (not normalized) and 'continuum'.
    """
    path = Path(path)
    weights = phoenix_weights(path, temp, logg, metallicity, 'HiRes')

    #Phoenix wavelengths, cut at the desired range
    wave_file = next(path.rglob('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'), None)
    if wave_file is None:
        raise FileNotFoundError('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits not found in %s. Download it from %s' % (path, PHOENIX_URL))
    with fits.open(wave_file) as hdul:
        wavelength = np.asarray(hdul[0].data, dtype=np.float64)
    idx_wv = (wavelength > wavelength_lower_limit - overhead) & (wavelength < wavelength_upper_limit + overhead)
    wv = wavelength[idx_wv]

    #trilinear interpolation: weighted sum of the corner models
    flux = np.zeros(len(wv))
    for model, w in weights.items():
        with fits.open(model) as hdul:
            flux += w * np.asarray(hdul[0].data, dtype=np.float64)[idx_wv]

    spectrum = {'wav': wv, 'intensity': flux}
    if normalize:
        continuum, x_bin, y_bin = phoenix_continuum(wv, flux, wavelength_lower_limit - overhead, wavelength_upper_limit + overhead)
        spectrum = {'wav': wv, 'intensity': flux / continuum, 'flux': flux, 'continuum': continuum}
        if plot:      #to check the normalisation
            plt.plot(wv, flux)
            plt.plot(x_bin, y_bin, 'ok')
            plt.plot(wv, continuum)
            plt.xlabel('wavelength [$\\AA$]')
            plt.show()
            plt.close()

    return {1.0: spectrum}


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