import tqdm
import numpy as np
import emcee
import gc
from pathlib import Path
from multiprocessing import Pool
import sys
import os
import corner
from configparser import ConfigParser
#import traceback
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib as mpl
import math as m
import collections
from astropy.io import fits
import json
from scipy.interpolate import interp1d
from scipy import optimize
import pandas as pd


from . import spectra
from . import nbspectra
from . import SA

from numba import jit
import starsim
#initialize numba
nbspectra.dummy()

class StarSim(object): 
    """
    Main Starsim class. Reads the configuration file and store the options into variables.
    """
    def __init__(self,conf_file_path='starsim.conf'):
            self.path = Path(__file__).parent #path of working directory
            self.conf_file_path = self.path / conf_file_path
            self.conf_file = self.__conf_init() 
            #files
            self.filter_name =  str(self.conf_file.get('files','filter_name'))
            self.orders_CRX_filename= str(self.conf_file.get('files','orders_CRX_filename'))
            #general
            self.simulation_mode = str(self.conf_file.get('general','simulation_mode'))
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit'))
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))
            self.n_grid_rings = int(self.conf_file.get('general','n_grid_rings'))
            self.spectral_library = str(self.conf_file.get('general','spectral_library'))
            self.mps_spectra_folder = str(self.conf_file.get('general','mps_spectra_folder'))
            self.stagger_spectra_folder = str(self.conf_file.get('general','stagger_spectra_folder'))
            self.mps_mu_grid = int(self.conf_file.get('general','mps_mu_grid'))
            self.ccf_norm = int(self.conf_file.get('general','ccf_norm'))
            self.flux_flat = int(self.conf_file.get('general','flux_flat'))
            self.highres_mu = int(self.conf_file.get('general','highres_mu'))
            self.smoothing_kernel = int(self.conf_file.get('general','smoothing_kernel'))
            self.remove_bis_spot = int(self.conf_file.get('general','remove_bis_spot'))


            #star
            self.radius = float(self.conf_file.get('star','radius')) #Radius of the star in solar radii
            self.mass = float(self.conf_file.get('star','mass')) #Mass of the star in solar radii
            self.rotation_period = float(self.conf_file.get('star','rotation_period')) #Rotation period in days
            self.inclination = np.deg2rad(90-float(self.conf_file.get('star','inclination'))) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
            self.temperature_photosphere = float(self.conf_file.get('star','temperature_photosphere'))
            self.spot_T_contrast = float(self.conf_file.get('star','spot_T_contrast'))
            self.facula_T_contrast = float(self.conf_file.get('star','facula_T_contrast'))
            self.convective_shift = float(self.conf_file.get('star','convective_shift'))#CB in m/s
            self.logg = float(self.conf_file.get('star','logg'))
            self.differential_rotation = float(self.conf_file.get('star','differential_rotation'))
            #rv
            self.ccf_template = str(self.conf_file.get('rv','ccf_template'))
            self.ccf_mask = str(self.conf_file.get('rv','ccf_mask'))
            self.ccf_weight_lines = int(self.conf_file.get('rv','ccf_weight_lines'))
            self.path_weight_lines = str(self.conf_file.get('rv','path_weight_lines'))
            self.ccf_rv_range= float(self.conf_file.get('rv','ccf_rv_range'))*1000 #in m/s
            self.ccf_rv_step= float(self.conf_file.get('rv','ccf_rv_step'))*1000 #in m/s
            self.kind_interp = 'cubic'#str(self.conf_file.get('rv','ccf_interpolation_spectra'))
            self.instrument_ccf = str(self.conf_file.get('rv','instrument'))
            #limb-darkening
            self.use_phoenix_limb_darkening = int(self.conf_file.get('LD','use_phoenix_limb_darkening'))
            self.limb_darkening_law = str(self.conf_file.get('LD','limb_darkening_law'))
            self.limb_darkening_q1= float(self.conf_file.get('LD','limb_darkening_q1'))
            self.limb_darkening_q2= float(self.conf_file.get('LD','limb_darkening_q2'))
            self.limb_extrapolation= str(self.conf_file.get('LD','limb_extrapolation'))
            self.limb_lim= float(self.conf_file.get('LD','limb_lim'))
            self.meunier = int(self.conf_file.get('LD','meunier'))



            #spots
            self.spots_evo_law = str(self.conf_file.get('spots','spots_evo_law'))
            self.plot_grid_map = int(self.conf_file.get('spots','plot_grid_map'))
            self.reference_time = float(self.conf_file.get('spots','reference_time'))
            self.sdo_input = int(self.conf_file.get('spots', 'sdo_input'))
            self.sdo_input_path=str(self.conf_file.get('spots', 'SDO_input_path'))


            #planet
            self.planet_period = float(self.conf_file.get('planet','planet_period')) #in days
            self.planet_transit_t0 = float(self.conf_file.get('planet','planet_transit_t0')) #in days
            self.planet_radius = float(self.conf_file.get('planet','planet_radius')) #in R* units
            self.planet_impact_param = float(self.conf_file.get('planet','planet_impact_param')) #from 0 to 1
            self.planet_spin_orbit_angle = float(self.conf_file.get('planet','planet_spin_orbit_angle'))*np.pi/180 #in deg
            self.simulate_planet=int(self.conf_file.get('planet','simulate_planet'))
            self.planet_semi_amplitude = float(self.conf_file.get('planet','planet_semi_amplitude')) #in m/s
            self.planet_esinw = float(self.conf_file.get('planet','planet_esinw')) 
            self.planet_ecosw = float(self.conf_file.get('planet','planet_ecosw')) 

            
            #from Òscar implementation
            self.return_Imu = 0 # Returns I(mu) in order to get limb darkening law.
            self.convolve_spec_with_filt = False # If True, it convolves self.results['spec'] with the selected filter in order to get the light curve, which is stored in self.results['spec_lc'].
            self.prova = "lol"


            #FUNCTIONS USED TO ADD BISECTORS TO THE PHOTOSPHERE AT DIFF ANGLES
            self.fun_coeff_bisectors_amu = spectra.cifist_coeff_interpolate
            self.fun_coeff_bisector_spots = spectra.dumusque_coeffs
            self.fun_coeff_bisector_faculae = spectra.dumusque_coeffs


            #initialize other variables to store results
            self.data = collections.defaultdict(dict) #dictionary to store input data
            self.instruments = []
            self.observables=[]
            self.results = {} #initialize results attribute. It will be a dictionary containing the results of the method forward (maybe also inverse)
            self.name_params = {'rv': 'RV\n[m/s]', 'contrast': 'CCF$_{{cont}}$', 'fwhm':'CCF$_{{FWHM}}$\n[m/s]', 'bis':'CCF$_{{BIS}}$\n[m/s]', 
            'lc':'Norm. flux', 'ff_sp': 'ff$_{{spot}}$ \n[%]','ff_ph': 'ff$_{{phot}}$ \n[%]','ff_fc': 'ff$_{{fac}}$ \n[%]','ff_pl': 'ff$_{{pl}}$ \n[%]',
            'crx':'CRX \n[m/s/Np]','ccx':'C$_{{Cont}}$X \n[1/Np]','cfx':'C$_{{FWHM}}$X \n[m/s/Np]','cbx':'C$_{{BIS}}$X \n[m/s/Np]'}
            self.rvo = None #initialize
            self.conto = None #initialize
            self.fwhmo = None #initialize
            self.planet_impact_paramiso = None #initialize

            #read and check spotmap
            pathspots = self.path / 'spotmap.dat' #path relatve to working directory 
            self.spot_map=np.loadtxt(pathspots)
            #TODO sort the spot map to have spot before

            if self.spot_map.ndim == 1:
                self.spot_map = np.array([self.spot_map]) #to avoid future errors
            elif self.spot_map.ndim == 0:
                sys.exit('The spot map file spotmap.dat is empty')
            self.spot_map = self.spot_map[np.argsort(self.spot_map[:,0]),:]
            self.active_region_types = [self.spot_map[i][0] for i in range(len(self.spot_map))]
            self.facular_area_ratio = [0.0 for i in range(len(self.spot_map))]

            # SDO input
            if self.sdo_input:
                spot_map_list = []
                faculae_map_list = []
                t = []

                for filename in os.listdir(self.sdo_input_path):
                    if filename.startswith("spot_map_") and filename.endswith(".txt"):
                        time_str = filename[len("spot_map_"):-len(".txt")]
                        time_val = float(time_str)
                        data = np.loadtxt(os.path.join(self.sdo_input_path, filename))
                        spot_map_list.append(data)
                        t.append(time_val)

    

                t = np.array(t)
                sort_idx = np.argsort(t)
                t = t[sort_idx]
                spot_map_list = [spot_map_list[i] for i in sort_idx]
                
                for t_i in t:
                        data = np.loadtxt(os.path.join(self.sdo_input_path, 'faculae_map_{}.txt').format(t_i))
                        # data = np.loadtxt(os.path.join(self.sdo_input_path, 'network_map_{}.txt').format(t_i))

                        faculae_map_list.append(data)

                self.obs_times = t
                self.maps_sp = spot_map_list
                self.maps_fc = faculae_map_list


            

            #select mode
            if self.simulation_mode == 'grid':
                pass
            elif self.simulation_mode == 'fast':
                pass
            else: 
                sys.exit('simulation_mode in configuration file is not valid. Valid modes are "fast" or "grid".')

            #mode to select the template used to compute the ccf. Model are Phoenix models, mask are custom masks. 
            if self.ccf_template == 'model': #use phoenix models
                pass
            elif self.ccf_template == 'mask': #use maske
                pathmask = self.path / 'masks' / self.ccf_mask
                try:
                    d = np.loadtxt(pathmask,unpack=True)
                    if len(d) == 2:
                        self.wvm = d[0]
                        self.fm = d[1]
                    elif len(d) == 3:
                        self.wvm = np.atleast_1d(spectra.air2vacuum((d[0]+d[1])/2)) #HARPS mask ar in air, not vacuum
                        self.fm = np.atleast_1d(d[2])
                    else:
                        sys.exit('Mask format not valid. Must have two (wv and weight) or three columns (wv1 wv2 weight).')

                except:
                    sys.exit('Mask file not found. Save it inside the masks folder.')


                if self.ccf_weight_lines:
                    pathweight = self.path / 'masks' / self.path_weight_lines
                    try:
                        order, order_weight, order_wvi, order_wvf = np.loadtxt(pathweight,unpack=True)
                        self.wvm, self.fm = nbspectra.weight_mask(order_wvi,order_wvf,order_weight,self.wvm,self.fm)
                    except:
                        sys.exit('File containing the weights of each order cold not be found/read. Make sure the file is inside the masks folder, and that it have 4 columns: order num, weight, initial wavelength, final wavelength')

                #Finally, set the wavlength range around the available lines
                self.wavelength_upper_limit=self.wvm.max()+1
                self.wavelength_lower_limit=self.wvm.min()-1

                self.fm /= np.sum(self.fm)
                


            else:
                sys.exit('ccf_template in configuration file is not valid. Valid modes are "model" or "mask".')

    @property
    def temperature_spot(self):
        return self.temperature_photosphere - self.spot_T_contrast

    @property
    def temperature_facula(self):
        return self.temperature_photosphere + self.facula_T_contrast

    @property
    def vsini(self):
        return 1000*2*np.pi*(self.radius*696342)*np.cos(self.inclination)/(self.rotation_period*86400) #vsini in m/s

    @property
    def planet_semi_major_axis(self):
        return 4.2097*self.planet_period**(2/3)*self.mass**(1/3)/self.radius #semi major axis in stellar radius units 


    def __conf_init(self):
        """creates an instance of class ConfigParser, and read the .conf file. Returns the object created
        ConfigParser, and read the .conf file. Returns the object created
        """
        conf_file_Object = ConfigParser(inline_comment_prefixes='#')
        if not conf_file_Object.read([self.conf_file_path]):
            print("The configuration file in" + str(self.conf_file_path) + " could not be read, please check that the format and/or path is are correct")
            sys.exit()
        else:
            return conf_file_Object


    def set_stellar_parameters(self,p):
        """Set the stellar parameters that have been optimized.
        """
        self.temperature_photosphere = p[0]
        self.spot_T_contrast = p[1]
        self.facula_T_contrast = p[2]
        self.facular_area_ratio = p[3]
        self.convective_shift = p[4]
        self.rotation_period = p[5]
        self.inclination = np.deg2rad(90-p[6]) #axis inclinations in rad (inc=0 has the axis pointing up). The input was in deg defined as usual.
        self.radius = p[7] #in Rsun
        self.limb_darkening_q1 = p[8]
        self.limb_darkening_q2 = p[9]
        self.planet_period = p[10]
        self.planet_transit_t0 = p[11]
        self.planet_semi_amplitude = p[12]
        self.planet_esinw = p[13]
        self.planet_ecosw = p[14]
        self.planet_radius = p[15]
        self.planet_impact_param = p[16]
        self.planet_spin_orbit_angle = p[17]*np.pi/180 #deg2rad    


    def compute_forward(self,observables=['lc'],t=None,inversion=False, w='eq_2'):
    # ##############################################################################
    # #SDO: takes spot & faculae maps as input
    # ##############################################################################

    #     if self.sdo_input:
    #         sdo.forward_model_sdo(self, observables, inversion, t)
                
    # #########################################################################################
    # ### Standard simulations
    # #########################################################################################
    #     else:
    #         regions.forward_model(self, observables, inversion, t)

        if inversion==False:
            self.wavelength_lower_limit = float(self.conf_file.get('general','wavelength_lower_limit')) #Repeat this just in case CRX has modified the values
            self.wavelength_upper_limit = float(self.conf_file.get('general','wavelength_upper_limit'))

        if self.sdo_input==0:
            if t is None:
                sys.exit('Please provide a valid time in compute_forward(observables,t=time)')

            self.obs_times = t


        Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(self.n_grid_rings)

        vec_grid = np.array([xs,ys,zs]).T #coordinates in cartesian
        theta, phi = np.arccos(zs*np.cos(-self.inclination)-xs*np.sin(-self.inclination)), np.arctan2(ys,xs*np.cos(-self.inclination)+zs*np.sin(-self.inclination))#coordinates in the star reference 


        #Main core of the method. Dependingon the observables you want, use lowres or high res spectra
        if 'lc' in observables: #use LR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            
            if self.spectral_library == 'phoenix':
                #Interpolate PHOENIX intensity models, only spot and photosphere
                acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
                acd_ph, wvp_lc_ph, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)
                flnp_lc_ph = flnp_lc

                if self.mps_mu_grid:
                    new_flns_lc = []
                    new_flnp_lc = []
                    for mu_t in np.linspace(0.1,1.0,10): #Loop for each ring, to compute the flux of the star.   

                    #Interpolate Phoenix intensity models to correct projected ange:
                        acd_low=np.max(acd[acd<mu_t]) #angles above and below the proj. angle of the grid
                        acd_upp=np.min(acd[acd>=mu_t])
                        idx_low=np.where(acd==acd_low)[0][0]
                        idx_upp=np.where(acd==acd_upp)[0][0]

                        new_flns_lc.append(flns_lc[idx_low]+(flns_lc[idx_upp]-flns_lc[idx_low])*(mu_t-acd_low)/(acd_upp-acd_low))
                        new_flnp_lc.append(flnp_lc[idx_low]+(flnp_lc[idx_upp]-flnp_lc[idx_low])*(mu_t-acd_low)/(acd_upp-acd_low))

                    
                        
                    flnp_lc = new_flnp_lc
                    flns_lc = new_flns_lc
                    acd = np.linspace(0.1,1.0,10)
                flfc_lc = np.copy(flnp_lc)


                
            elif self.spectral_library == 'mps' or self.spectral_library == 'mps_highres':
                acd, wvp_lc, flnp_lc = spectra.load_MPS_ATLAS_spectra_lc(self, 'ph')

                flns_lc = spectra.black_body(wvp_lc, self.temperature_spot) / spectra.black_body(wvp_lc, self.temperature_photosphere) * flnp_lc
                
                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                    acd, wvp_lc, flfc_lc = spectra.load_MPS_ATLAS_spectra_lc(self, 'fc')
                else:
                    flfc_lc = flnp_lc




            if self.limb_lim < acd.min():
                self.limb_lim = acd.min()
                print('Limb limit too small, set to the minimal mu of the spectra librairy: ', self.limb_lim)

            
            

            #Read filter and interpolate it in order to convolve it with the spectra
            f_filt = spectra.interpolate_filter(self)

            
            
            if self.simulation_mode == 'grid':
                brigh_grid_ph, flx_ph = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,f_filt,wvp_lc,'ph') #returns spectrum of grid in ring N, its brightness, and the total fl
                brigh_grid_sp, flx_sp = spectra.compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flns_lc,f_filt,wvp_lc,'sp') #returns spectrum of grid in ring N, its brightness, and the total flux
                brigh_grid_fc, flx_fc = brigh_grid_sp, flx_ph #if there are no faculae
                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                    brigh_grid_fc, flx_fc = spectra.compute_immaculate_facula_lc(self,Ngrid_in_ring,acd,amu,pare,flfc_lc,f_filt,wvp_lc) #returns spectrum of grid in ring N, its brightness, and the total flux
                
                if self.sdo_input:
                    FLUX,ff_ph,ff_sp,ff_fc,ff_pl, typ=spectra.generate_rotating_photosphere_lc_sdo(self,Ngrid_in_ring,pare,rs,brigh_grid_ph,brigh_grid_sp,brigh_grid_fc,flx_ph,inversion)
                else:
                    t,FLUX_n,FLUX, ff_ph,ff_sp,ff_fc,ff_pl,typ=spectra.generate_rotating_photosphere_lc(self,Ngrid_in_ring,pare,amu,brigh_grid_ph,brigh_grid_sp,brigh_grid_fc,flx_ph,vec_grid,inversion,plot_map=self.plot_grid_map)
                
            else:
                sys.exit("Invalid simulation mode. Only 'grid' mode is available")

            self.results['time']=t
            if self.sdo_input==0:
                self.results['lc']=FLUX_n
            self.results['intensity']=FLUX
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc
            self.results['typ'] = typ
            self.results['flux'] =FLUX
            self.results['flns_lc'] = flns_lc
            self.results['flnp_lc'] = flnp_lc
            self.results['wvp_lc'] = wvp_lc
            self.results['acd'] = acd







        ### OSCAR PART: MERGING IN PROGRESS 
        if 'spec' in observables:#Oscar: based on the procedure for the 'lc' simulation, simulate the spectra as a function of time. At the moment using LR templates as the 'lc' simulation.
            
            #Interpolate PHOENIX intensity models, only spot and photosphere
            acd, wvp_lc, flnp_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
            acd, wvp_lc, flns_lc =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)

            
            
            if self.simulation_mode == 'grid':
                spec_rings_ph, spec_ph = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,wvp_lc,'ph') #returns spectrum of grid in ring N, its brightness, and the total flux
                spec_rings_sp, spec_sp = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flns_lc,wvp_lc,'sp') #returns spectrum of grid in ring N, its brightness, and the total flux
                spec_rings_fc, spec_fc = spec_rings_ph, spec_ph #if there are no faculae
                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                    spec_rings_fc, spec_fc = spectra.compute_immaculate_spec(self,Ngrid_in_ring,acd,amu,pare,flnp_lc,wvp_lc,'fc')

                t,SPEC,ff_ph,ff_sp,ff_fc,ff_pl=spectra.generate_rotating_photosphere_spec(self,Ngrid_in_ring,pare,amu,spec_rings_ph,spec_rings_sp,spec_rings_fc,spec_ph,vec_grid,inversion,plot_map=self.plot_grid_map)
            
            
            elif self.simulation_mode == 'fast':
                sys.exit("Simulation of 'spec' is unavailable for the 'fast' mode, please use the 'grid' mode.")
            
            
            else:
                sys.exit("Invalid simulation mode. Only 'grid' mode is available for simulation of 'spec'.")
            
            
            
            self.results['time']=t
            self.results['spec']=SPEC
            self.results['spec_wv']=wvp_lc
            self.results['ff_ph']=ff_ph
            self.results['ff_sp']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc']=ff_fc
            
            
            if self.convolve_spec_with_filt: # I added this parameter to ask if you want to convolve self.results['spec'] with the selected filter in order to get the light curve.
                self.results['spec_lc']=spectra.convolve_spec_with_specified_filter(self)
                
                # #Read filter and interpolate it in order to convolve it with the spectra
                # f_filt = spectra.interpolate_filter(self)
                # spec_lc=np.zeros(len(t)) #brightness for each time step
                # spec_conv_filt=np.zeros([len(t),len(wvp_lc)]) #spectra for each time step convolved with filter
                
                # for k in range(len(t)):
                #     spec_conv_filt[k,:]=SPEC[k,:]*f_filt(wvp_lc) #convolve with filter.
                #     spec_lc[k]=np.sum(spec_conv_filt[k,:])
                
                # self.results['spec_lc']=spec_lc




        if 'rv' in observables or 'bis' in observables or 'fwhm' in observables or 'contrast' in observables: #use HR templates. Interpolate for temperatures and logg for different elements. Cut to desired wavelength.
            rvel=self.vsini*np.sin(theta)*np.sin(phi)#*np.cos(self.inclination) #radial velocities of each grid
            if self.spectral_library == 'phoenix':
                wv_rv, flnp_rv, flp_rv =spectra.interpolate_Phoenix(self,self.temperature_photosphere,self.logg) #returns norm spectra and no normalized, interpolated at T and logg
                wv_rv, flns_rv, fls_rv =spectra.interpolate_Phoenix(self,self.temperature_spot,self.logg)

                

                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                    wv_rv, flnf_rv, flf_rv =spectra.interpolate_Phoenix(self,self.temperature_facula,self.logg)
                else:
                    flnf_rv = flnp_rv
                spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized

                #for spots
                wv_rv_ph, flnp_rv_ph, flp_rv_ph = wv_rv, flnp_rv, flp_rv
                flns_rv_ph = 0

                #Interpolate also Phoenix intensity models to the Phoenix wavelength. 
                # if self.use_phoenix_limb_darkening: 
                acd, wv_rv_LR, flpk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_photosphere,self.logg) #acd is the angles at which the model is computed. 
                acd, wv_rv_LR, flsk_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_spot,self.logg)
                #acd, wv_rv_LR, flfc_rv =spectra.interpolate_Phoenix_mu_lc(self,self.temperature_facula,self.logg)

                acd_ph, wv_rv_LR_ph, flpk_rv_ph = acd, wv_rv_LR, flpk_rv 

                if self.mps_mu_grid:

                    new_flpk_rv = []
                    new_flsk_rv = []


                    for i, mu_t in enumerate(np.linspace(0.1,1,10)):


                        #Interpolate Phoenix intensity models to correct projected ange:
                        acd_low=np.max(acd[acd<mu_t]) #angles above and below the proj. angle of the grid
                        acd_upp=np.min(acd[acd>=mu_t])
                        idx_low=np.where(acd==acd_low)[0][0]
                        idx_upp=np.where(acd==acd_upp)[0][0]

                        new_flpk_rv.append(flpk_rv[idx_low]+(flpk_rv[idx_upp]-flpk_rv[idx_low])*(acd[i]-acd_low)/(acd_upp-acd_low))
                        new_flsk_rv.append(flsk_rv[idx_low]+(flsk_rv[idx_upp]-flsk_rv[idx_low])*(acd[i]-acd_low)/(acd_upp-acd_low))



                    acd = np.linspace(0.1,1,10)

                    flpk_rv = np.array(new_flpk_rv)
                    flsk_rv = np.array(new_flsk_rv)


                    
            elif self.spectral_library == 'mps':
                wv_rv, flnp_rv, flp_rv =spectra.interpolate_mps(self, 'ph') #returns norm spectra and no normalized, interpolated at T and logg
                flns_rv = spectra.black_body(wv_rv, self.temperature_spot) / spectra.black_body(wv_rv, self.temperature_photosphere) * flnp_rv

                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                    wv_rv, flnf_rv, flf_rv =spectra.interpolate_mps(self, 'fc')
                spec_ref = flnp_rv #reference spectrum to compute CCF. Normalized
                acd, wv_rv_LR, flpk_rv =spectra.load_MPS_ATLAS_spectra_lc(self, 'ph') #acd is the angles at which the model is computed. 
                # spot's spectra not available

                flsk_rv = spectra.black_body(wv_rv_LR, self.temperature_spot) / spectra.black_body(wv_rv_LR, self.temperature_photosphere) * flpk_rv

                acd, wv_rv_LR, flfc_rv =spectra.load_MPS_ATLAS_spectra_lc(self, 'fc')
            elif self.spectral_library == 'mps_highres':
                acd, wv_rv, flnp_rv, coeff = spectra.interpolate_mps_mu(self, 'ph')
                acd, wv_rv, flns_rv = spectra.interpolate_mps_mu(self, 'sp', coeff)
                acd, wv_rv, flfc_rv = spectra.interpolate_mps_mu(self, 'fc', coeff)

                

            #Compute the CCF of the spectrum of each element againts the reference template (photosphere)
            rv = np.arange(-self.ccf_rv_range,self.ccf_rv_range+self.ccf_rv_step,self.ccf_rv_step)
            #CCF with low res recipe
            if self.ccf_template == 'mask':
                if wv_rv.max()<(self.wvm.max()+1) or wv_rv.min()>(self.wvm.min()-1):
                    sys.exit('Selected wavelength must cover all the mask wavelength range, including 1A overhead covering RV shifts. Units in Angstroms.')

                if self.spectral_library == 'phoenix' or self.spectral_library == 'mps':
                    ccf_ph = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnp_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'), self.spectral_library, self.ccf_norm)
                    ccf_sp = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flns_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'), self.spectral_library, self.ccf_norm)
                    if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                        ccf_fc = nbspectra.cross_correlation_mask(rv,np.asarray(wv_rv,dtype='float64'),np.asarray(flnf_rv,dtype='float64'),np.asarray(self.wvm,dtype='float64'),np.asarray(self.fm,dtype='float64'), self.spectral_library, self.ccf_norm)
                    else:
                        ccf_fc=ccf_ph*0.0
                    
                
            if self.spectral_library == 'mps_highres':
                rv_ph = rv 
                rv_sp = rv 
                rv_fc = rv
            else:
                #Compute the bisector of the three reference CCF and return a cubic spline f fiting it, such that rv=f(ccf).
                fun_bis_ph = spectra.bisector_fit(self,rv,ccf_ph,plot_test=True,kind_interp=self.kind_interp)

                fun_bis_sp = spectra.bisector_fit(self,rv,ccf_sp,plot_test=True,kind_interp=self.kind_interp)
                if np.sum(self.active_region_types)>0 and self.sdo_input != 1:          
                        fun_raw_xbisc = spectra.bisector_fit(self,rv,ccf_fc,plot_test=False,kind_interp=self.kind_interp)  
                if self.spectral_library == 'phoenix':
                    rv_ph = rv - fun_bis_ph(ccf_ph) #subtract the bisector from the CCF.
                    rv_sp = rv - fun_bis_sp(ccf_sp)
                    rv_fc = rv_ph
                    if np.sum(self.active_region_types)>0 and self.sdo_input != 1:          
                        rv_fc = rv - fun_raw_xbisc(ccf_fc)
                elif self.spectral_library == 'mps':
                    rv_ph = rv 
                    if self.remove_bis_spot:
                        rv_sp = rv - fun_bis_sp(ccf_sp)
                    else:
                        rv_sp = rv
                    rv_fc = rv_ph


                


            if self.simulation_mode == 'grid':

                if self.spectral_library == 'mps_highres':
                    if self.instrument_ccf == 'HARPS-N':
                        print('BLAZE: HARPS')
                        #Load blaze fct
                        blaze = fits.open('/Users/sophiestucki/Downloads/HARPS.2025-12-16T23-21-06.037_blaze_A.fits')
                        waves = fits.open('/Users/sophiestucki/Downloads/HARPS.2024-10-17T21-57-53.302_wave_A.fits')
                        wv_orders = waves[0].data
                        blaze_orders = blaze[0].data        

                        #TO IMPROVE
                        data_eff = pd.read_csv('/Users/sophiestucki/Downloads/plot-data-3.csv')  

                        x = data_eff['x'] * 10
                        y = data_eff[' y']

                        coeff_eff = np.polyfit(x, y, 10)
                        for i in range(len(wv_orders)):
                            eff = np.poly1d(coeff_eff)(wv_orders[i])
                            blaze_orders[i] *= eff

                        ccf_ph_g = spectra.compute_immaculate_sphere_rv_by_order(self, Ngrid_in_ring, acd, amu, pare, flnp_rv, rv, wv_orders, blaze_orders, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'ph')
                        ccf_sp_g = spectra.compute_immaculate_sphere_rv_by_order(self, Ngrid_in_ring, acd, amu, pare, flns_rv, rv, wv_orders, blaze_orders,np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'sp')
                        ccf_fc_g = spectra.compute_immaculate_sphere_rv_by_order(self, Ngrid_in_ring, acd, amu, pare, flfc_rv, rv, wv_orders, blaze_orders,np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'fc')
                        ccf_ph_tot = np.sum(ccf_ph_g,axis=0) #?????
                    else:
                        print('BLAZE: None')
                        if self.highres_mu:
                            ccf_ph_g, norm = spectra.compute_immaculate_sphere_rv(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flnp_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'ph')
                            ccf_sp_g, _ = spectra.compute_immaculate_sphere_rv(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flns_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'sp', norm=norm)
                            ccf_fc_g, _ = spectra.compute_immaculate_sphere_rv(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flfc_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'fc', norm=norm)
                            ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                        else:
                            print('High res. smooth ratio')
                            ccf_ph_g = spectra.compute_immaculate_sphere_rv_smooth(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flnp_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'ph')
                            ccf_sp_g = spectra.compute_immaculate_sphere_rv_smooth(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flns_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'sp')
                            ccf_fc_g = spectra.compute_immaculate_sphere_rv_smooth(self, Ngrid_in_ring, acd, amu, pare, np.asarray(flfc_rv), rv, np.asarray(wv_rv,dtype='float64'), np.asarray(self.wvm,dtype='float64'), np.asarray(self.fm,dtype='float64'), rvel, 'fc')
                            ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                    

                else:
                    ccf_ph_g, flxph = spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_ph,rv,ccf_ph,rvel) #return ccf of each grid, and also the integrated ccf
                    # _, flxph_ph= spectra.compute_immaculate_photosphere_rv(self,Ngrid_in_ring,acd_ph,amu,pare,flpk_rv_ph,rv_ph_ph,rv,ccf_ph_ph,rvel)
                    plt.plot(ccf_ph_g[0])
                    plt.title('ccf_ph_g')
                    plt.show()
                    ccf_ph_tot = np.sum(ccf_ph_g,axis=0)
                    ccf_sp_g = spectra.compute_immaculate_spot_rv(self,Ngrid_in_ring,acd,amu,pare,flsk_rv,rv_sp,rv,ccf_sp,flxph,rvel)
                    ccf_fc_g = ccf_ph_g #to avoid errors, not used
                    if np.sum(self.active_region_types)>0 and self.sdo_input != 1:
                        # print('Computing facula. Limb brightening is hard coded. Luke Johnson 2021 maybe is better millor.')
                        if self.spectral_library == 'phoenix':
                            ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flpk_rv,rv_fc,rv,ccf_fc,flxph,rvel, wv_rv_LR)
                        elif self.spectral_library == 'mps':
                            ccf_fc_g = spectra.compute_immaculate_facula_rv(self,Ngrid_in_ring,acd,amu,pare,flfc_rv,rv_fc,rv,ccf_fc,flxph,rvel, wv_rv_LR)
                    
                    self.results['ccf_quiet'] = ccf_ph_tot

                # OLD
                #  if self.instrument_ccf != 'None':#tocheck
                #     ccf_ph_tot_new = spectra.add_resol(rv, ccf_ph_tot, self.instrument_ccf)
                # else:
                ccf_ph_tot_new = ccf_ph_tot
                plt.plot(rv, ccf_ph_tot_new)
                plt.show()
                RV0, C0, F0, B0,_,_ =spectra.compute_ccf_params(self,rv,[ccf_ph_tot_new],plot_test=False) #compute 0 point of immaculate photosphere

                #integrate the ccfs with doppler shifts at each time stamp
                if self.sdo_input:
                    CCF,ff_ph,ff_sp,ff_fc,ff_pl, vec_pos=spectra.generate_rotating_photosphere_rv_sdo(self,Ngrid_in_ring,pare,amu,rs,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g, inversion)
                else:
                    t,CCF,ff_ph,ff_sp,ff_fc,ff_pl, vec_pos=spectra.generate_rotating_photosphere_rv(self,Ngrid_in_ring,pare,amu,rv,ccf_ph_tot,ccf_ph_g,ccf_sp_g,ccf_fc_g,vec_grid,inversion,plot_map=self.plot_grid_map) 
                

            #FAST MODE ONLY WORKS FOR NON-OVERLAPPING SPOTS. 
            else:
                sys.exit("Invalid simulation mode. Only 'grid' mode is available")

            if self.simulate_planet:
                rvkepler = spectra.keplerian_orbit(t,[self.planet_period,self.planet_semi_amplitude,self.planet_esinw,self.planet_ecosw,self.planet_transit_t0])
            else:
                rvkepler = 0.0



            ccf_params=spectra.compute_ccf_params(self,rv,CCF,plot_test=False)
            self.results['time']=self.obs_times
            self.results['rv']=ccf_params[0] - RV0 + rvkepler #subtract rv of immaculate photosphere
            self.results['contrast']=ccf_params[1]/C0
            self.results['fwhm']=ccf_params[2]
            self.results['bis']=ccf_params[3]
            self.results['ff_ph_rv']=ff_ph
            self.results['ff_sp_rv']=ff_sp
            self.results['ff_pl']=ff_pl
            self.results['ff_fc_rv']=ff_fc
            self.results['CCF']=np.vstack((rv,CCF))
            self.results['raw_xbis'] = ccf_params[4]
            self.results['raw_ybis'] = ccf_params[5]
            self.results['pos'] = vec_pos
            self.results['flnp_rv'] = flnp_rv
            self.results['wv_rv'] = wv_rv
            self.results['ccf_quiet'] = ccf_ph_tot

