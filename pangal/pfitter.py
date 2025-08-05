import numpy as np
import astropy
from bisect import bisect_left
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator
import itertools
import os

from dynesty import NestedSampler


from .base import Spectrum
from .filters import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps

from .pfitter_methods.fit_utils import model_grid_interpolator, load_nebular_tables, load_dust_emission_models, dust_attenuation_curve, airtovac
from .pfitter_methods.fit_core import synthetic_spectrum, synthetic_phot



class PFitter(): # Parametric fitter

    def __init__(self, model_list, model_pars, cosmo=None, dustemimodel='dh02', leitatt=True, uv_bump=True, emimetal=0.0017, emimodel='2018'):


        self.model_pars = model_pars  # list the variables defyning the model grid

        # Menages model grid
        if not isinstance(model_list, list):
            raise ValueError("Error: model_grid is not a numpy array.")

        self.interp_young_flux, self.interp_old_flux, self.model_pars_arr, self.short_model_wl  = self.model_grid_interpolator(model_list, self.model_pars)

        #get the wavelength information for the dust templates to stitch it to the stellar model wl 
        this_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(this_dir, 'data', f'spectra_{dustemimodel}.dat')
        dust_wl = np.loadtxt(file_path, usecols=(0,)) * 1e4   # Load wavelength grid (in Angstroms)

        #print(dust_wl[0],dust_wl[-1])        

        #expand wavelength grid to include range covered by dust templates
        # Find the extended part of dust_wl
        extra_wl_mask = dust_wl > self.short_model_wl.max()
        extra_wl = dust_wl[extra_wl_mask]

        # Extend wavelength grid
        self.model_wl = np.r_[self.short_model_wl, extra_wl]

        # Extend resolution array with constant value matching the number of extra wavelengths
        last_res = model_list[0].resolution[-1]
        extra_res = np.full(extra_wl.size, last_res)

        # Concatenate the original model resolution with the extended constant resolution
        self.model_res = np.r_[model_list[0].resolution, extra_res]

        # PRECOMPUTES DUST ATTENUATION CURVE (Calzetti, 2000 + optional Leitherer+2002 + optional UV bump)
        self.k_cal = dust_attenuation_curve(self.model_wl, leitatt, uv_bump)

        # LOADS NEBULAR LINE TABLES
        self.nebular_func, self.nebular_ions, self.nebular_ages = self.load_nebular_tables(self.model_wl,self.model_res, emimetal,emimodel)

        # LOADS DUST EMISSION SPECTRA
        # This can be moved above for computing dust_wl only once?
        self.dustem_func, self.dustem_alpha = self.load_dust_emission_models(self.model_wl,dustemimodel)

    # methods
    load_nebular_tables = load_nebular_tables
    load_dust_emission_models = load_dust_emission_models
    dust_attenuation_curve = dust_attenuation_curve
    model_grid_interpolator = model_grid_interpolator
    
    synthetic_spectrum = synthetic_spectrum
    synthetic_phot = synthetic_phot


    def run_fit(self,
            
        spec= None,
        phot= None,

        bands= None,            # if I want to use only some of the bands in phot dict

        fix_pars=None,          # dictionary 

        polymax = 7,    

        nlive = 500,

        ):
        """
        Fits photometry and/or spectra using a parametric model and returns Fit object(s).

        Supports:
        - spectrum only
        - photometric_dict only
        - both
        - lists of both (must have the same length)

        Returns:
        - a single Fit instance or a list of Fit instances
        """

        self.nlive = nlive
        self.fix_pars = fix_pars

        if bands: 
            for b in bands:
                if b not in map_filter_names.keys():
                    raise ValueError(f'Unrecognized filter: {b}. Abort')


        # Handle list vs. single inputs
        if isinstance(spec, list) or isinstance(phot, list):
            spec_list = spec if isinstance(spec, list) else [None] * len(phot)
            phot_list = phot if isinstance(phot, list) else [None] * len(spec)

            if len(spec_list) != len(phot_list):
                raise ValueError("Spectrum and photometric_dict lists must have the same length.")

            fits = [self.run_single_fit(s, p, bands) for s, p in zip(spec_list, phot_list)]

            # menages writing on file
            return fits
        else:
            
            fit = self.run_single_fit(spec, phot, bands)

            # menages writing on file
            return fit


    # This function is internal
    def run_single_fit(self, spec, phot, bands):

        #if phot and hasattr(phot, 'header') and 'units' in phot.header:
            # convert everything to mJy!
            #if 


        if phot and hasattr(phot, 'header') and 'redshift' in phot.header:
            self.redshift = phot.header['redshift']
        elif spec and hasattr(spec, 'header') and 'redshift' in spec.header:
            self.redshift = spec.header['redshift']
        else:
            print("Redshift not provided and not found in phot dictionary or spectrum header.")
            print("Redshift set to 0.")
            self.redshift = 0
        
        #Derive luminosity distance in Mpc
        if phot and 'dl' in phot.header.keys():
            self.dl = phot.header['dl']
        elif spec and hasattr(spec, 'header') and 'dl' in spec.header:
            self.dl = spec.header['dl']
        else:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70,Om0=0.3)        
            self.dl = cosmo.luminosity_distance(self.redshift).value

        if spec:
            self.spec = spec
            print('Fitting spectrum')

        if phot: 

            self.phot = phot

            # PRECOMPUTES FILTERS

            # Filter bands to only those present in map_filter_names
            if bands: 
                self.bands = bands
            else:
                self.bands = [b for b in phot.photometry.keys() if b in map_filter_names.keys()]

            filters = {b:Filter(b) for b in self.bands}
            self.transmission_curves_funcs = {b:filters[b].transmission_curve for b in self.bands}
            self.transmission_lims = {b:filters[b].wavelength_range for b in self.bands}
            self.pivot_wl = {b:filters[b].effective_wavelength for b in self.bands} 

            # Print which filters are used
            print(f"Using the following photometric filters: {', '.join(self.bands)}")


            # Handles PARAMETERS: removes 
            free_pars = [p for p in self.model_pars + ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]
             if p not in self.fix_pars]

            
        # Create the sampler
        log_likelihood = self.make_log_likelihood()
        prior_transform = self.make_prior_tranform()

        print("Initializing live points")
        sampler = NestedSampler(log_likelihood, prior_transform, ndim=len(free_pars),nlive=self.nlive)

        # Run the nested sampling
        print("Running dynesty")
        sampler.run_nested(dlogz=0.01, print_progress=True)

        # Access results
        res = sampler.results

        return res
    

    def make_log_likelihood(self):
        def log_likelihood(pars):

            spec_lhood = 0
            phot_lhood = 0

            if self.phot:

                phot_points = np.array([self.phot.data[b][0] for b in self.bands])
                phot_errors = np.array([self.phot.data[b][1] for b in self.bands])

                upper_lims = (phot_points / phot_errors < 5).astype(int)


                # Unpacks parameters
                idx = 0
                n_model_pars = len(self.model_pars)
                model_pars = pars[idx : idx + n_model_pars]
                idx += n_model_pars

                param_names = ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]

                # Store in a local dict, then unpack manually
                param_values = {}
                for name in param_names:
                    if name in self.fix_pars:
                        param_values[name] = self.fix_pars[name]
                    else:
                        param_values[name] = pars[idx]
                        idx += 1

                # Now unpack them to true local variables (1 line)
                fesc, ion_gas, age_gas, av, av_ext, alpha, m_star = [param_values[n] for n in param_names]
                # parameters of the model grid
                kwargs = {key: value for key, value in zip(self.model_pars, model_pars)}
                
                # build total spectrum
                synth_spec = self.synthetic_spectrum(**kwargs, fesc=fesc, ion_gas=ion_gas, age_gas=age_gas, av=av, av_ext=av_ext, alpha=alpha, m_star=m_star, redshift=self.redshift, dl=self.dl)
                # get the synthetic photometric points out of the synthetic spectrum
                model_phot = self.synthetic_phot(synth_spec,)


                print('CHIAMA: ',pars)
                print((np.array(phot_points)-np.array(model_phot))) #**2 / np.array(phot_errors)**2

                for i in range(len(phot_points)):

                    if upper_lims[i] == 0:

                        phot_lhood += -0.5 * ( (phot_points[i]-model_phot[i])**2 / phot_errors[i]**2 + np.log(phot_errors[i]**2) + np.log(2.*np.pi) ) 

                    else: # upper limit: error function (integrated Gaussian probability)

                        terf = 0.5* ( 1 + erf( (phot_points[i]-model_phot[i])/np.sqrt(2.)/phot_errors[i]) ) # P(flux<upper limit)
                        phot_lhood += np.log(terf)
            

            print(pars, spec_lhood + phot_lhood )

            if not np.isfinite(phot_lhood):
                return -1e100  # NOT -np.inf

            return spec_lhood + phot_lhood 
    
        return log_likelihood
    


    def make_prior_tranform(self):
        def prior_transform(u):
            x = np.zeros_like(u)

            

            # model_pars: assume these are between 0 and 1 — flat prior
            n_model_pars = len(self.model_pars)  # or hardcode if known
            for i in range(n_model_pars):
                low = self.model_pars_arr[i][0]
                high = self.model_pars_arr[i][-1]
                x[i] = low + u[i] * (high - low)

            idx = n_model_pars

            if 'fesc' not in self.fix_pars:
                x[idx] = u[idx]
                idx += 1

            if 'ion_gas' not in self.fix_pars:
                x[idx] = u[idx] * (self.nebular_ions[-1] - self.nebular_ions[0]) + self.nebular_ions[0]
                idx += 1

            if 'age_gas' not in self.fix_pars:
                x[idx] = self.emm_ages[0] + u[idx] * (self.emm_ages[-1]+1 - self.emm_ages[0]) # allows age gas to run to 11 in order to have zero emission lines
                idx += 1

            if 'av' not in self.fix_pars:
                x[idx] = u[idx] * 1.0
                idx += 1

            if 'av_ext' not in self.fix_pars:
                x[idx] = u[idx] * 1.0
                idx += 1

            if 'alpha' not in self.fix_pars:
                x[idx] = u[idx] * (self.dustem_alpha[-1] - self.dustem_alpha[0]) + self.dustem_alpha[0]
                idx += 1

            if 'm_star' not in self.fix_pars:
                x[idx] = u[idx] * (11 - 7) + 7
                idx += 1

            return x
        return prior_transform




    

