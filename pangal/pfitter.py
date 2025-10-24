import numpy as np
import astropy
from bisect import bisect_left
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator
import itertools
import os

from dynesty import NestedSampler


from .image import Spectrum
from .filters import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps





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
            self.pivot_wl = {b:filters[b].pivot_wavelength for b in self.bands} 

            # Print which filters are used
            print(f"Using the following photometric filters: {', '.join(self.bands)}")


            # Handles PARAMETERS: removes 
            free_pars = [p for p in self.model_pars + ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]
             if p not in self.fix_pars]

            
        # Create the sampler
        log_likelihood = self.make_log_likelihood()
        prior_transform = self.make_prior_tranform()

        """
        print("Initializing live points")
        sampler = NestedSampler(log_likelihood, prior_transform, ndim=len(free_pars),nlive=self.nlive)

        # Run the nested sampling
        print("Running dynesty")
        sampler.run_nested(dlogz=0.01, print_progress=True)
        """

        sampler =  NestedSampler(
            log_likelihood, prior_transform, ndim=len(free_pars),
            nlive=500,
            sample="rwalk",   # or "rslice"
            walks=50,
            bound="multi"
        )
        sampler.run_nested(dlogz=0.1, print_progress=True)

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


                for i in range(len(phot_points)):

                    if upper_lims[i] == 0:

                        phot_lhood += -0.5 * ( (phot_points[i]-model_phot[i])**2 / phot_errors[i]**2 + np.log(phot_errors[i]**2) + np.log(2.*np.pi) ) 

                    else: # upper limit: error function (integrated Gaussian probability)

                        terf = 0.5* ( 1 + erf( (phot_points[i]-model_phot[i])/np.sqrt(2.)/phot_errors[i]) ) # P(flux<upper limit)
                        phot_lhood += np.log(terf)
            

            #print(pars, spec_lhood + phot_lhood )

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
                low = 10 #self.model_pars_arr[i][0]
                high = 200 #self.model_pars_arr[i][-1]
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


    def synthetic_spectrum(self, fesc, ion_gas, age_gas, av, av_ext, alpha, m_star, redshift, dl, multi_component=False, **kwargs,):
        """
        Generate a physically motivated synthetic galaxy spectrum, combining stellar populations, nebular lines, 
        dust attenuation, and re-emission, redshifted and scaled for observational comparison.

        Parameters
        ----------
        fesc : float
            Escape fraction of ionizing photons (0 = fully absorbed by gas; 1 = no nebular emission).
        
        ion_gas : float
            Ionization parameter of the nebular gas.

        age_gas : float
            Age of the nebular region in Myr (affects emission line spectrum).

        av : float
            V-band attenuation for the old stellar population (in magnitudes).

        av_ext : float
            Extra V-band attenuation applied only to the young stellar population.

        alpha : float
            Parameter controlling the shape of the dust emission SED (Dale & Helou 2002 models).

        m_star : float
            Logarithmic stellar mass of the galaxy (in solar masses, log₁₀[M*/M☉]).

        redshift : float
            Redshift of the galaxy.

        dl : float
            Luminosity distance to the galaxy in parsecs (used for flux scaling).

        multi_component : bool, optional
            If True, returns sub-components (e.g., dust, young/old spectra) in the output Spectrum object for inspection or plotting.

        **kwargs :
            Passed to stellar population model interpolators (e.g., age, metallicity, SFH parameters).

        Returns
        -------
        Spectrum
            A synthetic `Spectrum` object with:
                - wavelength grid in observed frame (Angstrom),
                - flux in erg/s/cm²/Å,
                - optional components if `multi_component=True` (e.g., dust emission, attenuated spectra).

        Notes
        -----
        The returned spectrum is built in the following order:
        1. Interpolate young and old stellar templates on model grid.
        2. Compute ionizing photon rates to scale nebular emission.
        3. Add nebular emission lines based on gas parameters and fesc.
        4. Apply dust attenuation using a Calzetti-like law.
        5. Compute energy absorbed by dust and add thermal IR emission from templates.
        6. Apply redshifting and flux scaling for observer's frame.

        The resulting spectrum is suitable for SED fitting, photometry synthesis, or mock observation generation.
        """
        
        young_stellar_spec = self.interp_young_flux(**kwargs)
        old_stellar_spec = self.interp_old_flux(**kwargs)

        # Resample the flux from short_wl to model_wl
        young_stellar_spec = np.interp(self.model_wl, self.short_model_wl, young_stellar_spec)
        old_stellar_spec = np.interp(self.model_wl, self.short_model_wl, old_stellar_spec)


        # Number of lyman continuum photons
        # Spectra are corrected for lyman continuum absorption including escape fraction

        #get number of lyman continuum photons from the young stellar population and the old stellar population
        #spec in erg/s/cm2/A so it returns the flux of ionizing photons not the rate 
        index_lyman = np.searchsorted(self.model_wl, 912, side='left') #wavelength just above Lyman limit

        lycont_wls = np.r_[self.model_wl[:index_lyman], np.array([912])]

        lycont_spec = np.interp(lycont_wls, self.model_wl, young_stellar_spec) 
        
        nlyman_young = np.trapz(lycont_spec*lycont_wls, lycont_wls)/6.626e-27/2.998e18 * (1-fesc)

        lycont_spec = np.interp(lycont_wls, self.model_wl, old_stellar_spec)                        # WHY THIS? WHAT CHANGES?
        nlyman_old = np.trapz(lycont_spec*lycont_wls, lycont_wls)/6.626e-27/2.998e18 * (1-fesc)

        #modify input spec to remove photons 
        if fesc>=0:
            young_stellar_spec[:index_lyman] *= fesc
            old_stellar_spec[:index_lyman] *= fesc             # is this necessary ?? 

        nebular_lines = self.nebular_func(ion_gas,age_gas) 

        # to get the full spec normalized line spectra must be multiplied by the un-escaped number of lyman photons from the stellar model
        nebular_young = nebular_lines * nlyman_young 
        nebular_old = nebular_lines * nlyman_old

        young_stellar_nebular_spec = young_stellar_spec + nebular_young
        old_stellar_nebular_spec = old_stellar_spec + nebular_old
        stellar_nebular_spec = young_stellar_nebular_spec + old_stellar_nebular_spec


        # Dust attenuation

        #attenuate photometry spec, k_cal is precomputed in the rest frame as it should be
        att_young_stellar_nebular_spec = (10**(-(av+av_ext)*self.k_cal) * (young_stellar_nebular_spec))
        att_old_stellar_nebular_spec   = (10**(-av*self.k_cal) * (old_stellar_nebular_spec))
        att_stellar_nebular_spec  = att_young_stellar_nebular_spec + att_old_stellar_nebular_spec


        #### THERMAL DUST EMISSION ####
        lbol_init = np.trapz(stellar_nebular_spec, self.model_wl)
        lbol_att  = np.trapz(att_stellar_nebular_spec, self.model_wl)

        dust_bol = (lbol_init - lbol_att)
        
        # raw interpolated dust emission template. independent of your specific galaxy model. 
        # generic spec for a given α (which controls dust heating intensity).

        tdust_spec = self.dustem_func(alpha)

        # remove stellar component which is included in dale models
        # otherwise it is double-counted.
        mask_pixels = (self.model_wl >= 2.5e4) & (self.model_wl <= 3e4)
        scale = np.sum(stellar_nebular_spec[mask_pixels]*tdust_spec[mask_pixels]) / np.sum(stellar_nebular_spec[mask_pixels]*stellar_nebular_spec[mask_pixels])
        tdust_spec -= scale*stellar_nebular_spec
        tdust_spec[(self.model_wl < 2.5e4) | (tdust_spec < 0.)] = 0.

        # scale to the correct total energy to re-emit
        norm = np.trapz(tdust_spec, self.model_wl)     
        dust_spec = tdust_spec * dust_bol / norm

        total_spec = att_stellar_nebular_spec + dust_spec

        # move to the observed frame
        model_red_wl = self.model_wl * (1+redshift)
        total_spec /= (1+redshift)

        # this models are scaled to 1 solar mass at 10 pc
        # Rescale
        fscale = 10**m_star / (dl * 1e5)**2

        total_spec *= fscale 

        spec = Spectrum(wl=model_red_wl,resolution=self.model_res,flux=total_spec)

        if multi_component:
            # arrays usefull for plotting
            for s in [young_stellar_nebular_spec, old_stellar_nebular_spec, att_young_stellar_nebular_spec, att_old_stellar_nebular_spec, dust_spec]:
                s /= (1+redshift)
                s *= fscale 
            spec.young_stellar_nebular = young_stellar_nebular_spec
            spec.old_stellar_nebular = old_stellar_nebular_spec
            spec.att_young_stellar_nebular = att_young_stellar_nebular_spec
            spec.att_old_stellar_nebular = att_old_stellar_nebular_spec
            spec.dust = dust_spec

        return spec
        
  

    def synthetic_phot(self, spectrum, bands=None):
        """
        Computes synthetic photometric points (in mJy) from a spectrum (in erg/s/cm2/A) and a list of bands.
        If bands is None, uses internal self.bands and self.transmission_curves_funcs.
        """
        if bands:
            invalid = [b for b in bands if b not in map_filter_names]
            if invalid:
                raise ValueError(f"Unrecognized filters: {', '.join(invalid)}")

        spec_func = interp1d(spectrum.wl, spectrum.flux, bounds_error=False, fill_value=0.0)
        synthetic_points = []

        # Decide mode: internal (fast) or external (general)
        if bands is None:
            band_list = self.bands
            get_curve = lambda b: self.transmission_curves_funcs[b]
            get_lims = lambda b: self.transmission_lims[b]
            get_pivot = lambda b: self.pivot_wl[b]
        else:
            band_list = bands
            get_curve = lambda b: Filter(b).transmission_curve
            get_lims = lambda b: Filter(b).wavelength_range
            get_pivot = lambda b: Filter(b).pivot_wavelength  

        for b in band_list:

            trans_curve = get_curve(b)
            lmin, lmax = get_lims(b)

            num_int, _ = quad(lambda l: trans_curve(l) * spec_func(l), lmin, lmax)
            norm_int, _ = quad(lambda l: trans_curve(l), lmin, lmax)

            phot_point = (num_int / norm_int) * get_pivot(b)**2 / 3.99e18 / 1e-26
            synthetic_points.append(phot_point)

        return synthetic_points











# UTILS

# Base directory containing the “DISKEVOL.RES_L0.05_V*” files
this_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to the data directory
data_dir = os.path.join(this_dir, "..", "data")
data_dir = os.path.normpath(data_dir)


def load_nebular_tables(self,wl,resolution,emimetal,emimodel):
    """
    Load and process precomputed nebular emission line tables from Byler et al. models.

    This function reads tabulated emission line luminosities (in erg/s per ionizing photon)
    for a given metallicity and emission model (2017 or 2018), and convolves the lines with
    the desired spectral resolution and wavelength grid. The resulting emission line spectra 
    are suitable for adding to stellar SEDs generated by codes like CIGALE.

    The lines are modeled as Gaussians, and the convolution simulates how they would appear
    in an observed spectrum, taking into account the resolution and pixel binning. An 
    interpolation function is returned, allowing the user to compute the full nebular 
    emission spectrum for any given ionization parameter and stellar population age.

    Parameters:
        wl (ndarray): Wavelength array in Angstroms.
        resolution (ndarray): Spectral resolution (R = λ/Δλ) array matching `wl`.
        emimetal (float): Metallicity of the model to load (closest available will be used).
        emimodel (str): Emission line model version ('2017' or '2018').

    Returns:
        nebular_emission_spectrum (function): Callable(ion, age) → emission spectrum on wl grid.
        emm_ions (ndarray): Ionization parameter grid points used in the model.
        emm_ages (ndarray): Age grid points (in Myr) used in the model.
    """
    
    wl_lyman = 912.
    ilyman = np.searchsorted(wl, wl_lyman, side='left') #wavelength just above Lyman limit
    lycont_wls = np.r_[wl[:ilyman], np.array([wl_lyman])]
    clyman_young = None #A list of two elements, first is for phot, the other for spec
    clyman_old   = None #A list of two elements, first is for phot, the other for spec

    # Emission line wavelengths have to be in VACUUM
    
    if emimodel == '2018':
        byler_bins = [382,9,7]
        byler_fname = 'nebular_Byler_mist_2018.lines'
        metallist = np.array([6.3245e-05,2.0000e-04,3.5565e-04,6.3245e-04,1.1246e-03,2.0000e-03,3.5565e-03,6.3245e-03,1.1246e-02,2.000e-02,3.5565e-02,6.3245e-02])
        metalstrg = np.array(['-2.5000e+00','-2.0000e+00','-1.7500e+00','-1.5000e+00','-1.2500e+00','-1.0000e+00','-7.5000e-01','-5.0000e-01','-2.5000e-01','0.0000e+00','2.5000e-01','5.0000e-01'])
    else:
        byler_bins = [128,10,7]
        byler_fname = 'nebular_Byler_mist_2017.lines'
        metallist = np.array([0.00020,0.00063246,0.00209426,0.00526054,0.00814761,0.01002374,0.01261915,0.01588656,0.02,0.02517851,0.03169786])
        metalstrg = np.array(['-1.9800e+00','-1.5000e+00','-9.8000e-01','-5.8000e-01','-3.9000e-01','-3.0000e-01','-2.0000e-01','-1.0000e-01','0.0000e+00','1.0000e-01','2.0000e-01'])
    
    print("Loading nebular lines templates from ",byler_fname)

    emm_scales = np.zeros((byler_bins[2],byler_bins[1],byler_bins[0]), dtype=float)
    emm_wls    = np.zeros(byler_bins[0],        dtype=float)
    emm_ages   = np.zeros(byler_bins[1],        dtype=float)
    emm_ions   = np.zeros(byler_bins[2],        dtype=float)
    icnt = 0
    rline = 0
    iline = 0
    
    
    metind = np.argmin(np.abs(metallist-emimetal))  # Select the closest metallicity available
    print(' Emission line metallicity requested {}, found {:5.4f}'.format(emimetal,metallist[metind]))
    emimetal = metallist[metind]


    # load emission lines tables
    file_path = os.path.join(data_dir, byler_fname)

    with open(file_path, 'r') as file:
        
        for line in file:
            if line[0] != '#':
                temp = (line.strip()).split(None)
                if not iline: #Read wave line
                    emm_wls[:] = np.array(temp, dtype=float)
                    iline = 1
                else:
                    if rline: #Read line fluxes
                        emm_scales[icnt%7,icnt//7,:] = np.array(temp, dtype=float)*3.839e33 #erg/s #output should be in erg/s/QHO
                        icnt += 1
                    if len(temp) == 3 and temp[0] == metalstrg[metind]:
                        rline = 1
                        emm_ages[icnt//7] = float(temp[1])/1e6
                        emm_ions[icnt%7]  = float(temp[2])
                    else:
                        rline = 0


    print(f"Tabulated values for ionized gas age (Myr): {emm_ages}")
    print(f"Tabulated values for ionization coefficient: {emm_ions}")

    # Filters out non-physical or extremely high wavelengths ?? Why ??
    keep_scale = (emm_wls<1E5)        
    emm_scales = emm_scales[:,:,keep_scale]
    emm_wls    = emm_wls[keep_scale]

    # Convolve lines to model resolution
    # Emission lines are modeled as Gaussians centered at their rest wavelengths.
    # To add these lines realistically, you integrate the Gaussian emission line profile over each pixel's wavelength bin.
    # Simulates how nebular emission lines would appear in a spectrum observed at a given resolution and wavelength sampling
    
    dpix = np.diff(wl)
    wl_edges  = np.r_[np.array([wl[0]-dpix[0]/2.]), np.r_[wl[1:]-dpix/2., np.array([wl[-1]+dpix[-1]/2.])]]
    delta_lambda = wl/resolution
    res_lines = np.interp(emm_wls, wl,delta_lambda)/2.355
    

    emm_lines_all = np.zeros((len(emm_ions), len(emm_ages), len(wl)), dtype=float)

    for jj in range(len(emm_ions)):
        for ii in range(len(emm_ages)):
            this_scale = emm_scales[jj,ii,:]
            emm_lines_all[jj,ii,:] = np.sum(this_scale[:,None]*\
                np.diff(0.5*(1.+erf((wl_edges[None,:]-emm_wls[:,None])/\
                np.sqrt(2.*res_lines**2)[:,None])), axis=1)/np.diff(wl_edges), axis=0)


    # Create interpolator: maps (ion, age) -> spectrum on wl grid
    interp_func = RegularGridInterpolator(
        points=(emm_ions, emm_ages),  # shape (n_ions, n_ages)
        values=emm_lines_all,         # shape (n_ions, n_ages, n_wl)
        bounds_error=False,
        fill_value=0.0
    )

    # Define a function for convenience
    def nebular_emission_spectrum(ion, age):
        input_point = np.array([[ion, age]])  # shape (1, 2)
        return interp_func(input_point)[0]    # returns (n_wl,)
    
    print("Loading of nebular lines templates completed.")
    
    return nebular_emission_spectrum, emm_ions, emm_ages



def load_dust_emission_models(self, wl, dustemimodel):
    """
    Loads dust emission templates from Dale & Helou 2002 (dustemimodel = 'dh02') and return a function of alpha
    or Draine & Li 2014 (dustemimodel = 'dl14'), interpolated on the input wavelength grid.
    """

    alpha_file = os.path.join(data_dir, f'alpha_{dustemimodel}.dat')
    spectra_file = os.path.join(data_dir, f'spectra_{dustemimodel}.dat')

    # Check files exist
    if not os.path.exists(alpha_file):
        raise FileNotFoundError(f"Missing file: {alpha_file}")
    if not os.path.exists(spectra_file):
        raise FileNotFoundError(f"Missing file: {spectra_file}")

    #print(f"Loading dust emission templates from: {alpha_file}")

    # Load alpha values and spectra
    dustem_alpha = np.loadtxt(alpha_file, usecols=(0,))
    dustem_wl = np.loadtxt(spectra_file, usecols=(0,)) * 1e4  # assuming microns to Angstroms
    full_spectra = np.loadtxt(spectra_file)

    # Initialize template matrix
    tdustem = np.zeros((len(dustem_alpha), len(wl)), dtype=float)

    for ii in range(len(dustem_alpha)):
        tdust = 10 ** full_spectra[:, ii + 1]  # column 1 is wavelength, so ii+1
        tdustem[ii, :] = np.interp(wl, dustem_wl, tdust, left=0.0, right=0.0) / wl

    # Normalize each template to Lbol = 1
    norm = np.trapz(tdustem, wl, axis=1)
    tdustem /= norm[:, None]

    dustem_func = interp1d(dustem_alpha, tdustem, axis=0, bounds_error=True)

    # Define a function for convenience
    def dust_emission_spectrum(alpha):
        return dustem_func(alpha)

    print(f"Tabulated alpha values from 'alpha_{dustemimodel}.dat': {dustem_alpha}")
    print("Loading of dust emission templates completed.")

    return dust_emission_spectrum, dustem_alpha




def model_grid_interpolator(self, model_list, param_names, extra_pars_to_interpolate=None):
    """
    Build N-dimensional interpolators for young and old fluxes.

    Parameters:
        model_list (list): Flat list of Spectrum models, each with `.flux_young`, `.flux_old`, `.header`
        param_names (list of str): Names of parameters to define the grid (e.g. ['age', 'tau'])

    Returns:
        interp_flux_young: Callable f(**params) -> flux_young (shape: n_wl)
        interp_flux_old: Callable f(**params) -> flux_old (shape: n_wl)
        grid_axes: List of sorted arrays for each parameter axis
        wl: Wavelength array (from first model)
    """

    # Step 1: Build param space
    param_tuples = []
    param_to_model = {}
    for model in model_list:
        try:
            values = tuple(model.header[k] for k in param_names)
        except KeyError as e:
            raise KeyError(f"Model is missing parameter {e} in header.")
        param_tuples.append(values)
        param_to_model[values] = model

    # Step 2: Build grid axes
    grid_axes = [np.array(sorted(set(p[i] for p in param_tuples))) for i in range(len(param_names))]

    # Step 3: Allocate grids for young and old fluxes
    grid_shape = tuple(len(ax) for ax in grid_axes)
    n_wl = model_list[0].flux_young.size

    flux_young_grid = np.zeros(grid_shape + (n_wl,))
    flux_old_grid = np.zeros(grid_shape + (n_wl,))

    # Step 4: Fill the grids
    for idxs in np.ndindex(*grid_shape):
        key = tuple(grid_axes[i][idxs[i]] for i in range(len(grid_axes)))
        model = param_to_model[key]
        flux_young_grid[idxs] = model.flux_young
        flux_old_grid[idxs] = model.flux_old

    # Step 5: Build interpolators
    interp_young = RegularGridInterpolator(points=grid_axes, values=flux_young_grid, bounds_error=False, fill_value=None)
    interp_old = RegularGridInterpolator(points=grid_axes, values=flux_old_grid, bounds_error=False, fill_value=None)

    def interp_flux_young(**kwargs):
        x = [kwargs[name] for name in param_names]
        return interp_young(x)[0]

    def interp_flux_old(**kwargs):
        x = [kwargs[name] for name in param_names]
        return interp_old(x)[0]
    
    return interp_flux_young, interp_flux_old, grid_axes, model_list[0].wl






















# Dust attenuation curve based on Calzetti et al. (2000) law
# Takes in imput an array of wavelengths and returns the attenuation function

def dust_attenuation_curve(wl, leitatt, uv_bump):
    
    k_cal = np.zeros(len(wl), dtype=float)  # cal for Calzetti

    #compute attenuation assuming Calzetti+ 2000 law
    #single component 

    R = 4.05
    div = wl.searchsorted(6300., side='left')
    
    #Longer than 6300
    k_cal[div:] = 2.659*( -1.857 + 1.04*(1e4/wl[div:])) + R
    #Shorter than 6300
    k_cal[:div] = 2.659*(-2.156 + 1.509*(1e4/wl[:div]) - 0.198*(1e4/wl[:div])**2 + 0.011*(1e4/wl[:div])**3) + R
    

    #IF REQUESTED Use leitherer 2002 formula below 1500A
    if leitatt:
        div = wl.searchsorted(1500., side='left')
        #Shorter than 1500
        k_cal[:div] = (5.472 + 0.671e4 / wl[:div] - 9.218e5 / wl[:div] ** 2 + 2.620e9 / wl[:div] ** 3)

    #Prevents negative attenuation, which can arise from extrapolation or math artifacts
    zero = bisect_left(-k_cal, 0.)
    k_cal[zero:] = 0.

    #2175A bump
    if uv_bump:
        eb = 1.0
        k_bump = np.zeros(len(wl), dtype=float)
        k_bump[:] = eb*(wl*350)**2 / ((wl**2 - 2175.**2)**2 + (wl*350)**2)
        k_cal += k_bump

            
    return 0.4 * k_cal / R





def vactoair(wl):
    """Convert vacuum wavelengths to air wavelengths using the conversion
    given by Morton (1991, ApJS, 77, 119).

    """
    wave2 = np.asarray(wl, dtype=float)**2
    fact = 1. + 2.735182e-4 + 131.4182/wave2 + 2.76249e8/(wave2*wave2)
    return wl/fact


def airtovac(wl):
    """Convert air wavelengths to vacuum wavelengths using the conversion
    given by Morton (1991, ApJS, 77, 119).

    """
    sigma2 = np.asarray(1E4/wl, dtype=float)**2
    
    fact = 1. + 6.4328e-5 + 2.949281e-2/(146.-sigma2) + 2.5540e-4/(41.-sigma2)
    fact[wl < 2000] = 1.0
    
    return wl*fact
