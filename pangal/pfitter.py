import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from bisect import bisect_left
from scipy.special import erf
from scipy.interpolate import RegularGridInterpolator, interp1d
import itertools
import os


from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import corner

from .spectrum import Spectrum
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps

from .pfitter_utils import load_nebular_tables, load_dust_emission_models, dust_attenuation_curve, model_grid_interpolator



class PFitter(): # Parametric fitter

    def __init__(self, model_list, model_pars, cosmo=None, dustemimodel='dh02', leitatt=True, uv_bump=True, emimetal=0.0017, emimodel='2018'):

        self.model_pars = model_pars  # list the variables defyning the model grid
        print(f"Initializing PFitter object")
        print(f"   Model parameters: {', '.join(self.model_pars)}\n")

        # Menages model grid
        if not isinstance(model_list, list):
            raise ValueError("Error: model_grid is not a numpy array.")

        self.interp_young_flux, self.interp_old_flux, self.model_pars_arr, self.short_model_wl  = self.model_grid_interpolator(model_list, self.model_pars)

        print(f" Pre-computed model grid:")
        for name, arr in zip(self.model_pars, self.model_pars_arr):
            print(f"   - {name}: {len(arr)} values, min={arr[0]}, max={arr[-1]}")
        print("\n")
        
        #get the wavelength information for the dust templates to stitch it to the stellar model wl 
        #this_dir = os.path.dirname(os.path.abspath(__file__))
        #file_path = os.path.join(this_dir, 'data', f'spectra_{dustemimodel}.dat')
        file_path = '/home/utente/Desktop/Pangal/pangal/data/'+f'spectra_{dustemimodel}.dat'
        dust_wl = np.loadtxt(file_path, usecols=(0,)) * 1e4   # Load wavelength grid (in Angstroms)

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


    
    def run_fit(self,
            
        spec= None,
        phot= None,

        bands= None,          # if I want to use only some of the bands in phot dict

        fix_pars={},          # dictionary 
        custom_priors={},        # dictionary

        polymax = 7,    

        nlive = 500,
        dlogz = 0.01,

        ):

        # Filter bands to only those present in map_filter_names
        if bands: 
            for b in bands:
                if b not in map_filter_names.keys():
                    raise ValueError(f'Unrecognized filter: {b}. Abort')
            self.bands = bands
        else:
            self.bands = [b for b in phot.photometry.keys() if b in map_filter_names.keys()]
        # Print which filters are used
        print(f"Using the following photometric filters: {', '.join(self.bands)}")
        

        # Redshift and luminosity distance
        if phot and hasattr(phot, 'header') and 'redshift' in phot.header:
            self.redshift = phot.header['redshift']
        elif spec and hasattr(spec, 'header') and 'redshift' in spec.header:
            self.redshift = spec.header['redshift']
        else:
            print("Redshift not provided and not found in phot dictionary or spectrum header. Set to 0.")
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


        
        # Handles PARAMETERS: removes fixed parameters
        self.fix_pars = fix_pars
        free_pars = [p for p in self.model_pars + ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]
                     if p not in self.fix_pars]
        self.free_pars = free_pars
        
        # Create the sampler
        log_likelihood = self.make_log_likelihood(spec, phot, free_pars, bands)
        prior_transform = self.make_prior_transform(custom_priors)

        print("Initializing live points")
        sampler = NestedSampler(log_likelihood, prior_transform, ndim=len(free_pars),nlive=100) #self.nlive

        # Run the nested sampling
        sampler.run_nested(dlogz=dlogz, print_progress=True)

        # Access results
        res = sampler.results

        # safe to save internally the result
        self.fit = res

        return res

    
    def make_log_likelihood(self, spec, phot, free_pars, bands):

        # --- Precompute constants ---
        if spec:
            print("Fitting spectrum")
    
        if phot:
            # Convert PhotometryTable object into arrays
            phot_points = np.array([phot.data[b][0] for b in bands])
            phot_errors = np.array([phot.data[b][1] for b in bands])
            upper_lims = (phot_points / phot_errors < 5).astype(int)
    
            # Precompute filter transmission curves on the model wavelength grid
            mask = {}
            trans_arrays = {}
            pivot_wls = {}
            for b in bands:
                F = Filter(b)
                lmin, lmax = F.wavelength_range
                mask[b] = (self.model_wl >= lmin) & (self.model_wl <= lmax)
                trans_arrays[b] = F.transmission_curve(self.model_wl[mask[b]])
                pivot_wls[b] = F.pivot_wavelength
    
        # --- Define log-likelihood ---
        def log_likelihood(pars):
            
            spec_lhood = 0
            phot_lhood = 0


            if spec: 
                print('SPEC!')

                
            
            if phot:
                idx = 0
                n_model_pars = len(self.model_pars)
                model_pars = pars[idx : idx + n_model_pars]
                idx += n_model_pars
    
                param_names = ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]
                param_values = {}
                for name in param_names:
                    if name in self.fix_pars:
                        param_values[name] = self.fix_pars[name]
                    else:
                        param_values[name] = pars[idx]
                        idx += 1
    
                fesc, ion_gas, age_gas, av, av_ext, alpha, m_star = [param_values[n] for n in param_names]
                kwargs = {key: value for key, value in zip(self.model_pars, model_pars)}
    
                # Build synthetic spectrum
                synth_spec = self.synthetic_spectrum(**kwargs,
                                                     fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                                                     av=av, av_ext=av_ext, alpha=alpha,
                                                     m_star=m_star, redshift=0, dl=100)
    
                model_phot = []
                for b in bands:
                    spec_array = synth_spec.flux[mask[b]]
                    if len(spec_array) == 0:
                        model_phot.append(np.nan)
                        continue
    
                    num_int = np.trapz(trans_arrays[b] * spec_array, self.model_wl[mask[b]])
                    norm_int = np.trapz(trans_arrays[b], self.model_wl[mask[b]])
                    if norm_int == 0:
                        model_phot.append(np.nan)
                        continue
    
                    phot_point = num_int / norm_int

                    # CHECK THIS
                    # Convert to mJy if needed
                    c = 2.99792458e18  # Å/s
                    phot_point = phot_point * pivot_wls[b]**2 / c / 1e-26
    
                    model_phot.append(phot_point)
    
                model_phot = np.array(model_phot)
    
                if not np.all(np.isfinite(model_phot)):
                    return -1e100
    
                for i in range(len(phot_points)):
                    if upper_lims[i] == 0:
                        phot_lhood += -0.5 * (
                            (phot_points[i] - model_phot[i])**2 / phot_errors[i]**2
                            + np.log(phot_errors[i]**2)
                            + np.log(2. * np.pi)
                        )
                    else:
                        terf = 0.5 * (1 + erf((phot_points[i] - model_phot[i]) /
                                              (np.sqrt(2.) * phot_errors[i])))
                        phot_lhood += np.log(terf)
    
            if not np.isfinite(phot_lhood):
                return -1e100
            
            return spec_lhood + phot_lhood

            
    
        return log_likelihood
    
    
    def make_prior_transform(self, custom_priors):

        # --- Validate custom priors ---
        all_pars = list(self.model_pars) + [
            p for p in ["fesc", "ion_gas", "age_gas", "av", "av_ext", "alpha", "m_star"]
            if p not in self.fix_pars
        ]
        for key, val in custom_priors.items():
            if key not in all_pars:
                raise ValueError(f"Unknown parameter in prior_dict: '{key}'. Valid names: {all_pars}")
            if 'type' not in val:
                raise ValueError(f"Missing 'type' field for prior '{key}'. Must be 'uniform' or 'gaussian'.")
            if val['type'] == 'uniform' and not all(k in val for k in ['low', 'high']):
                raise ValueError(f"Uniform prior '{key}' must define 'low' and 'high'.")
            if val['type'] == 'gaussian' and not all(k in val for k in ['mean', 'sigma']):
                raise ValueError(f"Gaussian prior '{key}' must define 'mean' and 'sigma'.")

        # --- Default priors ---
        priors = {
            'fesc':    {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'ion_gas': {'type': 'uniform', 'low': self.nebular_ions[0], 'high': self.nebular_ions[-1]},
            'age_gas': {'type': 'uniform', 'low': self.nebular_ages[0], 'high': self.nebular_ages[-1] + 1},
            'av':      {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'av_ext':  {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'alpha':   {'type': 'uniform', 'low': self.dustem_alpha[0], 'high': self.dustem_alpha[-1]},
            'm_star':  {'type': 'uniform', 'low': 7.0, 'high': 11.0}
        }
        for i, p in enumerate(self.model_pars):
            lo, hi = self.model_pars_arr[i][0], self.model_pars_arr[i][-1]
            priors[p] = {'type': 'uniform', 'low': lo, 'high': hi}

        # --- Override with user-defined priors ---
        for p in custom_priors.keys():
            priors[p] = custom_priors[p]

        # --- Print priors ---
        print("\nPriors:")
        for p in all_pars:
            if priors[p]['type'] == 'uniform':
                print(f"  - {p}: Uniform({priors[p]['low']}, {priors[p]['high']})")
            elif priors[p]['type'] == 'gaussian':
                print(f"  - {p}: Gaussian(mean={priors[p]['mean']}, sigma={priors[p]['sigma']})")
            else:
                print(f"  - {p}: {priors[p]['type']}")

        # --- Define transform function based on priors ---
        def prior_transform(u):
            x = np.zeros_like(u)
            for i, name in enumerate(all_pars):
                prior = priors[name]
                if prior['type'] == 'uniform':
                    x[i] = prior['low'] + u[i] * (prior['high'] - prior['low'])
                elif prior['type'] == 'gaussian':
                    # Use inverse CDF of normal distribution
                    x[i] = prior['mean'] + prior['sigma'] * np.sqrt(2) * erfinv(2 * u[i] - 1)
                else:
                    raise ValueError(f"Unsupported prior type '{prior['type']}' for parameter '{name}'.")
            return x

        return prior_transform


    def synthetic_spectrum(
        self, fesc, ion_gas, age_gas, av, av_ext, alpha, m_star, redshift, dl,
        multi_component=False, **kwargs
    ):
        """
        Generate a physically motivated synthetic galaxy spectrum, combining stellar populations, nebular lines, 
        dust attenuation, and re-emission, redshifted and scaled for observational comparison.
        """

        young_stellar_spec = self.interp_young_flux(**kwargs)
        old_stellar_spec = self.interp_old_flux(**kwargs)

        young_stellar_spec = np.interp(self.model_wl, self.short_model_wl, young_stellar_spec)
        old_stellar_spec = np.interp(self.model_wl, self.short_model_wl, old_stellar_spec)

        # --- Ionizing photons
        index_lyman = np.searchsorted(self.model_wl, 912, side='left')
        lycont_wls = np.r_[self.model_wl[:index_lyman], np.array([912])]

        lycont_spec = np.interp(lycont_wls, self.model_wl, young_stellar_spec)
        nlyman_young = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        lycont_spec = np.interp(lycont_wls, self.model_wl, old_stellar_spec)
        nlyman_old = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        if fesc >= 0:
            young_stellar_spec[:index_lyman] *= fesc
            old_stellar_spec[:index_lyman] *= fesc

        nebular_lines = self.nebular_func(ion_gas, age_gas)
        nebular_young = nebular_lines * nlyman_young
        nebular_old = nebular_lines * nlyman_old

        young_stellar_nebular_spec = young_stellar_spec + nebular_young
        old_stellar_nebular_spec = old_stellar_spec + nebular_old
        stellar_nebular_spec = young_stellar_nebular_spec + old_stellar_nebular_spec

        # --- Dust attenuation
        att_young_stellar_nebular_spec = (10 ** (-(av + av_ext) * self.k_cal)) * young_stellar_nebular_spec
        att_old_stellar_nebular_spec = (10 ** (-av * self.k_cal)) * old_stellar_nebular_spec
        att_stellar_nebular_spec = att_young_stellar_nebular_spec + att_old_stellar_nebular_spec

        # --- Dust emission
        lbol_init = np.trapz(stellar_nebular_spec, self.model_wl)
        lbol_att = np.trapz(att_stellar_nebular_spec, self.model_wl)
        dust_bol = (lbol_init - lbol_att)

        tdust_spec = self.dustem_func(alpha)
        mask_pixels = (self.model_wl >= 2.5e4) & (self.model_wl <= 3e4)
        scale = np.sum(stellar_nebular_spec[mask_pixels] * tdust_spec[mask_pixels]) / \
                np.sum(stellar_nebular_spec[mask_pixels] * stellar_nebular_spec[mask_pixels])
        tdust_spec -= scale * stellar_nebular_spec
        tdust_spec[(self.model_wl < 2.5e4) | (tdust_spec < 0.)] = 0.

        norm = np.trapz(tdust_spec, self.model_wl)
        dust_spec = tdust_spec * dust_bol / norm

        total_spec = att_stellar_nebular_spec + dust_spec

        # --- Redshift and flux scaling
        model_red_wl = self.model_wl * (1 + redshift)
        total_spec /= (1 + redshift)

        fscale = 10 ** m_star / (dl * 1e5) ** 2
        total_spec *= fscale

        # --- FITS Header with parameters
        header = fits.Header()
        header['WUNITS'] = 'A'
        header['FUNITS'] = 'erg/s/cm2/A'

        # Add model parameters
        header['FESC'] = (fesc, 'Ionizing photon escape fraction')
        header['ION_GAS'] = (ion_gas, 'Gas ionization parameter')
        header['AGE_GAS'] = (age_gas, 'Nebular region age [Myr]')
        header['AV_OLD'] = (av, 'V-band attenuation of old stars [mag]')
        header['AV_EXT'] = (av_ext, 'Extra V-band attenuation for young stars [mag]')
        header['ALPHA'] = (alpha, 'Dust emission alpha (Dale+2002)')
        header['MSTAR'] = (m_star, 'Log10 stellar mass [M_sun]')
        header['REDSHIFT'] = (redshift, 'Redshift')
        header['DL_PC'] = (dl, 'Luminosity distance [pc]')
        header['MULTICMP'] = (multi_component, 'Includes subcomponents')
        for key, val in kwargs.items():
            # Store any extra parameters (SFH, metallicity, etc.)
            if isinstance(val, (int, float, str)):
                header[f'{key[:6].upper()}'] = val

        # --- Final Spectrum object
        spec = Spectrum(wl=model_red_wl, resolution=self.model_res, flux=total_spec, header=header)

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



    def cornerplot(self, fit_result, show_stats=True, cmap='viridis', alpha=0.7,):
        """
        Make a corner plot from Dynesty fit results.
    
        Parameters
        ----------
        fit_result : dynesty results object
            The nested sampling results.
        show_stats : bool
            Whether to display mean ± std on top of histograms.
        cmap : str
            Matplotlib colormap for 2D contours.
        alpha : float
            Transparency for 2D contour fills.
        """    
        #resample 
        samples, weights = fit_result.samples, np.exp(fit_result.logwt - fit_result.logz[-1])
        equal_samples = resample_equal(samples, weights)
    
        # Map parameter names to nicer LaTeX labels
        latex_labels = {
            "age": r"$\mathrm{Age~[Myr]}$",
            "tau_main": r"$\tau_\mathrm{main} \mathrm{~[Myr]}$",
            "age_trunc": r"$Q_\\text{AGE}\mathrm{~[Myr]}$",
            "tau_trunc": r"$\\tau_\\text{Q}\mathrm{~[Myr]}$",
            "fesc": r"$f_\mathrm{esc}$",
            "ion_gas": r"$U_\mathrm{ion}$",
            "age_gas": r"$\mathrm{Age_{gas}}$",
            "av": r"$A_V$",
            "av_ext": r"$A_{V,ext}$",
            "alpha": r"$\alpha$",
            "m_star": r"$\log(M_\star)$"
        }
        labels = [latex_labels.get(p, p) for p in self.free_pars]
    
        plt.close('all')
        fig = corner.corner(equal_samples, labels=self.free_pars,
                    color='k', plot_contours=True, fill_contours=True,
                    cmap=cmap, alpha=alpha)
        
        if show_stats:
            means = np.mean(equal_samples, axis=0)
            stds = np.std(equal_samples, axis=0)
            # corner creates n*n axes, diagonal axes are at positions 0, n+1, 2*(n+1), ...
            diag_axes = [fig.axes[i*(len(self.free_pars)+1)] for i in range(len(self.free_pars))]
            for i, ax in enumerate(diag_axes):
                ax.set_title(f"{means[i]:.3f} ± {stds[i]:.3f}", fontsize=10, pad=12)
    
        plt.show()




