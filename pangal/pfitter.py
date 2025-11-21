import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from bisect import bisect_left
from scipy.special import erf, erfinv
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d #?
from scipy.signal import fftconvolve
import itertools
import os

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import corner

from .spectrum import Spectrum
from .run import Run
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps

from .pfitter_utils import load_nebular_tables, load_dust_emission_models, dust_attenuation_curve, model_grid_interpolator

"""
PFitter
   ├── run_fit(...)    → creates a new Run() instance
   │                      fills it with settings + results
   │                      stores it in self.runs list
   ├── many other functions
   └── runs : list[Run]

Run
   ├── stores: data, phot, spectrum, priors
   ├── stores: processed observed spectrum (normalized)
   ├── stores: obs resolution on model grid
   ├── stores: sampler result
   ├── stores: likelihood function
   └── lightweight methods if needed
"""


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

        self.runs = []        # <- list of Run objects

    # methods

    load_nebular_tables = load_nebular_tables
    load_dust_emission_models = load_dust_emission_models
    dust_attenuation_curve = dust_attenuation_curve
    model_grid_interpolator = model_grid_interpolator



    # --- FIT Structure
    # run_fit()               →   sets up the fit, calls make_log_likelihood()
    # make_log_likelihood()   →   precomputes constants, returns log_likelihood()
    # log_likelihood()        →   called thousands of times by the sampler

    """
    run_fit()
        ├── crop + normalize obs spectrum
        ├── setup parameters
        ├── sampler = NestedSampler(log_likelihood, ...)
        └── run sampler

    make_log_likelihood()
        ├── precompute resolution interpolation
        ├── precompute photometry arrays
        ├── precompute masks/transmission filters
        ├── return log_likelihood()  # closure capturing the above
    """
            

    # MENAGANES COMPLEX PHOTOMETRIC TABLES FOR MULTIPLE OBJECTS.
    # MENAGES COMPLEX SPECTRA

    def run_fit(self,
                obj_id=None,
                spec=None,
                phot=None,
                bands=None,
                spectral_range=None,
                fix_pars=None,
                custom_priors=None,
                polymax=7,
                nlive=500,
                dlogz=0.01):
        

        
        if phot and not spec:
            if len(phot.data) == 1: 

                return self._run_single_fit(
                            obj_id=None,
                            spec=None,
                            phot=None,
                            bands,
                            spectral_range,
                            fix_pars,
                            custom_priors,
                            polymax,
                            nlive,
                            dlogz)
            
            elif obj_id:





                
            

        #if phot and not spec:


        #if phot and spec:
        

            



    # internal
    def _run_single_fit(self,
                obj_id,
                spec,
                phot,
                bands,
                spectral_range,
                fix_pars,
                custom_priors,
                polymax,
                nlive,
                dlogz):

        # ------- safe copies for mutable inputs -------
        fix_pars = {} if fix_pars is None else dict(fix_pars)
        custom_priors = {} if custom_priors is None else dict(custom_priors)

        run = Run()     # new Run container
        run.pfitter = self

        # store user inputs
        run.spec = spec
        run.phot = phot
        run.spectral_range = spectral_range
        run.polymax = polymax
        run.custom_priors = custom_priors

        # --- bands selection ---
        if bands:
            for b in bands:
                if b not in map_filter_names.keys():
                    raise ValueError(f'Unrecognized filter: {b}. Abort')
            run.bands = list(bands)
        else:
            run.bands = [b for b in phot.photometry.keys() if b in map_filter_names.keys()]

        print(f"Using the following photometric filters: {', '.join(run.bands)}")

        # --- preprocess observed spectrum once (done here) ---
        if spec is not None and spectral_range is not None:
            run.norm_wl, run.norm_flux, run.norm_flux_err, run.continuum = \
                self._preprocess_observed_spectrum(spec, spectral_range, polymax)
        else:
            run.norm_wl = run.norm_flux = run.norm_flux_err = run.continuum = None

        # --- redshift and luminosity distance ---
        z_keys = ("REDSHIFT", "redshift", "z")
        if phot and hasattr(phot, "header"):
            run.redshift = next((phot.header[k] for k in z_keys if k in phot.header), None)
        elif spec and hasattr(spec, "header"):
            run.redshift = next((spec.header[k] for k in z_keys if k in spec.header), None)
        else:
            run.redshift = None
        if run.redshift is None:
            run.redshift = 0
            print("Redshift not provided; set to 0.")
        if phot and hasattr(phot, 'header') and 'dl' in phot.header:
            run.dl = phot.header['dl']
        elif spec and hasattr(spec, 'header') and 'dl' in spec.header:
            run.dl = spec.header['dl']
        else:
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            run.dl = cosmo.luminosity_distance(run.redshift).value

        # ------- fix_pars handling (copy, don't mutate caller) -------
        run.fix_pars = dict(fix_pars)  # safe copy

        # If there is no spectrum, treat vel params as fixed/unavailable
        if spec is None:
            # do not mutate caller dict (we already copied)
            run.fix_pars.setdefault("vel_sys", None)
            run.fix_pars.setdefault("sigma_vel", None)

        # ------- build lists of free parameters (separate model and global) -------
        run.free_model_pars = [p for p in self.model_pars if p not in run.fix_pars]

        global_pars = ["fesc", "ion_gas", "age_gas", "av", "av_ext",
                    "alpha", "m_star", "vel_sys", "sigma_vel"]
        run.free_global_pars = [p for p in global_pars if p not in run.fix_pars]

        run.free_pars = run.free_model_pars + run.free_global_pars

        # --- build likelihood and prior (pass run.bands) ---
        run.log_likelihood = self.make_log_likelihood(run, spec, phot, run.free_pars, run.bands)
        run.prior_transform = self.make_prior_transform(run, run.custom_priors)

        print("Initializing live points")
        run.sampler = NestedSampler(run.log_likelihood, run.prior_transform, ndim=len(run.free_pars), nlive=nlive)

        # --- run sampler ---
        run.sampler.run_nested(dlogz=dlogz, print_progress=True)
        run.result = run.sampler.results

        # Save run into PFitter
        self.runs.append(run)
        return run



    def make_log_likelihood(self, run, spec, phot, free_pars, bands):
        # precompute and attach to run for transparency

        # --- resolution interpolation for model grid (if spectrum provided) ---
        if spec is not None:
            run.obs_resolution_on_model_grid = interp1d(
                spec.wl, spec.resolution, kind="linear",
                bounds_error=False, fill_value="extrapolate"
            )(self.model_wl)
        else:
            run.obs_resolution_on_model_grid = None

        # --- photometric precomputations ---
        if phot is not None:
            run.phot_points = np.array([phot.data[][b][0] for b in bands])
            run.phot_errors = np.array([phot.data[][b][1] for b in bands])
            run.upper_lims = (run.phot_points / run.phot_errors < 5).astype(int)

            run.trans_mask = {}
            run.trans_arrays = {}
            run.pivot_wls = {}
            for b in bands:
                F = Filter(b)
                lmin, lmax = F.wavelength_range
                mask_b = (self.model_wl >= lmin) & (self.model_wl <= lmax)
                run.trans_mask[b] = mask_b
                run.trans_arrays[b] = F.transmission_curve(self.model_wl[mask_b])
                run.pivot_wls[b] = F.pivot_wavelength

        # --- closure used by the sampler ---
        def log_likelihood(pars):
            spec_lhood = 0.0
            phot_lhood = 0.0

            idx = 0

            # ---- MODEL PARAMETERS: only read the *free* model params from pars ----
            free_model_names = run.free_model_pars
            n_free_model = len(free_model_names)
            model_free_values = np.array([]) if n_free_model == 0 else pars[idx: idx + n_free_model]
            idx += n_free_model

            # Build model kwargs including fixed model params
            model_kwargs = {}
            for name in self.model_pars:
                if name in run.fix_pars:
                    model_kwargs[name] = run.fix_pars[name]
                else:
                    # find index in free_model_names
                    pos = free_model_names.index(name)
                    model_kwargs[name] = model_free_values[pos]

            # ---- GLOBAL / NAMED PARAMETERS ----
            named_order = ["fesc", "ion_gas", "age_gas", "av", "av_ext",
                        "alpha", "m_star", "vel_sys", "sigma_vel"]
            named_values = {}
            for name in named_order:
                if name in run.fix_pars:
                    named_values[name] = run.fix_pars[name]
                else:
                    named_values[name] = pars[idx]
                    idx += 1

            # unpack named params for readability
            fesc = named_values["fesc"]
            ion_gas = named_values["ion_gas"]
            age_gas = named_values["age_gas"]
            av = named_values["av"]
            av_ext = named_values["av_ext"]
            alpha = named_values["alpha"]
            m_star = named_values["m_star"]
            vel_sys = named_values["vel_sys"]
            sigma_vel = named_values["sigma_vel"]

            # --- Build synthetic spectrum (pass run.obs_resolution_on_model_grid) ---
            synth_spec = self.synthetic_spectrum(**model_kwargs,
                                                fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                                                av=av, av_ext=av_ext, alpha=alpha,
                                                m_star=m_star,
                                                vel_sys=vel_sys, sigma_vel=sigma_vel,
                                                redshift=0, dl=100,
                                                observed_spectrum_resolution=run.obs_resolution_on_model_grid)

            # ----------------- Photometric likelihood -----------------
            if phot is not None:
                model_phot = []
                for b in bands:
                    mask_b = run.trans_mask[b]
                    spec_array = synth_spec.flux[mask_b]
                    if len(spec_array) == 0:
                        model_phot.append(np.nan)
                        continue
                    num_int = np.trapz(run.trans_arrays[b] * spec_array, self.model_wl[mask_b])
                    den = np.trapz(run.trans_arrays[b], self.model_wl[mask_b])
                    if den == 0:
                        model_phot.append(np.nan)
                        continue
                    phot_point = num_int / den
                    # convert to mJy if needed
                    c = 2.99792458e18  # Å/s
                    phot_point = phot_point * run.pivot_wls[b]**2 / c / 1e-26
                    model_phot.append(phot_point)
                model_phot = np.array(model_phot)
                if not np.all(np.isfinite(model_phot)):
                    return -1e100
                for i in range(len(run.phot_points)):
                    if run.upper_lims[i] == 0:
                        phot_lhood += -0.5 * (
                            (run.phot_points[i] - model_phot[i])**2 / run.phot_errors[i]**2
                            + np.log(run.phot_errors[i]**2)
                            + np.log(2. * np.pi)
                        )
                    else:
                        terf = 0.5 * (1 + erf((run.phot_points[i] - model_phot[i]) /
                                            (np.sqrt(2.) * run.phot_errors[i])))
                        if terf <= 0:
                            return -1e100
                        phot_lhood += np.log(terf)

            # ----------------- Spectral likelihood (features only) -------------
            if spec is not None:
                wl_obs = run.norm_wl
                flux_obs = run.norm_flux
                err_obs = run.norm_flux_err

                # interpolate model to observed wl (use interp1d so we can extrapolate if needed)
                interp_model = interp1d(synth_spec.wl, synth_spec.flux, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
                model_flux_interp = interp_model(wl_obs)

                # remove continuum from model with same polynomial degree used on observed data
                coeff_model = np.polyfit(wl_obs, model_flux_interp, deg=run.polymax)
                model_continuum = np.polyval(coeff_model, wl_obs)
                model_flux_norm = model_flux_interp / model_continuum

                # compute residuals and gaussian log-likelihood
                residual = flux_obs - model_flux_norm
                inv_var = 1.0 / (err_obs**2)
                spec_lhood = -0.5 * np.sum(residual**2 * inv_var + np.log(2 * np.pi / inv_var))

            return spec_lhood + phot_lhood

        return log_likelihood


    def make_prior_transform(self, run, custom_priors):
        # safe copy
        custom_priors = {} if custom_priors is None else dict(custom_priors)

        # validate custom priors against the free parameters
        for key in custom_priors.keys():
            if key not in run.free_pars:
                raise ValueError(f"Unknown parameter in prior_dict: {key}")

        # default priors
        priors = {
            'fesc':    {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'ion_gas': {'type': 'uniform', 'low': self.nebular_ions[0], 'high': self.nebular_ions[-1]},
            'age_gas': {'type': 'uniform', 'low': self.nebular_ages[0], 'high': self.nebular_ages[-1] + 1},
            'av':      {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'av_ext':  {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'alpha':   {'type': 'uniform', 'low': self.dustem_alpha[0], 'high': self.dustem_alpha[-1]},
            'm_star':  {'type': 'uniform', 'low': 7.0, 'high': 11.0},
            'vel_sys': {'type': 'uniform', 'low': -500.0, 'high': 500.0},
            'sigma_vel': {'type': 'uniform', 'low': 1.0, 'high': 200.0},
        }
        for i, p in enumerate(self.model_pars):
            lo, hi = self.model_pars_arr[i][0], self.model_pars_arr[i][-1]
            priors[p] = {'type': 'uniform', 'low': lo, 'high': hi}

        # override with custom
        for p, v in custom_priors.items():
            priors[p] = v

        # print priors for the free params only
        print("\nPriors:")
        for p in run.free_pars:
            pr = priors[p]
            if pr['type'] == 'uniform':
                print(f"  - {p}: Uniform({pr['low']}, {pr['high']})")
            elif pr['type'] == 'gaussian':
                print(f"  - {p}: Gaussian(mean={pr['mean']}, sigma={pr['sigma']})")
            else:
                print(f"  - {p}: {pr}")

        # transform from unit cube to physical space
        def prior_transform(u):
            x = np.zeros_like(u)
            for i, name in enumerate(run.free_pars):
                prior = priors[name]
                if prior['type'] == 'uniform':
                    x[i] = prior['low'] + u[i] * (prior['high'] - prior['low'])
                elif prior['type'] == 'gaussian':
                    x[i] = prior['mean'] + prior['sigma'] * np.sqrt(2) * erfinv(2 * u[i] - 1)
                else:
                    raise ValueError(f"Unsupported prior type {prior['type']} for {name}")
            return x

        return prior_transform

    def synthetic_spectrum(
        self,
        fesc,
        ion_gas,
        age_gas,
        av,
        av_ext,
        alpha,
        m_star,
        redshift,
        dl,
        vel_sys=None,                        # systemic velocity [km/s]
        sigma_vel=None,                      # LOS velocity dispersion [km/s]
        observed_spectrum_resolution=None,   # If provided (it's an array) builds the model spectrum to the required resolution
        multi_component=False,
        **kwargs,
    ):
        """
        Generate a physically motivated synthetic galaxy spectrum including stellar,
        nebular, dust, and kinematic effects (velocity shift and broadening).

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
            Logarithmic stellar mass of the galaxy (log₁₀[M*/M☉]).
        redshift : float
            Cosmological redshift of the galaxy.
        dl : float
            Luminosity distance to the galaxy in parsecs (used for flux scaling).
        vel_sys : float, optional
            Systemic velocity offset [km/s], adds an extra Doppler shift to the spectrum.
        sigma_vel : float, optional
            Line-of-sight velocity dispersion [km/s], broadens absorption/emission lines.
        multi_component : bool, optional
            If True, includes component arrays (dust, young/old spectra) in the output Spectrum.
        **kwargs :
            Additional parameters for stellar population model interpolation (e.g., age, Z, SFH).

        Returns
        -------
        Spectrum
            Synthetic `Spectrum` object with wavelength (Å) and flux (erg/s/cm²/Å),
            including optional subcomponents if `multi_component=True`.
        """

        # --- Stellar population spectra ---
        young_stellar_spec = self.interp_young_flux(**kwargs)
        old_stellar_spec   = self.interp_old_flux(**kwargs)

        # Resample to model wavelength grid
        young_stellar_spec = np.interp(self.model_wl, self.short_model_wl, young_stellar_spec)
        old_stellar_spec   = np.interp(self.model_wl, self.short_model_wl, old_stellar_spec)

        # --- Ionizing photons and nebular emission ---
        index_lyman = np.searchsorted(self.model_wl, 912, side="left")
        lycont_wls = np.r_[self.model_wl[:index_lyman], np.array([912])]

        # Young
        lycont_spec = np.interp(lycont_wls, self.model_wl, young_stellar_spec)
        nlyman_young = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        # Old
        lycont_spec = np.interp(lycont_wls, self.model_wl, old_stellar_spec)
        nlyman_old = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        # Remove escaped photons
        if fesc >= 0:
            young_stellar_spec[:index_lyman] *= fesc
            old_stellar_spec[:index_lyman]  *= fesc

        # Nebular emission
        nebular_lines = self.nebular_func(ion_gas, age_gas)
        nebular_young = nebular_lines * nlyman_young
        nebular_old   = nebular_lines * nlyman_old

        young_stellar_nebular_spec = young_stellar_spec + nebular_young
        old_stellar_nebular_spec   = old_stellar_spec + nebular_old
        stellar_nebular_spec       = young_stellar_nebular_spec + old_stellar_nebular_spec

        # --- Dust attenuation ---
        att_young_stellar_nebular_spec = (10 ** (-(av + av_ext) * self.k_cal)) * young_stellar_nebular_spec
        att_old_stellar_nebular_spec   = (10 ** (-av * self.k_cal)) * old_stellar_nebular_spec
        att_stellar_nebular_spec       = att_young_stellar_nebular_spec + att_old_stellar_nebular_spec

        # --- Thermal dust re-emission ---
        lbol_init = np.trapz(stellar_nebular_spec, self.model_wl)
        lbol_att  = np.trapz(att_stellar_nebular_spec, self.model_wl)
        dust_bol  = lbol_init - lbol_att

        tdust_spec = self.dustem_func(alpha)

        # Remove stellar leakage in Dale templates
        mask_pixels = (self.model_wl >= 2.5e4) & (self.model_wl <= 3e4)
        scale = np.sum(stellar_nebular_spec[mask_pixels] * tdust_spec[mask_pixels]) / \
                np.sum(stellar_nebular_spec[mask_pixels] ** 2)
        tdust_spec -= scale * stellar_nebular_spec
        tdust_spec[(self.model_wl < 2.5e4) | (tdust_spec < 0.)] = 0.

        # Scale dust emission
        norm = np.trapz(tdust_spec, self.model_wl)
        dust_spec = tdust_spec * dust_bol / norm

        # --- Total rest-frame spectrum ---
        total_spec = att_stellar_nebular_spec + dust_spec



        # BROADENING
        # --- Apply LOS velocity broadening (in log λ space) ---
        # --- Shift using systemic velocity

        #wl_shifted = self.model_wl
        c = 2.99792458e5  # km/s
        wl_shifted = self.model_wl.copy()

        if vel_sys and sigma_vel:
            
            # --- Estimate internal broadening (instrumental) ---
            sigma_v_internal = c / (self.model_res * 2.355)  # per-pixel σ_v array
            sigma_v_internal_med = np.median(sigma_v_internal)

            # --- Compute net broadening to apply ---
            if sigma_vel > sigma_v_internal_med:
                sigma_v_apply = np.sqrt(sigma_vel**2 - sigma_v_internal_med**2)
            else:
                sigma_v_apply = 0.0


            if sigma_v_apply > 0:
                total_spec = self._vel_convolve_fft(total_spec, self.model_wl, sigma_v_apply)

            wl_shifted *= (1 + vel_sys / c)



        # --- Redshift to observed frame ---
        model_red_wl = wl_shifted * (1 + redshift)
        total_spec /= (1 + redshift)


        # --- Match model resolution to observed resolution R(λ) ---
        if observed_spectrum_resolution is not None:
          
            # Convert observed R to σ_v(λ)
            fwhm_obs = model_red_wl / observed_spectrum_resolution
            sigma_obs_kms = (fwhm_obs / 2.355) * (c / model_red_wl)

            # Convert model intrinsic R to σ_v
            fwhm_model = model_red_wl / self.model_res
            sigma_model_kms = (fwhm_model / 2.355) * (c / model_red_wl)

            # σ to apply
            sigma_apply = np.sqrt(np.maximum(0, sigma_obs_kms**2 - sigma_model_kms**2))

            # Perform convolution in chunks. FFT convolution assumes a constant σ over the whole array.
            # Split the spectrum into chunks. Convolve each chunk with a constant σ (≈ the median in that chunk)
            N_chunks = 12
            idxs = np.array_split(np.arange(len(total_spec)), N_chunks)
            model_conv = np.empty_like(total_spec)

            for idx in idxs:
                sigma_local = np.median(sigma_apply[idx])
                wl_local    = model_red_wl[idx]
                spec_local  = total_spec[idx]

                model_conv[idx] = self._vel_convolve_fft(
                    spec_local, wl_local, sigma_local, vel=0.0
                )

            total_spec = model_conv




        # --- Flux scaling to given stellar mass and luminosity distance ---
        fscale = 10**m_star / (dl * 1e5)**2
        total_spec *= fscale

        # --- Build FITS header ---
        header = fits.Header()
        header["WUNITS"]  = "A"
        header["FUNITS"]  = "erg/s/cm2/A"
        header["REDSHIFT"] = (redshift, "Cosmological redshift")
        header["VEL_SYS"]  = (vel_sys, "Systemic velocity [km/s]")
        header["SIGMA_V"]  = (sigma_vel, "LOS velocity dispersion [km/s]")
        header["AV"]       = (av, "V-band attenuation (old stars)")
        header["AV_EXT"]   = (av_ext, "Extra attenuation (young stars)")
        header["FESC"]     = (fesc, "Escape fraction of ionizing photons")
        header["ION_GAS"]  = (ion_gas, "Ionization parameter")
        header["AGE_GAS"]  = (age_gas, "Nebular region age [Myr]")
        header["ALPHA"]    = (alpha, "Dust heating parameter")
        header["MSTAR"]    = (m_star, "log10 Stellar mass [Msun]")
        header["DL_PC"]    = (dl, "Luminosity distance [pc]")

        # Add kwargs for traceability
        for k, v in kwargs.items():
            key = k[:8].upper()
            try:
                header[key] = v
            except Exception:
                header[key] = str(v)

        # --- Create Spectrum object ---
        spec = Spectrum(wl=model_red_wl, resolution=self.model_res, flux=total_spec, header=header)

        if multi_component:
            # Save components (rescaled to observed frame)
            for s in [young_stellar_nebular_spec, old_stellar_nebular_spec,
                    att_young_stellar_nebular_spec, att_old_stellar_nebular_spec, dust_spec]:
                s /= (1 + redshift)
                s *= fscale
            spec.young_stellar_nebular = young_stellar_nebular_spec
            spec.old_stellar_nebular   = old_stellar_nebular_spec
            spec.att_young_stellar_nebular = att_young_stellar_nebular_spec
            spec.att_old_stellar_nebular   = att_old_stellar_nebular_spec
            spec.dust = dust_spec

        return spec


    def _losvd_rfft(self, vel, sigma, wl, npad=None, c=2.99792458e5):
        """
        Analytic Fourier transform of a Gaussian LOSVD, following Cappellari et al. (2016).
        """
        npix = len(wl)
        if npad is None:
            npad = 2 * npix

        nl = npad // 2 + 1  # size of RFFT array

        # log-lambda grid properties
        loglam = np.log(wl)
        dloglam = np.median(np.diff(loglam))
        kms_per_pix = c * dloglam

        # convert to pixel units
        vel_pix = vel / kms_per_pix
        sigma_pix = sigma / kms_per_pix

        # frequency vector for RFFT
        w = np.linspace(0, np.pi, nl)

        if sigma_pix == 0:
            losvd_rfft = np.ones_like(w, dtype=complex)
        else:
            a = vel_pix / sigma_pix
            losvd_rfft = np.exp(1j * a * w - 0.5 * (sigma_pix * w)**2)

        return np.conj(losvd_rfft)
        
    def _vel_convolve_fft(self, spec, wl, sigma, vel=0.0):
        """Velocity-space convolution using analytic LOSVD in Fourier domain."""
        npix = len(spec)
        npad = 2 * npix  # zero-padding to avoid wrap-around

        # make LOSVD kernel for same npad length
        fft_losvd = self._losvd_rfft(vel, sigma, wl, npad=npad)

        # FFT of the input spectrum
        fft_spec = np.fft.rfft(spec, n=npad)

        # multiply in Fourier space (convolution)
        spec_conv = np.fft.irfft(fft_spec * fft_losvd, n=npad)

        # trim to original size
        return spec_conv[:npix]



    def _preprocess_observed_spectrum(self,spec,spectral_range,polymax):
        
        # --- Crop wavelength range ---
        mask = (spec.wl >= spectral_range[0]) & (spec.wl <= spectral_range[1])
        wl_crop = spec.wl[mask]
        flux_crop = spec.flux[mask]
        err_crop = spec.flux_err[mask]

        # --- Filter out invalid pixels ---
        good = np.isfinite(wl_crop) & np.isfinite(flux_crop) & np.isfinite(err_crop) & (err_crop > 0)
        wl_crop = wl_crop[good]
        flux_crop = flux_crop[good]
        err_crop = err_crop[good]

        # --- Fit polynomial to the continuum ---
        # We use weighted fitting with 1/error^2 weights
        coeff = np.polyfit(wl_crop, flux_crop, deg=polymax, w=1./err_crop)
        continuum = np.polyval(coeff, wl_crop)

        # --- Normalize flux and errors ---
        normalized_obs_wl_for_likelihood = wl_crop
        normalized_obs_flux_for_likelihood = flux_crop / continuum
        normalized_obs_err_for_likelihood = err_crop / continuum

        return normalized_obs_wl_for_likelihood, normalized_obs_flux_for_likelihood, normalized_obs_err_for_likelihood, continuum















