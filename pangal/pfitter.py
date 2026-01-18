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

from .photometry_table import PhotometryTable
from .spectrum import Spectrum
from .run import Run
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps
from .data.spectral_lines import atmospheric_lines

from .pfitter_utils import load_nebular_tables, load_dust_emission_models, model_grid_interpolator, load_spectrum_models_from_fits

"""
PFitter
   ├── run_fit(...)    → creates a new Run() instance
   │                      fills it with settings + results
   │                      stores it in self.runs list
   ├── many other functions


Run
   ├── stores: data, phot, spectrum, priors
   ├── stores: processed observed spectrum (normalized)
   ├── stores: obs resolution on model grid
   ├── stores: sampler result
   ├── stores: likelihood function
   └── lightweight methods if needed
"""


class PFitter(): 
    
    # Parametric fitter: this first block loads a grid of models

    def __init__(self, model_file, model_pars, cosmo=None, dustemimodel='dh02', leitatt=False, uv_bump=False, emimetal=0.0017, emimodel='2018'):

        print(f"Initializing PFitter object")
        self.model_file = model_file
        self.model_pars = model_pars  # list the variables defyning the model grid
        self.cosmo = cosmo
        self.dustemimodel = dustemimodel
        self.leiatt = leitatt
        self.uv_bump = uv_bump
        self.emimetal = emimetal
        self.emimodel = emimodel

        print("Loading models from ",model_file)
        model_list = load_spectrum_models_from_fits(model_file)

        ### ADDED LATER!! JUST DIAGNOSTICS
        self.model_list = model_list

        # Menages model grid
        if not isinstance(model_list, list):
            raise ValueError("Error: model_grid is not a numpy array.")


        # Model wavelength
        sps_models_wl = model_list[0].wl
        
        #get the wavelength information for the dust templates to stitch it to the stellar model wl 
        #this_dir = os.path.dirname(os.path.abspath(__file__))
        #file_path = os.path.join(this_dir, 'data', f'spectra_{dustemimodel}.dat')
        file_path = '/home/utente/Desktop/Pangal/pangal/data/'+f'spectra_{dustemimodel}.dat'
        dust_wl = np.loadtxt(file_path, usecols=(0,)) * 1e4   # Load wavelength grid (in Angstroms)

        #expand wavelength grid to include range covered by dust templates
        # Find the extended part of dust_wl
        extra_wl_mask = dust_wl > sps_models_wl.max()
        extra_wl = dust_wl[extra_wl_mask]

        # Extend wavelength grid
        self.model_wl = np.r_[sps_models_wl, extra_wl]

        # Extend resolution array with constant value matching the number of extra wavelengths
        last_res = model_list[0].resolution[-1]
        extra_res = np.full(extra_wl.size, last_res)

        # Concatenate the original model resolution with the extended constant resolution
        self.model_resolution = np.r_[model_list[0].resolution, extra_res]


        # INTERPOLATION!
        self.interp_young_flux, self.interp_old_flux, self.model_pars_arr, _ = self.model_grid_interpolator(model_list, self.model_pars)

        print(f" Pre-computed model grid:")
        for name, arr in zip(self.model_pars, self.model_pars_arr):
            print(f"   - {name}: {len(arr)} values, min={arr[0]}, max={arr[-1]}")
        print("\n")
        



        # PRECOMPUTES DUST ATTENUATION CURVE (Calzetti, 2000 + optional Leitherer+2002 + optional UV bump)
        self.k_cal = self.dust_attenuation_curve(self.model_wl, leitatt, uv_bump)
        print('Dust attenuation curve from Calzetti 2000.')

        # LOADS NEBULAR LINE TABLES
        self.nebular_func, self.nebular_ions, self.nebular_ages = self.load_nebular_tables(self.model_wl,self.model_resolution, emimetal,emimodel)

        # LOADS DUST EMISSION SPECTRA
        # This can be moved above for computing dust_wl only once?
        self.dustem_func, self.dustem_alpha = self.load_dust_emission_models(self.model_wl,dustemimodel)


    # methods

    load_nebular_tables = load_nebular_tables
    load_dust_emission_models = load_dust_emission_models
    model_grid_interpolator = model_grid_interpolator






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
        luminosity_distance,
        vel_sys=None,                         # systemic velocity [km/s]
        sigma_vel=None,                       # LOS velocity dispersion [km/s]
        observed_spectrum_resolution=None,   # If provided (it's an array) builds the model spectrum to the required resolution
        multi_component=False,               # OTHERWISE THE RESOLUTION IS THE ONE OF THE INITAL MODELS
        likelihood_call=False,
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
        luminosity_distance : float
            Luminosity distance to the galaxy in Mpc (used for flux scaling).
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
        #young_stellar_spec = np.interp(self.model_wl, self.short_model_wl, young_stellar_spec)
        #old_stellar_spec   = np.interp(self.model_wl, self.short_model_wl, old_stellar_spec)

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
            
            # --- Estimate internal broadening (model resolution) ---
            sigma_v_internal = c / (self.model_resolution * 2.355)  # per-pixel σ_v array
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
            fwhm_model = model_red_wl / self.model_resolution
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
        fscale = 10**m_star / (luminosity_distance * 1e5)**2
        total_spec *= fscale


        if not likelihood_call:
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
            header["DL_MPC"]    = (luminosity_distance, "Luminosity distance [Mpc]")

            # Add kwargs for traceability
            for k, v in kwargs.items():
                key = k[:8].upper()
                try:
                    header[key] = v
                except Exception:
                    header[key] = str(v)

            # --- Create Spectrum object ---
            spec = Spectrum(wl=model_red_wl, resolution=self.model_resolution, flux=total_spec, header=header)

            
            if multi_component:
                # --- Rescale fluxes to observed frame ---
                # Avoid modifying the original arrays if needed later
                young_stellar_nebular_flux = young_stellar_nebular_spec / (1 + redshift) * fscale
                old_stellar_nebular_flux   = old_stellar_nebular_spec   / (1 + redshift) * fscale
                att_young_stellar_nebular_flux = att_young_stellar_nebular_spec / (1 + redshift) * fscale
                att_old_stellar_nebular_flux   = att_old_stellar_nebular_spec   / (1 + redshift) * fscale
                dust_flux = dust_spec * fscale / (1 + redshift)

                # --- Save main components ---
                spec.young_stellar_nebular = Spectrum(
                    wl=model_red_wl,
                    resolution=self.model_resolution,
                    flux=young_stellar_nebular_flux,
                    header={**header, "COMPONENT": "young stellar and nebular spectrum, not attenuated by dust"}
                )

                spec.old_stellar_nebular = Spectrum(
                    wl=model_red_wl,
                    resolution=self.model_resolution,
                    flux=old_stellar_nebular_flux,
                    header={**header, "COMPONENT": "old stellar and nebular spectrum, not attenuated by dust"}
                )

                spec.att_young_stellar_nebular = Spectrum(
                    wl=model_red_wl,
                    resolution=self.model_resolution,
                    flux=att_young_stellar_nebular_flux,
                    header={**header, "COMPONENT": "young stellar and nebular spectrum, attenuated by dust"}
                )

                spec.att_old_stellar_nebular = Spectrum(
                    wl=model_red_wl,
                    resolution=self.model_resolution,
                    flux=att_old_stellar_nebular_flux,
                    header={**header, "COMPONENT": "old stellar and nebular spectrum, attenuated by dust"}
                )

                spec.dust = Spectrum(
                    wl=model_red_wl,
                    resolution=self.model_resolution,
                    flux=dust_flux,
                    header={**header, "COMPONENT": "dust emission spectrum"}
                )

            return spec
        
        # likelihood call. Minimal
        else: 
            spec = Spectrum(wl=model_red_wl, flux=total_spec)
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




    # Dust attenuation curve based on Calzetti et al. (2000) law
    # wl must be in Angstrom
    # Takes in imput an array of wavelengths and returns the attenuation function

    def dust_attenuation_curve(self, wl, leitatt, uv_bump):
        
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



    ###################################################### --- FIT --- #################################################################



    # --- FIT Structure
    # run_fit()               →   sets up the fit, calls make_log_likelihood()
    # make_log_likelihood()   →   precomputes constants, returns log_likelihood()
    # log_likelihood()        →   called thousands of times by the sampler


    def run_fit(self,
                spec=None,   # observed spectrum (a spectrum object)
                phot=None,   # observed photometry 
                bands=None,
                spectral_range=None,
                fix_pars=None,
                custom_priors=None,
                polydeg=7,
                nlive=500,
                dlogz=0.01):
        
        if phot is not None:
            if not isinstance(phot, PhotometryTable):
                raise TypeError(
                    f"'phot' must be a PhotometryTable instance, got {type(phot).__name__} instead."
                )
        if spec is not None:
            if not isinstance(spec, Spectrum):
                raise TypeError(
                    f"'spec' must be a Spectrum instance, got {type(spec).__name__} instead."
                )
            if not spectral_range:
                spectral_range = [spec.wl[0],spec.wl[-1]]
        if phot is None and spec is None:
            raise ValueError(
                "At least one of 'phot' or 'spec' must be provided."
            )



        # ------- safe copies for mutable inputs -------
        fix_pars = {} if fix_pars is None else dict(fix_pars)
        custom_priors = {} if custom_priors is None else dict(custom_priors)
        
        run = Run()     # new Run container

        # store user inputs
        run.pfitter = self
        run.model_file = self.model_file
        run.model_pars = self.model_pars
        run.spectral_range = spectral_range
        run.nlive = nlive
        run.dlogz = dlogz
        run.polydeg = polydeg
        run.custom_priors = custom_priors
        run.fix_pars = dict(fix_pars) 
        

        # --- bands selection ---
        if phot is not None:
            if bands:
                for b in bands:
                    if b not in map_filter_names.keys():
                        raise ValueError(f'Unrecognized filter: {b}. Abort')
                run.bands = list(bands)
            else:
                run.bands = [b for b in phot.data.keys() if b in map_filter_names.keys()]

            print(f"Using the following photometric filters: {', '.join(run.bands)}")

            run.phot = phot
            

        # --- preprocess observed spectrum once (done here) ---
        if spec:
            run.spec = spec
            obs_spec_wl, obs_spec_flux, obs_spec_flux_err = self._preprocess_observed_spectrum(spec, spectral_range,atmospheric_lines)
            run.spec_crop = Spectrum(wl=obs_spec_wl,flux=obs_spec_flux,flux_err=obs_spec_flux_err,resolution=spec.resolution,header=spec.header)


        # If there is no spectrum, treat vel params as fixed/unavailable
        if spec is None:
            # do not mutate caller dict (we already copied)
            run.fix_pars.setdefault("vel_sys", None)
            run.fix_pars.setdefault("sigma_vel", None)
            run.fix_pars.setdefault("spec_noise_scale", None)
            for par in ("vel_sys", "sigma_vel", "spec_noise_scale"):
                if par in run.custom_priors:
                    run.custom_priors.pop(par)
                print(f"Warning: Removed prior for '{par}' because no spectrum is provided.")



        # ------- build lists of free parameters (separate model and global) -------
        run.free_model_pars = [p for p in self.model_pars if p not in run.fix_pars]

        global_pars = ["fesc", "ion_gas", "age_gas", "av", "av_ext",
                    "alpha", "m_star", "vel_sys", "sigma_vel", "luminosity_distance", "redshift", "spec_noise_scale"]
        run.free_global_pars = [p for p in global_pars if p not in run.fix_pars]

        run.free_pars = run.free_model_pars + run.free_global_pars

        # --- build likelihood and prior ---
        run.log_likelihood = self.make_log_likelihood(run, spec, phot, run.free_pars)
        run.prior_transform = self.make_prior_transform(run, run.custom_priors)

        print("Initializing live points")
        run.sampler = NestedSampler(run.log_likelihood, run.prior_transform, ndim=len(run.free_pars), nlive=nlive)

        # --- run sampler ---
        run.sampler.run_nested(dlogz=dlogz, print_progress=True)
        run.result = run.sampler.results

        return run





    def make_log_likelihood(self, run, spec, phot, free_pars,):

        if phot: 
            phot_fluxes = np.array([run.phot.data[b][0] for b in run.bands])
            phot_errors = np.array([run.phot.data[b][1] for b in run.bands])
            upper_limits = (phot_fluxes / phot_errors < 5).astype(int)
            phot_units = phot.header["UNITS"]
            
            trans_mask = {}
            trans_arrays = {}
            pivot_wls = {}
            for b in run.bands:
                F = Filter(b)
                lmin, lmax = F.wavelength_range
                mask_b = (self.model_wl >= lmin) & (self.model_wl <= lmax)
                trans_mask[b] = mask_b
                trans_arrays[b] = F.transmission_curve(self.model_wl[mask_b])
                pivot_wls[b] = F.pivot_wavelength

            N_phot = len(phot_fluxes)
            obs_resolution_on_model_grid = None
        
        if spec: 
                obs_resolution_on_model_grid = interp1d( 
                    run.spec_crop.wl, run.spec_crop.resolution, kind="linear",
                    bounds_error=False, fill_value="extrapolate"
                )(self.model_wl)

                # Autocorrelation of spectral pixels. Information is not independent
                #N_spec_eff = np.sum(run.spec_crop.resolution / run.spec_crop.wl * np.gradient(run.spec_crop.wl))

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
            names = ["fesc", "ion_gas", "age_gas", "av", "av_ext",
                        "alpha", "m_star", "vel_sys", "sigma_vel","luminosity_distance","redshift","spec_noise_scale"]
            val = {}
            for name in names:
                if name in run.fix_pars:
                    val[name] = run.fix_pars[name]
                else:
                    val[name] = pars[idx]
                    idx += 1

            # unpack named params for readability
            fesc = val["fesc"]
            ion_gas = val["ion_gas"]
            age_gas = val["age_gas"]
            av = val["av"]
            av_ext = val["av_ext"]
            alpha = val["alpha"]
            m_star = val["m_star"]
            vel_sys = val["vel_sys"]
            sigma_vel = val["sigma_vel"]
            luminosity_distance = val["luminosity_distance"]
            redshift = val["redshift"]
            spec_noise_scale = val["spec_noise_scale"]

            # --- Build synthetic spectrum (pass obs_resolution_on_model_grid) ---
            synth_spec = self.synthetic_spectrum(**model_kwargs,
                                                fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                                                av=av, av_ext=av_ext, alpha=alpha,
                                                m_star=m_star,
                                                vel_sys=vel_sys, sigma_vel=sigma_vel,
                                                luminosity_distance=luminosity_distance,
                                                redshift=redshift,
                                                observed_spectrum_resolution=obs_resolution_on_model_grid,
                                                likelihood_call=True,)



            # ----------------- Photometric likelihood -----------------
            if phot is not None:

                model_phot_array = synth_spec.get_phot(bands=run.bands,trans_arrays=trans_arrays,trans_mask=trans_mask,pivot_wls=pivot_wls)

                if not np.all(np.isfinite(model_phot_array)):
                    return -1e100
                
                for i in range(len(phot_fluxes)):
                    if upper_limits[i] == 0:
                        var = phot_errors[i]**2
                        residual = (phot_fluxes[i] - model_phot_array[i]) / phot_errors[i]
                        phot_lhood += -0.5 * (residual**2 + np.log(2*np.pi*var)) 
                    else:
                        terf = 0.5 * (1 + erf((phot_fluxes[i] - model_phot_array[i]) / 
                                            (np.sqrt(2.) * phot_errors[i])))
                        if terf <= 0:
                            return -1e100
                        phot_lhood += np.log(terf)



            # ----------------- Spectral likelihood (features only) -------------
            if spec is not None:
                
                # Returns the model spectrum ready to be compared with the observed spectrum!
                normalization_factor, normalization_factor_smoothed, model = self._adapt_model_spectrum_to_observed_spectrum(run.spec_crop,synth_spec,run.spectral_range,run.polydeg)
                
                # Likelihood mask 
                mask_like = (
                    np.isfinite(model)
                    & np.isfinite(run.spec_crop.flux)
                    & (run.spec_crop.flux_err > 0)
                )
                flux_obs = run.spec_crop.flux[mask_like]
                flux_err = run.spec_crop.flux_err[mask_like]
                model = model[mask_like]


                # Free nuisance parameter: This is not arbitrary — it is equivalent to marginalizing over unknown variance.
                # Spectroscopy → many points (hundreds–thousands), often correlated, continuum-dominated
                # Photometry → few points (∼5–30), independent, broadband
                                
                flux_err_corr = flux_err * np.exp(spec_noise_scale)

                residuals = (flux_obs - model) / (flux_err_corr)
                
                chi2 = np.sum(residuals**2)
                
                spec_lhood = - 0.5 * chi2  - np.sum (np.log( np.sqrt(2 * np.pi) * flux_err_corr))
                            
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
            'luminosity_distance': {'type': 'uniform', 'low': 1, 'high': 1e4}, # in Mpc
            'redshift': {'type': 'uniform', 'low': 0, 'high': 6},    
            'spec_noise_scale': {'type': 'uniform', 'low': -2, 'high': 2}        
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
















    def _preprocess_observed_spectrum(self, spec, spectral_range, atmospheric_lines):
        """
        Preprocess observed spectrum:
        - Accepts one range [wl1, wl2] or multiple ranges [[wl1, wl2], [wl3, wl4], ...]
        - Zeroes out flux and err outside allowed ranges
        - Removes ±25 Å around each atmospheric line
        """

        wl = spec.wl.copy()
        flux = spec.flux.copy()
        err = spec.flux_err.copy()

        # ------------------------------
        # 1. Normalize spectral_range to list of intervals
        # ------------------------------
        if isinstance(spectral_range[0], (int, float)):
            # Single range -> wrap it
            ranges = [spectral_range]
        else:
            # Already list of lists
            ranges = spectral_range

        # ------------------------------
        # 2. Build mask for inside ANY allowed spectral window
        # ------------------------------
        mask_ranges = np.zeros_like(wl, dtype=bool)
        for r in ranges:
            wl1, wl2 = r
            mask_ranges |= (wl >= wl1) & (wl <= wl2)

        # ------------------------------
        # 3. Remove atmospheric lines ±25 Å
        # ------------------------------
        mask_atm = np.ones_like(wl, dtype=bool)
        for line in atmospheric_lines:
            lam = atmospheric_lines[line]
            bad = (wl >= lam - 25) & (wl <= lam + 25)
            mask_atm &= ~bad  # remove these pixels

        # ------------------------------
        # 4. Combined mask for valid pixels
        # ------------------------------
        mask_valid = mask_ranges & mask_atm

        # ------------------------------
        # 5. Zero out invalid pixels (requested behavior)
        # ------------------------------
        flux[~mask_valid] = 0.0
        err[~mask_valid] = 0.0   # error=0 means "ignore these pixels"

        # ------------------------------
        # 6. Remove NaNs, infs, negative errors INSIDE valid regions
        # ------------------------------
        good = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(err) & (err >= 0)
        # Keep invalid pixels zero, but we only mask bad valid ones:
        mask_final = mask_valid & good

        # Apply final mask:
        flux[~mask_final] = np.nan
        err[~mask_final] = np.nan

        return wl, flux, err



    def _adapt_model_spectrum_to_observed_spectrum(self, spec_crop, synth_spec,spectral_range,polydeg):

        wl_obs = spec_crop.wl
        flux_obs = spec_crop.flux
        err_obs = spec_crop.flux_err

        # Interpolate model on observed wl grid WITHOUT extrapolation 
        flux_model = interp1d(synth_spec.wl, synth_spec.flux,
                                kind='linear', bounds_error=False, fill_value=np.nan)(wl_obs) 

        # --- User-selected wavelength range for continuum fit ---
        wl_fit_min = spectral_range[0][0] if isinstance(spectral_range[0], (list, tuple)) else spectral_range[0]
        wl_fit_max = spectral_range[-1][1] if isinstance(spectral_range[-1], (list, tuple)) else spectral_range[1]


        # ---------------- 1) Build a safe mask for continuum fitting ----------------
        mask = (np.isfinite(flux_model) & np.isfinite(flux_obs) & np.isfinite(err_obs) & (err_obs > 0))
        
        # Restrict mask to the wavelength fitting window 
        mask &= (wl_obs >= wl_fit_min) & (wl_obs <= wl_fit_max)


        n_valid = mask.sum()

        # ---------------- 2) Determine polynomial degree ----------------
        if n_valid < max(10, polydeg + 1):
            polydeg = max(1, min(polydeg, n_valid - 1))
        else:
            polydeg = min(polydeg, n_valid - 1)

        # ---------------- 3) Compute raw normalization factor ----------------
        normalization_factor = np.full_like(flux_obs, np.nan)
        normalization_factor[mask] = flux_obs[mask] / flux_model[mask]


        # ---------------- 4) Fit polynomial only on masked region ----------------
        if n_valid >= polydeg + 1:

            # weights
            w = 1.0 / err_obs[mask]

            # wavelength scaling for stability
            wl_masked = wl_obs[mask]
            wl_med = np.median(wl_masked)
            wl_span = wl_masked.ptp() if wl_masked.ptp() != 0 else 1.0
            wl_scaled = (wl_masked - wl_med) / wl_span

            # polyfit
            coeff = np.polyfit(wl_scaled, normalization_factor[mask], deg=polydeg, w=w)

            # evaluate across full wavelength range:
            wl_scaled_full = (wl_obs - wl_med) / wl_span
            normalization_factor_smoothed = np.polyval(coeff, wl_scaled_full)

        else:
            # fallback: constant continuum
            median_val = np.nanmedian(normalization_factor[mask])
            normalization_factor_smoothed = np.full_like(wl_obs, median_val)

        # ---------------- 5) Scale the model ----------------
        model_scaled = flux_model * normalization_factor_smoothed

        
        return normalization_factor, normalization_factor_smoothed, model_scaled

































