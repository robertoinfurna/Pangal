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
import copy

from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from multiprocessing import Pool
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import corner

from .photometry_table import PhotometryTable
from .spectrum import Spectrum
from .run import Run
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps
from .data.spectral_lines import spectral_lines,atmospheric_lines

from .pfitter_utils import load_nebular_tables, load_dust_emission_models, model_grid_interpolator, load_spectrum_models_from_fits, dust_attenuation_curve

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

    def __init__(self, model_file, model_pars, burst_model_file=None, burst_model_pars=None, cosmo=None, dustemimodel='dh02', leitatt=False, uv_bump=False, emimetal=0.017, emimodel='nebular_Byler_mist_2018.lines'):

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


        if burst_model_file and burst_model_pars:
            self.burst_model_pars = burst_model_pars
            print("Burst component: loading models from ",burst_model_file)
            burst_model_list = load_spectrum_models_from_fits(burst_model_file)          

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
        
        self.interp_young_flux, self.interp_old_flux, self.interp_current_stellar_mass, self.model_pars_arr, _ = self.model_grid_interpolator(model_list, model_pars)
        
        print(f" Pre-computed model grid:")
        for name, arr in zip(self.model_pars, self.model_pars_arr):
            print(f"   - {name}: {len(arr)} values, min={arr[0]}, max={arr[-1]}")   
        
        if burst_model_file and burst_model_pars:
            self.interp_burst_young_flux, self.interp_burst_old_flux, self.interp_burst_current_stellar_mass, self.burst_model_pars_arr, _ = self.model_grid_interpolator(burst_model_list, burst_model_pars)

            for name, arr in zip(self.burst_model_pars, self.burst_model_pars_arr):
                print(f"   - {name}: {len(arr)} values, min={arr[0]}, max={arr[-1]}")       


        # PRECOMPUTES DUST ATTENUATION CURVE (Calzetti, 2000 + optional Leitherer+2002 + optional UV bump)
        self.k_calzetti = self.dust_attenuation_curve(leitatt, uv_bump)
        self.k_calzetti_array = self.k_calzetti(self.model_wl)
        print('Dust attenuation curve from Calzetti 2000.')

        # LOADS NEBULAR LINE TABLES
        self.nebular_func, self.nebular_scales, self.nebular_wls, self.nebular_ions, self.nebular_ages  = self.load_nebular_tables(self.model_wl,self.model_resolution, emimetal,emimodel)


        # LOADS DUST EMISSION SPECTRA
        # This can be moved above for computing dust_wl only once?
        self.dustem_func, self.dustem_alpha = self.load_dust_emission_models(self.model_wl,dustemimodel)


    # methods

    load_nebular_tables = load_nebular_tables
    load_dust_emission_models = load_dust_emission_models
    model_grid_interpolator = model_grid_interpolator
    dust_attenuation_curve = dust_attenuation_curve



    def synthetic_spectrum(
        self,
        burst_mass_fraction,
        fesc,
        ion_gas,
        age_gas,
        av,
        av_ext,
        alpha,
        log_m_star,
        redshift,
        luminosity_distance,
        vel_sys,
        sigma_vel,
        sigma_gas,
                 
        likelihood_call=False,
        bands=[],
        trans_arrays=None,
        trans_mask=None,
        pivot_wls=None,

        obs_spec=None,
        polydeg=None,
        spec_noise_scale=0,

        multi_component=False,     

        **kwargs,
    ):

        # PHOTOMETRY

        # --- Stellar population spectra ---

        Fym = self.interp_young_flux(**kwargs)
        Fom = self.interp_old_flux(**kwargs)
        eta_m = self.interp_current_stellar_mass(**kwargs)

        Fyb = self.interp_burst_young_flux(BURSTAGE=kwargs["TRUNCAGE"])
        Fob = self.interp_burst_old_flux(BURSTAGE=kwargs["TRUNCAGE"])
        eta_b = self.interp_burst_current_stellar_mass(BURSTAGE=kwargs["TRUNCAGE"])

        a = 1.0 / (eta_m + burst_mass_fraction * eta_b)   # makes total current mass = 1 Msun

        young_stellar = a * (Fym + burst_mass_fraction * Fyb)
        old_stellar   = a * (Fom + burst_mass_fraction * Fob)


        # --- Ionizing photons and nebular emission ---
        index_lyman = np.searchsorted(self.model_wl, 912, side="left")
        lycont_wls = np.r_[self.model_wl[:index_lyman], np.array([912])]

        # Young
        lycont_spec = np.interp(lycont_wls, self.model_wl, young_stellar)
        nlyman_young = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        # Old
        lycont_spec = np.interp(lycont_wls, self.model_wl, old_stellar)
        nlyman_old = np.trapz(lycont_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)



        # Remove escaped photons
        if fesc >= 0:
            young_stellar[:index_lyman] *= fesc
            old_stellar[:index_lyman]  *= fesc


        # Nebular emission
        nebular_lines = self.nebular_func(ion_gas, age_gas)
        young_nebular = nebular_lines * nlyman_young 
        old_nebular   = nebular_lines * nlyman_old   

        stellar_nebular = young_stellar + old_stellar + young_nebular + old_nebular

        # --- Dust attenuation ---
        att_young_stellar = (10 ** (-(av + av_ext) * self.k_calzetti_array)) * young_stellar
        att_old_stellar   = (10 ** (-av * self.k_calzetti_array)) * old_stellar
        att_stellar      = att_young_stellar + att_old_stellar

        att_young_nebular = (10 ** (-(av + av_ext) * self.k_calzetti_array)) * young_nebular
        att_old_nebular   = (10 ** (-av * self.k_calzetti_array)) * old_nebular
        att_nebular       = att_young_nebular + att_old_nebular

        att_stellar_nebular = att_stellar + att_nebular

        # --- Thermal dust re-emission ---
        lbol_init = np.trapz(stellar_nebular, self.model_wl)
        lbol_att  = np.trapz(att_stellar_nebular, self.model_wl)
        dust_bol  = lbol_init - lbol_att

        tdust_spec = self.dustem_func(alpha)

        # Remove stellar leakage in Dale templates
        mask_pixels = (self.model_wl >= 2.5e4) & (self.model_wl <= 3e4)
        scale = np.sum(stellar_nebular[mask_pixels] * tdust_spec[mask_pixels]) / \
                np.sum(stellar_nebular[mask_pixels] ** 2)
        tdust_spec -= scale * stellar_nebular
        tdust_spec[(self.model_wl < 2.5e4) | (tdust_spec < 0.)] = 0.

        # Scale dust emission
        norm = np.trapz(tdust_spec, self.model_wl)
        dust_spec = tdust_spec * dust_bol / norm

        # --- Total rest-frame spectrum ---
        final_spec_phot = att_stellar_nebular + dust_spec

        # --- Redshift to observed frame ---
        model_red_wl = self.model_wl * (1 + redshift)
        final_spec_phot /= (1 + redshift)

        # --- Flux scaling to given stellar mass and luminosity distance ---
        fscale = 10**log_m_star / (luminosity_distance * 1e5)**2
        final_spec_phot *= fscale



        # Mass formed 
        Mstar = 10**log_m_star
        Mform_main  = Mstar / (eta_m + burst_mass_fraction * eta_b)
        Mform_burst = burst_mass_fraction * Mform_main              # <-- total stellar mass formed in burst [Msun]
        Mform_total = Mform_main + Mform_burst

        
        # --- Compute synthetic photometric points
        model_phot = []
        for b in bands:
            mask_b = trans_mask[b]
            spec_array = final_spec_phot[mask_b]
            wl_b = model_red_wl[mask_b] 
            if len(spec_array) == 0:
                model_phot.append(np.nan)
                continue
            num_int = np.trapz(trans_arrays[b] * spec_array * wl_b, wl_b)
            den = np.trapz(trans_arrays[b] * wl_b, wl_b)
            if den == 0:
                model_phot.append(np.nan)
                continue
            phot_point = num_int / den
            
            # convert to mJy if needed
            c = 2.99792458e18  # Å/s
            phot_point = phot_point * pivot_wls[b]**2 / c / 1e-26
            model_phot.append(phot_point)

        
        # SPECTROSCOPY
        final_spec = None
        if obs_spec:

            c_kms = 2.99792458e5  # km/s
            
            # --- Kinematics of stellar spectrum: LOSVD + systemic shift ---
            
            wl_obs = self.model_wl * (1.0 + redshift) * (1.0 + vel_sys / c_kms) 
            att_stellar /= (1.0 + redshift)
            
            mask = (wl_obs > obs_spec.wl[0]) & (wl_obs < obs_spec.wl[-1])
            wl_obs = wl_obs[mask]
            att_stellar = att_stellar[mask]
            model_resolution = self.model_resolution[mask]

            sigma_model = c_kms / (model_resolution * 2.355)
            sigma_obs = c_kms / (obs_spec.resolution * 2.355)
            
            sigma_int = max(max(sigma_model),max(sigma_obs))

            # Final broadening to apply to the stellar spectrum
            sigma_tot = np.sqrt(sigma_int**2 + sigma_vel**2 - max(sigma_model)**2)

            # CONVOLVE MODEL SPECTRUM WITH LOSVD GAUSSIAN FUNCTION WITH SIGMA_TOT
            
            # Build a uniform log lambda grid
            loglam = np.log(wl_obs)
            loglam_unif = np.linspace(loglam[0],loglam[-1],len(loglam))

            # interpolate spectrum on new lambda grid
            logspec = np.interp(loglam_unif, loglam, att_stellar)

            # velocity in pixel space
            dloglam = np.median(np.diff(loglam_unif))
            kms_per_pix = 2.99792458e5 * dloglam
            vel_pix = vel_sys / kms_per_pix
            sigma_pix = sigma_tot / kms_per_pix    

            
            # Velocity-space convolution using analytic LOSVD in Fourier domain    
            
            npix = len(loglam_unif)
            npad = 2 * npix  # zero-padding to avoid wrap-around

            # make LOSVD kernel for same npad length
            
            nl = npad // 2 + 1                # size of RFFT array
            w = np.linspace(0, np.pi, nl)     # frequency vector for RFFT

            if sigma_pix == 0:
                losvd_rfft = np.ones_like(w, dtype=complex)
            else:   
                #a = v_pix / sigma_pix 
                # velocity shift already considered when shifting wl_obs
                a = 0
                losvd_rfft = np.exp(1j * a * w - 0.5 * (sigma_pix * w)**2)

            fft_losvd = np.conj(losvd_rfft)

            # FFT of the input spectrum
            fft_spec = np.fft.rfft(logspec, n=npad)

            # multiply in Fourier space (convolution)
            spec_conv = np.fft.irfft(fft_spec * fft_losvd, n=npad)

            # trim to original size
            stellar_conv = spec_conv[:npix]
            

            # Nebular emission lines kinematics
            nebular_scales = self.nebular_scales(age = age_gas, ion = ion_gas)
            nebular_wls = self.nebular_wls * (1.0 + redshift) * (1.0 + vel_sys / c_kms) 
            
            # remove emission lines out of range interest
            mask_nebular = (nebular_wls > obs_spec.wl[0]) & (nebular_wls < obs_spec.wl[-1])
            nebular_wls = nebular_wls[mask_nebular] 
            nebular_scales = nebular_scales[mask_nebular]

            # sigma to apply
            # changes with changing instrumental spectral resolution
            # interpolate instrumental resolution from obs_spec grid onto wl_unif
            R_unif = np.interp(np.exp(loglam_unif), obs_spec.wl, obs_spec.resolution)

            sigma_obs = c_kms / (R_unif * 2.355)
            sigma_gas_tot = np.sqrt(sigma_obs**2 + sigma_gas**2)
            sigma_gas_pix = sigma_gas_tot / kms_per_pix 
                
            # Build pixel edges
            pixels_left_borders = loglam_unif - dloglam/2
            pixels_right_borders = loglam_unif + dloglam/2

            nebular_conv = np.zeros_like(loglam_unif, dtype=float)
            delta_lambda = np.exp(pixels_right_borders) - np.exp(pixels_left_borders)
            sqrt2 = np.sqrt(2.0)

            # use a typical sigma in pixel space to set a window
            sig_med = np.median(sigma_gas_pix)
            win = int(np.ceil(6.0 * sig_med))  # +/- 6σ is plenty

            for w_line, scale in zip(nebular_wls, nebular_scales):
                mu = np.log(w_line)
                j0 = np.searchsorted(loglam_unif, mu)
                j1 = max(0, j0 - win)
                j2 = min(len(loglam_unif), j0 + win + 1)

                d_left  = (pixels_left_borders[j1:j2]  - mu) / dloglam
                d_right = (pixels_right_borders[j1:j2] - mu) / dloglam

                sig = sigma_gas_pix[j1:j2]

                frac = 0.5 * (
                    erf(d_right / (sqrt2 * sig)) -
                    erf(d_left  / (sqrt2 * sig))
                )

                nebular_conv[j1:j2] += scale * frac / delta_lambda[j1:j2]

            """
            # i runs over emission lines
            # j runs over pixels
            for i in range(len(nebular_wls)):
                for j in range(len(loglam_unif)):
                    # The fraction of line i’s total flux landing in pixel j is:
                    
                    d_ij_left = (pixels_left_borders[j] - np.log(nebular_wls[i]))/ dloglam
                    d_ij_right = (pixels_right_borders[j] - np.log(nebular_wls[i]))/ dloglam
                    fraction_flux_ij = 0.5 * (erf(d_ij_right /(np.sqrt(2)*sigma_gas_pix[j])) - erf(d_ij_left /(np.sqrt(2)*sigma_gas_pix[j])))
                    # convert to flux_density
                    delta_lambda_j = np.exp(pixels_right_borders[j]) - np.exp(pixels_left_borders[j])
                    nebular_conv[j] += nebular_scales[i] * fraction_flux_ij / delta_lambda_j
            """

            # restore wavelengths
            wl = np.exp(loglam_unif)
            
            # finally multiply for number of lyman photons and apply reddening
            young_nebular_conv = nebular_conv * nlyman_young *  (10.0 ** (-(av + av_ext) * self.k_calzetti(wl))) 
            old_nebular_conv = nebular_conv * nlyman_old * (10.0 ** (-av * self.k_calzetti(wl)))

            nebular_conv = young_nebular_conv + old_nebular_conv

            edge = int(np.ceil(4.0 * sigma_pix))
            wl = wl[edge:-edge]
            stellar_conv = stellar_conv[edge:-edge]
            nebular_conv = nebular_conv[edge:-edge]

            total_spec = stellar_conv+nebular_conv

            # Re sample on observed spectrum grid
            final_spec = np.interp(obs_spec.wl, wl, total_spec)




            #remove shape differences between spectrum and model
            eps = 1e-30  # choose for your flux units
            
            x = obs_spec.wl
            y = obs_spec.flux / final_spec
            
            # uncertainty on ratio (assuming model_spec exact)
            sigma_y = (obs_spec.flux_err * np.exp(spec_noise_scale)) / np.maximum(np.abs(final_spec), eps)
            
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_y)
            good &= (np.abs(final_spec) > eps) & (sigma_y > 0)

            # Need enough points
            if good.sum() < (polydeg + 1):
                raise ValueError(f"Not enough valid points for deg={polydeg}: {good.sum()} points")
            
            # Weights for polyfit are ~ 1/sigma (not sigma)
            w = 1.0 / sigma_y[good]
            
            coeff = np.polyfit(x[good], y[good], deg=polydeg, w=w)
            cont_poly = np.polyval(coeff, x)

            final_spec *= cont_poly

        if likelihood_call:
            return model_phot, final_spec
  
        
        if not likelihood_call:

            # --- Build FITS header ---
            header = fits.Header()
            header["MODFILE"]    = (self.model_file, "fits files of models")
            header["WUNITS"]  = "A"
            header["FUNITS"]  = "erg/s/cm2/A"
            header["REDSHIFT"] = (redshift, "Cosmological redshift")
            header["AV"]       = (av, "V-band attenuation (old stars)")
            header["AVEXT"]   = (av_ext, "Extra attenuation (young stars)")
            header["BURSTF"]   = (burst_mass_fraction, "Fraction of stellar mass formed during burst")
            header["FESC"]     = (fesc, "Escape fraction of ionizing photons")
            header["IONGAS"]  = (ion_gas, "Ionization parameter")
            header["AGEGAS"]  = (age_gas, "Nebular region age [Myr]")
            header["ALPHA"]    = (alpha, "Dust heating parameter")
            header["LOGMSTAR"]    = (log_m_star, "log10 Stellar mass [Msun]")
            header["MFORMTOT"]  = (np.log10(Mform_total),  "log10 total formed stellar mass [Msun]")
            if burst_mass_fraction>0: header["MFORMBST"]  = (np.log10(Mform_burst),  "log10 formed mass in burst [Msun]")
            header["MFORMMAN"]  = (np.log10(Mform_main),   "log10 formed mass in main SFH [Msun]")
            header["DLMPC"]    = (luminosity_distance, "Luminosity distance [Mpc]")


            # Add kwargs for traceability
            for k, v in kwargs.items():
                key = k[:8].upper()
                try:
                    header[key] = v
                except Exception:
                    header[key] = str(v)

            # --- Create Spectrum object ---
            spec = Spectrum(wl=model_red_wl, resolution=self.model_resolution, flux=final_spec_phot, header=header)

            if obs_spec:
                spec.obs_spec = obs_spec

                header_conv = header.copy()
                header_conv["VELSYS"]   = (vel_sys, "Systemic velocity [km/s]")
                header_conv["SIGMAVEL"] = (sigma_vel, "Stellar Doppler broadening [km/s]")
                header_conv["SIGMAGAS"] = (sigma_gas, "Gas Doppler broadening [km/s]")
            
                header_conv["POLYDEG"] = (polydeg, "Polynomial degree of normalization")
                header_conv["SPECNOI"]   = (spec_noise_scale, "Extra noise factor for spectrum")
            
                spec.model_spec_on_obs_grid = Spectrum(
                    wl=obs_spec.wl,
                    resolution=obs_spec.resolution,
                    flux=final_spec,
                    header=header_conv,
                )

            
            if multi_component:
                # --- Rescale fluxes to observed frame ---
                # Avoid modifying the original arrays if needed later
                young_stellar_nebular_flux = (young_stellar + young_nebular) / (1 + redshift) * fscale
                old_stellar_nebular_flux   = (old_stellar + old_nebular)  / (1 + redshift) * fscale
                att_young_stellar_nebular_flux = (att_young_stellar + att_young_nebular) / (1 + redshift) * fscale
                att_old_stellar_nebular_flux   = (att_old_stellar + att_old_nebular)  / (1 + redshift) * fscale
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



    ###################################################### --- FIT --- #################################################################



    # --- FIT Structure
    # run_fit()               →   sets up the fit, calls make_log_likelihood()
    # make_log_likelihood()   →   precomputes constants, returns log_likelihood()
    # log_likelihood()        →   called thousands of times by the sampler

    # ADD run.spec!!!!

    def run_fit(self,
                spec=None,   # observed spectrum (a spectrum object)
                phot=None,   # observed photometry 
                bands=None,
                treat_as_upper_limits = [],
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
        run.model_file = self.model_file
        run.model_pars = self.model_pars
        run.spectral_range = spectral_range
        run.nlive = nlive
        run.dlogz = dlogz
        run.polydeg = polydeg
        run.custom_priors = custom_priors
        run.fix_pars = dict(fix_pars) 

        run.bands = []

        run.spec_crop = None # initialized None
        

        # --- bands selection ---
        if phot:
            if bands:
                for b in bands:
                    if b not in map_filter_names.keys():
                        raise ValueError(f'Unrecognized filter: {b}. Abort')
                run.bands = list(bands)
            else:
                run.bands = [b for b in phot.data.keys() if b in map_filter_names.keys()]

            print(f"Using the following photometric filters: {', '.join(run.bands)}")
            
            run.treat_as_upper_limits = treat_as_upper_limits
            print(f"Setting as upper limits: {', '.join(run.bands)}")

            run.phot = phot
            

        # --- preprocess observed spectrum once (done here) ---
        if spec:
            run.spec = spec
            run.spec_crop = self._preprocess_observed_spectrum(spec, run.spectral_range, atmospheric_lines)


        # ------- build lists of free parameters (separate model and global) -------
        run.free_model_pars = [p for p in self.model_pars if p not in run.fix_pars]

        global_pars = ["burst_mass_fraction","fesc", "ion_gas", "age_gas", "av", "av_ext",
                    "alpha", "log_m_star", "vel_sys", "sigma_vel", "sigma_gas", "luminosity_distance", "redshift", "spec_noise_scale"]

        if not spec:
            print("No spectral data. Neglecting kinematics")
            run.fix_pars["vel_sys"] = 0
            run.fix_pars["sigma_vel"] = 0
            run.fix_pars["sigma_gas"] = 0
            run.fix_pars["spec_noise_scale"] = 0

        if not phot:
            print("No photometric data. Neglecting scaling")
            run.fix_pars["log_m_star"] = 1
            run.fix_pars["luminosity_distance"] = 1
            run.fix_pars["alpha"] = 1

        run.free_pars = run.free_model_pars + [p for p in global_pars if p not in run.fix_pars]

        # Printing
        for p in run.fix_pars.keys():
            print(f'Fixed parameter: {p} = {run.fix_pars[p]}')
        for p in run.free_pars: 
            print('Fitting: ',p)



        # --- build likelihood and prior ---
        run.log_likelihood = self.make_log_likelihood(run, spec, phot, run.free_pars) #self.
        run.prior_transform = self.make_prior_transform(run, run.custom_priors)

        print("Initializing live points")
        run.sampler = NestedSampler(run.log_likelihood, run.prior_transform, ndim=len(run.free_pars), nlive=nlive)
        
        # --- run sampler ---
        run.sampler.run_nested(dlogz=dlogz, print_progress=True)
        run.result = run.sampler.results

        return run


    def make_log_likelihood(self, run, spec, phot, free_pars,):

        if phot: 

            obs_phot = np.array([run.phot.data[b][0] for b in run.bands])
            obs_errors = np.array([run.phot.data[b][1] for b in run.bands])
            phot_units = phot.header["UNITS"]

            # Upper limits

            bad = [b for b in run.treat_as_upper_limits if b not in map_filter_names]
            if bad:
                raise ValueError(f'Unrecognized filter(s): {bad}. Abort')
            
            mask_ul = np.array([b in run.treat_as_upper_limits for b in run.bands], dtype=bool)
            
            # for upper limits: flux -> 0, error -> original flux (your logic)
            obs_errors[mask_ul] = obs_phot[mask_ul]
            obs_phot[mask_ul] = 0.0

            # Set filter trasmission curves
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


        # --- closure used by the sampler ---
        def log_likelihood(pars):
            
            spec_lhood = 0.0
            phot_lhood = 0.0
            
            idx = 0

            # ---- MODEL PARAMETERS: only read the *free* model params from pars ----
            # map vector → named parameters
            p = dict(zip(run.free_pars, pars))
            # inject fixed parameters
            p.update(run.fix_pars)
            model_kwargs = {k: p[k] for k in self.model_pars}

            # unpack named params for readability
            burst_mass_fraction = p["burst_mass_fraction"]
            fesc = p["fesc"]
            ion_gas = p["ion_gas"]
            age_gas = p["age_gas"]
            av = p["av"]
            av_ext = p["av_ext"]
            alpha = p["alpha"]
            log_m_star = p["log_m_star"]
            vel_sys = p["vel_sys"]
            sigma_vel = p["sigma_vel"]
            sigma_gas = p["sigma_gas"]
            luminosity_distance = p["luminosity_distance"]
            redshift = p["redshift"]
            spec_noise_scale = p["spec_noise_scale"]


            model_phot, final_spec = self.synthetic_spectrum(**model_kwargs,
                        burst_mass_fraction=burst_mass_fraction,
                        fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                        av=av, av_ext=av_ext, alpha=alpha,
                        log_m_star=log_m_star,
                        luminosity_distance=luminosity_distance,
                        redshift=redshift,
                        vel_sys=vel_sys,sigma_vel=sigma_vel,sigma_gas=sigma_gas,
                        likelihood_call=True,
                        bands=run.bands,trans_arrays=trans_arrays,trans_mask=trans_mask,pivot_wls=pivot_wls,
                        obs_spec=run.spec_crop,polydeg=run.polydeg,spec_noise_scale=spec_noise_scale,
                        )


            # ----------------- Photometric likelihood -----------------
            if model_phot:

                if not np.all(np.isfinite(model_phot)):
                    return -1e100
                
                for i in range(len(obs_phot)):
                    if obs_phot[i] > 0:
                        var = obs_errors[i]**2
                        residual = (obs_phot[i] - model_phot[i]) / obs_errors[i]
                        phot_lhood += -0.5 * (residual**2 + np.log(2*np.pi*var)) 
                    else:
                        terf = 0.5 * (1 + erf((obs_phot[i] - model_phot[i]) / 
                                            (np.sqrt(2.) * obs_errors[i])))
                        if terf <= 0:
                            return -1e100
                        phot_lhood += np.log(terf)


            # ----------------- Spectral likelihood (features only) -------------
            if final_spec is not None:

                # Likelihood mask 
                mask_like = (
                    np.isfinite(final_spec)
                    & np.isfinite(run.spec_crop.flux)
                    & (run.spec_crop.flux_err > 0)
                )
                flux_obs = run.spec_crop.flux[mask_like]
                flux_err = run.spec_crop.flux_err[mask_like]
                final_spec = final_spec[mask_like]


                # Free nuisance parameter: This is not arbitrary — it is equivalent to marginalizing over unknown variance.
                # Spectroscopy → many points (hundreds–thousands), often correlated, continuum-dominated
                # Photometry → few points (∼5–30), independent, broadband
                                
                flux_err_corr = flux_err * np.exp(spec_noise_scale)

                residuals = (flux_obs -final_spec) / (flux_err_corr)
                
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
            'burst_mass_fraction':    {'type': 'uniform', 'low': 0, 'high': 1.0},
            'fesc':    {'type': 'uniform', 'low': -1.0, 'high': 1.0},
            'ion_gas': {'type': 'uniform', 'low': self.nebular_ions[0], 'high': self.nebular_ions[-1]},
            'age_gas': {'type': 'uniform', 'low': self.nebular_ages[0], 'high': self.nebular_ages[-1] + 1},
            'av':      {'type': 'uniform', 'low': 0.0, 'high': 1.0},

            # MATCHES sps_spec_fitter: Av_ext ~ Normal(coef * Av, sigma)
            'av_ext':  {'type': 'gaussian_conditional', 'depends_on': 'av', 'coef': 1.17, 'sigma': 0.01,
                        'low': 0.0, 'high': 1.0},

            'alpha':   {'type': 'uniform', 'low': self.dustem_alpha[0], 'high': self.dustem_alpha[-1]},
            'log_m_star':  {'type': 'uniform', 'low': 4.0, 'high': 12.0},
            'vel_sys': {'type': 'uniform', 'low': -500.0, 'high': 500.0},
            'sigma_vel': {'type': 'uniform', 'low': 1.0, 'high': 200.0},
            'sigma_gas': {'type': 'uniform', 'low': 1.0, 'high': 500.0},
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
            elif pr['type'] == 'gaussian_conditional':
                lo = pr.get('low', None)
                hi = pr.get('high', None)
                clip = ""
                if lo is not None or hi is not None:
                    clip = f", truncated to [{lo}, {hi}]"
                print(f"  - {p}: Gaussian(mean={pr['coef']}*{pr['depends_on']}, sigma={pr['sigma']}){clip}")
            else:
                print(f"  - {p}: {pr}")

        # transform from unit cube to physical space
        def prior_transform(u):
            x = np.zeros_like(u)
            done = np.zeros(len(run.free_pars), dtype=bool)

            name_to_idx = {name: i for i, name in enumerate(run.free_pars)}

            # keep looping until everything is resolved (handles dependencies)
            progress = True
            while not np.all(done) and progress:
                progress = False

                for i, name in enumerate(run.free_pars):
                    if done[i]:
                        continue

                    prior = priors[name]

                    if prior['type'] == 'uniform':
                        x[i] = prior['low'] + u[i] * (prior['high'] - prior['low'])
                        done[i] = True
                        progress = True

                    elif prior['type'] == 'gaussian':
                        x[i] = prior['mean'] + prior['sigma'] * np.sqrt(2) * erfinv(2 * u[i] - 1)
                        done[i] = True
                        progress = True

                    elif prior['type'] == 'gaussian_conditional':
                        dep = prior['depends_on']
                        if dep not in name_to_idx:
                            raise ValueError(f"{name}: depends_on='{dep}' not in run.free_pars")

                        j = name_to_idx[dep]
                        if not done[j]:
                            continue  # wait until dependency is sampled

                        mu = prior['coef'] * x[j]
                        sig = prior['sigma']
                        val = mu + sig * np.sqrt(2) * erfinv(2 * u[i] - 1)

                        # optional truncation to match your old bounds behavior
                        lo = prior.get('low', None)
                        hi = prior.get('high', None)
                        if lo is not None:
                            val = max(lo, val)
                        if hi is not None:
                            val = min(hi, val)

                        x[i] = val
                        done[i] = True
                        progress = True

                    else:
                        raise ValueError(f"Unsupported prior type {prior['type']} for {name}")

            if not np.all(done):
                unresolved = [run.free_pars[i] for i in range(len(run.free_pars)) if not done[i]]
                raise ValueError(f"Could not resolve dependent priors for: {unresolved}")

            return x

        return prior_transform



    """
    ranges = spectral_range

    if isinstance(ranges, np.ndarray) and ranges.ndim == 1 and ranges.size == 2:
        ranges = [tuple(ranges.tolist())]
    elif isinstance(ranges, list) and len(ranges) == 2 and all(np.isscalar(x) for x in ranges):
        ranges = [tuple(ranges)]
    elif isinstance(ranges, tuple) and len(ranges) == 2 and all(np.isscalar(x) for x in ranges):
        ranges = [ranges]
    
    """


    # atmospheric mask not depending on resolution! CHANGE IN FUTURE
    from .pangal_utils import airtovac, vactoair

    def _preprocess_observed_spectrum(self, spec, spectral_range, atmospheric_lines):
        """
        Preprocess observed spectrum:
        - Accepts one range [wl1, wl2] or multiple ranges [[wl1, wl2], [wl3, wl4], ...]
        - Zeroes out flux and err outside allowed ranges
        - Removes ±5 Å around each atmospheric line
        """

        wl = spec.wl.copy()
        flux = spec.flux.copy()
        err = spec.flux_err.copy()
        resolution = spec.resolution.copy()

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
        # 3. Remove atmospheric lines ±10 Å
        # ------------------------------
        mask_atm = np.ones_like(wl, dtype=bool)
        for line in atmospheric_lines:
            lam = atmospheric_lines[line]

            if spec.header['WAVESYS'] == 'VACUUM  ':
                lam = self.airtovac(lam)

            bad = (wl >= lam - 10) & (wl <= lam + 10)
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

        # ------------------------------
        # 7. Trim spectrum to first/last valid pixel
        # ------------------------------
        valid = np.isfinite(flux) & (flux != 0)

        if np.any(valid):
            i0 = np.argmax(valid)                  # first True
            i1 = len(valid) - np.argmax(valid[::-1])  # last True + 1

            wl = wl[i0:i1]
            flux = flux[i0:i1]
            err = err[i0:i1]
            resolution = resolution[i0:i1]
        else:
            raise ValueError("No valid spectral pixels remain after preprocessing.")

        return Spectrum(wl=wl, flux=flux, flux_err=err, resolution=resolution, header=spec.header)







    def fit_diagnostic(
            self,
            params,
            spec=None,
            spectral_range=None,
            polydeg=7,
            phot=None,
            bands=None,
            treat_as_upper_limits=[],

            # plotting window
            winf_spec=None,
            wsup_spec=None,
        
            winf_phot=1e2,
            wsup_phot=1e7,
            ymin_phot=None,
            ymax_phot=None,

            # aestethics
            figsize=(10,10),
            color='crimson',
            show_errorbars=True,
                
            # optionals
            redshift=None,
            show_top_spectral_lines=False,
            show_all_spectral_lines=False,
            show_atmospheric_lines=False,
                
            spec_legend_pars=None,
            observed_spec_label=None,
            spec_legend_loc="upper right",
            phot_legend_loc="upper right",
            spec_legend_fontsize=14,
            title_fontsize=14,
            label_fontsize=14,
            spec_legend_title=None,
            phot_legend_title=None,
            title_color='black',
            title_fontweight='normal',
            
        ):
        

        if phot:
            
            if bands:
                for b in bands:
                    if b not in map_filter_names.keys():
                        raise ValueError(f'Unrecognized filter: {b}. Abort')
                bands = list(bands)
            else:
                bands = [b for b in phot.data.keys() if b in map_filter_names.keys()]

        
        if spec:

            spec_crop = self._preprocess_observed_spectrum(spec,spectral_range,atmospheric_lines)


        
        # keys that match the explicit signature of synthetic_spectrum
        spec_keys = {
            "burst_mass_fraction","fesc","ion_gas","age_gas","av","av_ext","alpha","log_m_star","redshift",
            "luminosity_distance","vel_sys","sigma_vel","sigma_gas","spec_noise_scale"
        }
            
        spec_args  = {k: params[k] for k in spec_keys if k in params}
        model_args = {k: v for k, v in params.items() if k not in spec_keys}  # e.g. TRUNCAGE/TRUNCTAU

        
        # Set filter trasmission curves
        trans_mask = {}
        trans_arrays = {}
        pivot_wls = {}
        for b in bands:
            F = Filter(b)
            lmin, lmax = F.wavelength_range
            mask_b = (self.model_wl >= lmin) & (self.model_wl <= lmax)
            trans_mask[b] = mask_b
            trans_arrays[b] = F.transmission_curve(self.model_wl[mask_b])
            pivot_wls[b] = F.pivot_wavelength

        
        synth_spec = self.synthetic_spectrum(
            **model_args,
            **spec_args,
            likelihood_call=False,
            bands=bands,
            trans_arrays=trans_arrays,
            trans_mask=trans_mask,
            pivot_wls=pivot_wls,
            obs_spec=spec_crop,
            polydeg=polydeg,
        )



        
        if phot:
        
            phot_fluxes = np.array([phot.data[b][0] for b in bands])
            phot_errors = np.array([phot.data[b][1] for b in bands]) 
            
            # apply upper limits locally
            bad = [b for b in treat_as_upper_limits if b not in map_filter_names]
            if bad:
                raise ValueError(f'Unrecognized filter(s): {bad}. Abort')
            
            mask_ul = np.array([b in treat_as_upper_limits for b in bands], dtype=bool)
            
            # for upper limits: flux -> 0, error -> original flux (your logic)
            phot_errors[mask_ul] = phot_fluxes[mask_ul]
            phot_fluxes[mask_ul] = 0.0
            
            # Just for plot
            phot_loc = copy.deepcopy(phot)
            for b in treat_as_upper_limits:
                appo = phot_loc.data[b][0] 
                phot_loc.data[b] = (0,appo)
            
            phot_units = phot.header["UNITS"]

            
            model_phot = synth_spec.get_phot(bands=bands,method='trapz',units='mJy' ) 
            model_phot_array = np.array([model_phot.data[b][0] for b in bands])
            
            if not np.all(np.isfinite(model_phot_array)):
                raise ValueError(f"Model spectrum {j:.0f} gives invalid photometry")
                            
            chi2 = 0.0
            logL_phot = 0.0
            
            for i in range(len(bands)):
            
                mod = model_phot_array[i]
                obs = phot_fluxes[i]
                err = phot_errors[i]
                var = err**2
        
                if obs > 0:
                    # Detection: Gaussian likelihood
                    resid = (obs - mod) / err
        
                    chi2 += resid**2
                    logL_phot += -0.5 * (resid**2 + np.log(2 * np.pi * var))
        
                    #residuals_per_band_dict[nice_filter_names[bands[i]]] =  resid**2
                    
                else:
                    # Upper limit: one-sided Gaussian
                    arg = (obs - mod) / (np.sqrt(2.0) * err)
                    cdf = 0.5 * (1 + erf(arg))
                    #if cdf <= 0:
                    #    logL = -1e100
                    #    break
                    logL_phot += np.log(cdf)
                    chi2 += - 2*np.log(cdf)

            
            
            #print('Photometric fitting:')
            #print('chi1: ',chi2,'  chi2_reduced: ',chi2/len(bands),'  logL: ',logL_phot)

            title = f'$A_v =${params["av"]:.2f},  ${{A_v}}_\\text{{, extra}} =${params["av_ext"]:.2f},  $\\alpha_\\text{{Dale}} =${params["alpha"]:.2f},  $\log M_* = ${params["log_m_star"]:.2f},  $d_L = ${params["luminosity_distance"]:.0f} Mpc'
            title += f',    $\chi_2 = ${chi2:.2f},     $\log \mathcal{{L}} =${logL_phot:.2f}'
            
            synth_spec.plot(
                per_wavelength=True,
                winf=winf_phot,
                wsup=wsup_phot,
                ymin=ymin_phot,
                ymax=ymax_phot,
        
                figsize=(20,7),
                color=color,
        
                redshift=None,
                show_top_spectral_lines=False,
                show_all_spectral_lines=False,
                show_atmospheric_lines=False,
                show_filters=False,

                label_fontsize=label_fontsize,
                phot=phot_loc,
                synth_phot=bands,
                spec_legend_pars=spec_legend_pars,
                show_phot_legend=True,
                show_spec_legend=True,
                spec_legend_loc=phot_legend_loc,
                spec_legend_title=None,
                spec_legend_fontsize=spec_legend_fontsize,

                title=title,
                title_fontsize=title_fontsize,
                title_fontweight=title_fontweight,
                title_loc='center',
                title_color=title_color,
                )
            
        
        ####################################
        
        if spec:

            wl = synth_spec.model_spec_on_obs_grid.wl
            flux_obs = spec_crop.flux
            spec_noise_scale = params["spec_noise_scale"]
            flux_err_corr = spec_crop.flux_err * np.exp(spec_noise_scale)
            flux_model = synth_spec.model_spec_on_obs_grid.flux

            residuals = (flux_obs - flux_model) / flux_err_corr

            chi2 = np.nansum(residuals**2)
            logL_spec = - 0.5 * chi2  - np.nansum (np.log( np.sqrt(2 * np.pi) * flux_err_corr ))

            #print('Spectral fitting:')
            #print('chi1: ',chi2,'  chi2_reduced: ',chi2/len(wl),'  logL: ',logL_spec)

            
            # Figure with shared x-axis
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                sharex=True,
                figsize=(20, 7),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0}
            )
            
            # -----------------------
            # Top panel: spectrum
            # -----------------------
            ax1.plot(wl, flux_obs, color="k", lw=1, label="Observed")      
            ax1.plot(wl, flux_model, color=color, lw=1.2, label="Model")  
            
            # Optional error band (nice but optional)
            ax1.fill_between(
                wl,
                flux_obs - flux_err_corr,
                flux_obs + flux_err_corr,
                color="k",
                alpha=0.2,
                lw=0,
            )

            ax1.set_xlim(winf_spec,wsup_spec)
            
            ax1.set_ylabel("Flux (erg/s/cm$^2$/Å)", fontsize=label_fontsize)
            ax1.legend(frameon=False)
            
            # -----------------------
            # Bottom panel: residuals
            # -----------------------
            ax2.axhline(0.0, color="k", lw=0.8, ls="--")
            ax2.plot(wl, residuals, color="steelblue", lw=0.8)
            
            ax2.set_xlabel("Wavelength (Å)", fontsize=label_fontsize)
            ax2.set_ylabel(r"$(F_{\rm obs}-F_{\rm model})/\sigma$", fontsize=label_fontsize)
            
            # Optional: symmetric limits for clarity
            rmax = np.nanpercentile(np.abs(residuals), 99)
            ax1.set_xlim(winf_spec,wsup_spec)
            ax2.set_ylim(-rmax, rmax)

            #title = f'$A_v =${params["av"]:.2f},  ${{A_v}}_\\text{{, extra}} =${params["av_ext"]:.2f},  '
            #title = f'$v_\\text{{sys}}=${params["vel_sys"]:.1f} km/s, $\sigma_v =${params["sigma_vel"]:.1f} km/s,  $\sigma_\\text{{gas}} =${params["sigma_gas"]:.1f}'
            title = f'$f_\\text{{esc}} =${params["fesc"]:.1f},  $\mathcal{{U}}_\\text{{ion}} =${params["ion_gas"]:.1f},  $\\text{{Age}}_\\text{{gas}} =${params["age_gas"]:.1f} Myr,  '
            title += f'    $\chi_2 = ${chi2:.2f},     $\log \mathcal{{L}} =${logL_spec:.2f}'

            ax1.set_title(title,fontsize=title_fontsize,color=title_color,fontweight=title_fontweight,loc='center')
            
            plt.show()




