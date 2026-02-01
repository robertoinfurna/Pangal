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

    def __init__(self, model_file, model_pars, cosmo=None, dustemimodel='dh02', leitatt=False, uv_bump=False, emimetal=0.017, emimodel='nebular_Byler_mist_2018.lines'):

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
        self.k_calzetti = self.dust_attenuation_curve(leitatt, uv_bump)
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
        multi_component=False,              
        likelihood_call=False,
        bands=None,
        trans_arrays=None,
        trans_mask=None,
        pivot_wls=None,
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
        att_young_stellar_nebular_spec = (10 ** (-(av + av_ext) * self.k_calzetti_array)) * young_stellar_nebular_spec
        att_old_stellar_nebular_spec   = (10 ** (-av * self.k_calzetti_array)) * old_stellar_nebular_spec
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

        # --- Redshift to observed frame ---
        model_red_wl = self.model_wl * (1 + redshift)
        total_spec /= (1 + redshift)

        # --- Flux scaling to given stellar mass and luminosity distance ---
        fscale = 10**m_star / (luminosity_distance * 1e5)**2
        total_spec *= fscale


        if not likelihood_call:

            # --- Build FITS header ---
            header = fits.Header()
            header["MODFILE"]    = (self.model_file, "fits files of models")
            header["WUNITS"]  = "A"
            header["FUNITS"]  = "erg/s/cm2/A"
            header["REDSHIFT"] = (redshift, "Cosmological redshift")
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
        
        # For likelihood
        # Photometric fluxes in mJy by default 
        else: 

            model_phot = []
            for b in bands:
                mask_b = trans_mask[b]
                spec_array = total_spec[mask_b]
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
                
            return np.array(model_phot)

       
    def synthetic_spectrum_for_fitting(
        self,
        obs_spec,
        polydeg,
        spec_noise_scale,
        
        fesc,
        ion_gas,
        age_gas,
        
        av,
        av_ext,
        alpha,
        redshift,
        vel_sys,                         # systemic velocity [km/s]
        sigma_vel,                         # LOS velocity dispersion [km/s]
        sigma_gas,
        
        likelihood_call=False,
        **kwargs,
    ):

        """
        mcspf-matched dust-emission handling:
        (1) Stellar-leakage subtraction in the DH02 templates uses the *continuum-only* spectrum
            (mcspf uses spec_model), not stellar+nebular.
        (2) Dust emission is attenuated by 10^(-Av*k_cal) and iterated to conserve energy,
            as in mcspf reconstruct_phot().
        """

        c_kms = 2.99792458e5  # km/s

        # --- Stellar population spectra (rest-frame, on self.model_wl) ---
        young_stellar = self.interp_young_flux(**kwargs)
        old_stellar   = self.interp_old_flux(**kwargs)
        
        # Number of ionizing photons for later nebular spectrum
        index_lyman = np.searchsorted(self.model_wl, 912, side="left")
        lycont_wls = np.r_[self.model_wl[:index_lyman], np.array([912])]
        lycont_young_spec = np.interp(lycont_wls, self.model_wl, young_stellar)
        nlyman_young = np.trapz(lycont_young_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)
        lycont_old_spec = np.interp(lycont_wls, self.model_wl, old_stellar)
        nlyman_old = np.trapz(lycont_old_spec * lycont_wls, lycont_wls) / 6.626e-27 / 2.998e18 * (1 - fesc)

        # Remove escaped photons
        young_stellar[:index_lyman] *= fesc
        old_stellar[:index_lyman]   *= fesc

        # --- Dust attenuation of stellar spectrum (rest-frame) ---
        att_young = (10.0 ** (-(av + av_ext) * self.k_calzetti_array)) * young_stellar
        att_old   = (10.0 ** (-av * self.k_calzetti_array))           * old_stellar
        att_stellar = att_young + att_old



        
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
        model_spec = np.interp(obs_spec.wl, wl, total_spec)



        #remove shape differences between spectrum and model
        eps = 1e-30  # choose for your flux units
        
        x = obs_spec.wl
        y = obs_spec.flux / model_spec
        
        # uncertainty on ratio (assuming model_spec exact)
        sigma_y = (obs_spec.flux_err * np.exp(spec_noise_scale)) / np.maximum(np.abs(model_spec), eps)
        
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_y)
        good &= (np.abs(model_spec) > eps) & (sigma_y > 0)

        # Need enough points
        if good.sum() < (polydeg + 1):
            raise ValueError(f"Not enough valid points for deg={polydeg}: {good.sum()} points")
        
        # Weights for polyfit are ~ 1/sigma (not sigma)
        w = 1.0 / sigma_y[good]
        
        coeff = np.polyfit(x[good], y[good], deg=polydeg, w=w)
        cont_poly = np.polyval(coeff, x)


        return model_spec*cont_poly





    # Dust attenuation curve based on Calzetti et al. (2000) law
    # wl must be in Angstrom
    # Takes in imput an array of wavelengths and returns the attenuation function

    def dust_attenuation_curve(self, leitatt: bool = False, uv_bump: bool = False):
        """
        return 0.4 * k_cal / R_V
        k_cal is the Calzetti (2000) selective attenuation curve with:
        - optional Leitherer (2002) replacement below 1500 Å
        - optional 2175 Å bump (Drude profile)
        The returned function accepts scalar or array wavelengths in Angstrom.
        """

        R_V = 4.05

        def k_lambda(wl):
            wl = np.asarray(wl, dtype=float)

            k_cal = np.zeros_like(wl, dtype=float)

            # Calzetti+ 2000: piecewise at 6300 Å
            m_long = wl >= 6300.0
            m_short = ~m_long

            # Longer than 6300 Å
            if np.any(m_long):
                k_cal[m_long] = 2.659 * (-1.857 + 1.04 * (1e4 / wl[m_long])) + R_V

            # Shorter than 6300 Å
            if np.any(m_short):
                x = 1e4 / wl[m_short]
                k_cal[m_short] = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + R_V

            # Optional Leitherer (2002) below 1500 Å
            if leitatt:
                m_uv = wl < 1500.0
                if np.any(m_uv):
                    w = wl[m_uv]
                    k_cal[m_uv] = (5.472 + 0.671e4 / w - 9.218e5 / w**2 + 2.620e9 / w**3)

            # Enforce non-negative attenuation pointwise
            k_cal = np.maximum(k_cal, 0.0)

            # Optional 2175 Å bump (Drude-like profile)
            if uv_bump:
                eb = 1.0
                bump = eb * (wl * 350.0) ** 2 / ((wl**2 - 2175.0**2) ** 2 + (wl * 350.0) ** 2)
                k_cal = k_cal + bump

            # Return 0.4 * A(lam)/A(V)
            return 0.4 * k_cal / R_V

        return k_lambda



    ###################################################### --- FIT --- #################################################################



    # --- FIT Structure
    # run_fit()               →   sets up the fit, calls make_log_likelihood()
    # make_log_likelihood()   →   precomputes constants, returns log_likelihood()
    # log_likelihood()        →   called thousands of times by the sampler


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
            run.spec_crop = self._preprocess_observed_spectrum(spec, run.spectral_range, atmospheric_lines)




        # ------- build lists of free parameters (separate model and global) -------
        run.free_model_pars = [p for p in self.model_pars if p not in run.fix_pars]

        global_pars = ["fesc", "ion_gas", "age_gas", "av", "av_ext",
                    "alpha", "m_star", "vel_sys", "sigma_vel", "sigma_gas", "luminosity_distance", "redshift", "spec_noise_scale"]

        if not spec:
            print("No spectral data. Neglecting kinematics")
            run.fix_pars["vel_sys"] = 0
            run.fix_pars["sigma_vel"] = 0
            run.fix_pars["sigma_gas"] = 0
            run.fix_pars["spec_noise_scale"] = 0

        if not phot:
            print("No photometric data. Neglecting scaling")
            run.fix_pars["m_star"] = 1
            run.fix_pars["luminosity_distance"] = 1
            run.fix_pars["alpha"] = 1

        run.free_pars = run.free_model_pars + [p for p in global_pars if p not in run.fix_pars]

        # Printing
        for p in run.fix_pars.keys():
            if p in global_pars: print(f'Fixed parameter: {p} = {run.fix_pars[p]}')
        for p in run.free_pars: 
            print('Fitting: ',p)




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
            phot_units = phot.header["UNITS"]

            # Upper limits

            bad = [b for b in run.treat_as_upper_limits if b not in map_filter_names]
            if bad:
                raise ValueError(f'Unrecognized filter(s): {bad}. Abort')
            
            mask_ul = np.array([b in run.treat_as_upper_limits for b in run.bands], dtype=bool)
            
            # for upper limits: flux -> 0, error -> original flux (your logic)
            phot_errors[mask_ul] = phot_fluxes[mask_ul]
            phot_fluxes[mask_ul] = 0.0

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


            # other very usefull stuff
            self.k_calzetti_array = self.k_calzetti(self.model_wl)
          
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
            fesc = p["fesc"]
            ion_gas = p["ion_gas"]
            age_gas = p["age_gas"]
            av = p["av"]
            av_ext = p["av_ext"]
            alpha = p["alpha"]
            m_star = p["m_star"]
            vel_sys = p["vel_sys"]
            sigma_vel = p["sigma_vel"]
            sigma_gas = p["sigma_gas"]
            luminosity_distance = p["luminosity_distance"]
            redshift = p["redshift"]
            spec_noise_scale = p["spec_noise_scale"]



            # ----------------- Photometric likelihood -----------------
            if phot is not None:

                model_phot_array = self.synthetic_spectrum(**model_kwargs,
                                        fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                                        av=av, av_ext=av_ext, alpha=alpha,
                                        m_star=m_star,
                                        luminosity_distance=luminosity_distance,
                                        redshift=redshift,
                                        likelihood_call=True,
                                        bands=run.bands,trans_arrays=trans_arrays,trans_mask=trans_mask,pivot_wls=pivot_wls)

                if not np.all(np.isfinite(model_phot_array)):
                    return -1e100
                
                for i in range(len(phot_fluxes)):
                    if phot_fluxes[i] > 0:
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

                model_spec_array = self.synthetic_spectrum_for_fitting(**model_kwargs,
                                        fesc=fesc, ion_gas=ion_gas, age_gas=age_gas,
                                        av=av, av_ext=av_ext, alpha=alpha,
                                        m_star=m_star,
                                        vel_sys=vel_sys, sigma_vel = sigma_vel, sigma_gas = sigma_gas,
                                        luminosity_distance=luminosity_distance,redshift=redshift,
                                        obs_spec=run.spec_crop,polydeg=run.polydeg,spec_noise_scale=spec_noise_scale)

                # Likelihood mask 
                mask_like = (
                    np.isfinite(model_spec_array)
                    & np.isfinite(run.spec_crop.flux)
                    & (run.spec_crop.flux_err > 0)
                )
                flux_obs = run.spec_crop.flux[mask_like]
                flux_err = run.spec_crop.flux_err[mask_like]
                model = model_spec_array[mask_like]


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
            'fesc':    {'type': 'uniform', 'low': -1.0, 'high': 1.0},
            'ion_gas': {'type': 'uniform', 'low': self.nebular_ions[0], 'high': self.nebular_ions[-1]},
            'age_gas': {'type': 'uniform', 'low': self.nebular_ages[0], 'high': self.nebular_ages[-1] + 1},
            'av':      {'type': 'uniform', 'low': 0.0, 'high': 1.0},

            # MATCHES sps_spec_fitter: Av_ext ~ Normal(coef * Av, sigma)
            'av_ext':  {'type': 'gaussian_conditional', 'depends_on': 'av', 'coef': 1.17, 'sigma': 0.01,
                        'low': 0.0, 'high': 1.0},

            'alpha':   {'type': 'uniform', 'low': self.dustem_alpha[0], 'high': self.dustem_alpha[-1]},
            'm_star':  {'type': 'uniform', 'low': 4.0, 'high': 12.0},
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




def _adapt_model_spectrum_to_observed_spectrum(wl_obs, flux_obs, err_obs, model_red_wl, synth_flux, spectral_range,polydeg):

    # Interpolate model on observed wl grid WITHOUT extrapolation 
    flux_model = interp1d(model_red_wl, synth_flux,
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



















