import numpy as np
from bisect import bisect_left
from scipy.special import erf


from ..methods.base_classes import Image, Cube, Region, Point, Contours, Spectrum
from ..methods.filters import map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps







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
    if fesc>0:
        young_stellar_spec[:index_lyman] *= fesc
        old_stellar_spec[:index_lyman] *= fesc

    emm_lines = self.emm_func(ion_gas,age_gas) 

    # to get the full spec normalized line spectra must be multiplied by the un-escaped number of lyman photons from the stellar model
    emm_young = emm_lines * nlyman_young 
    emm_old = emm_lines * nlyman_old

    young_stellar_emm_spec = young_stellar_spec + emm_young
    old_stellar_emm_spec = old_stellar_spec + emm_old
    stellar_emm_spec = young_stellar_emm_spec + old_stellar_emm_spec


    # Dust attenuation

    #attenuate photometry spec, k_cal is precomputed in the rest frame as it should be
    att_young_stellar_emm_spec = (10**(-(av+av_ext)*self.k_cal) * (young_stellar_emm_spec))
    att_old_stellar_emm_spec   = (10**(-av*self.k_cal) * (old_stellar_emm_spec))
    att_stellar_emm_spec  = att_young_stellar_emm_spec + att_old_stellar_emm_spec

    #### THERMAL DUST EMISSION ####
    lbol_init = np.trapz(stellar_emm_spec, self.model_wl)
    lbol_att  = np.trapz(att_stellar_emm_spec, self.model_wl)

    dust_emm = (lbol_init - lbol_att)
    
    # raw interpolated dust emission template. independent of your specific galaxy model. 
    # generic spec for a given α (which controls dust heating intensity).

    tdust_interp = interp1d(self.dh_alpha, self.dh_dustemm, axis=0, bounds_error=True)

    tdust_spec = tdust_interp(alpha)

    # remove stellar component which is included in dale models
    # otherwise it is double-counted.
    mask_pixels = (self.model_wl >= 2.5e4) & (self.model_wl <= 3e4)
    scale = np.sum(stellar_emm_spec[mask_pixels]*tdust_spec[mask_pixels]) / np.sum(stellar_emm_spec[mask_pixels]*stellar_emm_spec[mask_pixels])
    tdust_spec -= scale*stellar_emm_spec
    tdust_spec[(self.model_wl < 2.5e4) | (tdust_spec < 0.)] = 0.

    # scale to the correct total energy to re-emit
    norm = np.trapz(tdust_spec, self.model_wl)     
    dust_spec = tdust_spec * dust_emm / norm

    total_spec = att_stellar_emm_spec + dust_spec

    # move to the observed frame
    self.model_wl *= (1+redshift)
    total_spec /= (1+redshift)

    # this models are scaled to 1 solar mass at 10 pc
    # Rescale
    fscale = 10**m_star / (dl * 1e5)**2

    total_spec *= fscale 

    spec = Spectrum(wl=self.model_wl,resolution=self.model_res,flux=total_spec)

    if multi_component:
        # arrays usefull for plotting
        for s in [young_stellar_emm_spec, old_stellar_emm_spec, att_young_stellar_emm_spec, att_old_stellar_emm_spec, dust_spec]:
            s *= fscale 
        spec.young_stellar_emm = young_stellar_emm_spec
        spec.old_stellar_emm = old_stellar_emm_spec
        spec.att_young_stellar_emmc = att_young_stellar_emm_spec
        spec.att_old_stellar_emm = att_old_stellar_emm_spec
        spec.dust = dust_spec

    return spec
    
    
    
##############################################################################################################################
