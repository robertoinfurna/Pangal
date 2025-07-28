import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union, List

from pcigale.sed import SED
from pcigale.sed import utils
from pcigale.sed_modules import get_module

from ..methods.base_classes import Spectrum

from ..methods.utils import dust_attenuation_curve, airtovac, vactoair

def cigale_sps_sed(
    sfh='sfh2exp',
    metallicity=0.02,
    imf=1,
    stellar_library='cb19',
    include_nebular=True,
    include_dust=True,
    **sfh_params
):
    
    # Choose SFH
    if sfh == 'sfh2exp':
        sfh_module = get_module(
            sfh,
            age=sfh_params.get('age', 10000),
            tau_main=sfh_params.get('tau_main', 3000),
            normalise=True
        )
    elif sfh == 'sfhfromfile':
        sfh_module = get_module(
            sfh,
            filename=sfh_params['filename'],
            sfr_column=sfh_params.get('sfr_column', 1),
            age=sfh_params.get('age', 10000),
            normalise=True
        )
    else:
        sfh_module = get_module(sfh, **sfh_params)

    #if spectral_library == 'bc03':
        
        #METALLICITY


    # Stellar population synthesis
    ssp_module = get_module(
        stellar_library,
        imf=imf,
        metallicity=metallicity
    )

    sed = SED()

    sfh_module.process(sed)
    ssp_module.process(sed)


    wl = sed.wavelength_grid * 10   # Angstroms

    if stellar_library == 'bc03':
        #some information on the BC03 spectral resolution (FWHM is 3A in the optical)
        resolution = np.copy(wl)/300.
        resolution[(wl >= 3200) & (wl <= 9500)] = 3.
    elif stellar_library == 'cb19':
        resolution = np.zeros_like(wl)+2.
        resolution[(wl >= 912) & (wl <= 3540)] = 1.
        resolution[(wl >= 3540) & (wl <= 7350)] = 2.5
        resolution[(wl >= 7350) & (wl <= 9400)] = 1.
    else: 
        print('WARNING: Spectral library {} not understood, BC03 resolution will be used for spectral fits.'.format(spectral_library))
        resolution = np.copy(wl)/300.
        resolution[(wl >= 3200) & (wl <= 9500)] = 3.


    if stellar_library == 'bc03':   # convert wavelengths from air to vacuum # NOT NECESSARY FOR CB19 ??
        wl = airtovac(wl)


    young_luminosity = sed.luminosities['stellar.young']    # This is the luminosity density from the youn stellar components in W/nm
    old_luminosity = sed.luminosities['stellar.old']

    # Convert to convenient flux units
    parsec = 3.0857e18              # in cm
    young_flux = young_luminosity / ( 4*np.pi * (10*parsec)**2 )     # flux in W/nm at 10 pc
    old_flux = old_luminosity / ( 4*np.pi * (10*parsec)**2 )
    young_flux *= 100                     # convert in cgs   (erg/cm2/s/A) 1000 for w/m2 to erg/cm2/s and 0.1 for nm to A
    old_flux *= 100 

    # Output Spectrum
    spec = Spectrum(wl=wl,resolution=resolution,)  # two components model spectrum
    spec.flux_young = young_flux
    spec.flux_old = old_flux
    spec.flux = young_flux + old_flux

    spec.header['sfh_model'] = sfh
    spec.header['stellar_library'] = stellar_library

           
    return spec


"""

from pangal_methods.sps import cigale_sps_sed
from tqdm import tqdm

ages = [100, 200, 10000]
taus = [10,40, 500]

models = []
# Total iterations is len(ages)*len(taus)
total_iterations = len(ages) * len(taus)

with tqdm(total=total_iterations, desc="Computing models") as pbar:
    for i, age in enumerate(ages):
        for j, tau in enumerate(taus):
            s = cigale_sps_sed(
                sfh='sfh2exp',
                metallicity=0.008,
                imf=1,
                age=age, tau=tau,
                include_nebular=True,
            )
            s.header['age'] = age
            s.header['tau'] = tau

            if i + j > 0:
                s.wl = None           # No need to save all wavelenghts. Spece demanding
                s.resolution = None

            models.append(s)
            pbar.update(1)

"""













