import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import glob
import itertools
from tqdm import tqdm
import sys, os
sys.path.append(os.path.expanduser('~/Desktop/Pangal'))

#from pangal.pfitter import PFitter
from ..photometry_table import PhotometryTable

from ..spectrum import Spectrum
#from pangal.pfitter import PFitter
#from pangal.run import Run

import warnings
warnings.simplefilter("ignore")

import fsps
from astropy.io import fits



# https://dfm.io/python-fsps/current/stellarpop_api

def make_ssp_with_fsps(
    zmet=1,          # metallicity index
    imf_type=1,      # Chabrier

    ages=[10],
    filename=None,
):
    """
    Generate one or multiple FSPS models.
    ages can be scalars or lists.
    """

    if np.isscalar(ages):
        ages = [ages]
    
    sp = fsps.StellarPopulation(
        
        zmet=1,          # metallicity index
        imf_type=1,      # Chabrier
        sfh=0,            # SSP

        zcontinuous = 0,
        add_agb_dust_model = False,
        add_dust_emission = False,
        add_igm_absorption = False,
        add_neb_emission = False,
        add_neb_continuum =False,
        add_stellar_remnants= True,
        nebemlineinspec= False,
        cloudy_dust =False,
        agb_dust = False,
        tpagb_norm_type =0, # Padova 2007 isochrones
        
    )




    models = []
    for age in tqdm(ages):
        
        wl, flux = sp.get_spectrum(tage=age/1000,peraa=True)  # FSPS uses Gyr
        # wl in A
        # flux in L_sun / Å   per solar mass formed. no distance scaling in FSPS. 
        # https://dfm.io/python-fsps/current/stellarpop_api/?utm_source=chatgpt.com#fsps.StellarPopulation.get_spectrum
        L_sun = 3.828e33 # erg/s
        flux *= L_sun

        
        resolution = _compute_resolution(wl)

        # Spectrum object
        spec = Spectrum(wl=wl,resolution=resolution,flux=flux)

        # Metadata
        spec.header['WUNITS'] = 'A'
        spec.header['FUNITS'] = 'erg/s/A'
        spec.header['AGE'] = age

        models.append(spec)

    # Save to FITS 
    if filename is not None:
        save_to_fits(filename,models)

    return models if len(models) > 1 else models[0]

    
def _compute_resolution(wl):
    delta_lambda = np.zeros_like(wl, dtype=float)

    # --------------------------------------------------
    # UV: BaSeL
    # --------------------------------------------------
    uv = wl < 3500
    R_uv = 300.0                  # representative value (200–500)
    delta_lambda[uv] = wl[uv] / R_uv

    # --------------------------------------------------
    # Optical: MILES
    # --------------------------------------------------
    optical = (wl >= 3500) & (wl <= 7500)
    delta_lambda[optical] = 2.5   # Å, nearly constant FWHM

    # --------------------------------------------------
    # NIR: BaSeL
    # --------------------------------------------------
    nir = wl > 7500
    R_nir = 400.0                 # representative value (300–500)
    delta_lambda[nir] = wl[nir] / R_nir

    return wl / delta_lambda

def save_to_fits(filename, models):

    wl_data = models[0].wl.astype(np.float64)
    res_data = models[0].resolution.astype(np.float64)
    primary_hdu = fits.PrimaryHDU(data=[wl_data, res_data])

    model_hdus = []
    for i, spec in enumerate(models):
        cols = [
            fits.Column(
                name="FLUX",
                array=spec.flux.astype(np.float64),
                format="D"   # <-- FIX
            ),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, name=f"MODEL_{i}")

        for k, v in spec.header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass

        model_hdus.append(hdu)

    fits.HDUList([primary_hdu] + model_hdus).writeto(filename, overwrite=True)


def load_spectrum_models_from_fits(filename):
    with fits.open(filename) as hdul:
        # Wavelength array from primary HDU
        wl,resolution = hdul[0].data
        nmodels = len(hdul) - 1

        models = []
        for i in range(1, len(hdul)):
            data = hdul[i].data
            header = hdul[i].header


            spec = Spectrum(wl=wl,resolution=resolution,flux=data["FLUX"])
            
            # Keep the model's metadata
            spec.header.update(header)

            models.append(spec)

    return models
