import numpy as np

from scipy.constants import c, parsec
import astropy.io.fits as fits
import itertools
from copy import deepcopy

import copy, glob, re, os
import tempfile
from tqdm import tqdm

from pcigale.sed import SED
from pcigale.sed import utils
from pcigale.sed_modules import get_module

from ..spectrum import Spectrum
from ..filter import Filter


def make_cigale_models(
    sfh='sfh2exp',
    metallicity=0.02,
    imf=1,
    stellar_library='cb19',
    #include_nebular=True,
    #include_dust=True,
    filename=None,
    **sfh_params
):
    """
    Generate one or multiple CIGALE models.
    Parameters in sfh_params can be scalars or lists.
    """

    keys, values = zip(*sfh_params.items()) if sfh_params else ([], [])
    grid = list(itertools.product(*[v if isinstance(v, (list, tuple, np.ndarray)) else [v] for v in values]))

    models = []
    for combo in tqdm(grid):
        params = dict(zip(keys, combo))

        # --- Inner block: your single-model builder ---
        sfh_module = _build_sfh_module(sfh, params)
        ssp_module = get_module(stellar_library, imf=imf, metallicity=metallicity)

        sed = SED()
        sfh_module.process(sed)
        ssp_module.process(sed)

        wl = sed.wavelength_grid * 10   # Å
        resolution = _compute_resolution(wl, stellar_library)

        # Flux conversion
        young_lum = sed.luminosities['stellar.young']
        old_lum = sed.luminosities['stellar.old']
        parsec = 3.0857e18
        young_flux = (young_lum / (4*np.pi * (10*parsec)**2)) * 100
        old_flux = (old_lum / (4*np.pi * (10*parsec)**2)) * 100

        # Spectrum object
        spec = Spectrum(wl=wl, resolution=resolution)
        spec.flux_young = young_flux
        spec.flux_old = old_flux
        spec.flux = young_flux + old_flux

        # Metadata
        spec.header['WUNITS'] = 'A'
        spec.header['FUNITS'] = 'erg/s/cm2/A'
        spec.header['SFH'] = sfh
        spec.header['STARLIB'] = stellar_library
        for key, value in params.items():
            spec.header[key] = value

        models.append(spec)

    # Save to FITS 
    if filename is not None:
        save_to_fits(filename,models)

    return models if len(models) > 1 else models[0]



def _build_sfh_module(sfh, params):
    if sfh == "sfh2exp":
        required = ["age", "tau_main"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"sfh2exp requires {required}, missing {missing}")
        return get_module(
            'sfh2exp',
            age=params['age'],
            tau_main=params['tau_main'],
            tau_burst=params.get('tau_burst', 10),
            f_burst=params.get('f_burst', 0.0),
        )
    elif sfh == "sfhdelayed":
        required = ["age_main", "tau_main"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"sfhdelayed requires {required}, missing {missing}")
        return get_module(
            'sfhdelayed',
            age_main=params["age_main"],
            tau_main=params["tau_main"],
            f_burst=params.get("f_burst", 0.0),
            age_burst=params.get("age_burst", 50),
            normalise=params.get("normalise", True),
        )
    elif sfh == "sfhperiodic":
        required = ["age", "period", "burst_age", "burst_fwhm"]
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"sfhperiodic requires {required}, missing {missing}")
        return get_module(
            'sfhperiodic',
            age=params["age"],
            period=params["period"],
            burst_age=params["burst_age"],
            burst_fwhm=params["burst_fwhm"],
            normalise=params.get("normalise", True),
        )
    else:
        raise ValueError(f"Unsupported SFH '{sfh}'")

def _compute_resolution(wl, stellar_library):
    if stellar_library == 'bc03':
        delta_lambda = wl / 300
        delta_lambda[(wl >= 3200) & (wl <= 9500)] = 3.0
    elif stellar_library == 'cb19':
        delta_lambda = np.full_like(wl, 2.0)
        delta_lambda[(wl >= 912) & (wl <= 3540)] = 1.0
        delta_lambda[(wl >= 3540) & (wl <= 7350)] = 2.5
        delta_lambda[(wl >= 7350) & (wl <= 9400)] = 1.0
    else:
        print(f"WARNING: Unknown library {stellar_library}, using bc03 resolution.")
        delta_lambda = wl / 300
        delta_lambda[(wl >= 3200) & (wl <= 9500)] = 3.0
    return wl / delta_lambda



def save_to_fits(filename,models):
    
    # Primary HDU: wavelength array as data
    wl_data = models[0].wl.astype(np.float32)
    res_data = models[0].resolution.astype(np.float32)
    primary_hdu = fits.PrimaryHDU(data=[wl_data,res_data])
    primary_hdu.header["NMODEL"] = len(models)  # global metadata
    
    # Each model in its own extension
    model_hdus = []
    for i, spec in enumerate(models):
        cols = [
            fits.Column(name="FLUX",       array=spec.flux,       format="E"),
            fits.Column(name="FLUX_YOUNG", array=spec.flux_young, format="E"),
            fits.Column(name="FLUX_OLD",   array=spec.flux_old,   format="E"),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, name=f"MODEL_{i}")

        # Save model-specific metadata
        for k, v in spec.header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass  # ignore keywords that FITS can't store

        model_hdus.append(hdu)

    hdul = fits.HDUList([primary_hdu] + model_hdus)
    hdul.writeto(filename, overwrite=True)



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

            spec.flux_young = data["FLUX_YOUNG"]
            spec.flux_old   = data["FLUX_OLD"]
            
            # Keep the model's metadata
            spec.header.update(header)

            models.append(spec)

    return models