import numpy as np
import random
import glob
from copy import deepcopy
from collections import Counter, defaultdict

# Astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u

# Astropy modeling and regions
from astropy.modeling import models, fitting
from regions import (
    PixCoord,
    SkyRegion,
    EllipsePixelRegion,
    EllipseSkyRegion,
    EllipseAnnulusPixelRegion,
    CirclePixelRegion,
)

# Scipy
from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Skimage
from skimage import measure
from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.transform import rotate

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Dataclasses
from dataclasses import dataclass, field

from ..data.spectral_lines import spectral_lines, atmospheric_lines, UBGVRI_filters

from ..base import Image, Spectrum, Cube, Region, Point, Contours



def line_map(self, line, width, continuum_offset_1, continuum_offset_2, z=0, EBmV=0, cubes=['muse']):
    """
    Extracts an emission line map from a datacube, subtracts continuum and applies extinction correction.
    
    Parameters
    ----------
    line : str
        Name of the spectral line (must be in spectral_lines dict).
    width : float
        Width of the integration window around the line (in Angstrom).
    continuum_offset_1, continuum_offset_2 : float
        Offsets for the two continuum sidebands (in Angstrom).
    z : float
        Redshift of the galaxy.
    EBmV : float
        Color excess for extinction correction.
    cubes : str
        List of cubes in self.cubes to use. Default is muse
    """
    if isinstance(cubes, str):
        cubes = [cubes]

    for cube in cubes:
        if cube not in self.cubes.keys():
            raise ValueError(f"Cube '{cube}' not found in self.cubes.")

        selected_cube = self.cubes[cube]
        cube_data = selected_cube.cube
        var = selected_cube.var
        wl = selected_cube.wl
    
        # Rest-frame and redshifted wavelength
        rest_wavelength = spectral_lines[line]
        obs_wavelength = rest_wavelength * (1 + z)
    
        # Convert to channel indices
        ch_inf = np.digitize(obs_wavelength - width / 2, wl) - 1
        ch_sup = np.digitize(obs_wavelength + width / 2, wl) - 1
    
        cont_inf_1 = np.digitize(obs_wavelength - continuum_offset_1, wl) - 1
        cont_sup_1 = np.digitize(obs_wavelength - continuum_offset_1 + width, wl) - 1
        cont_inf_2 = np.digitize(obs_wavelength + continuum_offset_2, wl) - 1
        cont_sup_2 = np.digitize(obs_wavelength + continuum_offset_2 + width, wl) - 1
    
        print(f"{line} flux extracted in the range {(obs_wavelength - width/2):.2f}-{(obs_wavelength + width/2):.2f} Å")
        print(f"Continuum extracted in the ranges "
              f"{(obs_wavelength - continuum_offset_1):.2f}-{(obs_wavelength - continuum_offset_1 + width):.2f} Å and "
              f"{(obs_wavelength + continuum_offset_2):.2f}-{(obs_wavelength + continuum_offset_2 + width):.2f} Å")
    
        # Integrate flux and continuum
        line_cube = np.nansum(cube_data[ch_inf:ch_sup, :, :], axis=0)
        cont_cube_1 = np.nansum(cube_data[cont_inf_1:cont_sup_1, :, :], axis=0)
        cont_cube_2 = np.nansum(cube_data[cont_inf_2:cont_sup_2, :, :], axis=0)
        continuum = (cont_cube_1 + cont_cube_2) / 2
    
        # Optional extinction correction
        # AG = EBmV * MW_extinction(obs_wavelength)
        image_erg_s_cm2 = (line_cube - continuum) * selected_cube.dw * selected_cube.units  
    
        # Safe naming: avoid overwriting
        base_name = f"{cube}_{line}"
        image_name = base_name
        i = 1
        while image_name in self.images:
            image_name = f"{base_name}({i})"
            i += 1
    
        # Save
        self.images[image_name] = Image(
            image=image_erg_s_cm2,
            bunit='erg_s_cm2',
            wcs=selected_cube.wcs,
            dtheta_pix_deg=selected_cube.dtheta_pix_deg,
            area_pix_arcsec2=selected_cube.area_pix_arcsec2,
            pivot_wavelength=obs_wavelength,
            header=selected_cube.header
        )
           


def extract_spectra(self, regions, cube=None):
    """
    Extracts and stores spectra from defined spatial regions within data cube.

    Parameters
    ----------
    regions : list
        List of Region objects, each with a 'project' method 

    cube : Cube or None
        Optional cube object to use instead of self.cubes['muse'].

    Returns
    -------
    list of spectra objects
    """
    # Determine which cube to use
    if cube is not None:
        if not isinstance(self.cubes[cube], Cube):
            raise TypeError("Provided cube must be an instance of Cube class.")
        selected_cube = cube
    else:
        try:
            selected_cube = self.cubes['muse']
        except (KeyError, AttributeError):
            raise ValueError("No muse cube found in self.cubes")

    cube_data = selected_cube.cube
    var = selected_cube.var
    wl = selected_cube.wl
    resolution = selected_cube.resolution
    spectra_list = []

    for region in regions:
        # Generate a mask for the region using the WCS projection
        mask = region.project(cube_data[0, :, :], selected_cube.wcs).astype(bool)
        n_pixels = np.count_nonzero(mask)

        # Sum flux and variance over all pixels in the region
        flux = np.nansum(cube_data[:, mask], axis=1) / n_pixels * selected_cube.units
        variance = np.nansum(var[:, mask], axis=1) / n_pixels**2 * selected_cube.units**2
        error = np.sqrt(variance)
        
                
        spec = Spectrum(wl=wl,resolution=resolution,flux=flux,flux_err=error)

        spectra_list.append(spec)

    return spectra_list





def plot_spectra(
    self,
    spectra,
    z=0.0,
    log=False,
    winf=None,
    wsup=None,
    zoom_on_line=None,
    color='red',
    ecolor='green',
    show_snr=False,
    show_filters=False,
    show_spectral_lines=False,
    show_atmospheric_lines=False
):
    """
    Plot spectra extracted from regions, with optional annotations and features.

    Parameters
    ----------
    spectra : one spectrum or a list of spectrum

    z : float, optional
        Redshift to apply to spectral lines annotations, if shown.

    winf, wsup : float, optional
        Lower and upper wavelength limits for the x-axis.

    center_line : str, optional
        If provided and present in `spectral_lines`, centers the plot around this line.

    color : str, optional
        Color of the data points (default is 'red').

    ecolor : str, optional
        Color of the error bars (default is 'green').

    show_snr : bool, optional
        If True, display signal-to-noise ratio statistics on each subplot.

    show_filters : bool, optional
        If True, overlay filter bands using `UBGVRI_filters`.

    show_spectral_lines : bool, optional
        If True, annotate known spectral lines using `spectral_lines`.

    show_atmospheric_lines : bool, optional
        If True, annotate atmospheric absorption lines using `atmospheric_lines`.
    """
    # If input is a single Spectrum object, convert to list
    if not isinstance(spectra, (list, tuple)):
        spectra = [spectra]
    n_regions = len(spectra)

    fig, axes = plt.subplots(n_regions, 1, figsize=(7, 3 * n_regions), squeeze=False)
    axes = axes.flatten()

    for i, spec in enumerate(spectra):

        wl = spec.wl 
        flux = spec.flux 
        flux_err = spec.flux_err 
 
        ax = axes[i]


        if flux_err is not None:
            
            ax.errorbar(
                wl, flux, yerr=flux_err,
                fmt='o', color=color, ecolor=ecolor,
                elinewidth=1.5, capsize=3, capthick=1.5,
                linestyle='-', lw=0.4
            )
            
        else: # If error is zero the spectrum is a model

            ax.plot(
                wl, flux, color=color, linestyle='-', lw=1)

        
        # Center around specified line if requested
        if zoom_on_line and zoom_on_line in spectral_lines:
            central_wavelength = spectral_lines[zoom_on_line] * (1 + z)
            winf = central_wavelength - 150
            wsup = central_wavelength + 150

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')


        if not winf or not wsup:
            winf,wsup = wl[0],wl[-1]
        ax.set_xlim(winf, wsup)

        # Dynamic y-axis scaling
        mask_plot = (wl >= winf) & (wl <= wsup)
        flux_min = np.nanmin(flux[mask_plot])
        flux_max = np.nanmax(flux[mask_plot])
        ax.set_ylim(0.95 * flux_min, 1.5 * flux_max)

        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel(r"Flux (erg/s/cm$^2$/Å/arcsec$^2$)")

        if 'id' in spec.header or 'ID' in spec.header:
            ax.text(
                0.02, 0.96,
                f"{spec.header.get('id', spec.header.get('ID'))}",
                ha='left', va='top', fontsize=12,
                transform=ax.transAxes, color='black'
            )

        # --- SNR Calculation ---
        if show_snr and flux_err is not None:
            snr = flux / flux_err
            snr_median = np.nanmedian(snr)
            snr_min = np.nanmin(snr)
            snr_max = np.nanmax(snr)

            ax.text(
                0.95, 0.95,
                f"SNR (Median): {snr_median:.2f}\nMin: {snr_min:.2f}\nMax: {snr_max:.2f}",
                ha='right', va='top',
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='gray', facecolor='white', alpha=0.7)
            )

        # --- Filter Bands ---
        if show_filters:
            for name, props in UBGVRI_filters.items():
                wmin = props["pivot_wavelength"] - props["bandwidth"] / 2
                wmax = props["pivot_wavelength"] + props["bandwidth"] / 2
                ax.fill_betweenx(
                    [0.95 * flux_min, 1.5 * flux_max],
                    wmin, wmax,
                    alpha=0.2,
                    color=props["color"],
                    label=f"{name} filter"
                )

        # --- Spectral Lines ---
        if show_spectral_lines:
            for name, wavelength in spectral_lines.items():
                wavelength_shifted = wavelength * (1 + z)
                ax.axvline(wavelength_shifted, color="black", linestyle="dashed", alpha=0.7)
                ax.text(
                    wavelength_shifted + 2, 1.1 * flux_max, name,
                    rotation=90, verticalalignment="bottom",
                    fontsize=7, zorder=1, clip_on=True,
                    color='black', weight='bold', ha='center',
                    backgroundcolor='white'
                )

        # --- Atmospheric Lines ---
        if show_atmospheric_lines:
            for name, wavelength in atmospheric_lines.items():
                ax.axvline(wavelength, color="cyan", linestyle="dashed", alpha=0.7)
                ax.text(
                    wavelength + 2, flux_max * 0.8, name,
                    rotation=90, verticalalignment="bottom",
                    fontsize=8, zorder=1, clip_on=True,
                    color='cyan'
                )

    plt.tight_layout()
    plt.show()
        
        




        
        
        
        
# FIT LINES BLOCK
        
        
from scipy.stats import norm as stat_norm
from scipy.optimize import curve_fit



# All lines share the same velocity dispersion (same Gaussian width in km/s) because they come from the same gas region and are kinematically coupled
# [N II] λ6583 / λ6548 flux ratio = 2.96 (quantum mechanics. Storey & Zeippen (2000))

lambda_Ha = spectral_lines['Ha']
lambda_NII_6548 = spectral_lines['[N II] 6548']
lambda_NII_6583 = spectral_lines['[N II] 6583']

def model_Ha_emission(wl, z, continuum, amplitude_Ha, amplitude_NII_6583, sigma):

    amplitude_NII_6548 = amplitude_NII_6583 / 2.96

    ha_profile     = amplitude_Ha        * stat_norm(loc=lambda_Ha*(1+z),      scale=sigma).pdf(wl)
    nii_6583_prof  = amplitude_NII_6583  * stat_norm(loc=lambda_NII_6583*(1+z), scale=sigma).pdf(wl)
    nii_6548_prof  = amplitude_NII_6548  * stat_norm(loc=lambda_NII_6548*(1+z), scale=sigma).pdf(wl)

    return continuum + ha_profile + nii_6583_prof + nii_6548_prof


lambda_Hb = spectral_lines['Hb']

"""
def model_Hb_emission(wl, continuum, amplitude_Hb_em, amplitude_Hb_abs, z_abs, sigma_abs, *, z_em, sigma_em):
    return continuum + amplitude_Hb_em * stat_norm(loc=lambda_Hb*(1+z_em), scale=sigma_em).pdf(wl) + amplitude_Hb_abs * stat_norm(loc=lambda_Hb*(1+z_abs), scale=sigma_abs).pdf(w)
"""
def model_Hb_emission(wl, continuum, amplitude_Hb, *, z, sigma):
    return continuum + amplitude_Hb * stat_norm(loc=lambda_Hb*(1+z), scale=sigma).pdf(wl)


# --- [OIII] doublet ---
# Very strong in low-metallicity/star-forming galaxies
# [O III] λ4959	always 1/3 the intensity of λ5007 (quantum mechanics)

lambda_OIII_5007 = spectral_lines['[O III] 5007']
lambda_OIII_4959 = spectral_lines['[O III] 4959']

def model_OIII(wl, continuum, amplitude_OIII_5007, *, sigma, z):
    amplitude_OIII_4959 = amplitude_OIII_5007 / 3
    return continuum + amplitude_OIII_5007 * stat_norm(loc=lambda_OIII_5007*(1+z), scale=sigma).pdf(wl) + \
           amplitude_OIII_4959 * stat_norm(loc=lambda_OIII_4959*(1+z), scale=sigma).pdf(wl)


# --- [SII] doublet ---
# [SII] 6716 and [SII] 6731 intensity ratio is very sensitive to electron density (nₑ) in the range 10^2-10^4 cm-3

lambda_SII_6716 = spectral_lines['[S II] 6716']
lambda_SII_6731 = spectral_lines['[S II] 6731']

def model_SII(wl, continuum, amplitude_SII_6716, amplitude_SII_6731, *, sigma, z):
    return continuum + amplitude_SII_6716* stat_norm(loc=lambda_SII_6716*(1+z), scale=sigma).pdf(wl) + \
           amplitude_SII_6731 * stat_norm(loc=lambda_SII_6731*(1+z), scale=sigma).pdf(wl)

# --- [HeI] 5876 + Na I D doublet

lambda_HeI = spectral_lines['He I 5876']

def model_HeI(wl, continuum, amplitude_HeI, *, sigma, z):
    return continuum + amplitude_HeI* stat_norm(loc=lambda_HeI*(1+z), scale=sigma).pdf(wl) 


### --- Balmer absorption lines --- ###

from scipy.special import wofz

def voigt(x, amp, center, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# the σ (Gaussian width) and γ (Lorentzian width) in Voigt profiles are generally different across the Balmer series — not equal


def model_Balmer_absorption(wl, lambda_center, continuum, amplitude, sigma, gamma):
    line_profile = voigt(wl,amplitude,lambda_center,sigma, gamma)
    return continuum + line_profile 



from functools import partial

def fit_lines(self, spectra, z_mean=0, plot=True):

    lines = []
    for i, spec in enumerate(spectra):

        #print(f"Spectrum of region str(i+1))")
        
        wl = spec.wl 
        flux = spec.flux 
        err = spec.flux_err    


        mask_Ha = (wl > (lambda_Ha * (1 + z_mean) - 50)) & (wl< (lambda_Ha * (1 + z_mean) + 50))
        wlave_Ha = wl[mask_Ha]
        flux_Ha = flux[mask_Ha]
        err_Ha = err[mask_Ha]

        mask_Hb = (wl > (lambda_Hb * (1 + z_mean) - 50)) & (wl< (lambda_Hb * (1 + z_mean) + 50))
        wlave_Hb = wl[mask_Hb]
        flux_Hb = flux[mask_Hb]
        err_Hb = err[mask_Hb]

        mask_OIII = (wl > (lambda_OIII_4959 * (1 + z_mean) - 50)) & (wl< (lambda_OIII_5007 * (1 + z_mean) + 50))
        wlave_OIII = wl[mask_OIII]
        flux_OIII = flux[mask_OIII]
        err_OIII = err[mask_OIII]

        mask_SII = (wl > (lambda_SII_6716 * (1 + z_mean) - 50)) & (wl< (lambda_SII_6731 * (1 + z_mean) + 50))
        wlave_SII = wl[mask_SII]
        flux_SII = flux[mask_SII]
        err_SII = err[mask_SII]

        mask_HeI = (wl > (lambda_HeI * (1 + z_mean) - 50)) & (wl< (lambda_HeI * (1 + z_mean) + 50))
        wlave_HeI = wl[mask_HeI]
        flux_HeI = flux[mask_HeI]
        err_HeI = err[mask_HeI]
        

        median_Ha = np.median(flux_Ha)
        median_Hb = np.median(flux_Hb)


        # Check if the line is in emission or absorption
        #def mad(arr):
        #    return np.median(np.abs(arr - np.median(arr)))

        #peak_Ha = np.max(flux_Ha)
        #peak_Hb = np.max(flux_Hb)
        #cont_Ha = np.median(flux_Ha)
        #cont_Hb = np.median(flux_Hb)
        #noise_Ha = mad(flux_Ha)
        #noise_Hb = mad(flux_Hb)

        #if (peak_Ha - cont_Ha) / noise_Ha > 3 or (peak_Hb - cont_Hb) / noise_Hb > 3:
        #    print("LINES IN EMISSION")

        if flux_Ha[np.argmax(abs(flux_Ha - median_Ha))] >  median_Ha: # and flux_Hb[np.argmax(abs(flux_Hb - median_Hb))] > median_Hb:
            print('LINES IN EMISSION')

            q1, q2 = np.nanquantile(flux_Ha, [0.16, 0.84])
            subset = flux_Ha[(flux_Ha >= q1) & (flux_Ha <= q2)]
            estimated_err = np.std(subset)

            guess = [z_mean, np.median(flux_Ha), np.max(flux_Ha), np.max(flux_Ha)/2, 2.0]
            popt_Ha, pcov_Ha = curve_fit(model_Ha_emission, wave_Ha, flux_Ha, p0=guess, sigma=err_Ha, absolute_sigma=True)
            z_fit, cont_Ha, amp_Ha, amp_NII_6583, sigma_fit = popt_Ha

            model_vals = model_Ha_emission(wave_Ha, *popt_Ha)
            residuals = flux_Ha - model_vals
            chi2 = np.sum((residuals / estimated_err) ** 2)
            dof = len(wave_Ha) - len(popt_Ha)
            chi2_red_Ha = chi2 / dof


            # Initial guess for Hβ fit:
            # [continuum, amplitude_Hb_emission, amplitude_Hb_absorption, z_absorption, sigma_absorption]
            """
            guess_Hb = [
                np.median(flux_Hb),        # continuum
                np.max(flux_Hb),           # emission amplitude (positive)
                -np.abs(np.min(flux_Hb)),  # absorption amplitude (negative)
                z_fit,                     # absorption z guess (start with emission z)
                sigma_fit                  # absorption sigma guess (start with emission sigma)
            ]
            """
            guess_Hb = [np.median(flux_Hb), np.max(flux_Hb)]

            # Fix z_em and sigma_em from Hα fit via partial
            #model_Hb_fixed = partial(model_Hb_emission, z_em=z_fit, sigma_em=sigma_fit)
            model_Hb_fixed = partial(model_Hb_emission, z=z_fit, sigma=sigma_fit)


            # Define bounds to enforce absorption amplitude negative:
            bounds_Hb = (
                [-np.inf,       0,       -np.abs(np.min(flux_Hb)),  z_fit - 0.001,  0],   # lower bounds: continuum, amplitude_em >=0, amplitude_abs <0, small z range, sigma > 0
                [np.inf,  1.5*max(flux_Hb),          0,    z_fit + 0.001,  8]  # upper bounds: continuum, amplitude_abs < 0 so max 0
            )

            popt_Hb, pcov_Hb = curve_fit(
                model_Hb_fixed, wave_Hb, flux_Hb, p0=guess_Hb, sigma=err_Hb, absolute_sigma=True,) #bounds=bounds_Hb
            
            #cont_Hb, amp_Hb_em, amp_Hb_abs, z_abs, sigma_abs = popt_Hb
            cont_Hb, amp_Hb = popt_Hb

            model_vals = model_Hb_fixed(wave_Hb, *popt_Hb)
            residuals = flux_Hb - model_vals
            chi2 = np.sum((residuals / err_Hb) ** 2)
            dof = len(wave_Hb) - len(popt_Hb)
            chi2_red_Hb = chi2 / dof




            # --- Fit OIII doublet

            model_OIII_fixed = partial(model_OIII, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_OIII), np.max(flux_OIII)]
            popt_OIII, pcov_OIII = curve_fit(model_OIII_fixed, wave_OIII, flux_OIII, p0=guess, sigma=err_OIII, absolute_sigma=True)
            cont_OIII, amp_OIII = popt_OIII

            model_vals = model_OIII_fixed(wave_OIII, *popt_OIII)
            residuals = flux_OIII - model_vals
            chi2 = np.sum((residuals / err_OIII) ** 2)
            dof = len(wave_OIII) - len(popt_OIII)
            chi2_red_OIII = chi2 / dof


            # --- Fit SII doublet

            model_SII_fixed = partial(model_SII, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_SII), np.max(flux_SII), np.max(flux_SII)]
            popt_SII, pcov_SII = curve_fit(model_SII_fixed, wave_SII, flux_SII, p0=guess, sigma=err_SII, absolute_sigma=True)
            cont_SII, amp_SII_6716, amp_SII_6731 = popt_SII
            
            model_vals = model_SII_fixed(wave_SII, *popt_SII)
            residuals = flux_SII - model_vals
            chi2 = np.sum((residuals / err_SII) ** 2)
            dof = len(wave_SII) - len(popt_SII)
            print(chi2)
            print(dof)
            chi2_red_SII = chi2 / dof

            # --- Fit HeI + Na doublet

            model_HeI_fixed = partial(model_HeI, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_HeI), np.max(flux_HeI)]
            popt_HeI, pcov_HeI = curve_fit(model_HeI_fixed, wave_HeI, flux_HeI, p0=guess, sigma=err_HeI, absolute_sigma=True)
            cont_HeI, amp_HeI = popt_HeI
            
            model_vals = model_HeI_fixed(wave_HeI, *popt_HeI)
            residuals = flux_HeI - model_vals
            chi2 = np.sum((residuals / err_HeI) ** 2)
            dof = len(wave_HeI) - len(popt_HeI)
            print(chi2)
            print(dof)
            chi2_red_HeI = chi2 / dof

            if plot:
                fig, ax = plt.subplots(2, 3, figsize=(12, 6))
                ax = ax.flatten()

                w_array = np.linspace(wave_Ha[0], wave_Ha[-1], 1000)
                ax[0].errorbar(wave_Ha, flux_Ha, yerr=err_Ha, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[0].plot(wave_Ha, flux_Ha, linestyle='-', alpha=0.3, color='black')
                ax[0].plot(w_array, cont_Ha + amp_Ha*stat_norm(lambda_Ha*(1+z_fit),sigma_fit).pdf(w_array), color='red',label='$H\\alpha$')
                ax[0].plot(w_array, cont_Ha + (amp_NII_6583/2.96)*stat_norm(lambda_NII_6548*(1+z_fit),sigma_fit).pdf(w_array), color='orange',label='[NII] 6548')
                ax[0].plot(w_array, cont_Ha + amp_NII_6583*stat_norm(lambda_NII_6583*(1+z_fit),sigma_fit).pdf(w_array), color='darkgreen', label='[NII] 6583')
                ax[0].plot(w_array, np.full(w_array.shape, cont_Ha), color='navy', linewidth=1)
                ax[0].legend(loc='upper right', frameon=False)
                ax[0].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_Ha:.2f}", transform=ax[0].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_Hb[0], wave_Hb[-1], 1000)
                ax[1].errorbar(wave_Hb, flux_Hb, yerr=err_Hb, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[1].plot(wave_Hb, flux_Hb, linestyle='-', alpha=0.3, color='black')
                
                ax[1].plot(w_array, cont_Hb + amp_Hb*stat_norm(lambda_Hb*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='$H\\beta$')
                #ax[1].plot(w_array, cont_Hb + amp_Hb_abs*stat_norm(lambda_Hb*(1+z_abs), sigma_abs).pdf(w_array), color='red')
                ax[1].plot(w_array, np.full(w_array.shape, cont_Hb), color='navy', linewidth=1)
                ax[1].legend(loc='upper right', frameon=False)
                ax[1].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_Hb:.2f}", transform=ax[1].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_OIII[0], wave_OIII[-1], 1000)
                ax[2].errorbar(wave_OIII, flux_OIII, yerr=err_OIII, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[2].plot(wave_OIII, flux_OIII, linestyle='-', alpha=0.3, color='black')
                ax[2].plot(w_array, cont_OIII + amp_OIII*stat_norm(lambda_OIII_5007*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[OIII] 5007')
                ax[2].plot(w_array, cont_OIII + (amp_OIII/3)*stat_norm(lambda_OIII_4959*(1+z_fit), sigma_fit).pdf(w_array), color='lime',label='[OIII] 4959')
                ax[2].plot(w_array, np.full(w_array.shape, cont_OIII), color='navy', linewidth=1)
                ax[2].legend(loc='upper right', frameon=False)
                ax[2].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_OIII:.2f}", transform=ax[2].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_SII[0], wave_SII[-1], 1000)
                ax[3].errorbar(wave_SII, flux_SII, yerr=err_SII, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[3].plot(wave_SII, flux_SII, linestyle='-', alpha=0.3, color='black')
                ax[3].plot(w_array, cont_SII + amp_SII_6716*stat_norm(lambda_SII_6716*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[SII] 6716')
                ax[3].plot(w_array, cont_SII + amp_SII_6731*stat_norm(lambda_SII_6731*(1+z_fit), sigma_fit).pdf(w_array), color='lime',label='[SII] 6731')
                ax[3].plot(w_array, np.full(w_array.shape, cont_SII), color='navy', linewidth=1)
                ax[3].legend(loc='upper right', frameon=False)
                ax[3].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_SII:.2f}", transform=ax[3].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_HeI[0], wave_HeI[-1], 1000)
                ax[4].errorbar(wave_HeI, flux_HeI, yerr=err_HeI, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[4].plot(wave_HeI, flux_HeI, linestyle='-', alpha=0.3, color='black')
                ax[4].plot(w_array, cont_HeI + amp_HeI*stat_norm(lambda_HeI*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[HeI] 5876')
                ax[4].plot(w_array, np.full(w_array.shape, cont_HeI), color='navy', linewidth=1)
                ax[4].legend(loc='upper right', frameon=False)
                ax[4].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_HeI:.2f}", transform=ax[4].transAxes, ha='left', va='top', fontsize=8, color='black')

                for i in range(5):
                    ax[i].set_xlabel("Wavelength [Å]")
                ax[0].set_ylabel("Flux [erg/s/cm²/Å/arcsec²]")
                ax[3].set_ylabel("Flux [erg/s/cm²/Å/arcsec²]")

                sigma_v = (sigma_fit / lambda_Ha) * 2.99e5 * (z_fit-z_mean)
                fig.suptitle(f"Region {i+1}: $z = {z_fit:.5f} \pm {np.sqrt(pcov_Ha[0, 0]):.5f}$, $\sigma_v = {sigma_v:.5f} km/s$")

                plt.tight_layout()
                plt.show()

            lines.append(None)

        else:
            print("Line in absorption")

            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(12, 3))

            for i, balmer_line in enumerate(['Ha', 'Hb']):
                lambda_center = spectral_lines[balmer_line] * (1 + z_mean)
                mask_fit = (w > (lambda_center - 50)) & (w < (lambda_center + 50))
                wave_fit = w[mask_fit]
                flux_fit = flux[mask_fit]
                err_fit = err[mask_fit]

                guess = [lambda_center, np.median(flux_fit), np.max(flux_fit), 2.0, 2.0]
                popt, pcov = curve_fit(model_Balmer_absorption, wave_fit, flux_fit, p0=guess, sigma=err_fit, absolute_sigma=True)
                lambda_center_fit, cont_fit, amp_fit, sigma_fit, gamma_fit = popt
                z_fit = lambda_center_fit / spectral_lines[balmer_line] - 1

                model_fit = model_Balmer_absorption(wave_fit, *popt)

                if plot:
                    ax[i].errorbar(wave_fit, flux_fit, yerr=err_fit, fmt='o', markersize=3, color='black', alpha=0.8)
                    ax[i].plot(wave_fit, flux_fit, linestyle='-', alpha=0.3, color='black')
                    ax[i].plot(wave_fit, model_fit, 'r--', label='Voigt fit')
                    ax[i].set_title(f"{balmer_line} fit")
                    ax[i].set_xlabel("Wavelength [Å]")

    return lines
        
