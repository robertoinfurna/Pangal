# Standard Library
import glob
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from copy import deepcopy

# Numerical & Scientific
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Astropy
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# Regions
from regions import (
    PixCoord,
    SkyRegion,
    EllipsePixelRegion,
    EllipseAnnulusPixelRegion,
    EllipseSkyRegion,
    CirclePixelRegion,
)

# Skimage
from skimage.draw import polygon
from skimage import measure
from skimage.measure import find_contours
from skimage.transform import rotate

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from .pangal_methods.base_classes import Image, Cube, Region, Point, Contours
from .pangal_methods.filters import Filter, list_filters, plot_filters, map_filter_names, nice_filter_names
from .pangal_methods.plot import plot
from .pangal_methods.photometry import photometry, surface_brightness_profile, inspect_photometry
from .pangal_methods.spectral_analysis import line_map, extract_spectra, plot_spectra, fit_lines
from .pangal_methods.utils import area_pixel, dtheta_pixel, cut_and_rotate, correct_coords_offset, mosaic


# usefull data
# https://en.wikipedia.org/wiki/AB_magnitude                         AB magnitudes
# https://www.ipac.caltech.edu/2mass/overview/about2mass.html        2MASS
# https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html    Vega-to-AB conversions
# https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-5-uvis-spectral-elements   HST-UVIS 
# https://archive.stsci.edu/manuals/archive_handbook/chap4.html      GALEX




 
                
class PanGal:

    def __init__(self, fits_files, target_coords, fov, EBmV=0,):


        self.images = {}
        self.cubes = {}
        self.spectra = {}
        
        self.target_coords = target_coords
        self.fov = fov
        self.EBmV = EBmV
              
        for file in fits_files:
            
            # add images
            self.add_image(file)

            # add muse cube
            if 'muse' in file:
                self.add_cube(file)
            
            
                
    def add_image(self,file):
        """
        Adds observations (Image, Cube or Spectrum) to the PangalObject given a fits file
        fits file must be named using a particular convention
        """
            
        with fits.open(file) as hdul:
                                  
            if 'galex' in file.lower():

                print('Processing GALEX file: ',file)
            
                header = hdul[0].header
                bunit = 'counts_s' 
                    
                band = 'galex_nuv' if float(header['BAND']) == 1.0 else 'galex_fuv'

                ZP = 20.08 if band == 'galex_nuv' else 18.82
                ZP_err = 0.05
                pivot_wavelength = 2315 if band == 'galex_nuv' else 1538    

                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs) 
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
        
                image = hdul[0].data   # native units in counts/s
                image, wcs = cut_and_rotate(image, wcs, target_coords, fov)
        
                AG = self.EBmV * self.MW_extinction(pivot_wavelength)
                print(f'Milky Way dust extinction: {AG:.2f}, {10**(AG / 2.5)}')
                
                flux_conv_counts_s_to_mJy = 1e3 * 10**(-ZP / 2.5 + AG / 2.5 + 8.9 / 2.5)
                    

            elif 'sdss' in file.lower():
            
                print('Processing SDSS file: ',file)
        
                header = hdul[0].header
        
                band = 'sdss_'+header['FILTER']
                sdss_wavelengths = {'u': 3561, 'g': 4718, 'r': 6185, 'i': 7499, 'z': 8961}
                pivot_wavelength = sdss_wavelengths[header['FILTER']]
                conversion_factor = header['NMGY']            # counts/s to nanomaggy
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['SPA']
        
                image_nanomaggy = hdul[0].data
                image_nanomaggy, wcs = cut_and_rotate(image_nanomaggy, wcs, target_coords, fov, position_angle)
                
                # convert from nanomaggy to counts/s
                image = image_nanomaggy / conversion_factor
                bunit = 'counts_s'

                AG = self.EBmV * self.MW_extinction(pivot_wavelength)
                print(f'Milky Way dust extinction: {AG:.2f}, {10**(AG / 2.5)}')
                flux_conv_counts_s_to_mJy = 3.631e-6 * 1e3 * conversion_factor * 10**(AG / 2.5)
                
                ZP = None
                ZP_err = None
                
                
                
            elif 'hst' in file.lower():
            
                print('HST: ',file)
        
                header = fits.Header()
                header.update(hdul[0].header)
                header.update(hdul[1].header)

                band = f"hst_{header['INSTRUME'].strip().lower()}_"

                if header['INSTRUME'].strip() == 'WFC3':
                    band += header['DETECTOR'].strip().lower() + '_'

                # Determine the correct filter
                f = None
                for key in ['FILTER', 'FILTER2', 'FILTER1']:
                    val = header.get(key, '').strip()
                    if val.startswith('F'):
                        f = val
                        break

                if not f:
                    print(file.lower(), ': HST filter not recognized. Skipping')
                    
                band += f.lower()
                
        
                conv_factor_flambda = header['PHOTFLAM']
                pivot_wavelength = header['PHOTPLAM']
                ZP = header['PHOTZPT']
                
                if 'PHOTFNU' not in header:
                    header['PHOTFNU'] = conv_factor_flambda * (pivot_wavelength**2 / 2.99e18) / 1e-23  # Jy/e-
        
                AG = self.EBmV * self.MW_extinction(pivot_wavelength)
                flux_conv_counts_s_to_mJy = 1e3 * header['PHOTFNU'] * 10**(AG / 2.5)
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                
                position_angle = -header['ORIENTAT'] if 'ORIENTAT' in header else (-header['PA_APER'] if 'PA_APER' in header else None)
           
                bunit = 'counts_s'
                image = hdul[1].data
                image, wcs = cut_and_rotate(image, wcs, self.target_coords, self.fov, position_angle)
                image = np.nan_to_num(image, nan=0.0)

                ZP = None
                ZP_err = None
        
        
            elif any(x in file.lower() for x in ['2mass', '2masx']):
            
                print('2MASS: ',file)
        
                header = hdul[0].header
                
                bb = header['FILTER'].strip().lower()
                if bb == 'k': bb = 'ks'
                band = '2mass_'+bb
        
                header['EXPTIME'] = 7.8 # seconds
                sky_val = header['SKYVAL']
                sigma_sky = header['SKYSIG']
        
                Vega_to_AB_correction = {'j': 0.91, 'h': 1.39, 'ks': 1.85}
                ZP = header['MAGZP'] + Vega_to_AB_correction[bb]
                ZP_err = None
        
                two_mass_wavelengths = {'j': 1.2408, 'h': 1.6513, 'ks': 2.1655}
                pivot_wavelength = two_mass_wavelengths[bb] * 1e4
        
                flux_conv_counts_s_to_mJy = 1e3 * 10**(-ZP / 2.5 + 8.9 / 2.5)
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = - header['CROTA2']
        
                bunit = 'counts_s'
                image = hdul[0].data
                image, wcs = cut_and_rotate(image, wcs, target_coords, fov, position_angle)
                image = np.nan_to_num(image, nan=0.0)
        
                image = image - sky_val
                
            elif 'wise' in file.lower(): #https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/index.html
            
                print('WISE: ',file)
                
                header = hdul[0].header
                
                bb = header['BAND']
                band = 'wise_w'+str(bb)
                pivot_wavelenth = header['WAVELEN']
      
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['CROTA2']   
              
                Vega_to_AB_correction = {1: 2.699, 2: 3.339, 3: 5.174, 4:6.620} #https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
                ZP = header['MAGZP'] + Vega_to_AB_correction[bb]
                ZP_err = header['MAGZPUNC']
                
                header['EXPTIME'] = header['NUMFRMS'] * 7.7
                
                flux_conv_counts_s_to_mJy = 1e3 * 10**(-ZP / 2.5 + 8.9 / 2.5)     
                
                bunit = 'counts_s'
                image = hdul[0].data                 
        
            elif 'irac' in file.lower():
            
                print('IRAC: ',file)
        
                header = hdul[0].header
        
                channel = header['CHNLNUM']
                band = 'spitzer_irac_'+str(channel)
                pivot_wavelength = [3.6e4, 4.5e4, 5.8e4, 8.0e4][channel - 1]
                conversion_factor = header['FLUXCONV'] / header['GAIN']
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['PA']
                
                bunit = 'counts_s'
                image_MJy_sr = hdul[0].data
                image = image_MJy_sr / conversion_factor
        
                image, wcs = cut_and_rotate(image, wcs, target_coords, fov, position_angle)
                image = np.nan_to_num(image, nan=0.0)

                flux_conv_counts_s_to_mJy = conversion_factor * 1e9 * area_pix_deg2 * 3.04617e-4
                
                ZP = None
                ZP_err = None

            elif 'hpacs' in file.lower():
            
                print('Herschel PACS: ',file)
        
                header = fits.Header()
                header.update(hdul[0].header)
                header.update(hdul[1].header)
        
                pivot_wavelength = header['WAVELNTH'] 
                band = f'herschel_pacs_{pivot_wavelength:.0f}um'
                pivot_wavelength = pivot_wavelength * 1e4
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = 0
        
                bunit = 'mJy'
                image = hdul[1].data * 1e3 # mJy
                image, wcs = cut_and_rotate(image, wcs, target_coords, fov)

                flux_conv_counts_s_to_mJy = None
                ZP = None
                ZP_err = None

            #else:
                #print(f'Warning. Unrecognized file: {file}')
                
                
            # manages multiple images from same band (mosaics)
            # first band 
            image_name = band  
            ii = 1
            while image_name in self.images:
                image_name = f"{band}({ii})"
                print(image_name)
                ii += 1
        
            self.images[image_name] = Image(
                image=image,
                bunit=bunit,
                wcs=wcs,
                dtheta_pix_deg=dtheta_pix_deg,
                area_pix_arcsec2=area_pix_arcsec2,
                pivot_wavelength=pivot_wavelength,
                flux_conv_counts_s_to_mJy=flux_conv_counts_s_to_mJy,
                ZP=ZP,
                ZP_err=ZP_err,
                header=header,
                filter=Filter(band)
            )

    
    def add_muse_cube(self,file):

        print('Processing MUSE cube: ',file)
        
        with fits.open(file) as hdul:

            header = fits.Header()
            header.update(hdul[0].header)
            header.update(hdul[1].header)
                
            units = 1e-20        # converts in erg/s/cm2/A
            
            cube = hdul[1].data
            var = hdul[2].data 
    
            crval3 = header['CRVAL3']
            dw = header['CD3_3']
            channels = np.arange(0, header['NAXIS3'])
            w = crval3 + channels * dw
            
            # Compute dust attenuation
            AG = self.EBmV * self.MW_extinction(w)
            MW_dust_extinction = 10**(AG/2.5)
            if self.EBmV != 0:
                print("Correcting MUSE cube for Milky Way dust extinction")
                cube *= MW_dust_extinction[:, np.newaxis, np.newaxis]
            
                
            
            with fits.open('Extracted_resolution_muse.fits') as hdul:
                w_r = np.arange(0,hdul[0].header['NAXIS1'],1) * hdul[0].header['CD1_1'] + hdul[0].header['CRVAL1']
                r = hdul[0].data                           # spectral resolution FWHM, in Angstroms
            r_fun = interp1d(w_r, r, kind='linear', bounds_error=False, fill_value='extrapolate')
            R = r_fun(w)

            dtheta_pix_deg = abs(header['CD1_1'])  # degrees per pixel
            area_pix_arcsec2 = (dtheta_pix_deg * 3600)**2

            wcs = WCS(header).celestial

            cube_name = 'muse'
            ii = 1
            while cube_name in self.cubes:
                cube_name = f"muse({ii})"
                ii += 1
            
            self.cubes[cube_name] = Cube(cube=cube,var=var,units=units,w=w,dw=dw,R=R,wcs=wcs,dtheta_pix_deg=dtheta_pix_deg,area_pix_arcsec2=area_pix_arcsec2)

        
            for band in [key for key in map_filter_names if key.startswith('muse')]:

                filter = Filter(band)
                bunit = 'erg_s_cm2_A'
        
                w_inf = max(w[0],filter.wavelength_range[0])
                w_sup = filter.wavelength_range[1]
                channel_inf = np.digitize(w_inf, w) - 1
                channel_sup = np.digitize(w_sup, w) - 1
                bandwidth = w_sup - w_inf
                w_pivot = filter.effective_wavelength

                integrated_band = np.nansum(cube[channel_inf:channel_sup, :, :], axis=0)

                image = integrated_band * dw / bandwidth * units #* 10**(AG / 2.5)

                image_name = f"{band}"  # e.g., muse_red
                ii = 1
                while image_name in self.images:
                    image_name = f"{band}({ii})"
                    ii += 1

                self.images[image_name] = Image(
                    image=image,
                    bunit=bunit,
                    wcs=wcs,
                    dtheta_pix_deg=dtheta_pix_deg,
                    area_pix_arcsec2=area_pix_arcsec2,
                    pivot_wavelength=w_pivot,
                    header=header,
                    filter=filter
                    )
            
    
                
                
                
                    
    # methods
    plot = plot
    photometry = photometry
    inspect_photometry = inspect_photometry
    surface_brightness_profile = surface_brightness_profile
    line_map = line_map                                               # Extracts and plots a emission line surface brightness map Image object
    extract_spectra = extract_spectra                                 # Extracts a Spectrum object from a cube, for a given a Region object
    plot_spectra = plot_spectra
    fit_lines = fit_lines
    correct_coords_offset = correct_coords_offset
    mosaic = mosaic
        
    list_filters = list_filters      
    plot_filters = plot_filters
    


    def MW_extinction(self, lam):
        # Return Milky Way extinction curve A(lam)/E(B-V) via interpolation.
        anchor_points = np.array((1111, 1176, 1250, 1316, 1393, 1490, 1600, 1701, 1799, 1901, 2000,
                                2101, 2188, 2299, 2398, 2500, 2740, 3436, 4000, 4405, 5495,
                                6993, 9009, 12500, 16390, 22200))
        
        anchor_values = np.array((11.53, 10.53, 9.63, 8.99, 8.44, 8.07, 7.81, 7.68, 7.73, 7.98, 8.60,
                                9.31, 9.65, 8.85, 7.98, 7.27, 6.18, 4.88, 4.38, 4.08, 3.08,
                                2.30, 1.48, 0.85, 0.50, 0.32))
        
        interpolator = interp1d(anchor_points, anchor_values, kind='cubic', fill_value="extrapolate")
        
        return interpolator(lam)


    def plot_Milky_Way_dust_absorption(self,):
        wavelengths = np.linspace(1000, 23000, 1000)  # Angstroms
        extinction = self.MW_extinction(wavelengths)

        fig,ax = plt.subplots(figsize=(8, 5))

        ax.plot(wavelengths, extinction, color='tab:blue', lw=2)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Extinction')
        ax.set_title(f'Milky Way Extinction Curve, RA: {self.target_coords[0]}, Dec: {self.target_coords[1]}')
        ax.text(0.02,0.9,f'$E(B-V)=${self.EBmV}',transform=ax.transAxes)
        plt.tight_layout()
        plt.show()
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
