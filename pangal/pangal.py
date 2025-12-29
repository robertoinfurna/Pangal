# Standard Library
import glob
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from copy import deepcopy
import os

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


from .image import Image
from .cube import Cube
from .region import Region
from .filter import Filter, list_filters, plot_filters, map_filter_names, nice_filter_names

from .plot import plot, make_false_color, plot_false_color, Point, Contours
from .photometry import photometry, surface_brightness_profile, inspect_photometry

from .pangal_utils import area_pixel, dtheta_pixel, cut_and_rotate, correct_coords_offset, mosaic, MW_extinction


# usefull data
# https://en.wikipedia.org/wiki/AB_magnitude                         AB magnitudes
# https://www.ipac.caltech.edu/2mass/overview/about2mass.html        2MASS
# https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html    Vega-to-AB conversions
# https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-5-uvis-spectral-elements   HST-UVIS 
# https://archive.stsci.edu/manuals/archive_handbook/chap4.html      GALEX
# https://www.iiap.res.in/projects/uvit/instrument/filters/          UVIT



class DotDict(dict):                      # Dictionary that supports attribute-style access.
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    def __dir__(self):
        # Include dict keys in autocomplete
        return list(self.keys()) + list(super().__dir__())

                 
class PanGal:

    def __init__(self, 
                 directory=None,
                 fits_files=None, 
                 target_coords=None, 
                 fov=0, 
                 EBmV=0,
                 EBmV_err=0,
                 spectral_range_muse=None,
                 correct_dust_muse=True,
                 ): 

        self.images = DotDict()
        self.cubes = DotDict()
        self.spectra = DotDict()
        
        self.target_coords = target_coords
        self.fov = fov
        self.EBmV = EBmV
        self.EBmV_err = EBmV_err
              

        # --- resolve input files ---
        if fits_files is None and directory is not None:
            # find FITS files in directory
            fits_files = sorted(glob.glob(os.path.join(directory, "*.fits")))
            fits_files += sorted(glob.glob(os.path.join(directory, "*.fit")))
            fits_files += sorted(glob.glob(os.path.join(directory, "*.fz")))  # ESO/MUSE compressed

        # --- upload data ---
        for file in fits_files:        
            self.add_image(file)
            self.add_cube(file, spectral_range=spectral_range_muse, correct_for_dust=correct_dust_muse)
            # self.add_spectrum(file)




    # METHODS

    @property
    def overview(self):
        print('Images: \n')  
        print(list(self.images.keys()))
        print('\nCubes: \n')  
        print(list(self.cubes.keys()))
              
    # methods
    plot = plot
    make_false_color = make_false_color
    plot_false_color = plot_false_color
    photometry = photometry
    inspect_photometry = inspect_photometry
    surface_brightness_profile = surface_brightness_profile
    
        
    list_filters = list_filters    # lists all available filters    
    plot_filters = plot_filters


    # Utils
    MW_extinction = MW_extinction
    correct_coords_offset = correct_coords_offset
    mosaic = mosaic



    # Core function of Pangal       
    def add_image(self,file):
        """
        Adds Image observations to the Pangal object given a fits file
        fits file must be named using a particular convention
        """
            
        with fits.open(file) as hdul:
                                  
            if 'galex' in file.lower():

                print('Processing GALEX file: ',file)
            
                header = hdul[0].header
                #bunit = 'counts_s' 
                    
                band = 'galex_nuv' if float(header['BAND']) == 1.0 else 'galex_fuv'

                # https://galex.stsci.edu/gr6/?page=faq&utm_source=chatgpt.com
                ZP = 20.08 if band == 'galex_nuv' else 18.82
                ZP_err = 0.05
                pivot_wavelength = 2297 if band == 'galex_nuv' else 1524    

                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs) 
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
        
                image = hdul[0].data   # native units in counts/s
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov)

                exptime = float(header['EXPTIME'])
        



            elif 'sdss' in file.lower():
            
                print('Processing SDSS file: ',file)
        
                header = hdul[0].header
        
                band = 'sdss_'+header['FILTER']
                sdss_wavelengths = {'u': 3561, 'g': 4718, 'r': 6185, 'i': 7499, 'z': 8961}
                pivot_wavelength = sdss_wavelengths[header['FILTER']]

                # native image are given in nanomaggy
                # header['NMGY'] gives counts/s to nanomaggy
                # m_AB = 22.5 - 2.5 log (F_nMgy) = 22.5 - 2.5 log (counts/s * header['NMGY'])
        
                # convert from nanomaggy to counts/s
                image = hdul[0].data / header['NMGY'] 
                ZP = 22.5 - 2.5 * np.log10(header['NMGY'])
                ZP_err = {'u': 0.04, 'g': 0.01, 'r': 0.01, 'i': 0.01, 'z': 0.02}.get(band[5], 0.01)
 
                exptime = float(header['EXPTIME']) # 57 s

                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['SPA']
        
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)
                
                
            elif 'hst' in file.lower():
            
                print('Processing HST files: ',file)
        
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
                    return None

                band += f.lower()
                
        
                conv_factor_flambda = header['PHOTFLAM']   # Inverse sensitivity, ergs/cm2/A/e- 
                pivot_wavelength = header['PHOTPLAM']
                
                ZP = - 2.5 * np.log10(pivot_wavelength**2/2.998e18) -2.5*np.log10(conv_factor_flambda) - 48.60
                ZP_err = 0.011                              # CONSERVATIVE 1% error on the flux conversion (Calamida et al, 2022) 
                
                exptime = float(header['EXPTIME'])
                               
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                
                position_angle = -header['ORIENTAT'] if 'ORIENTAT' in header else (-header['PA_APER'] if 'PA_APER' in header else None)

                image = hdul[1].data
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)
                
                image[np.isnan(image)] = -999
        
            # 2MASS
            elif any(x in file.lower() for x in ['2mass', '2masx']):
            
                print('Processing 2MASS: ',file)
        
                header = hdul[0].header
                
                bb = header['FILTER'].strip().lower()
                if bb == 'k': bb = 'ks'
                band = '2mass_'+bb
        
                header['EXPTIME'] = 7.8 # seconds
                sky_val = header['SKYVAL']
                sigma_sky = header['SKYSIG']
        
                Vega_to_AB_correction = {'j': 0.91, 'h': 1.39, 'ks': 1.85}
                ZP = header['MAGZP'] + Vega_to_AB_correction[bb]
                ZP_err = 0.011        # conservative 1% error
        
                two_mass_wavelengths = {'j': 1.2408, 'h': 1.6513, 'ks': 2.1655}
                pivot_wavelength = two_mass_wavelengths[bb] * 1e4

                exptime = float(header['EXPTIME'])
        
                #flux_conv_counts_s_to_mJy = 1e3 * 10**(-ZP / 2.5 + 8.9 / 2.5)
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = - header['CROTA2']
        
                image = hdul[0].data
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)
                image = np.nan_to_num(image, nan=0.0)
        
                image = image - sky_val
                
            # WISE
            elif 'wise' in file.lower(): #https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/index.html
            
                print('Processing WISE: ',file)
                
                header = hdul[0].header
                
                bb = header['BAND']
                band = 'wise_w'+str(bb)
                pivot_wavelength = header['WAVELEN']
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['CROTA2']   
                
                Vega_to_AB_correction = {1: 2.699, 2: 3.339, 3: 5.174, 4:6.620} #https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
                ZP = header['MAGZP'] + Vega_to_AB_correction[bb]
                ZP_err = header['MAGZPUNC']
                
                exptime = header['NUMFRMS'] * 7.7

                image = hdul[0].data     
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)            
        
        
            # Spitzer IRAC
            elif 'irac' in file.lower():
            
                print('Processing IRAC: ',file)
        
                header = hdul[0].header
        
                wcs = WCS(header)
                dtheta_pix_deg = dtheta_pixel(wcs)
                area_pix_deg2 = area_pixel(wcs)
                area_pix_arcsec2 = area_pix_deg2 * 3600**2
                position_angle = -header['PA']

                channel = header['CHNLNUM']
                band = 'spitzer_irac_'+str(channel)
                pivot_wavelength = [3.6e4, 4.5e4, 5.8e4, 8.0e4][channel - 1]
                

                # native image in MJy/sr. CONVERT TO counts/s
                conversion_factor = header['FLUXCONV'] / header['GAIN']  # brings MJy/sr in counts/s
                image = hdul[0].data / conversion_factor

                flux_conv_counts_s_to_Jy = conversion_factor * 1e6 * area_pix_deg2 * 3.04617e-4
                # recall that m_AB = ZP - 2.5 log (counts/s) = - 2.5 log f + 8.90
                # Jy = counts/s * flux_conv_counts_s_to_Jy

                ZP = -2.5*np.log10(flux_conv_counts_s_to_Jy) + 8.90
                ZP_err = 0.011        # conservative 1% error

                exptime = float(header['EXPTIME'])
        

                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)
                
                image[np.isnan(image)] = -999

            
            # Herschel PACS
            elif 'hpacs' in file.lower():
            
                print('Processing Herschel PACS: ',file)
        
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
        
                exptime = None
             
                image = hdul[1].data  # NATIVE UNITS ARE GIVEN IN Jy/pixel
                image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov, position_angle=position_angle)

                # Recall that m_ab = - 2.5 log f + 8.90 = ZP - 2.5 log counts/s by definition, with f in Jy
                ZP = 8.90      # smart
                ZP_err = 0     # VERY IMPORTANT! THIS SAYS THAT PACS IS NOT A PHOTON COUNTING DEVICE. NO POISSON ERROR

            else:
                return None
            

            
            # Correct for Milky way foreground dust extinction
            if pivot_wavelength < 22200:
                AG = self.EBmV * self.MW_extinction(pivot_wavelength)           # Milky way foreground dust extinction
                AG_err = self.EBmV_err * self.MW_extinction(pivot_wavelength)
                ZP -= AG                                                        # corrects for foreground extinction
                ZP_err = np.sqrt(ZP_err**2 + AG_err**2)                         # error in extinction estimation

            # manages multiple images from same band (mosaics)
            # first band 
            image_name = band  
            ii = 1
            while image_name in self.images:
                image_name = f"{band}({ii})"
                print(image_name)
                ii += 1
        
            self.images[image_name] = Image(
                data=image,
                wcs=wcs,
                dtheta_pix_deg=dtheta_pix_deg,
                area_pix_arcsec2=area_pix_arcsec2,
                pivot_wavelength=pivot_wavelength,
                ZP=ZP,
                ZP_err=ZP_err,
                exptime=exptime,
                header=header,
                filter=Filter(band)
            )



    def add_cube(self,file,spectral_range,correct_for_dust):
        """
        Adds Cube observations to the PangalObject given a fits file
        fits file must be named using a particular convention

        spectral_range : saves RAM space cropping the spectra
        """
            
        with fits.open(file) as hdul:   

            if 'muse' in file.lower():
            
                print('Processing MUSE: ',file)
            
                with fits.open(file) as hdul:

                    header = fits.Header()
                    header.update(hdul[0].header)
                    header.update(hdul[1].header)
                    
                    crval3 = header['CRVAL3']
                    dw = header['CD3_3']
                    channels = np.arange(0, header['NAXIS3'])
                    wl = crval3 + channels * dw


                    if spectral_range:
                        wl_min, wl_max = spectral_range
                    
                        # Clamp limits to cube range
                        wl_min = max(wl_min, wl[0])
                        wl_max = min(wl_max, wl[-1])
                    
                        # Find indices
                        mask = (wl >= wl_min) & (wl <= wl_max)
                        idx = np.where(mask)[0]
                    
                        if len(idx) == 0:
                            raise ValueError(f"No overlap with cube spectral range "
                                            f"({wl[0]:.1f}–{wl[-1]:.1f} Å)")
                    
                        i_min, i_max = idx[0], idx[-1] + 1
                    else: 
                        i_min, i_max = 0, len(channels)

                    # Slice cube
                    cube = hdul[1].data[i_min:i_max, :, :] 
                    var  = hdul[2].data[i_min:i_max, :, :]
                    wl   = wl[i_min:i_max]

                    # convert to erg/s/cm-2/A
                    cube *= 1e-20
                    var *= 1e-40
                    header['FUNITS'] = 'erg/s/cm2/A'
                    header['WUNITS'] = 'A'

                    # Update header if saving
                    header['CRVAL3'] = wl[0]
                    header['NAXIS3'] = len(wl)

                    #shape = cube.shape
                            
                    # Compute dust attenuation
                    if self.EBmV != 0:
                        # 1D extinction curve
                        ext = self.MW_extinction(wl) * self.EBmV
                        AG = 10 ** (ext / 2.5)

                        # In-place multiply without creating a giant broadcasted array
                        np.multiply(cube, AG[:, None, None], out=cube)
                    
                        
                    module_dir = os.path.dirname(os.path.abspath(__file__))
                    muse_resolution_file = os.path.join(module_dir, 'data', 'Extracted_resolution_muse.fits')
                    with fits.open(muse_resolution_file) as hdul:
                        w_r = np.arange(0,hdul[0].header['NAXIS1'],1) * hdul[0].header['CD1_1'] + hdul[0].header['CRVAL1']
                        r = hdul[0].data                           # spectral resolution FWHM, in Angstroms
                    r_fun = interp1d(w_r, r, kind='linear', bounds_error=False, fill_value='extrapolate')
                    resolution = r_fun(wl)
        
                    dtheta_pix_deg = abs(header['CD1_1'])  # degrees per pixel
                    area_pix_arcsec2 = (dtheta_pix_deg * 3600)**2
        
                    wcs = WCS(header).celestial


                    cube_name = 'muse'
                    ii = 1
                    while cube_name in self.cubes:
                        cube_name = f"muse({ii})"
                        ii += 1
                    
                    self.cubes[cube_name] = Cube(
                        cube=cube,
                        var=var,
                        wl=wl,
                        dw=dw,
                        resolution=resolution,
                        wcs=wcs,
                        dtheta_pix_deg=dtheta_pix_deg,
                        area_pix_arcsec2=area_pix_arcsec2,
                        header=header,
                        id='muse')

                
                    for band in [key for key in map_filter_names if key.startswith('muse')]:

                        filter = Filter(band)
                
                        w_inf = max(wl[0],filter.wavelength_range[0])
                        w_sup = filter.wavelength_range[1]
                        channel_inf = np.digitize(w_inf, wl) - 1
                        channel_sup = np.digitize(w_sup, wl) - 1
                        bandwidth = w_sup - w_inf
                        pivot_wavelength = filter.pivot_wavelength

                        integrated_band = np.nansum(cube[channel_inf:channel_sup, :, :], axis=0)
                        
                
                        image = integrated_band * dw / bandwidth 

                        ZP =  - 2.5 * np.log10(pivot_wavelength**2/2.998e18) - 48.60
                        ZP_err = 0
                        exptime = None

                        image_name = f"{band}"  # e.g., muse_red
                        ii = 1
                        while image_name in self.images:
                            image_name = f"{band}({ii})"
                            ii += 1

                        self.images[image_name] = Image(
                            data=image,
                            wcs=wcs,
                            dtheta_pix_deg=dtheta_pix_deg,
                            area_pix_arcsec2=area_pix_arcsec2,
                            pivot_wavelength=pivot_wavelength,
                            ZP=ZP,
                            ZP_err=ZP_err,
                            exptime=exptime,
                            header=header,
                            filter=filter
                            )
                        
                        

                
     
                    
                    
                    
                    
"""

elif 'uvit' in file.lower():

    print('Processing UVIT file: ',file)

    header = hdul[0].header
    bunit = 'counts_s' 
    header['EXPTIME'] = header['EXP_TIME']
        
    if header['DETECTOR'] == 'FUV' and header['FILTER'] == 'F2': 
        band = 'uvit_fuv_f154w' 
        ZP = 17.778
        ZP_err = 0.01
        pivot_wavelength = 1541    
    elif header['DETECTOR'] == 'NUV' and header['FILTER'] == 'F3':
        band = 'uvit_nuv_n245m' 
        ZP = 18.50
        ZP_err = 0.07
        pivot_wavelength = 2447   
    elif header['DETECTOR'] == 'NUV' and header['FILTER'] == 'F5':
            band = 'uvit_nuv_n263m' 
            ZP = 18.18
            ZP_err = 0.01
            pivot_wavelength = 2632 
    else:
        return None

    wcs = WCS(header)
    dtheta_pix_deg = dtheta_pixel(wcs) 
    area_pix_deg2 = area_pixel(wcs)
    area_pix_arcsec2 = area_pix_deg2 * 3600**2


    position_angle = -header['ROTATN']


    image = hdul[0].data / header['EXP_TIME']   # native units in counts
    image, wcs = cut_and_rotate(image, wcs=wcs, target_coords=self.target_coords, fov=self.fov,position_angle=position_angle)

    AG = self.EBmV * self.MW_extinction(pivot_wavelength)
    print(f'Milky Way dust extinction: {AG:.2f}, {10**(AG / 2.5)}')
    
    flux_conv_counts_s_to_mJy = 1e3 * 10**(-ZP / 2.5 + AG / 2.5 + 8.9 / 2.5)
                

"""
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
