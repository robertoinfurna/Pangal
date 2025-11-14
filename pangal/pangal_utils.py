import numpy as np
from dataclasses import dataclass, field
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import SkyRegion, PixCoord, EllipsePixelRegion

from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from skimage.draw import polygon
from skimage import measure
from skimage.measure import find_contours
from skimage.transform import rotate

from astropy.coordinates import Angle
from astropy import units as u


from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from bisect import bisect_left

from copy import deepcopy

from ..image import Image
from ..cube import Cube
from ..region import Region, Point, Contours

# Area of one pixel in deg^2 or arcsec^2 
def area_pixel(wcs,arcsec=False):
    if hasattr(wcs.wcs, 'cd') and wcs.wcs.cd is not None:
        CD = wcs.wcs.cd
    else:
        CD = wcs.pixel_scale_matrix
    det_CD = np.linalg.det(CD)
    area_pix = np.abs(det_CD)
    if arcsec == True:
        area_pix = area_pix * 3600**2
    return area_pix
    
# Side of one pixel in deg or arcsec 
def dtheta_pixel(wcs,arcsec=False):
    area_pix = area_pixel(wcs,arcsec)
    dtheta_pix = np.sqrt(area_pix)
    return dtheta_pix
    

def cut_and_rotate(image,wcs,target_coords=None,fov=None,position_angle=None):
    
    if target_coords and fov:
    
        ra_t, dec_t = target_coords

        # Convert input coordinates to degrees if in string format
        if isinstance(ra_t, str):
            ra_t = Angle(ra_t, unit='hourangle').degree 
        if isinstance(dec_t, str):
            dec_t = Angle(dec_t, unit='deg').degree
            
        # Convert target coords to pixel coordinates
        x_t, y_t = wcs.all_world2pix([[ra_t, dec_t]], 0)[0]

        dtheta_pix_deg = dtheta_pixel(wcs)
    
        fov_pixel = fov / dtheta_pix_deg
        
        # Define cutout window
        x0 = max(0, int(x_t - fov_pixel / 2))
        x1 = min(image.shape[1], int(x_t + fov_pixel / 2))
        y0 = max(0, int(y_t - fov_pixel / 2))
        y1 = min(image.shape[0], int(y_t + fov_pixel / 2))

        image = image[int(y0):int(y1), int(x0):int(x1)]

        wcs.wcs.crpix[0] -= x0
        wcs.wcs.crpix[1] -= y0
        

    # rotation
    if position_angle:
        
        old_shape = image.shape
        image = rotate(image, position_angle, resize=True, mode='constant', cval=-999, order=0, preserve_range=True) 
        new_shape = image.shape
                

        # Fallback to building a CD matrix if not present
        if not hasattr(wcs.wcs, 'cd') or wcs.wcs.cd is None:
            CD = wcs.pixel_scale_matrix
            wcs.wcs.cd = CD
            
        crpix_x = old_shape[1] / 2 
        crpix_y = old_shape[0] / 2 
        crval_ra, crval_dec = wcs.all_pix2world([[crpix_x,crpix_y]],0)[0]

        # overwrite wcs: from now on the wcs is the new wcs
        wcs.wcs.crpix[0] = new_shape[1] / 2 
        wcs.wcs.crpix[1] = new_shape[0] / 2 
        wcs.wcs.crval[0] = crval_ra
        wcs.wcs.crval[1] = crval_dec
                
        rotation_matrix = np.array([[np.cos(np.radians(position_angle)), -np.sin(np.radians(position_angle))],
                                            [np.sin(np.radians(position_angle)),  np.cos(np.radians(position_angle))]])
        wcs.wcs.cd = wcs.wcs.cd @ rotation_matrix    

    return image,wcs


def correct_coords_offset(self, bands, offset):
    """
    Apply a WCS coordinate offset to all bands whose names contain any of the given substrings.
    
    Parameters
    ----------
    bands : str or list of str
        Substring(s) to match band names in `self.images` or `self.cubes`. For example, [sdss_r,galex_nuv] or 'muse' or ['muse'] or ['f200w', 'f356'].
        
    offset : tuple of float
        A tuple (delta_ra, delta_dec) specifying the WCS shift to apply in degrees.
    """
    
    # Validate offset
    if not (isinstance(offset, tuple) and len(offset) == 2 and 
            all(isinstance(v, (float, int)) for v in offset)):
        raise ValueError("offset must be a tuple of two floats (delta_ra, delta_dec) in degrees.")
    
    delta_ra, delta_dec = offset

    # Normalize input: convert to list of substrings to match
    if isinstance(bands, str):
        substrings = [bands]
    elif isinstance(bands, list) and all(isinstance(b, str) for b in bands):
        substrings = bands
    else:
        raise TypeError("bands must be a string or a list of strings.")

    # Find all matching keys in both images and cubes
    matched_bands = []
    for substr in substrings:
        matched_bands.extend([key for key in self.images if substr in key])
        matched_bands.extend([key for key in self.cubes if substr in key])
    matched_bands = list(set(matched_bands))  # remove duplicates

    if not matched_bands:
        raise KeyError(f"No bands matching {bands} found in images or cubes.")

    # Apply WCS offset
    for band in matched_bands:
        for source in (self.images, self.cubes):
            if band in source:
                try:
                    wcs_copy = deepcopy(source[band].wcs)
                    crval = wcs_copy.wcs.crval
                    new_crval = [crval[0] - delta_ra, crval[1] - delta_dec]
                    wcs_copy.wcs.crval = new_crval
                    source[band].wcs = wcs_copy
                    print(f"{band} CRVAL updated by offset (RA: {delta_ra} deg, Dec: {delta_dec} deg). "
                          f"Old CRVAL: {crval} New CRVAL: {new_crval}")
                except AttributeError as e:
                    raise AttributeError(f"WCS is missing or malformed in band '{band}': {e}")
   
   
def mosaic(self, image_list, weight_list=None):
    """
    Combines several Image objects (self.images[..]) in a single mosaic 
    
    Parameters
    ----------
    image_list : list of str
        List of Image objects (stored in self.images)
        Firs image in the list dictates the resolution, wavelength

    weight_list : list of weights
    """
    # Allow single string as keyword
    if isinstance(image_list, str):
        keyword = image_list
        image_list = [k for k in self.images if keyword in k]
        if not image_list:
            raise ValueError(f"No images found in self.images containing '{keyword}'")
        print(f"Mosaicing {len(image_list)} images matching '{keyword}': {image_list}")
    
    elif not isinstance(image_list, list):
        raise ValueError("image_list must be a string or a list of strings")
    
    ## do i still need this?
    for ID in image_list:
        if ID not in self.images.keys():
            raise ValueError("Images must be contained in self.images")
        
    # Check! They must all be similar!
    for ID in image_list:
        ZP = self.images[ID].ZP
        ZP_err = self.images[ID].ZP_err   
        
    dtheta = self.images[image_list[0]].dtheta_pix_deg
    area_pix_arcsec2 = self.images[image_list[0]].area_pix_arcsec2 
    w_pivot = self.images[image_list[0]].pivot_wavelength
    
    min_ra, max_ra, min_dec, max_dec = [], [], [], []

    # Step 1: Determine global RA/Dec bounding box
    for ID in image_list:

        Im = self.images[ID]
        image = Im.image  # suppose all with same bunit
        wcs = Im.wcs
        
        ny, nx = image.shape
        corners = np.array([[0, 0], [0, ny - 1], [nx - 1, 0], [nx - 1, ny - 1]])
        world = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        min_ra.append(world.ra.deg.min())
        max_ra.append(world.ra.deg.max())
        min_dec.append(world.dec.deg.min())
        max_dec.append(world.dec.deg.max())

    ra0, ra1 = min(min_ra), max(max_ra)
    dec0, dec1 = min(min_dec), max(max_dec)
    ny, nx = int((dec1 - dec0) / dtheta), int((ra1 - ra0) / dtheta)

    # Step 2: Create output WCS
    wcs_new = WCS(naxis=2)
    wcs_new.wcs.crval = [(ra0 + ra1) / 2, (dec0 + dec1) / 2]
    wcs_new.wcs.cdelt = [-dtheta, dtheta]
    wcs_new.wcs.crpix = [nx / 2, ny / 2]
    wcs_new.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    mosaic_image = np.zeros((ny, nx))
    weight = np.zeros((ny, nx))  # To count valid contributions
    
    for ID in image_list:

        Im = self.images[ID]
        image = Im.image  # suppose all with same bunit
        wcs = Im.wcs
        
        ny_in, nx_in = image.shape
        x, y = np.meshgrid(np.arange(nx_in), np.arange(ny_in))
        
        sky = wcs.pixel_to_world(x, y)
        ra = sky.ra.deg
        dec = sky.dec.deg
    
        h, k = wcs_new.world_to_pixel_values(ra, dec)
        h = h.astype(int)
        k = k.astype(int)
    
        # Valid if inside bounds and image is not NaN
        valid = (
            (h >= 0) & (h < nx) &
            (k >= 0) & (k < ny) &
            ~np.isnan(image)
        )
    
        np.add.at(mosaic_image, (k[valid], h[valid]), image[valid])
        np.add.at(weight, (k[valid], h[valid]), 1)
    
    # Avoid division by zero
    #with np.errstate(invalid='ignore', divide='ignore'):
    #    mosaic_image /= np.where(weight == 0, np.nan, weight)
        
    mosaic_Image = Image(
        image=mosaic_image,
        wcs=wcs_new,
        dtheta_pix_deg=dtheta,
        area_pix_arcsec2=area_pix_arcsec2,
        pivot_wavelength=w_pivot,
        ZP=ZP,
        ZP_err=ZP_err,
        #header = header
    )
    
    #name = "_".join(Image_list) + "_maic" #This must go in the header
    
    # Remove the original component images
    #for im_name in image_list:
    #    self.images.pop(im_name, None)  # Safe removal
    
    name = image_list[0]
    self.images[name] = mosaic_Image 

    return mosaic_image   
             






# FIX THIS FUNCTION!!!
def MW_extinction(self,lam,plot=False):
    # Return Milky Way extinction curve A(lam)/E(B-V) via interpolation.
    anchor_points = np.array((1111, 1176, 1250, 1316, 1393, 1490, 1600, 1701, 1799, 1901, 2000,
                                  2101, 2188, 2299, 2398, 2500, 2740, 3436, 4000, 4405, 5495,
                                  6993, 9009, 12500, 16390, 22200))
    anchor_values = np.array((11.53, 10.53, 9.63, 8.99, 8.44, 8.07, 7.81, 7.68, 7.73, 7.98, 8.60,
                                  9.31, 9.65, 8.85, 7.98, 7.27, 6.18, 4.88, 4.38, 4.08, 3.08,
                                  2.30, 1.48, 0.85, 0.50, 0.32))
    interpolator = interp1d(anchor_points, anchor_values, kind='cubic',fill_value="extrapolate")

    if plot==False:
        return interpolator(lam)
    else:
        wavelengths = np.linspace(1000, 23000, 1000)  # Angstroms
        extinction = interpolator(wavelengths)

        fig,ax = plt.subplots(figsize=(8, 5))

        ax.plot(wavelengths, extinction, color='tab:blue', lw=2)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Extinction')
        ax.set_title(f'Milky Way Extinction Curve, RA: {self.target_coords[0]}, Dec: {self.target_coords[1]}')
        ax.text(0.02,0.9,f'$E(B-V)=${self.EBmV}',transform=ax.transAxes)
        plt.tight_layout()
        plt.show()




