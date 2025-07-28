import numpy as np
import random
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skimage.draw import polygon
from skimage import measure
from skimage.measure import find_contours
from skimage.transform import rotate

from regions import (
    PixCoord,
    SkyRegion,
    CirclePixelRegion,
    EllipsePixelRegion,
    EllipseAnnulusPixelRegion,
    EllipseSkyRegion,
    CircleSkyRegion
)

from .filters import Filter, map_filter_names, nice_filter_names

@dataclass
class Image:
    image: np.ndarray=None                              # image in native units
    bunit: str=None
    wcs: WCS=None                                       # Astropy WCS object
    dtheta_pix_deg: float=None                          # Pixel scale (degrees/pixel)
    area_pix_arcsec2: float=None                        # Pixel area (arcsec²)
    
    pivot_wavelength: float=None                        # Pivot wavelength (Angstroms)
    ZP: float=None                                      # ZP
    ZP_err: float=None                                  # ZP error
    flux_conv_counts_s_to_mJy: float=None               # Flux conversion factor, from counts_s to mJy or from erg/s/cm2/A to mJy
    
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    filter: Filter=None
    
    @property
    def counts_s(self):
        if self.bunit == 'counts_s':
            return self.image

    @property
    def mJy(self):
        if self.bunit == 'mJy':
            return self.image
        elif self.bunit == 'counts_s' and self.flux_conv_counts_s_to_mJy:
            return self.image * self.flux_conv_counts_s_to_mJy                   
        elif self.bunit == 'erg_s_cm2_A':
            return self.image * self.pivot_wavelength**2 / 3.99e18 / 1e-26 #23                                        
        else:                                                     
            raise ValueError("No data available for specific flux calculation.")

        
    @property
    def mJy_arcsec2(self):
        return self.mJy / self.area_pix_arcsec2

    @property
    def MJy_sr(self):
        return self.mJy / self.area_pix_arcsec2 / 0.0235045

    @property
    def mag(self):
        return -2.5 * np.log10(self.mJy /1000 / 3631)

    @property
    def mag_arcsec2(self):
        return -2.5 * np.log10(self.mJy_arcsec2 /1000 / 3631)

    @property
    def erg_s_cm2_A(self):
        if self.bunit == 'erg_s_cm2_A':
            return self.image
        elif self.bunit == 'counts_s' and self.flux_conv_counts_s_to_mJy:
            return self.image * self.flux_conv_counts_s_to_mJy  
        elif self.bunit == 'mJy':
            return self.image / self.pivot_wavelength**2 * 3.99e18 * 1e-26     
        else:                                                     
            raise ValueError("No data available for specific flux calculation.")
    
    @property
    def erg_s_cm2_A_arcsec2(self):
        return self.erg_s_cm2_A / self.area_pix_arcsec2                       
    
    
    @property
    def erg_s_cm2(self):
        if self.bunit == 'erg_s_cm2':
            return self.image  
        else:                                                     
            raise ValueError("No data available for integrated flux calculation.")
    
    @property
    def erg_s_cm2_arcsec2(self):
        return self.erg_s_cm2 / self.area_pix_arcsec2                                        

           
@dataclass
class Spectrum:
    wl: np.ndarray=None                                       # wavelength
    resolution: np.ndarray=None                               # spectral resolution
    flux: np.ndarray=None                                     # flux
    flux_err:  np.ndarray=None                                # flux error
    header: fits.Header = field(default_factory=fits.Header)  # FITS header (IF SPECTRUM IS DOWNLOADED FROM AN ARCHIVE, or if a model)

    @property
    def res_delta_lambda(self):
        """
        delta lambda (lambda): spectral resoolution
        """    
        return self.wl / self.resolution         
            




@dataclass
class Cube:
    cube: np.ndarray=None                              # cube in native units
    var: np.ndarray=None                               # var in native units
    units: float=None                                  # convert to erg/s/cm2/A
    wl: np.array=None                                   # wavelength array
    resolution: np.array=None                                   # resolution array
    dw: float=None                                     # spectral channel width                     
    wcs: WCS=None                                       # Astropy WCS object
    dtheta_pix_deg: float=None                          # Pixel scale (degrees/pixel)
    area_pix_arcsec2: float=None                        # Pixel area (arcsec²)

    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    
    @property
    def delta_lambda(self):
        """
        delta lambda (lambda): spectral resoolution
        """
        return self.wl / self.resolution  
    
    
    
@dataclass
class Region:
    # Either a mask or a str ('ellipse','circle','elliptical_annulus') can be provided
    # Final result is always a mask
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    id: str=None                                    # id
    sky_region: str=None                            # string
    mask: np.ndarray=None                           # Native mask 
    wcs: np.ndarray=None                            # Native image wcs
    galactocentric_radius: float=None               # if provided, distance of region centroid from the center of the galaxy, in kpc
    image_shape: tuple=None
    
    # Values for astropy regions
    ra_center: float=None
    dec_center: float=None
    radius: float=None
    semimajor_axis: float=None
    axis_ratio: float=None
    dr: float=None
    angle: float=None
    
    # aesthetic defaults for plotting
    color: str = 'cyan'
    linestyle: str = '-'
    linewidth: float = 2
    alpha: float = 1
    label: str = None
    caption_coords: tuple = None
    caption_fontsize: int = 10

    def __post_init__(self):
        if self.mask is None and self.sky_region is None:
            raise ValueError("Must provide either a 'sky_region' or a 'mask'.")
        
        # If a mask is provided directly, use it
        if self.mask is not None:
            self.mask = self.mask.astype(bool)
            
        elif self.sky_region is not None:
            if self.wcs is None or self.image_shape is None:
                raise ValueError("WCS and image_shape must be provided to convert SkyRegion to mask.")

            ### --- Circle ---
            if self.sky_region == 'Circle':
                
                if self.ra_center is None or self.dec_center is None or self.radius is None:
                    raise ValueError(
                        "To define a Circle region, you must provide the following parameters: "
                        "ra_center, dec_center, and radius."
                    )
                
                sky_region = CircleSkyRegion(center=SkyCoord(ra=self.ra_center * u.deg, dec=self.dec_center * u.deg, frame='fk5'),
                                radius=self.radius * u.deg)

                self.mask = sky_region.to_pixel(self.wcs).to_mask(mode='center').to_image(self.image_shape).astype(bool)
 
            ### --- Ellipse ---
            elif self.sky_region == 'Ellipse':
                
                if self.ra_center is None or self.dec_center is None or self.semimajor_axis is None or self.axis_ratio is None or self.angle is None:
                    raise ValueError(
                        "To define an Ellipse region, you must provide the following parameters: "
                        "ra_center, dec_center, semimajor_axis, axis_ratio and angle."
                    )
                    
                sky_region = EllipseSkyRegion(center=SkyCoord(ra=self.ra_center*u.deg, dec=self.dec_center*u.deg, frame='fk5'),
                                  width=2*self.semimajor_axis * u.deg ,          
                                  height=2*self.semimajor_axis * u.deg * self.axis_ratio,        
                                  angle=self.angle * u.deg)

                self.mask = sky_region.to_pixel(self.wcs).to_mask(mode='center').to_image(self.image_shape).astype(bool)

            ### --- Ellipse Annulus---
            elif self.sky_region == 'EllipseAnnulus':
                
                if self.ra_center is None or self.dec_center is None or self.semimajor_axis is None or self.axis_ratio is None or self.angle is None or self.dr is None:
                    raise ValueError(
                        "To define an Ellipse region, you must provide the following parameters: "
                        "ra_center, dec_center, semimajor_axis, axis_ratio and width dr (in deg)"
                    )

                if self.semimajor_axis - self.dr/2 <= 0:
                    self.semimajor_axis = self.dr/2 + 0.00001 ###

                sky_region = EllipseAnnulusSkyRegion(
                    center=SkyCoord(ra=self.ra_center * u.deg, dec=self.dec_center * u.deg, frame='fk5'),
                    inner_width=2 * (self.semimajor_axis - self.dr/2) * u.deg,
                    inner_height=2 * (self.semimajor_axis - self.dr/2) * u.deg * self.axis_ratio,
                    outer_width=2 * (self.semimajor_axis + self.dr/2) * u.deg,
                    outer_height=2 * (self.semimajor_axis + self.dr/2) * u.deg * self.axis_ratio,
                    angle=self.angle * u.deg
                )
            
                self.mask = sky_region.to_pixel(self.wcs).to_mask(mode='center').to_image(self.image_shape).astype(bool)

            else:
                raise ValueError(
                        "sky region must be 'Circle', 'Ellipse' or 'ELlipseAnnulus'"
                    )


    def project(self,new_image,new_wcs):

        old_resolution = proj_plane_pixel_scales(self.wcs)[0]  # or 1? If your image has square pixels (which is common), both are equal and it doesn't matter.
        new_resolution = proj_plane_pixel_scales(new_wcs)[0]   
        
        # get mask pixel coordinates ra dec
        y_indices, x_indices = np.where(self.mask)
        ra_dec = self.wcs.all_pix2world(np.column_stack((x_indices, y_indices)), 0)
            
        # Convert RA/Dec to pixel coordinates of the new image
        x_y = new_wcs.all_world2pix(ra_dec, 0)
        new_mask = np.zeros_like(new_image, dtype=float)

        # If WCS are the same, return original mask
        if self.wcs.wcs.compare(new_wcs.wcs):  # compare WCS objects directly
            new_mask = self.mask.astype(float)
        
        elif new_resolution >= old_resolution: 
    
            x_grid = np.arange(new_image.shape[1])
            y_grid = np.arange(new_image.shape[0])
            X, Y = np.meshgrid(x_grid, y_grid)
                                
            counter = Counter((np.digitize(y, Y[:, 0]), np.digitize(x, X[0, :])) for x, y in x_y)  #-1  #-1
            max_count = max(counter.values()) if counter else 1
            for (i, j), n in counter.items():
                new_mask[i, j] = n / max_count

        
        elif new_resolution < old_resolution: # and self.mask:  # HST versus MUSE case
            
            ra_values = ra_dec[:, 0]   # of low resolution mask
            dec_values = ra_dec[:, 1]  # of low resolution mask
            ra_min_mask, ra_max_mask = np.min(ra_values), np.max(ra_values)
            dec_min_mask, dec_max_mask = np.min(dec_values), np.max(dec_values)
            (x0, y0), (x1, y1) = new_wcs.all_world2pix([[ra_max_mask, dec_min_mask], [ra_min_mask, dec_max_mask]], 0)

            x_vals = np.arange(int(x0), int(x1))
            y_vals = np.arange(int(y0), int(y1))
            X, Y = np.meshgrid(x_vals, y_vals)
            XY = np.column_stack([X.ravel(), Y.ravel()])

            RA_DEC = new_wcs.all_pix2world(XY, 0)

            XY_old = self.wcs.all_world2pix(RA_DEC, 0)

            X_old = np.clip(XY_old[:, 0].astype(int), 0, self.mask.shape[1]) #-1
            Y_old = np.clip(XY_old[:, 1].astype(int), 0, self.mask.shape[0]) #-1

            new_mask[Y.ravel(), X.ravel()] = self.mask[Y_old, X_old]
        
        return new_mask

    def plot_aesthetic(self, color=None, linestyle=None, linewidth=None, alpha=None, label=None, name=None, caption_coords=None, caption_fontsize=None):
        for key, val in {'color': color, 'linestyle': linestyle, 'linewidth': linewidth, 'alpha': alpha,
                         'label': label, 'name': name, 'caption_coords': caption_coords,
                         'caption_fontsize': caption_fontsize}.items():
            setattr(self, key, val if val is not None else getattr(self, key))
            
            
            
@dataclass
class Point:
    coords: tuple = None
    color: str = 'white'
    m: str = '+'
    s: float = 50
    caption: str = None
    caption_fontsize: float = None

    target_bands: np.array=None      # List of bands where to plot point

    def plot_aesthetic(self,color=None,m=None,s=None,caption=None,caption_fontsize=None):
        # If the argument is None, use the existing value of the attribute (self.attribute)
        if color is None: color = self.color
        if m is None: m = self.m
        if s is None: s = self.s
        if name is None: name = self.caption
        if caption_fontsize is None: caption_fontsize = self.caption_fontsize

        # Update the class attributes with the new (or unchanged) values
        self.color = color
        self.m = m
        self.s = s
        self.caption = caption
        self.caption_fontsize = caption_fontsize
        
        
@dataclass
class Contours:
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    image: np.ndarray=None                              # image to compute contours on 
    wcs: WCS=None                                       # Astropy WCS object of the image
    levels: np.array=None                               

    target_bands: np.array=None                         # List of bands where to plot contours
    
    # aesthetic defaults for plotting
    color: str = 'red'
    linestyle: str = '-'
    linewidth: float = 2
    alpha: float = 1
    label: str = None
    clabel_fmt: str = None
    clabel_fontsize: float = 12


    def plot_aesthetic(self, color=None, linestyle=None, linewidth=None, alpha=None, label=None, clabel_fmt=None, clabel_fontsize=None):
        for key, val in {'color': color, 'linestyle': linestyle, 'linewidth': linewidth, 'alpha': alpha,
                         'label': label, 'clabel_fmt': clabel_fmt, 'clabel_fontsize': clabel_fontsize}.items():
            setattr(self, key, val if val is not None else getattr(self, key))


