import numpy as np
import random
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import pandas as pd

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter, median_filter
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
    EllipseAnnulusSkyRegion,
    EllipseSkyRegion,
    CircleSkyRegion
)

from .filters import Filter, map_filter_names, nice_filter_names


@dataclass
class Region:
    """
    Generic region class. A region is always represented internally as a mask.
    Use one of the named constructors (from_circle, from_ellipse, from_ellipse_annulus)
    to build a region from sky geometry.
    """
    header: fits.Header = field(default_factory=fits.Header)
    id: str = None
    mask: np.ndarray = None
    wcs: np.ndarray = None
    image_shape: tuple = None
    galactocentric_radius: float = None
    target_bands: np.ndarray = None

    # Aesthetic defaults for plotting
    color: str = 'cyan'
    linestyle: str = '-'
    linewidth: float = 2
    alpha: float = 1
    label: str = None
    caption_coords: tuple = None
    caption_fontsize: int = 10

    # ------------------------------
    # Named constructors
    # ------------------------------
    @classmethod
    def circle(cls, ra_center, dec_center, radius, wcs, image_shape, **kwargs):
        if wcs is None or image_shape is None:
            raise ValueError("WCS and image_shape must be provided for CircleRegion.")

        sky_region = CircleSkyRegion(
            center=SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='fk5'),
            radius=radius * u.deg
        )
        mask = sky_region.to_pixel(wcs).to_mask(mode='center').to_image(image_shape).astype(bool)
        return cls(mask=mask, wcs=wcs, image_shape=image_shape, **kwargs)

    @classmethod
    def ellipse(cls, ra_center, dec_center, semimajor_axis, axis_ratio, angle, wcs, image_shape, **kwargs):
        if wcs is None or image_shape is None:
            raise ValueError("WCS and image_shape must be provided for EllipseRegion.")

        sky_region = EllipseSkyRegion(
            center=SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='fk5'),
            width=2 * semimajor_axis * u.deg,
            height=2 * semimajor_axis * u.deg * axis_ratio,
            angle=angle * u.deg
        )
        mask = sky_region.to_pixel(wcs).to_mask(mode='center').to_image(image_shape).astype(bool)
        return cls(mask=mask, wcs=wcs, image_shape=image_shape, **kwargs)

    @classmethod
    def ellipse_annulus(cls, ra_center, dec_center, semimajor_axis, axis_ratio, angle, dr, wcs, image_shape, **kwargs):
        if wcs is None or image_shape is None:
            raise ValueError("WCS and image_shape must be provided for EllipseAnnulusRegion.")

        if semimajor_axis - dr / 2 <= 0:
            semimajor_axis = dr / 2 + 1e-5  # avoid negative inner radius

        sky_region = EllipseAnnulusSkyRegion(
            center=SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame='fk5'),
            inner_width=2 * (semimajor_axis - dr/2) * u.deg,
            inner_height=2 * (semimajor_axis - dr/2) * u.deg * axis_ratio,
            outer_width=2 * (semimajor_axis + dr/2) * u.deg,
            outer_height=2 * (semimajor_axis + dr/2) * u.deg * axis_ratio,
            angle=angle * u.deg
        )
        mask = sky_region.to_pixel(wcs).to_mask(mode='center').to_image(image_shape).astype(bool)
        return cls(mask=mask, wcs=wcs, image_shape=image_shape, **kwargs)


    # Core function of this class
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
                                
            counter = Counter((np.digitize(y, Y[:, 0])-1, np.digitize(x, X[0, :])-1) for x, y in x_y)  
            normalization_factor = (new_resolution / old_resolution)**2
            for (i, j), n in counter.items():
                mask_pixel_value = min(1, n / normalization_factor)
                if mask_pixel_value > 0.9: mask_pixel_value = 1    # UVIT needs this
                new_mask[i, j] = mask_pixel_value
            new_mask = median_filter(new_mask, size=3)

        
        elif new_resolution < old_resolution:  # HST versus MUSE case
            
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

            X_old = np.clip(XY_old[:, 0].astype(int), 0, self.mask.shape[1] - 1)
            Y_old = np.clip(XY_old[:, 1].astype(int), 0, self.mask.shape[0] - 1)

            new_mask[Y.ravel(), X.ravel()] = self.mask[Y_old, X_old]
        
        return new_mask
    
    

    def plot_aesthetic(self, color=None, linestyle=None, linewidth=None, alpha=None, label=None, caption_coords=None, caption_fontsize=None):
        for key, val in {'color': color, 'linestyle': linestyle, 'linewidth': linewidth, 'alpha': alpha,
                         'label': label, 'caption_coords': caption_coords,
                         'caption_fontsize': caption_fontsize}.items():
            setattr(self, key, val if val is not None else getattr(self, key))
            
            



####################################################################################################################à



            
@dataclass
class Point:
    coords: tuple = None
    color: str = 'white'
    m: str = '+'
    s: float = 50
    caption: str = None
    caption_fontsize: float = 12
    caption_offset: tuple = (0.02,0.02)

    target_bands: np.array=None      # List of bands where to plot point

    def plot_aesthetic(self, color=None, m=None, s=None, caption=None, caption_fontsize=None):
        self.color = color or self.color
        self.m = m or self.m
        self.s = s or self.s
        self.caption = caption if caption is not None else self.caption
        self.caption_fontsize = caption_fontsize or self.caption_fontsize
        self.caption_offset = caption_fontsize or self.caption_offset

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


