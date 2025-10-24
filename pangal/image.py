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
    EllipseSkyRegion,
    CircleSkyRegion
)

from .filters import Filter, map_filter_names, nice_filter_names

@dataclass
class Image:
    image: np.ndarray=None                              # image in native units
    wcs: WCS=None                                       # Astropy WCS object
    dtheta_pix_deg: float=None                          # Pixel scale (degrees/pixel)
    area_pix_arcsec2: float=None                        # Pixel area (arcsec²)
    pivot_wavelength: float=None                        # Pivot wavelength (Angstroms)
    ZP: float=None                                      # ZP
    ZP_err: float=0                                     # ZP error
    exptime: float=None
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    filter: Filter=None
    
    @property
    def mag(self):
        return self.ZP - 2.5 * np.log10(self.image)

    @property
    def mag_arcsec2(self):
        return self.ZP - 2.5 * np.log10(self.image / self.area_pix_arcsec2)

    @property
    def Jy(self):
        return 10**(0.4*(-self.ZP+8.90)) * self.image
        
    @property
    def Jy_arcsec2(self):
        return self.Jy / self.area_pix_arcsec2
    
    @property
    def mJy(self):
        return 1e3 * self.Jy
        
    @property
    def mJy_arcsec2(self):
        return self.mJy / self.area_pix_arcsec2

    @property
    def MJy_sr(self):
        return self.mJy / self.area_pix_arcsec2 / 0.0235045

    @property
    def erg_s_cm2_A(self):
        return 1e-23 * 2.998e18/self.pivot_wavelength**2 * self.Jy
         
    @property
    def erg_s_cm2_A_arcsec2(self):
        return self.erg_s_cm2_A / self.area_pix_arcsec2                       
    
    
    # This is problematic!
    @property
    def erg_s_cm2(self):
        return self.image
                                             
        #raise ValueError("No data available for integrated flux calculation.")
    
    @property
    def erg_s_cm2_arcsec2(self):
        return self.erg_s_cm2 / self.area_pix_arcsec2    


    """
    def plot(
            units=None,
            scale_lims,
            colorbar, 
            ...
    ):                      
    
           
    if units is None:
    """











