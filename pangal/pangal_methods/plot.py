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


from ..base import Image, Cube, Region, Point, Contours
from ..filters import map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps



# INTERACTIVE SKY COORDS NOT WORKING!!
# BBOX STILL BELOW COLORBAR!


default_cmaps = {
    'galex_fuv': 'gnuplot', 'galex_nuv': 'gnuplot2', 
    'muse_blue': 'bone', 'muse_green': 'viridis','muse_red': 'hot', 'muse_nir': 'copper', 'muse_white_aoe': 'gray',
    'muse_Ha': 'plasma', 'muse_Hb': 'inferno', 'muse_SII': 'viridis', 
    **{band: 'gray' for band in map_filter_names if band.startswith('hst')},
    'sdss_u': 'Blues_r','sdss_g': 'Greens_r', 'sdss_r': 'Reds_r', 'sdss_i': 'YlOrBr_r', 'sdss_z': 'Greys_r',
    '2mass_j': 'PuRd_r', '2mass_h': 'BuPu_r', '2mass_ks': 'Oranges_r', 
    **{f'wise_w{band}': 'viridis' for band in ['1','2','3','4']},
    **{band: 'nipy_spectral' for band in map_filter_names if band.startswith('spitzer_irac')},
    'herschel_pacs_70um': 'Blues_r', 'herschel_pacs_100um': 'Greens_r', 'herschel_pacs_160um': 'Reds_r'
    }
    
default_filter_colors = { 
    'galex_fuv': 'violet', 'galex_nuv': 'darkblue', 
    'muse_blue': 'blue', 'muse_green': 'green', 'muse_red': 'red', 'muse_nir': 'gold', 'muse_white_aoe':'gray',
    'muse_Ha': 'orange', 'muse_Hb': 'purple', 'muse_sii': 'lime',
    **{band: 'purple' for band in map_filter_names if band.startswith('hst')},
    'sdss_u': 'midnightblue', 'sdss_g': 'cyan', 'sdss_r': 'orangered', 
    'sdss_i': 'red', 'sdss_z': 'darkred',
    '2mass_j': 'pink', '2mass_h': 'purple', '2mass_ks': 'darkviolet',
    **{f'wise_w{band}': 'brown' for band in ['1','2','3','4']},
    'spitzer_irac_1': 'teal', 'spitzer_irac_2': 'lightgreen', 
    'spitzer_irac_3': 'orange', 'spitzer_irac_4': 'darkred',
    'herschel_pacs_70um': 'lime', 'herschel_pacs_100um': 'lime', 'herschel_pacs_160um': 'lime'
}    

        
default_plot_units = {
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('galex')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('uvit')},
    'muse_blue': 'mag_arcsec2','muse_green': 'mag_arcsec2', 'muse_red': 'mag_arcsec2', 'muse_nir': 'mag_arcsec2', 'muse_white_aoe': 'mag_arcsec2',
    'muse_Ha': 'erg_s_cm2_arcsec2', 'muse_Hb': 'erg_s_cm2_arcsec2',
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('hst')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('sdss')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('2mass')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('wise')},
    **{band: 'MJy_sr' for band in map_filter_names if band.startswith('spitzer')},
    **{band: 'MJy_sr' for band in map_filter_names if band.startswith('herschel')},
}

default_plot_scale_lims = {
    'FUV': (22, 27), 'NUV': (22, 27), 
    'galex_fuv': (22, 27), 'galex_nuv': (22, 27), 
    'muse_blue': (20, 27),'muse_green': (20, 27), 'muse_red': (20, 27), 'muse_nir': (20, 27), 'muse_white_aoe': (20,27),
    'muse_Ha': (1e-19, 1e-14), 'muse_Hb': (1e-19, 1e-14),
    **{band: (19, 27) for band in map_filter_names if band.startswith('hst')},
    **{f'sdss_{band}': (20, 27) for band in ['u', 'g', 'r', 'i', 'z']},
    '2mass_j': (15, 22), '2mass_h': (15, 22), '2mass_ks': (15, 21), 
    **{f'wise_w{band}': (15, 22) for band in ['1','2','3','4']},
    'spitzer_irac_1': (6e-2, 10), 'spitzer_irac_2': (1e-1, 10), 'spitzer_irac_3': (1, 10), 'spitzer_irac_4': (5, 100),
    'herschel_pacs_70um': (10, 100), 'herschel_pacs_100um': (10, 100), 'herschel_pacs_160um': (10, 100)
}
         


def plot(self,
        bands = None,                            # List of bands to be plotted. If None, all bands are plotted

        regions=[],                              # List of Region objects
        contours=[],                             # List of Contour objects
        points=[],                               # List of Point objects
                 
        set_cmaps: dict | None          = None,
        set_units: dict | None          = None,
        set_scale_lims: dict | None     = None,
                 
        # --- Coordinate cuts (world coordinates)
        ra_min=None, ra_max=None,             # RA limits (deg or sexagesimal string, e.g., '12:59:00.6')
        dec_min=None, dec_max=None,           # Dec limits (deg or sexagesimal string, e.g., '28:07:35.0')
    
        # --- Display options
        plots_per_row=3,                       # Number of plots per row
        figsize=None,                          # Overall figure size (width, height) in inches
    
        # --- Axis ticks
        n_xticks=5, n_yticks=5,                 # Number of ticks along x and y axes
        axis_label_fontsize=12,                      # Font size for axis labels
        axis_ticks_fontsize=10,                      # Font size for tick labels

        # ---cbar ticks
        show_cbar = True,
        cbar_label_fontsize=12,
        cbar_ticks_fontsize=11,
        
        auto_titles=True,                        # Enables authomatic titles 
        auto_title_fontsize=12,
        auto_title_color='black',
        auto_titles_style=0,                     # Autotitles style
        autotitle_loc = [0.03, 0.97],
        
        titles_dict=[],                          # List of manual titles

        legend = True,
        legend_loc = 'lower right',
        legend_fontsize = 14,
        show_region_captions = True, 
        
    ):

    # Bands 
    if bands is None: bands = list(self.images.keys())             
    n_images = len(bands)

    # ---- Create the plot ----    
    if figsize is None: figsize=(8*plots_per_row, 8*np.ceil(n_images/plots_per_row))
    
    if n_images == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax = [ax]
    else:
        nrows = int(np.ceil(n_images / plots_per_row))
        fig, ax = plt.subplots(nrows, plots_per_row, figsize=figsize)
        ax = ax.flatten()


    # ---- CMAPS, UNITS AND SCALE LIMS SETTINGS

    cmaps = {**default_cmaps,**(set_cmaps or {})}
    plot_units = {**default_plot_units,**(set_units or {})}
    plot_scale_lims = set_scale_lims or {}
    

    # ---- prepare a legend
    legend_handles = [[] for b in bands]


    # ---- Loop through images ----   
    for i, band in enumerate(bands):
    
        # manages multiple images of the same band (mosaics)
        image_id = band
        if '(' in image_id:
            idx = image_id.find('(')
            band = image_id[:idx]

        units = plot_units[band]

        if units == 'mag_arcsec2':
            image = self.images[image_id].mag_arcsec2
            label = 'mag/arcsec$^2$'
        elif units == 'Jy':
            image = self.images[image_id].Jy
            label = 'Jy/pixel'
        elif units == 'MJy_sr':
            image = self.images[image_id].MJy_sr
            label = 'MJy/sr'
        elif units == 'erg_s_cm2_A_arcsec2':  # integrated line flux
            image = self.images[image_id].erg_s_cm2_arcsec2
            label = 'erg/s/cm$^2$/Å/arcsec$^2$'
        elif units == 'erg_s_cm2_arcsec2':  # integrated line flux
            image = self.images[image_id].erg_s_cm2_arcsec2
            label = 'erg/s/cm$^2$/arcsec$^2$'
        else:
            print(f"Warning: No recognized measurement found for {units}. Skipping.")
            break

        if band in plot_scale_lims:
            vmin, vmax = plot_scale_lims[band]
        else:
            finite_vals = image[np.isfinite(image)]
            if units != 'mag_arcsec2':
                vmin, vmax = np.nanquantile(finite_vals[finite_vals>0], 0.01), np.nanquantile(finite_vals, 0.9999)
            else:
                vmin, vmax = np.nanquantile(finite_vals, 0.0001), np.nanquantile(finite_vals, 0.9)
        
        ratio = vmax / vmin
        
        # Choose normalization
        if ratio >= 100:  # large dynamic range
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:  # small range, linear works better
            norm = Normalize(vmin=vmin, vmax=vmax)
            
        cmap = plt.get_cmap(cmaps[band])
        min_color = cmap(0.0)           # 0.0 corresponds to the minimum end of the colormap
        cmap.set_bad(color=min_color)   # set bad pixels to appear as minimum color
        if units == 'mag_arcsec2': cmap = cmap.reversed()

        
        im = ax[i].imshow(image, origin='lower', cmap=cmap, norm=norm, interpolation='none')
        if show_cbar:
            cbar = fig.colorbar(im, ax=ax[i], location='right', shrink=0.6, pad=0.01)
            cbar.ax.tick_params(labelsize=cbar_label_fontsize)
            cbar.set_label(label, fontsize=cbar_ticks_fontsize)
                
        wcs = self.images[image_id].wcs
                
        # ---- Ticks and Labels ----
        add_ra_dec_ticks(ax[i], image, wcs, n_xticks, n_yticks,axis_ticks_fontsize, axis_label_fontsize,ra_min, ra_max, dec_min, dec_max)
        
        # ---- Regions ----
        # This works for masks 
        for Reg in regions:
            ny, nx = Reg.mask.shape
            ra_max_reg, dec_min_reg = Reg.wcs.all_pix2world([[0, 0]], 0)[0]
            ra_min_reg, dec_max_reg = Reg.wcs.all_pix2world([[nx, ny]], 0)[0]
            x_min, y_min = wcs.all_world2pix([[ra_max_reg, dec_min_reg]], 0)[0]
            x_max, y_max = wcs.all_world2pix([[ra_min_reg, dec_max_reg]], 0)[0]
                
            X_low = np.linspace(x_min, x_max, Reg.mask.shape[1])
            Y_low = np.linspace(y_min, y_max, Reg.mask.shape[0])
            X_low, Y_low = np.meshgrid(X_low, Y_low)
            ax[i].contour(X_low,Y_low,Reg.mask,levels=[0.5],
                colors=Reg.color,linestyles=Reg.linestyle,linewidths=Reg.linewidth,alpha=Reg.alpha) 

            if Reg.caption_coords and (Reg.id or 'id' in Reg.header.keys() or 'ID' in Reg.header.keys()) and show_region_captions:
                x, y = wcs.all_world2pix([[Reg.caption_coords[0], Reg.caption_coords[1]]], 0)[0]
                ax[i].text(x, y, Reg.id, color='white',fontsize=Reg.caption_fontsize, ha='center', va='center', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            if Reg.label:
                    handle = Line2D([0], [0], color=Reg.color, linestyle=Reg.linestyle, linewidth = Reg.linewidth, alpha=Reg.alpha, label=Reg.label)
                    legend_handles[i].append(handle)


        
    # ---- Contours -----
    for Con in contours:

        image = gaussian_filter(Con.image, sigma=2.5)
        (ra1, dec0), (ra0, dec1) = Con.wcs.all_pix2world([[0, 0], [image.shape[1], image.shape[0]]], 0)

        target_bands = Con.target_bands if Con.target_bands else bands
            
        for band in target_bands:
            j = bands.index(band)
            target_wcs = self.images[band].wcs

                
            (x0, y0), (x1, y1) = target_wcs.all_world2pix([[ra1, dec0], [ra0, dec1]], 0)

            X = np.linspace(x0, x1, image.shape[1])
            Y = np.linspace(y0, y1, image.shape[0])
            X, Y = np.meshgrid(X, Y)

            cs = ax[j].contour(X, Y, image, levels=Con.levels, colors=Con.color, linewidths=Con.linewidth, alpha=Con.alpha)
                    
            if Con.clabel_fmt:
                ax[j].clabel(cs, inline=True, fontsize=Con.clabel_fontsize, fmt=Con.clabel_fmt, colors=Con.color)

            if Con.label:
                    handle = Line2D([0], [0], color=Con.color, linestyle=Con.linestyle, linewidth=Con.linewidth, alpha=Con.alpha, label=Con.label)
                    legend_handles[j].append(handle)

    # ---- Scatter points (RA/Dec to pixels) ----
    for P in points:
        coords = P.coords
        target_bands = P.target_bands if P.target_bands else bands
        for band in target_bands:
            j = bands.index(band)
            wcs = self.images[band].wcs
            x, y = wcs.all_world2pix([[coords[0], coords[1]]], 0)[0]

            ax[j].scatter(x, y, c=P.color, marker=P.m, s=P.s)
            if P.caption:
                x_ax, y_ax = ax[j].transData.transform((x, y))  # pixels in figure
                x_ax, y_ax = ax[j].transAxes.inverted().transform((x_ax, y_ax))  # 0-1 in axes
                ax[j].text(x_ax+P.caption_offset[0], y_ax+P.caption_offset[1], P.caption,color=P.color, fontsize=P.caption_fontsize,
                           ha='center', va='center', fontweight='bold',transform=ax[j].transAxes)

    # ---- Legend ----
    if legend:
        for i in range(len(bands)):
            if legend_handles[i]:
                ax[i].legend(handles=legend_handles[i], loc=legend_loc, fontsize=legend_fontsize)

    # ---- Authomatic titles ----
    for i, band in enumerate(bands):
        if auto_titles and not titles_dict:       
            if auto_titles_style == 0:     
                ax[i].text(autotitle_loc[0], autotitle_loc[1], nice_filter_names[band], 
                           color=auto_title_color, fontsize=auto_title_fontsize, 
                           ha='left', va='top', 
                           transform=ax[i].transAxes,
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            if auto_titles_style == 1:     
                ax[i].text(autotitle_loc[0], autotitle_loc[1], nice_filter_names[band], 
                           color=auto_title_color, fontsize=auto_title_fontsize, fontweight='bold',
                           ha='left', va='top', 
                           transform=ax[i].transAxes)

    # ---- Manual titles
    for title_dict in titles_dict: 
        band = next(b for b in title_dict if b in bands)
        title = title_dict[band]
        j = bands.index(band)
        title_location = title_dict.get('location', [0.02, 0.98])
        title_color = title_dict.get('color', 'black')
        title_fontsize = title_dict.get('fontsize', 14)
        title_ha = title_dict.get('ha', 'left')
        title_va = title_dict.get('va', 'top')
        ax[j].text(title_location[0], title_location[1], title,
                   color=title_color, fontsize=title_fontsize,
                    ha=title_ha, va=title_va, transform=ax[j].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        
    for i in range(n_images, len(ax)):
         ax[i].axis('off')

    plt.tight_layout()
    plt.show()




def add_ra_dec_ticks(ax,           # matplotlib axis on which to add ticks
                     image,        # 2D numpy array representing the image (must be aligned with RA/Dec grid)
                     wcs,          # astropy WCS object corresponding to the image
                     n_xticks,     # int; number of tick marks along the RA (x) axis
                     n_yticks,     # int; number of tick marks along the Dec (y) axis
                     ticks_fontsize,   # int or float; font size to use for tick labels
                     label_fontsize,   # int or float; font size to use for the axis labels (e.g., 'RA (deg)', 'Dec (deg)')
                     ra_min_cut=None, ra_max_cut=None,  # optional; strings for RA limits (in hourangle format, e.g., '12:59:00.0'); if provided, used to crop the axis
                     dec_min_cut=None, dec_max_cut=None # optional; strings for Dec limits (in sexagesimal or decimal format, e.g., '28:07:35.0'); if provided, used to crop the axis
                     ):
    """
    Adds right ascension (RA) and declination (Dec) tick marks and labels to a WCS-aligned image.
    
    If the RA/Dec cut limits (ra_min_cut, ra_max_cut, dec_min_cut, dec_max_cut) are provided,
    the displayed portion of the image is cropped to these limits by adjusting the x/y axis limits.
    """

    hourangle = False
    
    ### Find x0,y0 and x1,y1, the corners of the image   
    if all([ra_min_cut, ra_max_cut, dec_min_cut, dec_max_cut]):
        if any(isinstance(val, str) for val in [ra_min_cut, ra_max_cut, dec_min_cut, dec_max_cut]):
            hourangle = True
            # convert to degrees (for wcs)
            ra_min_cut = Angle(ra_min_cut, unit='hourangle').degree
            ra_max_cut = Angle(ra_max_cut, unit='hourangle').degree
            dec_min_cut = Angle(dec_min_cut, unit='deg').degree
            dec_max_cut = Angle(dec_max_cut, unit='deg').degree

        pix_coords = wcs.all_world2pix([[ra_max_cut, dec_min_cut], [ra_min_cut, dec_max_cut]], 1)
        (x0, y0), (x1, y1) = pix_coords 
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        x0,y0 = 0,0
        y1,x1 = image.shape
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

    
    ### Find corresponding ra dec
    ra_max, dec_min = wcs.all_pix2world([[x0, y0]], 1)[0]  # Get RA, DEC
    ra_min, dec_max = wcs.all_pix2world([[x1, y1]], 1)[0]

    ra_ticks = np.linspace(ra_max, ra_min, n_xticks+2)
    dec_ticks = np.linspace(dec_min, dec_max, n_yticks+2) 
    
    ra_span = abs(ra_max - ra_min)
    dec_span = abs(dec_max - dec_min)

    ra_prec = dynamic_precision(ra_span)
    dec_prec = dynamic_precision(dec_span)

    if hourangle:
        ra_ticks_str = Angle(ra_ticks, unit='deg').to_string(unit='hour', sep=':', precision=ra_prec)
        dec_ticks_str = Angle(dec_ticks, unit='deg').to_string(unit='deg', sep=':', precision=dec_prec)
    else:
        ra_ticks_str = [f'{ra:.{ra_prec}f}' for ra in ra_ticks]
        dec_ticks_str = [f'{dec:.{dec_prec}f}' for dec in dec_ticks]


    ra_ticks_str[0], ra_ticks_str[-1] = '', ''
    dec_ticks_str[0], dec_ticks_str[-1] = '', ''

    ax.set_aspect('equal')

    ax.set_xticks(np.linspace(x0, x1, n_xticks+2))
    ax.set_yticks(np.linspace(y0, y1, n_yticks+2))
        
    ax.set_xticklabels(ra_ticks_str, fontsize=ticks_fontsize)
    ax.set_yticklabels(dec_ticks_str, fontsize=ticks_fontsize, va='center', rotation=90)
    
    ax.set_xlabel('RA (deg)', fontsize=label_fontsize)
    ax.set_ylabel('Dec (deg)', fontsize=label_fontsize)
    
# Function useful for add_ra_dec_ticks only 
def dynamic_precision(span_deg):
    """
    Returns number of decimal places based on the degree span.
    """
    if span_deg > 30:
        return 0
    elif span_deg > 10:
        return 1
    elif span_deg > 1:
        return 2
    elif span_deg > 0.1:
        return 3
    else:
        return 4







