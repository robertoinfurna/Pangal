import numpy as np
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from skimage.draw import polygon
from skimage import measure
from skimage.measure import find_contours

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from sklearn.mixture import GaussianMixture

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u

from .image import Image
from .cube import Cube
from .region import Region
from .filter import Filter, list_filters, plot_filters, map_filter_names, nice_filter_names, default_filter_colors

from .plot import add_ra_dec_ticks, Point, Contours

from .photometry_table import PhotometryTable, PhotometryCatalog



def photometry(self,              
               region,
               bands=None,
               units='mJy',
               threshold_quantile=0.9,
               n_background_regions=500,
               background_exclude_regions=[],
               print_output=True):


    if not isinstance(region, Region):
        raise TypeError(
            f"'region' must be a Region object, got {type(region).__name__}"
        )

    if not isinstance(background_exclude_regions, (list, tuple)):
        background_exclude_regions = [background_exclude_regions]

    if bands is None:
        bands = self.images.keys()

    phot_table = PhotometryTable()
    phot_table.header['UNITS'] = units
    # other info
    # phot_table.header['Region'] = centroid


    # ---------------------------
    # Photometry loop
    # ---------------------------
    for b, band in enumerate(bands):

        # mosaic naming
        exposure_id = None
        if '(' in band:
            idx = band.find('(')
            exposure_id = band[idx+1:-1]
            band = band[:idx]
        else:
            band = band


        image = self.images[band].data
        wcs = self.images[band].wcs

        if print_output:
            print(nice_filter_names[band])

        # ---------------------------
        # Build source mask
        # ---------------------------
        source_mask = np.zeros_like(image, dtype=bool)

        if background_exclude_regions:
            for reg in background_exclude_regions:
                mask = reg.project(image, wcs).astype(bool)
                source_mask |= mask
            outer_mask = ~source_mask
            source_area = np.sum(source_mask)
            outer_area = np.sum(outer_mask)
        else:
            source_area = 0
            outer_area = 1

        # ---------------------------
        # Aperture loop
        # ---------------------------
        mask = region.project(image, wcs)
        n = np.nansum(mask)

        # ---------------------------
        # BACKGROUND ESTIMATION
        # ---------------------------
        if outer_area > 10 * source_area:

            back_flux_arr = []
            threshold = np.nanquantile(image[image > -999], threshold_quantile)

            mask_indices = np.argwhere(mask > 0)
            mask_values = mask[mask > 0]

            while len(back_flux_arr) < n_background_regions:

                dy = random.randint(-(image.shape[0] - 5), image.shape[0] - 5)
                dx = random.randint(-(image.shape[1] - 5), image.shape[1] - 5)
                shifted = mask_indices + [dy, dx]

                if (shifted[:, 0].min() < 0 or shifted[:, 0].max() >= image.shape[0] or
                    shifted[:, 1].min() < 0 or shifted[:, 1].max() >= image.shape[1]):
                    continue

                shifted_vals = image[shifted[:, 0], shifted[:, 1]]

                if np.any(source_mask[shifted[:, 0], shifted[:, 1]]):
                    continue
                if np.any(np.isclose(shifted_vals, -999)):
                    continue

                back_flux = np.nansum(shifted_vals * mask_values)
                if back_flux < threshold * n:
                    back_flux_arr.append(back_flux)

            mean_background_flux = np.mean(back_flux_arr)
            background_standard_dev = np.std(back_flux_arr)

        else:
            masked_image = np.where(source_mask, image, np.nan)
            flattened = masked_image[(~np.isnan(masked_image)) & (masked_image != -999)]
            bins = np.linspace(np.nanquantile(flattened, 0.005),
                                np.nanquantile(flattened, 0.9))
            hist, bins = np.histogram(flattened, bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            peak_idx = np.argmax(hist)
            peak = bin_centers[peak_idx]

            neg = flattened[flattened < peak]
            symmetric = np.concatenate((neg, 2*peak - neg))
            mean, std = stat_norm.fit(symmetric)

            mean_background_flux = mean * n
            background_standard_dev = std * np.sqrt(n)

        # -------------------
        # FLUX + UNCERTAINTY 
        # -------------------
        flux = np.nansum(image * mask)
        flux_sky_subtracted = flux - mean_background_flux

        exptime = self.images[band].exptime
        if exptime:
            total_counts = flux_sky_subtracted * exptime
            poisson_err = np.sqrt(total_counts) / exptime
        else:
            poisson_err = 0

        total_flux_err = np.sqrt(background_standard_dev**2 + poisson_err**2)

        if print_output: 
            print(f"Total flux: {flux:.5f}, poisson unc: {poisson_err:.5f}, " f"sky flux: {mean_background_flux:.5f}, sky std: {background_standard_dev}")

        if flux_sky_subtracted < 0:
            flux_sky_subtracted = background_standard_dev
        if total_flux_err / flux_sky_subtracted < 0.01:
            total_flux_err = 0.01 * flux_sky_subtracted

        if 'arcsec2' in units:
            area_arcsec2 = n * self.images[band].area_pix_arcsec2
            flux_sky_subtracted /= area_arcsec2
            total_flux_err /= area_arcsec2

        # ----------------
        # UNIT CONVERSION 
        # ----------------
        ZP = self.images[band].ZP
        ZP_err = self.images[band].ZP_err

        mag = ZP - 2.5*np.log10(flux_sky_subtracted)
        mag_err = np.sqrt(ZP_err**2 +
                            (2.5/(np.log(10)*flux_sky_subtracted))**2 * total_flux_err**2)

        if 'mag' in units:
            flux_sky_subtracted = mag
            total_flux_err = mag_err

        if 'mJy' in units:
            flux_sky_subtracted = 1e3 * 10**(0.4*(-mag+8.90))
            total_flux_err = 1e3 * 0.4*np.log(10) * 10**(0.4*(-mag+8.90)) * mag_err

        if 'erg_s_cm2_A' in units:
            wl = self.images[band].pivot_wavelength
            factor = 2.998e18 / (wl**2)
            flux_sky_subtracted = factor * 10**(0.4*(-mag-48.60))
            total_flux_err = factor * 0.4*np.log(10) * 10**(0.4*(-mag-48.60)) * mag_err

        # ---------------------------
        # Print output 
        # ---------------------------
        if print_output: fmt = '.5f' if 'mJy' in units else '.2f' if 'mag' in units else '.2e' 
        print(f"\tFlux = {format(flux_sky_subtracted, fmt)} ± {format(total_flux_err, fmt)} " f"{units}, SNR = {flux_sky_subtracted / total_flux_err:.2f}")

        # ---------------------------
        # SAVE RESULT INTO REGION ENTRY
        # ---------------------------
        band = band if exposure_id is None else f"{band}({exposure_id})"

        phot_table.data[band] = (flux_sky_subtracted,total_flux_err)
 
    return phot_table
    





def surface_brightness_profile(self,regions,units='mJy_arcsec2',bands=None,threshold_quantile=0.9,n_background_regions=500, background_exclude_regions = [],
        function=None, initial_parameters=None):             # nice to put also into big fov!):

    radii = []
    try:
        for Reg in regions:
            radii.append(Reg.galactocentric_radius)    
    except AttributeError:
        raise AttributeError("One or more regions do not have the 'galactocentric_radius' attribute.")
          
    fig, ax = plt.subplots()
        
    if bands is None: bands = self.images.keys()
        
    phot_list = self.photometry(regions,bands,units='mJy_arcsec2',
                                    print_output=False,
                                    threshold_quantile=threshold_quantile,
                                    n_background_regions=n_background_regions,
                                    background_exclude_regions=background_exclude_regions)
        
    for band in bands:
        
        flux = [phot_list[r].data[band][0] for r in range(len(regions))]
        err = [phot_list[r].data[band][1] for r in range(len(regions))]
            
        ax.errorbar(
                radii, flux, yerr=err,
                fmt='o',
                color=default_filter_colors[band],
                markersize=6,
                capsize=4,
                capthick=1,
                ecolor=default_filter_colors[band],
                alpha=0.8,
                linestyle='',
                label=f"{band}" if function is None else None
            )
        
        if function:

            # Fit the model
            popt, _ = curve_fit(function, radii, flux, p0=initial_parameters)
                
            r = np.linspace(0.8*min(radii),1.2*max(radii))
            fun = function(r,*popt)
            ax.plot(r,fun, c=default_filter_colors[band], label=f"{nice_filter_names[band]}: fit ($\Sigma_0$={popt[0]:.1e}, $R_d$={popt[1]:.2f} kpc)")

    if 'mag' in units:
        import matplotlib.ticker as ticker

        def mJy_to_mag(y):
            return -2.5 * np.log10(y / 1000 / 3631)

        ax.set_yscale('log')
        ax.set_ylabel("$\Sigma$ [mag/arcsec²]")

        # Custom formatter for mag labels
        def mag_formatter(y, pos):
            try:
                return f"{mJy_to_mag(y):.2f}" if y > 0 else ""
            except:
                return ""

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(mag_formatter))
    
    elif 'erg_s_cm2_A' in units:
        pivot_wavelength = self.images[band].pivot_wavelength  
        def mJy_to_erg(y):
            return y * 1e-23 * (2.99e18  / pivot_wavelength**2)
        ax.set_ylabel("$\Sigma$ [erg/s/cm²/Å/arcsec²]")
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{mJy_to_erg(y):.2e}" if y > 0 else "" for y in yticks])
    
    elif 'mJy' in units:
        ax.set_ylabel("$\Sigma$ [mJy/arcsec²]")
               
    else: 
        raise ValueError('Unrecognized units. Allowed units are: mag_arcsec2, erg_s_cm2_A_arcsec2, mJy_arcsec2')
            
    ax.set_xlabel('Galactocentric radius (kpc)')
    ax.legend()

    plt.show()

           
 






 











from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogFormatter, NullFormatter, FuncFormatter

def inspect_photometry(      
               galaxy,
               band,              # One band only
               region,            # One region only
               units='mJy',        # units of output 
               threshold_quantile=0.9,
               n_background_regions=500,
               N_background_gaussians=3,
               background_exclude_regions = [],     
               region_plot_cmap='plasma',
               background_plot_cmap='viridis',
               bad_pixel_color='black',
               ):
    """
    Perform aperture photometry with sky subtraction and error estimation.

    Parameters
    ----------
    data : dict of BandData
        Dictionary where each band is a `BandData` object.
    
    regions : list of array-like
        List of regions, each containing a list of [RA, Dec] pairs.
    plot_photometry : bool, optional
        If True, generates photometry plots.
    hdu : FITS HDU, optional
        If provided, photometry results will be stored in the HDU.
    """
    self=galaxy
    
    if band not in self.images.keys():
        raise ValueError(f"Band '{band}' not found in self.images. Available bands: {list(self.images.keys())}")

    print(f"### PHOTOMETRY, BAND: {band} ###")

    if not isinstance(background_exclude_regions, (list, tuple)):
        background_exclude_regions = [background_exclude_regions]
            
    # Check units
    if units not in ['mJy', 'mJy_arcsec2', 'mag', 'mag_arcsec2', 'erg_s_cm2_A', 'erg_s_cm2_A_arcsec2']: #, 'erg_s_cm2', 'erg_s_cm2_arcsec2']:
        raise ValueError('Unrecognized output units: must be "mJy", "mJy_arcsec2", "mag", "mag_arcsec2", "erg_s_cm2_A", "erg_s_cm2_A_arcsec2"') #, "erg_s_cm2", "erg_s_cm2_arcsec2"')
    
    print("Units of output: ",units)

    image = self.images[band].data

    wcs = self.images[band].wcs


    

    # ---- Plotting ----
    fig, ax = plt.subplots(1,2,figsize=(18, 10),constrained_layout=True)
            
    cmap = cm.get_cmap(region_plot_cmap).copy()
    cmap.set_bad(color=bad_pixel_color)   
    vmin, vmax = np.nanquantile(image[image > 0], 0.001), np.nanquantile(image, 0.999)
    ratio = vmax / vmin
    
    # Choose normalization
    if ratio >= 20:  # large dynamic range
        cmap_norm = LogNorm(vmin=vmin, vmax=vmax)
    else:  # small range, linear works better
        cmap_norm = Normalize(vmin=vmin, vmax=vmax)
    
    im0 = ax[0].imshow(image, origin='lower', cmap=cmap, norm=cmap_norm)
    
    # --- Create colorbar ---
    
    units_label = 'counts/s' if self.images[band].ZP_err != 0 else 'mJy'
    cbar = fig.colorbar(im0, ax=ax[0], orientation='horizontal', pad=0.02, shrink=0.98, label=units_label+'/pixel')
    cbar.ax.tick_params(labelsize=12)


    # --- Inset
    ax_reg = ax[0].inset_axes([0.68, 0.02, 0.3, 0.3])

    ny, nx = region.mask.shape
    ra_max, dec_min = region.wcs.all_pix2world([[0, 0]], 0)[0]
    ra_min, dec_max = region.wcs.all_pix2world([[nx, ny]], 0)[0]
    x_min, y_min = wcs.all_world2pix([[ra_max, dec_min]], 0)[0]
    x_max, y_max = wcs.all_world2pix([[ra_min, dec_max]], 0)[0]               
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    X_low = np.linspace(x_min + dx/2, x_max - dx/2, nx)
    Y_low = np.linspace(y_min + dy/2, y_max - dy/2, ny)
    X_low, Y_low = np.meshgrid(X_low, Y_low)
    for j in [0,1]:
        ax[j].contour(X_low,Y_low,region.mask,levels=[0.5],
            colors=region.color,linestyles=region.linestyle,linewidths=region.linewidth,alpha=region.alpha)    
    
    
    mask = region.project(image, wcs)
    
    im_reg = ax_reg.imshow(mask,origin='lower')
    ax_reg.contour(X_low,Y_low,region.mask,levels=[0.5],
        colors=region.color,linestyles=region.linestyle,linewidths=region.linewidth,alpha=region.alpha) 
    ax_reg.set_xticks([])
    ax_reg.set_yticks([])

    # Get bounding box of the True values in the region mask
    mask_indices = np.argwhere(mask > 0)
    y_min_mask, x_min_mask = mask_indices.min(axis=0)
    y_max_mask, x_max_mask = mask_indices.max(axis=0)
    pad = 35
    x_min = max(x_min_mask - pad, 0)
    x_max = x_max_mask + pad
    y_min = max(y_min_mask - pad, 0)
    y_max = y_max_mask + pad
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)   
    ax_reg.set_xlim(x_min_mask-2, x_max_mask+2)
    ax_reg.set_ylim(y_min_mask-2, y_max_mask+2)
    ra_max_crop, dec_min_crop = wcs.all_pix2world([[x_min, y_min]], 0)[0]
    ra_min_crop, dec_max_crop = wcs.all_pix2world([[x_max, y_max]], 0)[0]
    add_ra_dec_ticks(ax[0],image,wcs,4,4,12,12,ra_min_crop,ra_max_crop,dec_min_crop,dec_max_crop)
    


    
    # Background plot!
    cmap = cm.get_cmap(background_plot_cmap).copy()
    cmap.set_bad(color=bad_pixel_color)   
    vmin, vmax = np.nanquantile(image[image > 0], 0.01), np.nanquantile(image, 0.999)
    ratio = vmax / vmin
    
    # Choose normalization
    if ratio >= 10:  # large dynamic range
        cmap_norm = LogNorm(vmin=vmin, vmax=vmax)
    else:  # small range, linear works better
        cmap_norm = Normalize(vmin=vmin, vmax=vmax)
    im1 = ax[1].imshow(image, origin='lower', cmap=cmap, norm=cmap_norm, alpha=1)
    cbar = fig.colorbar(im1, ax=ax[1], orientation='horizontal', pad=0.02, shrink=0.98, label=units_label+'/pixel')
    cbar.ax.tick_params(labelsize=12)

    # --- Add Ra Dec
    add_ra_dec_ticks(ax[1],image,wcs,4,4,12,12,)

    
    ax_hist = ax[1].inset_axes([1.1, 0., 1.05, 0.85])
    ax_hist.set_xlabel(units_label, fontsize=12)
    ax_hist.tick_params(axis='both', which='major', labelsize=10)

    # Create initial mask for sources to exclude from background estimation
    source_mask = np.zeros_like(image, dtype=bool)

    # Build source exclusion mask
    if background_exclude_regions:
        for reg in background_exclude_regions:
            if hasattr(reg, 'project'):
                mask = reg.project(image, wcs).astype(bool)
            else:
                raise TypeError("Each region must be a custom Region")
            source_mask |= mask

        outer_mask = ~source_mask
        source_area = np.sum(source_mask)
        outer_area = np.sum(outer_mask)

        contours = measure.find_contours(mask, level=0.5)
        for contour in contours:
            ax[1].plot(contour[:, 1], contour[:, 0], color='darkred', linewidth=1)

    else:
        print("No detected sources are masked during background estimation.")
        source_area = 0
        outer_area = 1      # Avoid divide-by-zero later



    


    # region mask
    mask = region.project(image,wcs)
    
    # weighted number of pixels
    n = np.nansum(mask)

    # number of pixels where the mask is non zero 
    nonzero_pixels = np.sum((mask != 0) & ~np.isnan(mask))

    print("Mask number of pixels: ",nonzero_pixels)
    print("Mask weighted number of pixels: ",n)


    # If the field of view is very big compared to the source background can be estimated moving around the regions
    if outer_area > 10 * source_area:

        back_flux_arr = []
        
        threshold = np.nanquantile(image[image>-999], threshold_quantile)

        mask_indices = np.argwhere(mask > 0)
        mask_values = mask[mask > 0]

        good_regions_plotted = 0
        bad_regions_plotted = 0
        max_good = 50
        max_bad = 20

        while len(back_flux_arr) < n_background_regions:
            dy = random.randint(- (image.shape[0] - 5), image.shape[0] - 5)
            dx = random.randint(- (image.shape[1] - 5), image.shape[1] - 5)
            shifted = mask_indices + [dy, dx]
            
            # check if all shifted pixels are inside the image
            if (shifted[:,0].min() < 0 or shifted[:,0].max() >= image.shape[0] or
                shifted[:,1].min() < 0 or shifted[:,1].max() >= image.shape[1]):
                continue  # reject this shift
            shifted_vals = image[shifted[:, 0], shifted[:, 1]]

            # extra checks
            # background region must not overlap with the source region
            if np.any(source_mask[shifted[:, 0], shifted[:, 1]]):
                continue

            # All pixels outside the original image that appear after rotation will be set to -999
            # Exclude regions that follow even partially outside original image
            if np.any(np.isclose(shifted_vals, -999, atol=1e-6)):
                continue

            
            back_flux = np.nansum(shifted_vals * mask_values)

            if back_flux < threshold * n:
                back_flux_arr.append(back_flux)

            # ---- plotting
            back_reg_plot = np.column_stack((shifted[:, 0], shifted[:, 1]))
            mask_plot = np.zeros_like(image, dtype=float)
            mask_plot[back_reg_plot[:, 0], back_reg_plot[:, 1]] = 1.0
            contours = measure.find_contours(mask_plot, level=0.5)

            if back_flux < threshold * n and good_regions_plotted < max_good:
                color = 'lime'
                good_regions_plotted += 1
            elif back_flux >= threshold * n and bad_regions_plotted < max_bad:
                color = 'red'
                bad_regions_plotted += 1
            else:
                continue

            for contour in contours:
                ax[1].plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.5, alpha=1)
            # -----

        back_flux_arr = np.array(back_flux_arr)
        mean_background_flux = np.mean(back_flux_arr)
        background_standard_dev = np.std(back_flux_arr)

        
        # ---- plotting
        
        # bin width using Freedman Diaconis rule
        q_75 = np.nanquantile(back_flux_arr, 0.75)
        q_25 = np.nanquantile(back_flux_arr, 0.25)
        q_01 = np.nanquantile(back_flux_arr, 0.01)
        q_99 = np.nanquantile(back_flux_arr, 0.99)
        bin_width = 2 * (q_75 - q_25) / (len(back_flux_arr))**(1/3)
        bins = np.arange(q_01, q_99, bin_width)
        
        # Compute histogram
        hist, bins, _ = ax_hist.hist(
            back_flux_arr, bins=bins, alpha=0.6,
            color='skyblue', edgecolor='black', label="Background Flux Histogram"
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]  # for scaling
        
        # Prepare data for GMM
        valid_data = back_flux_arr[~np.isnan(back_flux_arr)].reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=int(N_background_gaussians), covariance_type='full', random_state=42)
        gmm.fit(valid_data)
        
        # Extract parameters
        means = gmm.means_.flatten()
        weights = gmm.weights_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        
        # Sort by mean for consistency
        sorted_idx = np.argsort(means)
        means = means[sorted_idx]
        weights = weights[sorted_idx]
        stds = stds[sorted_idx]
        
        # X-axis for PDF plotting
        x_vals = np.linspace(np.nanquantile(valid_data, 0.005), np.nanquantile(valid_data, 0.995), 500).reshape(-1, 1)
        gmm_pdf = np.exp(gmm.score_samples(x_vals))
        gmm_scaled = gmm_pdf * len(valid_data) * bin_width
        
        # Plot GMM total PDF
        ax_hist.plot(x_vals, gmm_scaled, 'navy', lw=2, label=f'{N_background_gaussians:.0f}-Gaussian Mixture')
        
        # Plot individual Gaussian components
        colors = ['red', 'green', 'purple']
        for i, idx in enumerate(sorted_idx):
            mu = means[i]
            sigma = stds[i]
            weight = weights[i]
            
            # Individual Gaussian scaled to histogram
            component_pdf = stat_norm.pdf(x_vals.flatten(), mu, sigma)
            component_scaled = component_pdf * weight * len(valid_data) * bin_width
            
            ax_hist.plot(
                x_vals, component_scaled, linestyle='--', lw=2, color=colors[i],
                label=rf"$\mu$={mu:.2e}, $\sigma$={sigma:.2e}"
            )
        
        ax_hist.legend(fontsize=12)

    else:
        # SMALL FOV
        masked_image = np.where(source_mask, image, np.nan)
        flattened_data = masked_image[~np.isnan(masked_image)].flatten()  
        bins = np.linspace(np.nanquantile(flattened_data, 0.005), np.nanquantile(flattened_data, 0.9))
        hist, bins = np.histogram(flattened_data, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
                
        peak_idx = np.argmax(hist)
        peak_value = bin_centers[peak_idx]
        flattened_data_neg = flattened_data[flattened_data < peak_value]
        symmetric_data = np.concatenate((flattened_data_neg, 2 * peak_value - flattened_data_neg))
        mean, std = stat_norm.fit(symmetric_data)

        mean_background_flux = mean * n
        background_standard_dev = std * n

        # ---- plotting
        ax_hist.hist(flattened_data, bins=bins) 
        normalization = max(hist) * (std*np.sqrt(2*np.pi))
        gaussian_curve = stat_norm.pdf(bin_centers, mean, std) * normalization
        ax_hist.plot(bin_centers, gaussian_curve, label=f"Gaussian Fit\nmean=0\nstd={std:.2e}", color='black', lw=2)
        ax_hist.axvline(x=peak_value,linestyle=':',c='black',lw=2)




    # photometry

    flux = np.nansum(image * mask)
    print(f"Detected flux: {flux} {units_label}")

    # --- Compute Poissonian error
    exptime = self.images[band].exptime
    if exptime and flux > 0:
        exptime = float(self.images[band].header['EXPTIME'])
        N = flux * exptime
        poisson_err = np.sqrt(N) / exptime
    else:
        poisson_err = 0

    print(f"Poissonian Error {poisson_err} {units_label}")

    # --- Compute Sky Background Error
    print(f"Background: {mean_background_flux} {units_label}")
    print(f"Background std: {background_standard_dev} {units_label}")
    
    flux_sky_subtracted = flux - mean_background_flux

    print(f"Sky subtracted flux: {flux_sky_subtracted} {units_label}")

    if flux_sky_subtracted < 0:
        print(f"Negative flux: setting flux to 0") 
        flux_sky_subtracted = 0

    
    total_flux_err = np.sqrt(background_standard_dev**2 + poisson_err**2)

    print(f"Total Error {total_flux_err} {units_label}")
                
    if total_flux_err / flux_sky_subtracted < 0.01:
        total_flux_err = 0.01 * flux_sky_subtracted

    if 'arcsec2' in units: 
        flux_sky_subtracted = flux_sky_subtracted / (n * self.images[band].area_pix_arcsec2)
        total_flux_err = total_flux_err / (n * self.images[band].area_pix_arcsec2)


    # --- UNIT CONVERSION ---
    # So far we worked on counts/s, apart for Herschel
    ZP = self.images[band].ZP
    ZP_err = self.images[band].ZP_err
    print(f"ZP {ZP} ± {ZP_err}")

    #convert all to mag
    mag = ZP - 2.5*np.log10(flux_sky_subtracted)
    mag_err = np.sqrt(ZP_err**2 + (2.5/(np.log(10)*flux_sky_subtracted))**2 * total_flux_err**2)   #  Here I add calibration error!

    print(f"Flux in AB magnitudes = {mag:.4f}  ± {mag_err:.4f}")
    
    if 'mag' in units:
        flux_sky_subtracted = mag
        total_flux_err = mag_err
        
    if 'mJy' in units:
        flux_sky_subtracted = 1e3 * 10**(0.4*(-mag+8.90))
        total_flux_err = 1e3 * 0.4*np.log(10)*10**(0.4*(-mag+8.90)) * mag_err

    if 'erg_s_cm2_A' in units:
        flux_sky_subtracted = 2.998e18 / self.images[band].pivot_wavelength**2 * 10**(0.4*(-mag-48.60))
        total_flux_err = 2.998e18 / self.images[band].pivot_wavelength**2  * 0.4*np.log(10)*10**(0.4*(-mag-48.60)) * mag_err


    fmt = '.5f' if 'mJy' in units else '.2f' if 'mag' in units else '.2e'
    print(f"### Final value ### \n Flux = {format(flux_sky_subtracted, fmt)} ± {format(total_flux_err, fmt)} {units}, SNR = {flux_sky_subtracted / total_flux_err:.2f}")
    print('\n')
    
    ax[0].text(
        0.02, 0.98,
        f"{nice_filter_names[band]}, region {region.id}\n"
        f"Flux and Poisson: {flux:g} ± {poisson_err:g}",
        transform=ax[0].transAxes,
        fontsize=15,
        va="top",  # anchor top since you're near y=0.98
        bbox=dict(
            facecolor="white",   # box fill color
            alpha=0.9,
            edgecolor="cyan",   # border color
            boxstyle="square,pad=0.3"  # squared corners with small padding
        )
    )

    ax[1].text(
        0.02, 0.98,
        f"Background: mean = {np.mean(back_flux_arr):g}, std = {np.std(back_flux_arr):g}",
        transform=ax[1].transAxes,
        fontsize=15,
        va="top",  # anchor top since you're near y=0.98
        bbox=dict(
            facecolor="white",   # box fill color
            alpha=0.9,
            edgecolor="cyan",   # border color
            boxstyle="square,pad=0.3"  # squared corners with small padding
        )
    )


    plt.show()


            
