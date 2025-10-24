import numpy as np
from dataclasses import dataclass, field

from astropy.io import fits
from astropy import units as u

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D

from .data.spectral_lines import spectral_lines, UBGVRI_filters, atmospheric_lines


@dataclass
class Spectrum:
    wl: np.ndarray=None                                       # wavelength
    resolution: np.ndarray=None                               # spectral resolution
    flux: np.ndarray=None                                     # flux
    flux_err:  np.ndarray=None                                # flux error
    header: fits.Header = field(default_factory=fits.Header)  # FITS header (IF SPECTRUM IS DOWNLOADED FROM AN ARCHIVE, or if a model)
    id: str=None                                              # id

    @property
    def res_delta_lambda(self):
        """
        delta lambda (lambda): spectral resoolution
        """    
        return self.wl / self.resolution         
            

    def __call__(self, wavelength: float, kind: str = "linear") -> float:
        """
        Interpolate the flux at a given wavelength.
        
        Parameters
        ----------
        wavelength : float
            Wavelength (same units as self.wl)
        kind : str
            Interpolation kind ("linear", "nearest", etc.)
        
        Returns
        -------
        float
            Interpolated flux value.
        """
        if self.wl is None or self.flux is None:
            raise ValueError("Spectrum has no wavelength or flux defined.")
        
        return np.interp(wavelength, self.wl, self.flux)


    def plot(
        self,
        spectra=None,
        z=0.0,
        log=True,
        per_wavelength=False,
        winf=None,
        wsup=None,
        ymin=None,
        ymax=None,
        zoom_on_line=None,
        color='red',
        ecolor='green',
        show_snr=False,
        show_filters=False,
        show_spectral_lines=False,
        show_atmospheric_lines=False,
        figsize=None,
        ):
        """
        Plot spectra extracted from regions, with optional annotations and features.

        Parameters
        ----------
        spectra : list of spectra, option to plot more spectra in the same plot

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

        if spectra is None:
            spectra = [self]
        elif not isinstance(spectra, (list, tuple)):
            spectra = [spectra]

        if zoom_on_line is None: 
            zoom_on_line = []
        if isinstance(zoom_on_line, str): 
            zoom_on_line = [zoom_on_line]

        n_cols = max(1, len(zoom_on_line))
        figsize = figsize if figsize else (6*n_cols, 4)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)

        cols = zoom_on_line if zoom_on_line else [None]

        for j, line in enumerate(cols):
            ax = axes[0, j]

            for i, spec in enumerate(spectra):

                
                wl, flux, flux_err = spec.wl, spec.flux, spec.flux_err

                if per_wavelength:
                    flux = flux * wl
                    flux_err = flux * wl


                # Determine x-limits
                if line:
                    if line in spectral_lines:
                        center = spectral_lines[line] * (1 + z) 
                    elif line in atmospheric_lines:
                        center = atmospheric_lines[line]  # no redshift
                    else:
                        raise ValueError(f"Line {line} not recognized")
                    x0, x1 = center-70, center+70
                else:
                    x0, x1 = (winf if winf is not None else wl[0], 
                            wsup if wsup is not None else wl[-1])

                # Pick color if list of spectra
                if isinstance(color, (list, tuple)):
                    c = color[i % len(color)]
                elif color is None:
                    c = f"C{i}"
                else:
                    c = color

                if flux_err is not None:
                    ax.errorbar(wl, flux, yerr=flux_err, fmt='o', markersize=2,
                                color=c, ecolor=ecolor, elinewidth=1.0,
                                capsize=2, capthick=1.0, linestyle='-', lw=0.5,
                                label=spec.id or f"Spectrum {i+1}")
                else:
                    ax.plot(wl, flux, color=c, lw=1,
                            label=spec.id or f"Spectrum {i+1}")

                if log:
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                mask = (wl >= x0) & (wl <= x1)
                mask = (wl >= x0) & (wl <= x1)
                fmax = np.nanmax(flux[mask])
                fmin = np.nanmin(flux[mask])

                if ymin is None and ymax is None:
                    ymax = 1.2 * fmax
                    ymin = 1e-4*fmax if fmin < 1e-4*fmax else 0.8*fmin

                ax.set_xlim(x0, x1)
                ax.set_ylim(ymin, ymax)

            # Shared axis labels
            ax.set_xlabel("Wavelength (Å)")
            
            if per_wavelength == False:
                ax.set_ylabel(r"Flux (erg/s/cm$^2$/Å/arcsec$^2$)")
            else:
                ax.set_ylabel(r"$\lambda \times F_\lambda$ (erg/s/cm$^2$/arcsec$^2$)")

            ax.legend()

            # Add optional annotations (filters, lines, etc.)
            if show_filters:
                for name, props in UBGVRI_filters.items():
                    wmin = props["pivot_wavelength"]-props["bandwidth"]/2
                    wmax = props["pivot_wavelength"]+props["bandwidth"]/2
                    ax.axvspan(wmin, wmax, alpha=0.2, color=props["color"], label=f"{name} filter")

            if show_spectral_lines:
                for name, wavelength in spectral_lines.items():
                    wl_shift = wavelength*(1+z)
                    ax.axvline(wl_shift, color="black", linestyle="dashed", alpha=0.7)
                    name_map = {'Ha':'$H\\alpha$', 'Hb':'$H\\beta$', 'Hg':'$H\\gamma$', 'Hd':'$H\\delta$', 'Lya':'Ly$\\alpha$'}
                    ax.text(wl_shift, 1.1*fmax, name_map.get(name,name),
                            rotation=90, va="bottom", fontsize=10, ha='center', clip_on=True,
                            bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))

            if show_atmospheric_lines:
                for name, wavelength in atmospheric_lines.items():
                    ax.axvline(wavelength, color="cyan", linestyle="dashed", alpha=0.7)
                    ax.text(wavelength, 1.1*fmax, name, rotation=90, color='cyan',
                            va="bottom", fontsize=9, ha='center', clip_on=True,
                            bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))

        plt.tight_layout()
        plt.show()







