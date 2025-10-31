import numpy as np
from dataclasses import dataclass, field

from astropy.io import fits
from astropy import units as u

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
from scipy.integrate import quad

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D

from .photometry_table import PhotometryTable
from .filter import Filter, map_filter_names, nice_filter_names
from .data.spectral_lines import spectral_lines, UBGVRI_filters, atmospheric_lines


@dataclass
class Spectrum:
    wl: np.ndarray=None                                       # wavelength
    resolution: np.ndarray=None                               # spectral resolution
    flux: np.ndarray=None                                     # flux
    flux_err:  np.ndarray=None                                # flux error
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 

    
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


    

    # Given a spectrum and a set of filters, it does the convolution with the transmission curve 
    # If I pass the bands it build new Filters at each call
    # If I pass directly the filters objects, it doesn't have to load it each time
    def get_phot(self, 
                bands=None,
                filters=None,
                units='erg/s/cm2/A'):
        """
        Computes synthetic photometric points from a spectrum and a list of bands.

        Parameters
        ----------
        bands : list
            List of band names (e.g., ['g', 'r', 'i']).
        filters : dict
            Optional preloaded Filter objects to speed up processing.
        units : str
            Desired output flux units (e.g., 'erg/s/cm2/A' or 'mJy').

        Notes
        -----
        The internal integration is always done in erg/s/cm²/Å and Å.
        The output is then converted to the requested `units`.
        """
        if not bands and not filters:
            raise ValueError('Pass list of bands or dictionary of filters')
        if not bands:
            bands = filters.keys()

        invalid = [b for b in bands if b not in map_filter_names]
        if invalid:
            raise ValueError(f"Unrecognized filters: {', '.join(invalid)}")

        # Normalize units before interpolation
        self.convert_units(new_wl_units='A', new_flux_units='erg/s/cm2/A')

        # Build filter objects if not provided
        if not filters:
            filters = {b: Filter(b) for b in bands}

        spec_func = interp1d(self.wl, self.flux, bounds_error=False, fill_value=0.0)
        data = {}

        for b in bands:
            trans_curve = filters[b].transmission_curve
            lmin, lmax = filters[b].wavelength_range

            num_int, _ = quad(lambda l: trans_curve(l) * spec_func(l), lmin, lmax)
            norm_int, _ = quad(lambda l: trans_curve(l), lmin, lmax)

            phot_point = num_int / norm_int
            pivot_w = filters[b].pivot_wavelength  # in Å

            # Convert to requested output units if needed
            if units == 'mJy':
                # Convert erg/s/cm²/Å → mJy using Fν = Fλ * λ² / c
                c = 2.99792458e18  # Å/s
                phot_point = phot_point * pivot_w**2 / c / 1e-26

            data[b] = (phot_point, np.nan)

        header = fits.Header()
        header['units'] = units

        return PhotometryTable(data=data, header=header)


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
        color=None,
        ecolor='green',
        show_snr=False,
        show_filters=False,
        show_H_lines=False,
        show_all_spectral_lines=False,
        show_atmospheric_lines=False,
        figsize=None,
        phot_points=None,
        x_units=None,   # e.g. 'A', 'nm', 'um', 'eV', 'Hz'
        y_units=None,   # e.g. 'erg/s/cm2/A', 'Jy', 'mJy'
    ):
        """
        Plot spectra with optional annotations and full unit conversion.

        Parameters
        ----------
        x_units : str, optional
            Desired wavelength units for plotting. If None, uses the spectrum's native wl_units.
        y_units : str, optional
            Desired flux units for plotting. If None, uses the spectrum's native flux_units.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import interp1d
        from scipy.integrate import quad

        # Prepare spectra list
        if spectra is None:
            spectra = [self]
        elif not isinstance(spectra, (list, tuple)):
            spectra = [spectra]

        if zoom_on_line is None:
            zoom_on_line = []
        if isinstance(zoom_on_line, str):
            zoom_on_line = [zoom_on_line]

        n_cols = max(1, len(zoom_on_line))
        figsize = figsize if figsize else (6 * n_cols, 4)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)

        cols = zoom_on_line if zoom_on_line else [None]

        for j, line in enumerate(cols):
            ax = axes[0, j]

            for i, spec in enumerate(spectra):

                # --- Determine requested plotting units ---
                x_u = x_units or spec.header.get('WUNITS', 'A')
                y_u = y_units or spec.header.get('FUNITS', 'erg/s/cm2/A')

                # --- Create converted working copy ---
                spec_conv = Spectrum(
                    wl=spec.wl.copy(),
                    flux=spec.flux.copy(),
                    flux_err=spec.flux_err.copy() if getattr(spec, 'flux_err', None) is not None else None,
                    header=spec.header.copy(),
                ).convert_units(new_wl_units=x_u, new_flux_units=y_u)

                wl = spec_conv.wl
                flux = spec_conv.flux
                flux_err = spec_conv.flux_err

                # --- Optional λ × Fλ scaling ---
                if per_wavelength:
                    flux = flux * wl
                    if flux_err is not None:
                        flux_err = flux_err * wl

                # --- Determine plotting window ---
                if line:
                    if line in spectral_lines:
                        center = spectral_lines[line] * (1 + z)
                    elif line in atmospheric_lines:
                        center = atmospheric_lines[line]
                    else:
                        raise ValueError(f"Line {line} not recognized")
                    x0, x1 = center - 70, center + 70
                else:
                    x0 = winf if winf is not None else wl[0]
                    x1 = wsup if wsup is not None else wl[-1]

                # --- Color selection ---
                if isinstance(color, (list, tuple)) and len(color) > 0:
                    # User passed a list or tuple of colors
                    c = color[i % len(color)]
                elif color is None:
                    # Use Matplotlib's default color cycle
                    c = f"C{i % 10}"  # ensures valid index (Matplotlib default has 10 colors)
                else:
                    # Fixed single color for all spectra
                    c = color

                # --- Plot spectrum with errorbars if available ---
                label = spec.header.get('ID', getattr(spec, 'id', None)) or f"Spectrum {i + 1}"

                if flux_err is not None:
                    ax.errorbar(
                        wl, flux, yerr=flux_err, fmt='o', markersize=2,
                        color=c, ecolor=ecolor, elinewidth=1.0,
                        capsize=2, capthick=1.0, linestyle='-', lw=0.5,
                        label=label
                    )
                else:
                    ax.plot(wl, flux, color=c, lw=1, label=label)

                # --- Overlay synthetic photometry points ---
                if phot_points:

                    model_phot = spec.get_phot(bands=phot_points, units=y_u)
                    
                    for b in phot_points:
                        wl_b = self._angstrom_to_wl(np.array([Filter(b).pivot_wavelength]), x_u)
                        f_b = model_phot.data[b][0]
                        if per_wavelength:
                            f_b *= wl_b
                        ax.scatter(wl_b, f_b, 
                                c=c, edgecolors='black', linewidths=0.5,
                                zorder=3,
                                label=nice_filter_names.get(b, b))
                    ax.legend()

                # --- Axis scaling and limits ---
                if log:
                    ax.set_xscale('log')
                    ax.set_yscale('log')

                mask = (wl >= x0) & (wl <= x1)
                fmax = np.nanmax(flux[mask])
                fmin = np.nanmin(flux[mask])

                if ymin is None and ymax is None:
                    y1 = 1.2 * fmax
                    y0 = 1e-4 * fmax if fmin < 1e-4 * fmax else 0.8 * fmin
                else:
                    y0, y1 = ymin, ymax

                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

            # --- Axis labels (with correct units) ---
            ax.set_xlabel(f"Wavelength ({x_u})")
            if per_wavelength:
                ax.set_ylabel(f"$\\lambda F_\\lambda$ ({y_u})")
            else:
                ax.set_ylabel(f"Flux ({y_u})")

            ax.legend()

            # --- Optional overlays ---
            if show_filters:
                for name, props in UBGVRI_filters.items():
                    wmin = props["pivot_wavelength"] - props["bandwidth"] / 2
                    wmax = props["pivot_wavelength"] + props["bandwidth"] / 2
                    wmin = self._angstrom_to_wl(wmin, x_u)
                    wmax = self._angstrom_to_wl(wmax, x_u)
                    ax.axvspan(wmin, wmax, alpha=0.2, color=props["color"], label=f"{name} filter")

            if show_H_lines:
                for name in ['Lya','Ha','Hb','Hg','Hd']:
                    wavelength = spectral_lines[name]
                    wl_shift = self._angstrom_to_wl(wavelength * (1 + z), x_u)
                    ax.axvline(wl_shift, color="black", linestyle="dashed", alpha=0.4)
                    name_map = {'Ha': '$H\\alpha$', 'Hb': '$H\\beta$', 'Hg': '$H\\gamma$', 'Hd': '$H\\delta$', 'Lya': 'Ly$\\alpha$'}
                    ax.text(wl_shift, 1.1 * fmax, name_map.get(name, name),
                            rotation=90, va="bottom", fontsize=10, ha='center', clip_on=True,
                            bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))

            if show_all_spectral_lines:
                for name, wavelength in spectral_lines.items():
                    wl_shift = self._angstrom_to_wl(wavelength * (1 + z), x_u)
                    ax.axvline(wl_shift, color="black", linestyle="dashed", alpha=0.4)
                    name_map = {'Ha': '$H\\alpha$', 'Hb': '$H\\beta$', 'Hg': '$H\\gamma$', 'Hd': '$H\\delta$', 'Lya': 'Ly$\\alpha$'}
                    ax.text(wl_shift, 1.1 * fmax, name_map.get(name, name),
                            rotation=90, va="bottom", fontsize=10, ha='center', clip_on=True,
                            bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))

            if show_atmospheric_lines:
                for name, wavelength in atmospheric_lines.items():
                    wl_plot = self._angstrom_to_wl(wavelength, x_u)
                    ax.axvline(wl_plot, color="cyan", linestyle="dashed", alpha=0.7)
                    ax.text(wl_plot, 1.1 * fmax, name, rotation=90, color='cyan',
                            va="bottom", fontsize=9, ha='center', clip_on=True,
                            bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))

        plt.tight_layout()
        plt.show()




    # CONVERSION

    # ---- CONSTANTS ----
    C_ANGS_PER_S = 2.99792458e18  # Speed of light [Å/s]
    H_EV_S = 4.135667696e-15      # Planck constant [eV·s]

    # ---- MAIN CONVERTER ----
    def convert_units(self, new_wl_units=None, new_flux_units=None):
        """
        Manual conversion of wavelength and flux units.

        Supported wl_units: A, nm, um, cm, m, Hz, eV
        Supported flux_units: erg/s/cm2/A, erg/s/cm2/nm, W/m2/um, Jy, mJy
        """
        if not self.header.get('WUNITS') or not self.header.get('FUNITS'):
            raise ValueError('No units found in header')

        # ---- WAVELENGTH CONVERSIONS ----
        if new_wl_units and new_wl_units != self.header['WUNITS']:
            old_units = self.header['WUNITS']
            wl_A = self._wl_to_angstrom(self.wl, old_units)
            self.wl = self._angstrom_to_wl(wl_A, new_wl_units)
            self.header['WUNITS'] = new_wl_units

        # ---- FLUX CONVERSIONS ----
        if new_flux_units and new_flux_units != self.header['FUNITS']:
            old_flux_units = self.header['FUNITS']
            wl_units = self.header['WUNITS']
            self.flux = self._convert_flux(
                self.flux, self.wl, old_flux_units, new_flux_units, wl_units
            )
            self.header['FUNITS'] = new_flux_units

        return self


    # -- internal
    def _wl_to_angstrom(self, wl, units):
        """Convert wavelength-like quantities to Å."""
        c = self.C_ANGS_PER_S
        h = self.H_EV_S
        wl = np.asarray(wl, dtype=float)

        if units == 'A':
            return wl
        elif units == 'nm':
            return wl * 10
        elif units == 'um':
            return wl * 1e4
        elif units == 'cm':
            return wl * 1e8
        elif units == 'm':
            return wl * 1e10
        elif units == 'Hz':
            return c / wl
        elif units == 'eV':
            return (c * h) / wl
        else:
            raise ValueError(f"Unsupported wavelength unit: {units}")

    def _angstrom_to_wl(self, wl_A, units):
        """Convert Å to target wavelength-like unit."""
        c = self.C_ANGS_PER_S
        h = self.H_EV_S
        wl_A = np.asarray(wl_A, dtype=float)

        if units == 'A':
            return wl_A
        elif units == 'nm':
            return wl_A / 10
        elif units == 'um':
            return wl_A / 1e4
        elif units == 'cm':
            return wl_A / 1e8
        elif units == 'm':
            return wl_A / 1e10
        elif units == 'Hz':
            return c / wl_A
        elif units == 'eV':
            return (c * h) / wl_A
        else:
            raise ValueError(f"Unsupported wavelength unit: {units}")

    def _convert_flux(self, flux, wl, old_units, new_units, wl_units):
        """Manual flux conversion with wavelength dependence."""
        c = self.C_ANGS_PER_S
        wl_A = self._wl_to_angstrom(wl, wl_units)
        flux = np.asarray(flux, dtype=float)

        # --- Convert old flux to erg/s/cm²/Å ---
        if old_units == 'erg/s/cm2/A':
            f_lambda = flux
        elif old_units == 'erg/s/cm2/nm':
            f_lambda = flux / 10.0
        elif old_units == 'W/m2/um':
            # W/m²/μm → erg/s/cm²/Å
            f_lambda = flux * 1e7 / 1e4 / 1e4  # = *1e-1
        elif old_units == 'Jy':
            f_lambda = flux * 1e-23 * c / (wl_A**2)
        elif old_units == 'mJy':
            f_lambda = flux * 1e-26 * c / (wl_A**2)
        else:
            raise ValueError(f"Unsupported flux unit: {old_units}")

        # --- Convert erg/s/cm²/Å to target ---
        if new_units == 'erg/s/cm2/A':
            return f_lambda
        elif new_units == 'erg/s/cm2/nm':
            return f_lambda * 10.0
        elif new_units == 'W/m2/um':
            return f_lambda / 1e7 * 1e4 * 1e4
        elif new_units == 'Jy':
            return f_lambda * (wl_A**2) / c / 1e-23
        elif new_units == 'mJy':
            return f_lambda * (wl_A**2) / c / 1e-26
        else:
            raise ValueError(f"Unsupported flux unit: {new_units}")