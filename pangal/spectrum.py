import numpy as np
from dataclasses import dataclass, field
import itertools

from astropy.io import fits
from astropy import units as u

from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.signal import savgol_filter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D

from .photometry_table import PhotometryTable
from .filter import Filter, map_filter_names, nice_filter_names
from .data.spectral_lines import spectral_lines, top_galaxy_lines, UBGVRI_filters, atmospheric_lines


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
    def get_phot(self, 
                bands=None,
                units='erg/s/cm2/A',
                trans_arrays=None,
                trans_mask=None,
                pivot_wls=None,
                ):
        """
        Computes synthetic photometric points from a spectrum and a list of bands.

        Parameters
        ----------
        bands : list
            List of band names (e.g., ['g', 'r', 'i']).
        units : str
            Desired output flux units (e.g., 'erg/s/cm2/A' or 'mJy').

        Notes
        -----
        The internal integration is always done in erg/s/cm²/Å and Å.
        The output is then converted to the requested `units`.
        """
        if not trans_arrays:

            invalid = [b for b in bands if b not in map_filter_names]
            if invalid:
                raise ValueError(f"Unrecognized filters: {', '.join(invalid)}")

            # Normalize units before interpolation
            self.convert_units(new_wl_units='A', new_flux_units='erg/s/cm2/A')



            spec_func = interp1d(self.wl, self.flux, bounds_error=False, fill_value=0.0)
            data = {}

            for b in bands:
                filter = Filter(b) 
                trans_curve = filter.transmission_curve
                lmin, lmax = filter.wavelength_range

                num_int, _ = quad(lambda l: trans_curve(l) * spec_func(l), lmin, lmax)
                norm_int, _ = quad(lambda l: trans_curve(l), lmin, lmax)

                phot_point = num_int / norm_int
                pivot_w = filter.pivot_wavelength  # in Å

                # Convert to requested output units if needed
                if units == 'mJy':
                    # Convert erg/s/cm²/Å → mJy using Fν = Fλ * λ² / c
                    c = 2.99792458e18  # Å/s
                    phot_point = phot_point * pivot_w**2 / c / 1e-26

                data[b] = (phot_point, np.nan)

            header = fits.Header()
            header['units'] = units

            return PhotometryTable(data=data, header=header)
        

        # FOR LIKELIHOOD! self is the synthetic spectrum!
        else: 

            model_phot = []
            for b in bands:
                mask_b = trans_mask[b]
                spec_array = self.flux[mask_b]
                if len(spec_array) == 0:
                    model_phot.append(np.nan)
                    continue
                num_int = np.trapz(trans_arrays[b] * spec_array, self.wl[mask_b])
                den = np.trapz(trans_arrays[b], self.wl[mask_b])
                if den == 0:
                    model_phot.append(np.nan)
                    continue
                phot_point = num_int / den
                # convert to mJy if needed
                c = 2.99792458e18  # Å/s
                phot_point = phot_point * pivot_wls[b]**2 / c / 1e-26
                model_phot.append(phot_point)

            return np.array(model_phot)

        
   

    def plot(
        self,
        spectra=None,

        # main
        log=True,
        normalized=False,
        remove_continuum=False,
        draw_continuum=False,
        per_wavelength=False,

        # units
        x_units=None,   # e.g. 'A', 'nm', 'um', 'eV', 'Hz'
        y_units=None,   # e.g. 'erg/s/cm2/A', 'Jy', 'mJy'
        
        # plotting window
        winf=None,
        wsup=None,
        ymin=None,
        ymax=None,
        zoom_on_line=None,

        custom_ylabel=None,
        
        # aestethics
        figsize=None,
        color=None,
        linestyle_1='-',
        linestyle_2='-',
        ecolor='green',
        show_snr=False,
        
        # optionals
        redshift=None,
        show_top_spectral_lines=False,
        show_all_spectral_lines=False,
        show_atmospheric_lines=False,
        show_filters=False,
        
        equivalent_widths=False,
        ew_window=5,

        phot=None,
        synth_phot=None,

        spec_legend_pars=None,
        show_phot_legend=True,
        show_spec_legend=True,
        spec_legend_loc="upper left",
        spec_legend_title=None,
        spec_legend_fontsize=10,
    ):


        if normalized or remove_continuum or zoom_on_line:
            phot_points = False
            synth_phot_points = False

        if equivalent_widths:
            remove_continuum = True


        # --- Prepare spectra list ---
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
            global_fmax = -np.inf
            global_fmin = np.inf

            spec_handles = []
            spec_labels = []

            # --- For photometric legend ---
            band_handles = {}
            band_labels = {}

            for i, spec in enumerate(spectra):
                # --- Units ---
                x_u = x_units or spec.header.get('WUNITS', 'A')
                y_u = y_units or spec.header.get('FUNITS', 'erg/s/cm2/A')

                spec_conv = Spectrum(
                    wl=spec.wl.copy(),
                    flux=spec.flux.copy(),
                    flux_err=spec.flux_err.copy() if getattr(spec, 'flux_err', None) is not None else None,
                    header=spec.header.copy(),
                ).convert_units(new_wl_units=x_u, new_flux_units=y_u)

                wl = spec_conv.wl
                flux = spec_conv.flux
                flux_err = spec_conv.flux_err

                if per_wavelength:
                    flux = flux * wl
                    if flux_err is not None:
                        flux_err = flux_err * wl

                if redshift:
                    z = redshift
                else:
                    z = next((spec.header[k] for k in ("REDSHIFT", "redshift", "z") if k in spec.header), 0)
                    if "vel_sys" in spec.header and spec.header["vel_sys"] is not None: 
                        v_sys = spec.header["vel_sys"]
                        z += v_sys / 2.99792458e5
    

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

                # --- Color ---
                if isinstance(color, (list, tuple)) and len(color) > 0:
                    c = color[i % len(color)]
                elif color is None:
                    c = f"C{i % 10}" 
                else:
                    c = color

                # --- Label ---
                label = None

                if isinstance(spec_legend_pars, str):
                    label = str(spec.header.get(spec_legend_pars))

                elif isinstance(spec_legend_pars, dict):
                    parts = []
                    for k, p in spec_legend_pars.items():
                        if k in spec.header:
                            parts.append(
                                f"{p['label']}={spec.header[k]:{p['fmt']}} {p['unit']}"
                            )
                    label = ", ".join(parts) if parts else None

                else:
                    label = spec.header.get("ID", getattr(spec, "id", None))

                if not label:
                    label = "observed spectrum"



                # --- Plot spectrum ---

                # optional normalizations

                if normalized:
                    normalization = np.median(flux) #### FIX HERE!!!

                elif remove_continuum:
                    # don't mutate outer flags; use local copies if needed
                    per_wavelength = False
                    log = False

                    # Fit continuum only inside the plotting window (x0, x1)
                    # Use deg=3 (or 1-4 depending on shape); sigma-clipping iter=6 recommended
                    continuum = self._fit_continuum(wl, flux, deg=7, sigma=2.0, iters=6, method='poly', fit_region=(x0, x1))
                    normalization = continuum
                else:
                    normalization = 1


                flux /= normalization
                if flux_err is not None:
                    flux_err /= normalization

                mask = (wl >= x0) & (wl <= x1)
                if len(flux[mask])>0:
                    fmax = np.nanmax(flux[mask])
                    fmin = np.nanmin(flux[mask])
                    global_fmax = max(global_fmax, fmax)
                    global_fmin = min(global_fmin, fmin)


                if flux_err is not None:
                    handle = ax.errorbar(
                        wl, flux, yerr=flux_err, fmt='o', markersize=2,
                        color=c, ecolor=ecolor, elinewidth=1.0,
                        capsize=2, capthick=1.0, linestyle=linestyle_1, lw=0.5,
                        label=label
                    )
                else:
                    handle = ax.plot(wl, flux, color=c, linestyle=linestyle_2,lw=1, label=label)[0]


                if draw_continuum:
                    continuum = self._fit_continuum(wl, flux, deg=7, sigma=2.0, iters=6, method='poly', fit_region=(x0, x1))
                    ax.plot(wl,continuum,color=c,linestyle=':',label='continuum')

                spec_handles.append(handle)
                spec_labels.append(label)

                # --- Equivalent widths --- #
                if equivalent_widths:
                    for name, wavelength in spectral_lines.items():

                        # observed-frame wavelength
                        wl_shift = self._angstrom_to_wl(wavelength * (1 + z), x_u)

                        if x0 < wl_shift < x1:

                            # define integration window around the line (example: ±5 Å)
                            lmin = wl_shift - ew_window
                            lmax = wl_shift + ew_window

                            # restrict to plotting range
                            lmin = max(lmin, x0)
                            lmax = min(lmax, x1)

                            # interpolated normalized flux function
                            spec_func = lambda l: np.interp(l, wl, flux)

                            # EW for continuum-normalized flux
                            EW, _ = quad(lambda l: 1.0 - spec_func(l), lmin, lmax)

                            print(f"Line {name} EW: {EW:.2f} Å")



                # --- Synthetic photometric points (larger, white-filled) ---
                if synth_phot:
                    markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>'])
                    model_phot = spec.get_phot(bands=synth_phot, units=y_u)
                    for band in synth_phot:
                        pivot_wl_A = Filter(band).pivot_wavelength
                        wl_plot = self._angstrom_to_wl([pivot_wl_A], x_u)[0]
                        flux_val = model_phot.data[band][0]
                        if per_wavelength:
                            flux_val *= wl_plot

                        # pick marker
                        marker = next(markers)
                        h = ax.scatter(
                            wl_plot, flux_val/normalization,
                            s=100,               # bigger marker size
                            facecolors='white',  # hollow center
                            edgecolors=c,  # black edge
                            marker=marker,
                            zorder=3,
                            label=None           # optional: don't add to legend to avoid duplicates
                        )
                        band_handles[band] = h
                        band_labels[band] = nice_filter_names.get(band, band)



            # --- Photometric points ---
            if phot:
                markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>'])
                phot_units = phot.header.get('units', 'mJy')
                
                for band in phot.data.keys():
                    flux_val, flux_err_val = phot.data[band]
                    pivot_wl_A = Filter(band).pivot_wavelength
                    wl_plot = self._angstrom_to_wl([pivot_wl_A], x_u)[0]
                    flux_val_conv = self._convert_flux(flux_val, pivot_wl_A, phot_units, y_u, 'A')
                    flux_err_conv = self._convert_flux(flux_err_val, pivot_wl_A, phot_units, y_u, 'A')
                    if per_wavelength:
                        flux_val_conv *= wl_plot
                        flux_err_conv *= wl_plot

                    marker = next(markers)
                    h = ax.errorbar(
                        wl_plot, flux_val_conv, yerr=flux_err_conv,
                        fmt=marker, color='black', ecolor='black', capsize=2, markersize=6,
                        label=f"{nice_filter_names.get(band, band)} phot"
                    )
                    if x0 <= wl_plot <= x1:
                        band_handles[band] = h
                        band_labels[band] = nice_filter_names.get(band, band)




            # --- Axis limits ---
            if zoom_on_line: log = False
            if log:
                ax.set_xscale('log')
                ax.set_yscale('log')

            y0 = ymin if ymin is not None else (0.8 * global_fmin if global_fmin > 0 else 1e-3 * global_fmax)
            y1 = ymax if ymax is not None else (1.4 * global_fmax if log is False else 10 * global_fmax)
            if ymin is None and y0 < 1e-4 * y1:
                y0 = 1e-4 * y1
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            # --- Axis labels ---
            if x_u == 'A': 
                unit_label = 'Å'
            else:
                unit_label = x_u 
            ax.set_xlabel(f"Wavelength ({unit_label})")
            ylabel = r"$\lambda \times F_\lambda$" if per_wavelength else "Flux"
           
            if y_u == 'erg/s/cm2/A':
                unit_label = 'erg/s/cm$^2/Å$' if per_wavelength == False else 'erg/s/cm$^2$'
            if y_u == 'erg/s/A':
                unit_label = 'erg/s/Å$' if per_wavelength == False else 'erg/s'
            else: unit_label = y_u
            ax.set_ylabel(f"{ylabel} ({unit_label})")
            if normalized or remove_continuum:
                ax.set_ylabel("arbitrary units")
            if custom_ylabel: 
                ax.set_ylabel(custom_ylabel)


            # --- Spectra legend ---
            if show_spec_legend:
                spec_legend = ax.legend(
                    spec_handles,
                    spec_labels,
                    fontsize=spec_legend_fontsize,
                    title=spec_legend_title,
                    loc=spec_legend_loc,
                    frameon=False
                )
                ax.add_artist(spec_legend)  

            # --- Phot legend ---
            if zoom_on_line: show_phot_legend = False
            if show_phot_legend and band_handles:
                handles = []
                for h in band_handles.values():
                    # extract Line2D from ErrorbarContainer or use scatter
                    if isinstance(h, matplotlib.container.ErrorbarContainer):
                        handles.append(h[0])
                    else:
                        handles.append(h)
                labels = list(band_labels.values())
                ax.legend(
                    handles, labels,
                    title=f"Photometry",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False
                )



            # --- Optionals ---
            name_map = {'Ha': '$H\\alpha$', 'Hb': '$H\\beta$', 'Hg': '$H\\gamma$', 'Hd': '$H\\delta$',
                'Lya': 'Ly$\\alpha$','Lyb': 'Ly$\\beta$','Lyg': 'Ly$\\gamma$',}
            top_galaxy_lines = ["Lya","Lyb","Lyman break","Ha","Hb","Hg","Hd","[O III] 5007","[O II] 3727","[S II] 6716","He II 4686"]


            if show_top_spectral_lines:
                for name in top_galaxy_lines:
                    wavelength = spectral_lines[name]
                    wl_shift = self._angstrom_to_wl(wavelength * (1 + z), x_u)
                    if wl_shift > x0 and wl_shift < x1:
                        ax.axvline(wl_shift, color="black", linestyle="dashed", alpha=0.4)
                        ax.text(wl_shift, 1.1 * global_fmax, name_map.get(name, name),
                                rotation=90, va="bottom", fontsize=10, ha='center', clip_on=True,
                                bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))


            if show_all_spectral_lines:
                for name, wavelength in spectral_lines.items():
                    wl_shift = self._angstrom_to_wl(wavelength * (1 + z), x_u)
                    if wl_shift > x0 and wl_shift < x1:
                        ax.axvline(wl_shift, color="black", linestyle="dashed", alpha=0.4)
                        ax.text(wl_shift, 1.1 * global_fmax, name_map.get(name, name),
                                rotation=90, va="bottom", fontsize=10, ha='center', clip_on=True,
                                bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))


            if show_atmospheric_lines:
                for name, wavelength in atmospheric_lines.items():
                    wl_plot = self._angstrom_to_wl(wavelength, x_u)
                    if wl_plot > x0 and wl_plot < x1:
                        ax.axvline(wl_plot, color="cyan", linestyle="dashed", alpha=0.7)
                        ax.text(wl_plot, 1.1 * global_fmax, name, rotation=90, color='cyan',
                                va="bottom", fontsize=9, ha='center', clip_on=True,
                                bbox=dict(facecolor='white', edgecolor='white', boxstyle='square,pad=0.2'))


        plt.tight_layout(rect=[0,0,0.85,1])
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
        


    # --------------------------------------------------
    # FITS EXPORT
    # --------------------------------------------------
    def to_fits(self, filename: str):
        """
        Save spectrum to a FITS file.
        """
        cols = []

        if self.wl is not None:
            cols.append(fits.Column(name="WAVELENGTH", format="D", array=self.wl))
        if self.resolution is not None:
            cols.append(fits.Column(name="RESOLUTION", format="D", array=self.resolution))
        if self.flux is not None:
            cols.append(fits.Column(name="FLUX", format="D", array=self.flux))
        if self.flux_err is not None:
            cols.append(fits.Column(name="FLUX_ERR", format="D", array=self.flux_err))

        hdu = fits.BinTableHDU.from_columns(cols, header=self.header)

        hdu.writeto(filename, overwrite=True)

    # --------------------------------------------------
    # FITS IMPORT
    # --------------------------------------------------
    def from_fits(self, filename: str):
        """
        Load spectrum from a FITS file.
        Returns a new Spectrum instance.
        """
        with fits.open(filename) as hdul:
            data = hdul[1].data
            hdr = hdul[1].header

            wl         = data["WAVELENGTH"] if "WAVELENGTH" in data.columns.names else None
            resolution = data["RESOLUTION"] if "RESOLUTION" in data.columns.names else None
            flux       = data["FLUX"]       if "FLUX"       in data.columns.names else None
            flux_err   = data["FLUX_ERR"]   if "FLUX_ERR"   in data.columns.names else None

            return Spectrum(
                wl=np.array(wl) if wl is not None else None,
                resolution=np.array(resolution) if resolution is not None else None,
                flux=np.array(flux) if flux is not None else None,
                flux_err=np.array(flux_err) if flux_err is not None else None,
                header=hdr
            )


    

    def _fit_continuum(self, wl, flux, deg=3, sigma=2.0, iters=5, method='poly',
                    fit_region=None, savgol_window=301, savgol_polyorder=3):
        """
        Estimate continuum, optionally using only data inside fit_region.

        Parameters
        ----------
        wl, flux : 1D arrays (floats)
        deg : polynomial degree (for method='poly')
        sigma, iters : sigma-clipping params
        method : 'poly' or 'savgol' (percentile method could be added)
        fit_region : tuple (xmin, xmax) or None. If provided, the iterative mask
                    uses only wl inside that region to determine the continuum fit.
                    The returned continuum is an array for the whole wl array.
        """
        wl = np.asarray(wl, dtype=float)
        flux = np.asarray(flux, dtype=float)

        # Basic finite-data mask
        good = np.isfinite(wl) & np.isfinite(flux)
        if not np.any(good):
            raise ValueError("No finite points in wl/flux")

        mask = good.copy()

        # If user requested a fit region, restrict the clipping/fitting to that region
        if fit_region is not None:
            xmin, xmax = fit_region
            region_mask = (wl >= xmin) & (wl <= xmax)
        else:
            region_mask = np.ones_like(wl, dtype=bool)

        # Start with mask restricted to the region (and finite points)
        mask = mask & region_mask

        # Iterative sigma clipping done only inside the region
        for _ in range(iters):
            if mask.sum() < (deg + 1):  # not enough points to fit
                break
            x_fit = wl[mask]
            y_fit = flux[mask]
            # scale x for numeric stability
            x_mean = x_fit.mean()
            x_std = x_fit.std() if x_fit.std() > 0 else 1.0
            x_scaled = (x_fit - x_mean) / x_std

            if method == 'poly':
                try:
                    coeffs = np.polyfit(x_scaled, y_fit, deg)
                    cont_est = np.polyval(coeffs, (wl - x_mean) / x_std)
                except np.linalg.LinAlgError:
                    cont_est = np.median(y_fit) * np.ones_like(flux)
            elif method == 'savgol':
                # savgol doesn't use masking directly, so compute a rough cont then clip
                w = int(savgol_window)
                if w % 2 == 0: w += 1
                w = min(w, max(3, len(flux) - (1 - len(flux) % 2)))
                cont_est = savgol_filter(flux, window_length=w, polyorder=savgol_polyorder, mode='interp')
            else:
                raise ValueError("Unsupported method")

            resid = flux - cont_est
            resid_mask = np.abs(resid - np.nanmedian(resid[mask])) < sigma * (np.nanstd(resid[mask]) if np.nanstd(resid[mask])>0 else 1.0)

            # keep only points that are good, inside the region, and not strong deviant
            new_mask = good & region_mask & resid_mask
            # If no change, break
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        # Final continuum: refit using final mask (if polynomial) or reuse cont_est (if savgol)
        if method == 'poly':
            if mask.sum() < (deg + 1):
                # fallback to median in region
                cont = np.median(flux[region_mask & good]) * np.ones_like(flux)
            else:
                x_fit = wl[mask]
                y_fit = flux[mask]
                x_mean = x_fit.mean(); x_std = x_fit.std() if x_fit.std()>0 else 1.0
                coeffs = np.polyfit((x_fit - x_mean) / x_std, y_fit, deg)
                cont = np.polyval(coeffs, (wl - x_mean) / x_std)
        else:  # savgol
            w = int(savgol_window)
            if w % 2 == 0: w += 1
            w = min(w, max(3, len(flux) - (1 - len(flux) % 2)))
            cont = savgol_filter(flux, window_length=w, polyorder=savgol_polyorder, mode='interp')

        # avoid zeros / negative continuum
        pos = cont > 0
        if not np.any(pos):
            cont = np.ones_like(cont) * np.nanmedian(flux[good])
        else:
            cont[~pos] = np.nanmedian(cont[pos])

        return cont





"""
import numpy as np
import random
import glob
from copy import deepcopy
from collections import Counter, defaultdict

# Astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u

# Astropy modeling and regions
from astropy.modeling import models, fitting
from regions import (
    PixCoord,
    SkyRegion,
    EllipsePixelRegion,
    EllipseSkyRegion,
    EllipseAnnulusPixelRegion,
    CirclePixelRegion,
)

# Scipy
from scipy.optimize import curve_fit
from scipy.stats import norm as stat_norm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

# Skimage
from skimage import measure
from skimage.draw import polygon
from skimage.measure import find_contours
from skimage.transform import rotate

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Dataclasses
from dataclasses import dataclass, field



from ..image import Image
from ..cube import Cube
from ..region import Region, Point, Contours
from ..filter import Filter, list_filters, plot_filters, map_filter_names, nice_filter_names

from ..data.spectral_lines import spectral_lines, UBGVRI_filters, atmospheric_lines


        
        
        
        
# FIT LINES BLOCK
        
        
from scipy.stats import norm as stat_norm
from scipy.optimize import curve_fit



# All lines share the same velocity dispersion (same Gaussian width in km/s) because they come from the same gas region and are kinematically coupled
# [N II] λ6583 / λ6548 flux ratio = 2.96 (quantum mechanics. Storey & Zeippen (2000))

lambda_Ha = spectral_lines['Ha']
lambda_NII_6548 = spectral_lines['[N II] 6548']
lambda_NII_6583 = spectral_lines['[N II] 6583']

def model_Ha_emission(wl, z, continuum, amplitude_Ha, amplitude_NII_6583, sigma):

    amplitude_NII_6548 = amplitude_NII_6583 / 2.96

    ha_profile     = amplitude_Ha        * stat_norm(loc=lambda_Ha*(1+z),      scale=sigma).pdf(wl)
    nii_6583_prof  = amplitude_NII_6583  * stat_norm(loc=lambda_NII_6583*(1+z), scale=sigma).pdf(wl)
    nii_6548_prof  = amplitude_NII_6548  * stat_norm(loc=lambda_NII_6548*(1+z), scale=sigma).pdf(wl)

    return continuum + ha_profile + nii_6583_prof + nii_6548_prof


lambda_Hb = spectral_lines['Hb']


def model_Hb_emission(wl, continuum, amplitude_Hb_em, amplitude_Hb_abs, z_abs, sigma_abs, *, z_em, sigma_em):
    return continuum + amplitude_Hb_em * stat_norm(loc=lambda_Hb*(1+z_em), scale=sigma_em).pdf(wl) + amplitude_Hb_abs * stat_norm(loc=lambda_Hb*(1+z_abs), scale=sigma_abs).pdf(w)

def model_Hb_emission(wl, continuum, amplitude_Hb, *, z, sigma):
    return continuum + amplitude_Hb * stat_norm(loc=lambda_Hb*(1+z), scale=sigma).pdf(wl)


# --- [OIII] doublet ---
# Very strong in low-metallicity/star-forming galaxies
# [O III] λ4959	always 1/3 the intensity of λ5007 (quantum mechanics)

lambda_OIII_5007 = spectral_lines['[O III] 5007']
lambda_OIII_4959 = spectral_lines['[O III] 4959']

def model_OIII(wl, continuum, amplitude_OIII_5007, *, sigma, z):
    amplitude_OIII_4959 = amplitude_OIII_5007 / 3
    return continuum + amplitude_OIII_5007 * stat_norm(loc=lambda_OIII_5007*(1+z), scale=sigma).pdf(wl) + \
           amplitude_OIII_4959 * stat_norm(loc=lambda_OIII_4959*(1+z), scale=sigma).pdf(wl)


# --- [SII] doublet ---
# [SII] 6716 and [SII] 6731 intensity ratio is very sensitive to electron density (nₑ) in the range 10^2-10^4 cm-3

lambda_SII_6716 = spectral_lines['[S II] 6716']
lambda_SII_6731 = spectral_lines['[S II] 6731']

def model_SII(wl, continuum, amplitude_SII_6716, amplitude_SII_6731, *, sigma, z):
    return continuum + amplitude_SII_6716* stat_norm(loc=lambda_SII_6716*(1+z), scale=sigma).pdf(wl) + \
           amplitude_SII_6731 * stat_norm(loc=lambda_SII_6731*(1+z), scale=sigma).pdf(wl)

# --- [HeI] 5876 + Na I D doublet

lambda_HeI = spectral_lines['He I 5876']

def model_HeI(wl, continuum, amplitude_HeI, *, sigma, z):
    return continuum + amplitude_HeI* stat_norm(loc=lambda_HeI*(1+z), scale=sigma).pdf(wl) 


### --- Balmer absorption lines --- ###

from scipy.special import wofz

def voigt(x, amp, center, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# the sigma (Gaussian width) and γ (Lorentzian width) in Voigt profiles are generally different across the Balmer series — not equal


def model_Balmer_absorption(wl, lambda_center, continuum, amplitude, sigma, gamma):
    line_profile = voigt(wl,amplitude,lambda_center,sigma, gamma)
    return continuum + line_profile 



from functools import partial

def fit_lines(self, spectra, z_mean=0, plot=True):

    lines = []
    for i, spec in enumerate(spectra):

        #print(f"Spectrum of region str(i+1))")
        
        wl = spec.wl 
        flux = spec.flux 
        err = spec.flux_err    


        mask_Ha = (wl > (lambda_Ha * (1 + z_mean) - 50)) & (wl< (lambda_Ha * (1 + z_mean) + 50))
        wlave_Ha = wl[mask_Ha]
        flux_Ha = flux[mask_Ha]
        err_Ha = err[mask_Ha]

        mask_Hb = (wl > (lambda_Hb * (1 + z_mean) - 50)) & (wl< (lambda_Hb * (1 + z_mean) + 50))
        wlave_Hb = wl[mask_Hb]
        flux_Hb = flux[mask_Hb]
        err_Hb = err[mask_Hb]

        mask_OIII = (wl > (lambda_OIII_4959 * (1 + z_mean) - 50)) & (wl< (lambda_OIII_5007 * (1 + z_mean) + 50))
        wlave_OIII = wl[mask_OIII]
        flux_OIII = flux[mask_OIII]
        err_OIII = err[mask_OIII]

        mask_SII = (wl > (lambda_SII_6716 * (1 + z_mean) - 50)) & (wl< (lambda_SII_6731 * (1 + z_mean) + 50))
        wlave_SII = wl[mask_SII]
        flux_SII = flux[mask_SII]
        err_SII = err[mask_SII]

        mask_HeI = (wl > (lambda_HeI * (1 + z_mean) - 50)) & (wl< (lambda_HeI * (1 + z_mean) + 50))
        wlave_HeI = wl[mask_HeI]
        flux_HeI = flux[mask_HeI]
        err_HeI = err[mask_HeI]
        

        median_Ha = np.median(flux_Ha)
        median_Hb = np.median(flux_Hb)


        # Check if the line is in emission or absorption
        #def mad(arr):
        #    return np.median(np.abs(arr - np.median(arr)))

        #peak_Ha = np.max(flux_Ha)
        #peak_Hb = np.max(flux_Hb)
        #cont_Ha = np.median(flux_Ha)
        #cont_Hb = np.median(flux_Hb)
        #noise_Ha = mad(flux_Ha)
        #noise_Hb = mad(flux_Hb)

        #if (peak_Ha - cont_Ha) / noise_Ha > 3 or (peak_Hb - cont_Hb) / noise_Hb > 3:
        #    print("LINES IN EMISSION")

        if flux_Ha[np.argmax(abs(flux_Ha - median_Ha))] >  median_Ha: # and flux_Hb[np.argmax(abs(flux_Hb - median_Hb))] > median_Hb:
            print('LINES IN EMISSION')

            q1, q2 = np.nanquantile(flux_Ha, [0.16, 0.84])
            subset = flux_Ha[(flux_Ha >= q1) & (flux_Ha <= q2)]
            estimated_err = np.std(subset)

            guess = [z_mean, np.median(flux_Ha), np.max(flux_Ha), np.max(flux_Ha)/2, 2.0]
            popt_Ha, pcov_Ha = curve_fit(model_Ha_emission, wave_Ha, flux_Ha, p0=guess, sigma=err_Ha, absolute_sigma=True)
            z_fit, cont_Ha, amp_Ha, amp_NII_6583, sigma_fit = popt_Ha

            model_vals = model_Ha_emission(wave_Ha, *popt_Ha)
            residuals = flux_Ha - model_vals
            chi2 = np.sum((residuals / estimated_err) ** 2)
            dof = len(wave_Ha) - len(popt_Ha)
            chi2_red_Ha = chi2 / dof


            # Initial guess for Hβ fit:
            # [continuum, amplitude_Hb_emission, amplitude_Hb_absorption, z_absorption, sigma_absorption]
            
            #guess_Hb = [
            #    np.median(flux_Hb),        # continuum
            #    np.max(flux_Hb),           # emission amplitude (positive)
            #    -np.abs(np.min(flux_Hb)),  # absorption amplitude (negative)
            #    z_fit,                     # absorption z guess (start with emission z)
            #    sigma_fit                  # absorption sigma guess (start with emission sigma)
            #]
            
            guess_Hb = [np.median(flux_Hb), np.max(flux_Hb)]

            # Fix z_em and sigma_em from Ha fit via partial
            #model_Hb_fixed = partial(model_Hb_emission, z_em=z_fit, sigma_em=sigma_fit)
            model_Hb_fixed = partial(model_Hb_emission, z=z_fit, sigma=sigma_fit)


            # Define bounds to enforce absorption amplitude negative:
            bounds_Hb = (
                [-np.inf,       0,       -np.abs(np.min(flux_Hb)),  z_fit - 0.001,  0],   # lower bounds: continuum, amplitude_em >=0, amplitude_abs <0, small z range, sigma > 0
                [np.inf,  1.5*max(flux_Hb),          0,    z_fit + 0.001,  8]  # upper bounds: continuum, amplitude_abs < 0 so max 0
            )

            popt_Hb, pcov_Hb = curve_fit(
                model_Hb_fixed, wave_Hb, flux_Hb, p0=guess_Hb, sigma=err_Hb, absolute_sigma=True,) #bounds=bounds_Hb
            
            #cont_Hb, amp_Hb_em, amp_Hb_abs, z_abs, sigma_abs = popt_Hb
            cont_Hb, amp_Hb = popt_Hb

            model_vals = model_Hb_fixed(wave_Hb, *popt_Hb)
            residuals = flux_Hb - model_vals
            chi2 = np.sum((residuals / err_Hb) ** 2)
            dof = len(wave_Hb) - len(popt_Hb)
            chi2_red_Hb = chi2 / dof




            # --- Fit OIII doublet

            model_OIII_fixed = partial(model_OIII, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_OIII), np.max(flux_OIII)]
            popt_OIII, pcov_OIII = curve_fit(model_OIII_fixed, wave_OIII, flux_OIII, p0=guess, sigma=err_OIII, absolute_sigma=True)
            cont_OIII, amp_OIII = popt_OIII

            model_vals = model_OIII_fixed(wave_OIII, *popt_OIII)
            residuals = flux_OIII - model_vals
            chi2 = np.sum((residuals / err_OIII) ** 2)
            dof = len(wave_OIII) - len(popt_OIII)
            chi2_red_OIII = chi2 / dof


            # --- Fit SII doublet

            model_SII_fixed = partial(model_SII, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_SII), np.max(flux_SII), np.max(flux_SII)]
            popt_SII, pcov_SII = curve_fit(model_SII_fixed, wave_SII, flux_SII, p0=guess, sigma=err_SII, absolute_sigma=True)
            cont_SII, amp_SII_6716, amp_SII_6731 = popt_SII
            
            model_vals = model_SII_fixed(wave_SII, *popt_SII)
            residuals = flux_SII - model_vals
            chi2 = np.sum((residuals / err_SII) ** 2)
            dof = len(wave_SII) - len(popt_SII)
            print(chi2)
            print(dof)
            chi2_red_SII = chi2 / dof

            # --- Fit HeI + Na doublet

            model_HeI_fixed = partial(model_HeI, z=z_fit, sigma=sigma_fit)

            guess = [np.median(flux_HeI), np.max(flux_HeI)]
            popt_HeI, pcov_HeI = curve_fit(model_HeI_fixed, wave_HeI, flux_HeI, p0=guess, sigma=err_HeI, absolute_sigma=True)
            cont_HeI, amp_HeI = popt_HeI
            
            model_vals = model_HeI_fixed(wave_HeI, *popt_HeI)
            residuals = flux_HeI - model_vals
            chi2 = np.sum((residuals / err_HeI) ** 2)
            dof = len(wave_HeI) - len(popt_HeI)
            print(chi2)
            print(dof)
            chi2_red_HeI = chi2 / dof

            if plot:
                fig, ax = plt.subplots(2, 3, figsize=(12, 6))
                ax = ax.flatten()

                w_array = np.linspace(wave_Ha[0], wave_Ha[-1], 1000)
                ax[0].errorbar(wave_Ha, flux_Ha, yerr=err_Ha, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[0].plot(wave_Ha, flux_Ha, linestyle='-', alpha=0.3, color='black')
                ax[0].plot(w_array, cont_Ha + amp_Ha*stat_norm(lambda_Ha*(1+z_fit),sigma_fit).pdf(w_array), color='red',label='$H\\alpha$')
                ax[0].plot(w_array, cont_Ha + (amp_NII_6583/2.96)*stat_norm(lambda_NII_6548*(1+z_fit),sigma_fit).pdf(w_array), color='orange',label='[NII] 6548')
                ax[0].plot(w_array, cont_Ha + amp_NII_6583*stat_norm(lambda_NII_6583*(1+z_fit),sigma_fit).pdf(w_array), color='darkgreen', label='[NII] 6583')
                ax[0].plot(w_array, np.full(w_array.shape, cont_Ha), color='navy', linewidth=1)
                ax[0].legend(loc='upper right', frameon=False)
                ax[0].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_Ha:.2f}", transform=ax[0].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_Hb[0], wave_Hb[-1], 1000)
                ax[1].errorbar(wave_Hb, flux_Hb, yerr=err_Hb, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[1].plot(wave_Hb, flux_Hb, linestyle='-', alpha=0.3, color='black')
                
                ax[1].plot(w_array, cont_Hb + amp_Hb*stat_norm(lambda_Hb*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='$H\\beta$')
                #ax[1].plot(w_array, cont_Hb + amp_Hb_abs*stat_norm(lambda_Hb*(1+z_abs), sigma_abs).pdf(w_array), color='red')
                ax[1].plot(w_array, np.full(w_array.shape, cont_Hb), color='navy', linewidth=1)
                ax[1].legend(loc='upper right', frameon=False)
                ax[1].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_Hb:.2f}", transform=ax[1].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_OIII[0], wave_OIII[-1], 1000)
                ax[2].errorbar(wave_OIII, flux_OIII, yerr=err_OIII, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[2].plot(wave_OIII, flux_OIII, linestyle='-', alpha=0.3, color='black')
                ax[2].plot(w_array, cont_OIII + amp_OIII*stat_norm(lambda_OIII_5007*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[OIII] 5007')
                ax[2].plot(w_array, cont_OIII + (amp_OIII/3)*stat_norm(lambda_OIII_4959*(1+z_fit), sigma_fit).pdf(w_array), color='lime',label='[OIII] 4959')
                ax[2].plot(w_array, np.full(w_array.shape, cont_OIII), color='navy', linewidth=1)
                ax[2].legend(loc='upper right', frameon=False)
                ax[2].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_OIII:.2f}", transform=ax[2].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_SII[0], wave_SII[-1], 1000)
                ax[3].errorbar(wave_SII, flux_SII, yerr=err_SII, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[3].plot(wave_SII, flux_SII, linestyle='-', alpha=0.3, color='black')
                ax[3].plot(w_array, cont_SII + amp_SII_6716*stat_norm(lambda_SII_6716*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[SII] 6716')
                ax[3].plot(w_array, cont_SII + amp_SII_6731*stat_norm(lambda_SII_6731*(1+z_fit), sigma_fit).pdf(w_array), color='lime',label='[SII] 6731')
                ax[3].plot(w_array, np.full(w_array.shape, cont_SII), color='navy', linewidth=1)
                ax[3].legend(loc='upper right', frameon=False)
                ax[3].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_SII:.2f}", transform=ax[3].transAxes, ha='left', va='top', fontsize=8, color='black')

                w_array = np.linspace(wave_HeI[0], wave_HeI[-1], 1000)
                ax[4].errorbar(wave_HeI, flux_HeI, yerr=err_HeI, fmt='o', markersize=3, color='black', alpha=0.8)
                ax[4].plot(wave_HeI, flux_HeI, linestyle='-', alpha=0.3, color='black')
                ax[4].plot(w_array, cont_HeI + amp_HeI*stat_norm(lambda_HeI*(1+z_fit), sigma_fit).pdf(w_array), color='purple',label='[HeI] 5876')
                ax[4].plot(w_array, np.full(w_array.shape, cont_HeI), color='navy', linewidth=1)
                ax[4].legend(loc='upper right', frameon=False)
                ax[4].text(0.02, 0.98, f"$\\tilde{{\chi^2}}=$ {chi2_red_HeI:.2f}", transform=ax[4].transAxes, ha='left', va='top', fontsize=8, color='black')

                for i in range(5):
                    ax[i].set_xlabel("Wavelength [Å]")
                ax[0].set_ylabel("Flux [erg/s/cm²/Å/arcsec²]")
                ax[3].set_ylabel("Flux [erg/s/cm²/Å/arcsec²]")

                sigma_v = (sigma_fit / lambda_Ha) * 2.99e5 * (z_fit-z_mean)
                fig.suptitle(f"Region {i+1}: $z = {z_fit:.5f} \pm {np.sqrt(pcov_Ha[0, 0]):.5f}$, $\sigma_v = {sigma_v:.5f} km/s$")

                plt.tight_layout()
                plt.show()

            lines.append(None)

        else:
            print("Line in absorption")

            if plot:
                fig, ax = plt.subplots(1, 2, figsize=(12, 3))

            for i, balmer_line in enumerate(['Ha', 'Hb']):
                lambda_center = spectral_lines[balmer_line] * (1 + z_mean)
                mask_fit = (w > (lambda_center - 50)) & (w < (lambda_center + 50))
                wave_fit = w[mask_fit]
                flux_fit = flux[mask_fit]
                err_fit = err[mask_fit]

                guess = [lambda_center, np.median(flux_fit), np.max(flux_fit), 2.0, 2.0]
                popt, pcov = curve_fit(model_Balmer_absorption, wave_fit, flux_fit, p0=guess, sigma=err_fit, absolute_sigma=True)
                lambda_center_fit, cont_fit, amp_fit, sigma_fit, gamma_fit = popt
                z_fit = lambda_center_fit / spectral_lines[balmer_line] - 1

                model_fit = model_Balmer_absorption(wave_fit, *popt)

                if plot:
                    ax[i].errorbar(wave_fit, flux_fit, yerr=err_fit, fmt='o', markersize=3, color='black', alpha=0.8)
                    ax[i].plot(wave_fit, flux_fit, linestyle='-', alpha=0.3, color='black')
                    ax[i].plot(wave_fit, model_fit, 'r--', label='Voigt fit')
                    ax[i].set_title(f"{balmer_line} fit")
                    ax[i].set_xlabel("Wavelength [Å]")

    return lines
        



"""




