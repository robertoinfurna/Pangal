import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from astropy.io import fits
from bisect import bisect_left
from scipy.special import erf, erfinv
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d 
from scipy.signal import fftconvolve
import itertools
import os

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import corner

from .photometry_table import PhotometryTable
from .spectrum import Spectrum
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps
from .data.spectral_lines import spectral_lines, UBGVRI_filters, atmospheric_lines
from .pfitter_utils import load_nebular_tables, load_dust_emission_models, model_grid_interpolator, load_spectrum_models_from_fits


from PIL import Image

"""
PFitter
   ├── run_fit(...)    → creates a new Run() instance
   │                      fills it with settings + results
   │                      stores it in self.runs list
   ├── many other functions
   └── runs : list[Run]

Run
   ├── stores: data, phot, spectrum, priors
   ├── stores: processed observed spectrum (normalized)
   ├── stores: obs resolution on model grid
   ├── stores: sampler result
   ├── stores: likelihood function
   └── lightweight methods if needed
"""

class Run:

    def __init__(self):

        # Fit configuration
        self.model_file = None
        self.model_pars = None
        self.bands = []
        self.spectral_range = None
        self.treat_as_upper_limits = []
        self.fix_pars = {}
        self.custom_priors = {}
        self.polydeg = None
        self.nlive = None
        self.dlogz = None

        # Input and preprocessed data
        self.spec = None
        self.spec_crop = None
        self.phot = None

        # Parameter bookkeeping
        self.free_model_pars = []
        self.free_pars = []

        # Likelihood, priors, sampler, posterior
        self.log_likelihood = None
        self.prior_transform = None
        self.sampler = None
        self.result = None

        # Posterior point estimates
        self.best_model_pars_MAP = None
        self.best_model_pars_ML = None
        self.best_model_pars_median = None

        # Evaluated arrays used by Run.plot_retrieval()
        self.best_models = {}



    def best_model(self, method='MAP', quantiles=None):

        if self.result is None:
            raise ValueError("No fit results found in this Run object.")
        if not hasattr(self, "free_pars"):
            raise ValueError("Run has no 'free_pars' attribute.")
        if self.fix_pars is None:
            self.fix_pars = {}

        samples = self.result.samples
        weights = np.exp(self.result.logwt - self.result.logz[-1])
        weights = weights / np.sum(weights)
        param_names = self.free_pars

        def weighted_quantile_1d(x, qs, w):
            x = np.asarray(x)
            w = np.asarray(w)
            qs = np.asarray(qs)

            sorter = np.argsort(x)
            x_sorted = x[sorter]
            w_sorted = w[sorter]

            cdf = np.cumsum(w_sorted)
            cdf = cdf / cdf[-1]

            cdf = np.concatenate([[0.0], cdf])
            x_sorted = np.concatenate([[x_sorted[0]], x_sorted])

            return np.interp(qs, cdf, x_sorted)

        # --- Select best parameters ---
        if method == 'MAP':
            logpost = self.result.logl + np.log(weights + 1e-300)
            idx = np.argmax(logpost)
            best_vals = samples[idx]

        elif method == 'ML':
            idx = np.argmax(self.result.logl)
            best_vals = samples[idx]

        elif method == 'median':
            best_vals = np.array([
                weighted_quantile_1d(samples[:, i], 0.5, weights)
                for i in range(samples.shape[1])
            ])
        else:
            raise ValueError(f"Unknown method '{method}'")

        best_params = {name: val for name, val in zip(param_names, best_vals)}
        best_params.update(self.fix_pars)

        # --- OLD behaviour ---
        if quantiles is None:
            return best_params

        # --- Quantiles requested ---
        if (not isinstance(quantiles, (tuple, list))) or len(quantiles) != 2:
            raise ValueError("quantiles must be (q_lo, q_hi)")
        q_lo, q_hi = quantiles

        out = {}

        # free parameters
        for i, name in enumerate(param_names):
            lo, hi = weighted_quantile_1d(samples[:, i], [q_lo, q_hi], weights)
            out[name] = (lo, best_params[name], hi)

        # fixed parameters
        for name, val in self.fix_pars.items():
            out[name] = (val, val, val)

        return out
        
            
        
    def plot_retrieval(
        self,
        solution="MAP",

        # spectral plotting window
        winf_spec=None,
        wsup_spec=None,

        # photometric plotting window
        winf_phot=1e2,
        wsup_phot=1e7,
        ymin_phot=None,
        ymax_phot=None,

        # aesthetics
        figsize=(12, 6),
        color="crimson",
        synth_phot_color="navy",
        show_errorbars=True,

        observed_spec_label="Observed spectrum",
        model_label="Best-fit model",
        spec_legend_pars = None,
        spec_legend_loc="upper right",
        phot_legend_loc="upper right",
        spec_legend_fontsize=14,
        label_fontsize=14,
        show_phot_legend=True,

        show_title=True,
        title=None,
        title_fontsize=14,
        title_color="black",
        title_fontweight="normal",

        filename=None,
    ):
        """
        Plot an already-evaluated posterior solution stored in ``run``.
        No PFitter instance is used and no model is recomputed.
        """

        if not hasattr(self, "best_models"):
            raise ValueError(
                "This Run object has no 'best_models'. "
                "Run evaluate_best_model() at the end of run_fit()."
            )

        if solution not in self.best_models:
            raise ValueError(
                f"Unknown solution '{solution}'. "
                f"Available: {list(self.best_models.keys())}"
            )

        best_model = self.best_models[solution]

        params = best_model["params"]
        spec_data = best_model["spectrum"]
        phot_data = best_model["photometry"]

        saved_figures = []

        # ==========================================================
        # Photometric SED
        # ==========================================================
        
        # Rebuild a Spectrum object only for its .plot() method.
        # No PFitter and no model evaluation are involved here.
        model_header = fits.Header()
        model_header["WUNITS"] = "A"
        model_header["FUNITS"] = "erg/s/cm2/A"

        # Parameter names used in synthetic_spectrum() FITS headers
        header_key_map = {
            "burst_mass_fraction": "BURSTF",
            "fesc": "FESC",
            "ion_gas": "IONGAS",
            "age_gas": "AGEGAS",
            "av": "AV",
            "av_ext": "AVEXT",
            "alpha": "ALPHA",
            "log_m_star": "LOGMSTAR",
            "redshift": "REDSHIFT",
            "luminosity_distance": "DLMPC",
            "vel_sys": "VELSYS",
            "sigma_vel": "SIGMAVEL",
            "sigma_gas": "SIGMAGAS",
            "spec_noise_scale": "SPECNOI",
        }
        
        for par_name, value in params.items():
        
            # Grid parameters such as TRUNCAGE, TRUNCTAU, V_DISC,
            # R_GAL, and METAL retain their original names.
            header_key = header_key_map.get(
                par_name,
                par_name[:8].upper(),
            )

            
            model_header[header_key] = value

        if phot_data is not None:
            
            synth_spec = Spectrum(
                wl=best_model["model_wl"],
                flux=best_model["model_flux"],
                resolution=np.ones_like(best_model["model_wl"]),
                header=model_header,
            )
        
            # Rebuild photometry in the format expected by Spectrum.plot().
            phot_header = fits.Header()
            phot_header["UNITS"] = "mJy"
        
            phot_loc_data = {}
        
            for band, flux, err, is_ul in zip(
                phot_data["bands"],
                phot_data["flux_obs"],
                phot_data["flux_err"],
                phot_data["upper_limits"],
            ):
        
                # Spectrum.plot convention for an upper limit: (0, limit)
                if is_ul:
                    phot_loc_data[band] = (0.0, flux)
                else:
                    phot_loc_data[band] = (flux, err)
        
            phot_loc = PhotometryTable(
                data=phot_loc_data,
                header=phot_header,
            )

            # Same photometric likelihood diagnostics as test_model()
            obs_phot_like = phot_data["flux_obs"].copy()
            obs_err_like = phot_data["flux_err"].copy()
            model_phot_like = phot_data["flux_model"]
            upper_limits = phot_data["upper_limits"]

            obs_err_like[upper_limits] = obs_phot_like[upper_limits]
            obs_phot_like[upper_limits] = 0.0

            chi2_phot = 0.0
            logl_phot = 0.0

            for obs, err, mod in zip(
                obs_phot_like,
                obs_err_like,
                model_phot_like,
            ):

                if obs > 0:
                    resid = (obs - mod) / err
                    chi2_phot += resid**2
                    logl_phot += -0.5 * (
                        resid**2
                        + np.log(2 * np.pi * err**2)
                    )

                else:
                    arg = (obs - mod) / (np.sqrt(2.0) * err)
                    cdf = 0.5 * (1.0 + erf(arg))

                    logl_phot += np.log(cdf)
                    chi2_phot += -2.0 * np.log(cdf)

            phot_title = "" if title is None else title

            phot_title += (
                rf"     $A_v={params['av']:.2f}$, "
                rf"${{A_v}}_{{\rm extra}}={params['av_ext']:.2f}$, "
                rf"$\alpha_{{\rm Dale}}={params['alpha']:.2f}$, "
                rf"$\log M_\star={params['log_m_star']:.2f}$"
            )

            phot_title += (
                rf",    $\chi^2={chi2_phot:.2f}$, "
                rf"$\log \mathcal{{L}}={logl_phot:.2f}$"
            )
        
            if filename is not None:
                filename_phot = filename + "_phot.png"
            else:
                filename_phot = None

            
            synth_spec.plot(
                per_wavelength=True,
                winf=winf_phot,
                wsup=wsup_phot,
                ymin=ymin_phot,
                ymax=ymax_phot,
        
                figsize=figsize,
                color=color,
                synth_phot_color=synth_phot_color,
        
                redshift=None,
                show_top_spectral_lines=False,
                show_all_spectral_lines=False,
                show_atmospheric_lines=False,
                show_filters=False,
        
                label_fontsize=label_fontsize,
                phot=phot_loc,
                synth_phot=phot_data["bands"],
        
                spec_legend_pars=spec_legend_pars,
                show_phot_legend=show_phot_legend,
                spec_legend_loc=phot_legend_loc,
                spec_legend_title=None,
                spec_legend_fontsize=spec_legend_fontsize,
                
                title=phot_title if show_title else None,
                title_fontsize=title_fontsize,
                title_fontweight=title_fontweight,
                title_loc="center",
                title_color=title_color,
        
                filename=filename_phot,
            )


        
        # ==========================================================
        # Spectrum + residuals
        # ==========================================================
        if spec_data is not None:

            wl = spec_data["wl"]
            flux_obs = spec_data["flux_obs"]
            flux_err = spec_data["flux_err"]
            flux_model = spec_data["flux_model"]
            residuals = spec_data["residuals"]
            chi2 = spec_data["chi2"]
            logl = spec_data["logl"]

            fig_spec, (ax1, ax2) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=figsize,
                gridspec_kw={
                    "height_ratios": [3, 1],
                    "hspace": 0.0,
                },
            )

            ax1.plot(
                wl,
                flux_obs,
                color="black",
                lw=1.0,
                label=observed_spec_label,
            )

            ax1.plot(
                wl,
                flux_model,
                color=color,
                lw=1.2,
                label=model_label,
            )

            if show_errorbars:
                ax1.fill_between(
                    wl,
                    flux_obs - flux_err,
                    flux_obs + flux_err,
                    color="black",
                    alpha=0.18,
                    lw=0,
                )

            ax1.set_ylabel(
                r"Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)",
                fontsize=label_fontsize - 1,
            )

            ax1.legend(
                frameon=False,
                loc=spec_legend_loc,
                fontsize=spec_legend_fontsize,
            )

            ax2.axhline(
                0.0,
                color="black",
                lw=0.8,
                ls="--",
            )

            ax2.plot(
                wl,
                residuals,
                color="steelblue",
                lw=0.8,
            )

            ax2.set_xlabel(
                r"Wavelength ($\AA$)",
                fontsize=label_fontsize,
            )
            ax2.set_ylabel(
                "Residuals",
                fontsize=label_fontsize - 1,
            )

            if winf_spec is None:
                winf_spec = np.nanmin(wl)

            if wsup_spec is None:
                wsup_spec = np.nanmax(wl)

            ax1.set_xlim(winf_spec, wsup_spec)

            finite_residuals = residuals[np.isfinite(residuals)]

            if len(finite_residuals) > 0:
                rmax = np.nanpercentile(
                    np.abs(finite_residuals),
                    99,
                )

                if rmax > 0:
                    ax2.set_ylim(-rmax, rmax)

            title_spec = (
                f"{solution}: "
                rf"$f_{{\rm esc}}={params.get('fesc', np.nan):.2f}$, "
                rf"$\mathcal{{U}}_{{\rm ion}}={params.get('ion_gas', np.nan):.2f}$, "
                rf"${{\rm Age}}_{{\rm gas}}={params.get('age_gas', np.nan):.1f}$ Myr"
                "\n"
                rf"$\chi^2={chi2:.2f}$, "
                rf"$\log \mathcal{{L}}={logl:.2f}$"
            )

            if title is not None:
                title_spec = title

            spec_pars_text = (
                rf"$v_{{\rm sys}}={params['vel_sys']:.1f}\ "
                rf"{{\rm km\,s^{{-1}}}}$,   "
                rf"$\sigma_v={params['sigma_vel']:.1f}\ "
                rf"{{\rm km\,s^{{-1}}}}$,   "
                rf"$\mathcal{{U}}_{{\rm ion}}={params['ion_gas']:.2f}$,   "
                rf"${{\rm Age}}_{{\rm gas}}={params['age_gas']:.1f}\ "
                rf"{{\rm Myr}}$,   "
                rf"$\sigma_{{\rm gas}}={params['sigma_gas']:.1f}\ "
                rf"{{\rm km\,s^{{-1}}}}$"
            )
            
            ax1.text(
                0.5,
                1.02,
                spec_pars_text,
                transform=ax1.transAxes,
                ha="center",
                va="bottom",
                fontsize=title_fontsize,
                color=title_color,
                fontweight=title_fontweight,
                clip_on=False,
            )

            ax1.tick_params(labelsize=label_fontsize - 2)
            ax2.tick_params(labelsize=label_fontsize - 2)

            fig_spec.tight_layout()

            if filename is not None:
            
                spec_filename = filename + "_spec.png"
            
                fig_spec.savefig(
                    spec_filename,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )
            
                saved_figures.append(spec_filename)


        if (
            filename is not None
            and phot_data is not None
            and spec_data is not None
        ):

            file_phot = filename + "_phot.png"
            file_spec = filename + "_spec.png"

            img_phot = Image.open(file_phot)
            img_spec = Image.open(file_spec)

            max_width = max(img_phot.width, img_spec.width)

            def resize_to_width(img, width):
                if img.width == width:
                    return img

                ratio = width / img.width
                new_height = int(img.height * ratio)

                return img.resize(
                    (width, new_height),
                    Image.LANCZOS,
                )

            img_phot = resize_to_width(img_phot, max_width)
            img_spec = resize_to_width(img_spec, max_width)

            combined = Image.new(
                "RGB",
                (max_width, img_phot.height + img_spec.height),
                color="white",
            )

            combined.paste(img_phot, (0, 0))
            combined.paste(img_spec, (0, img_phot.height))
            combined.save(filename + ".png")

            img_phot.close()
            img_spec.close()

            os.remove(file_phot)
            os.remove(file_spec)

            saved_figures = [filename + ".png"]

            plt.show()


            
            
    def cornerplot(
            self,
            parameters=None,
            show_stats=True,
            show_histograms=True,
            labels=None,
            cmap=None,
            contour_color='k',
            alpha=0.7,
            mark_point=None,
            mark_color="crimson",
            mark_lw=2.0,
            filename=None,
            label_fontsize=11,
            x_label_pad=0.02,
            y_label_pad=0.02,
            title=None,
            title_fontsize=10,
            figsize=None,
        ):
            if not hasattr(self, "result"):
                raise ValueError("Run has no .result attribute. Did you run run_fit()?")

            samples = self.result.samples
            logwt   = self.result.logwt
            logz    = self.result.logz

            weights = np.exp(logwt - logz[-1])
            equal_samples = resample_equal(samples, weights)

            # ---- labels ----
            if labels:
                latex_labels = labels
            else:
                latex_labels = {
                    "age": r"$\mathrm{Age~[Myr]}$",
                    "tau_main": r"$\tau_\mathrm{main}\ \mathrm{[Myr]}$",
                    "TRUNCAGE": r"$Q_\mathrm{AGE}\ \mathrm{[Myr]}$",
                    "TRUNCTAU": r"$\tau_Q\ \mathrm{[Myr]}$",
                    "BURST": r"burst factor",
                    "METAL": r"metallicity [$Z_\odot$]",
                    "fesc": r"$f_\mathrm{esc}$",
                    "ion_gas": r"$U_\mathrm{ion}$",
                    "age_gas": r"$\mathrm{Age_{gas}}$",
                    "av": r"$A_V$",
                    "av_ext": r"$A_{V,\mathrm{ext}}$",
                    "alpha": r"$\alpha$",
                    "log_m_star": r"$\log M_\star$",
                    "vel_sys": r"$v\ \mathrm{[km/s]}$",
                    "sigma_vel": r"$\sigma_v\ \mathrm{[km/s]}$",
                    "sigma_gas": r"$\sigma_\text{gas}\ \mathrm{[km/s]}$",
                    "redshift": r"redshift",
                    "luminosity_distance": r"luminosity distance",
                    "spec_noise_scale": r"spectral noise scaling parameter",
                    "R_GAL": r"$R_\text{gal}$",
                    "V_DISC": r"$v_\text{disc}$",
                    "burst_mass_fraction": r"$f_\text{burst}$",
                }

            # ---- parameter selection ----
            if parameters is None:
                selected_pars = list(self.free_pars)
                indices = list(range(len(self.free_pars)))
            else:
                missing = [p for p in parameters if p not in self.free_pars]
                if missing:
                    raise ValueError(f"Requested parameters not in free_pars: {missing}")
                selected_pars = list(parameters)
                indices = [self.free_pars.index(p) for p in selected_pars]
                equal_samples = equal_samples[:, indices]

            plot_labels = [latex_labels.get(p, p) for p in selected_pars]
            n = len(selected_pars)

            
            # ---- create figure ----
            plt.close("all")
            if figsize is None: figsize = (2*n,2*n)
            fig = plt.figure(figsize=figsize)

            # ---- corner plot ----
            fig = corner.corner(
                equal_samples,
                labels=plot_labels,
                range=[0.99] * n,
                plot_contours=True,
                fill_contours=True if cmap else None,
                contourf_kwargs={
                        "cmap": cm.get_cmap(cmap),
                        "colors": None,
                        "alpha": alpha,
                    } if cmap else None,
                contour_kwargs={"colors": contour_color} if cmap else None,
                color=contour_color if not cmap else None,
                fig=fig,
            )

            axes = np.array(fig.axes).reshape((n, n))

            # ---- hide upper triangle if needed ----
            if not show_histograms:
                for row in range(n):
                    for col in range(n):
                        if col >= row:
                            axes[row, col].set_visible(False)

            diag_axes = [axes[i, i] for i in range(n)]

            # ---- stats ----
            if show_stats and show_histograms:

                if mark_point is None:
                    vals = np.mean(equal_samples, axis=0)
                    errs = np.std(equal_samples, axis=0)

                elif isinstance(mark_point, dict):
                    param_names = np.array(parameters)  # if parameters is already plotted order
            
                    vals = np.array([mark_point[name] for name in param_names])
                    errs = None

                elif str(mark_point).lower() == "median":
                    vals = np.median(equal_samples, axis=0)
                    errs = np.std(equal_samples, axis=0)

                elif str(mark_point).lower() in ("maxlogl", "map"):
                    idx_best = int(np.argmax(self.result.logl))
                    pt_full = samples[idx_best]
                    vals = pt_full[indices]
                    errs = None

                for i, ax in enumerate(diag_axes):
                    if errs is None:
                        ax.set_title(f"{vals[i]:.3f}", fontsize=10, pad=12)
                    else:
                        ax.set_title(f"{vals[i]:.3f} ± {errs[i]:.3f}", fontsize=10, pad=12)

        
            # ---- crosshair ----
            if mark_point is not None:
            
                param_names = np.array(self.free_pars)[indices]
            
                # Allow single point or list of points
                if isinstance(mark_point, list):
                    mark_points = mark_point
                else:
                    mark_points = [mark_point]
            
                # Allow single color or list of colors
                if isinstance(mark_color, list):
                    mark_colors = mark_color
                else:
                    mark_colors = [mark_color] * len(mark_points)
            
                for mp, mc in zip(mark_points, mark_colors):
            
                    if isinstance(mp, dict):
                        # parameter order must match plotted dimensions
                        pt = np.array([mp[name] for name in param_names])
            
                    elif str(mp).lower() == "median":
                        pt = np.median(equal_samples, axis=0)
            
                    elif str(mp).lower() == "ml":
                        idx_best = np.argmax(self.result.logl)
                        pt = samples[idx_best][indices]
                    
                    elif str(mp).lower() == "map":
                        logpost = self.result.logl + np.log(weights + 1e-300)
                        idx_best = np.argmax(logpost)
                        pt = samples[idx_best][indices]
            
                    for i in range(n):
                        for j in range(i):
                            ax = axes[i, j]
                            #ax.axvline(pt[j], color=mc, lw=mark_lw)
                            #ax.axhline(pt[i], color=mc, lw=mark_lw)

                            ax.scatter(
                                pt[j], pt[i],
                                color=mc,
                                s=400,          
                                linewidths=3,
                                marker="+",    # or ".", "+", "x", etc.
                                zorder=100
                            )
                    
                    if show_histograms:
                        for i, ax in enumerate(diag_axes):
                            ax.axvline(pt[i], color=mc, lw=mark_lw)
            
            

        
            for ax in fig.axes:
                ax.tick_params(axis='x', labelrotation=0)
                ax.tick_params(axis='y', labelrotation=90)

            # remove default labels
            for row in range(n):
                for col in range(n):
                    axes[row, col].set_xlabel("")
                    axes[row, col].set_ylabel("")

            left, bottom = 0.10, 0.10
            right = left + ((n - 1) / n) * (1.0 - left)
            top   = bottom + ((n - 1) / n) * (0.95 - bottom)

            fig.subplots_adjust(left=left,bottom=bottom,right=right,top=top,hspace=0.05,wspace=0.05,)        
            fig.canvas.draw()

        
            if title is not None:
                # find first visible axis (top-left of your visible grid)
                for ax in fig.axes:
                    if ax.get_visible():
                        target_ax = ax
                        break
            
                pos = target_ax.get_position()
            
                fig.text(
                    pos.x0,           # align with left edge of first axis
                    pos.y1 + 0.02,    # slightly above it
                    title,
                    ha="left",
                    va="bottom",
                    fontsize=title_fontsize,
                )

        
            # x labels
            for col in range(n):
                ax = axes[n-1, col]
                if not ax.get_visible():
                    continue

                pos = ax.get_position()
                fig.text((pos.x0 + pos.x1) / 2, pos.y0 - x_label_pad, plot_labels[col],ha='center',va='top',fontsize=label_fontsize,)

            # y labels
            for row in range(n):
                ax = axes[row, 0]
                if not ax.get_visible():
                    continue

                pos = ax.get_position()
                fig.text(pos.x0 - y_label_pad,(pos.y0 + pos.y1) / 2,plot_labels[row],ha='right',va='center',rotation=90,fontsize=label_fontsize,)

            """
            k = selected_pars.index("av")
        
            # bottom axis
            axes[-1, k].set_xticks([0.0, 0.005, 0.010, 0.015])
            axes[-1, k].set_xticklabels(["0", "0.005", "0.01", "0.015"])
            axes[k, 0].set_yticks([0.005, 0.010, 0.015])
            axes[k, 0].set_yticklabels(["0.005", "0.01", "0.015"])
            """                       

            # ---- save ----
            if filename is not None:
                if not str(filename).lower().endswith(".png"):
                    filename = f"{filename}.png"
                fig.savefig(filename, dpi=300, bbox_inches="tight")

            plt.show()







                
    def save(self, filename):
        """Save dynesty samples, priors, and run metadata to a compressed NPZ file."""

        if self.result is None:
            raise ValueError("No fit results available. Run run_fit() first.")

        # Convert custom priors to a fully serializable form
        if self.custom_priors is not None:
            custom_priors_serial = np.array(self.custom_priors, dtype=object)
        else:
            custom_priors_serial = None

        save_dict = dict(
            samples=self.result.samples,
            logl=self.result.logl,
            logwt=self.result.logwt,
            logz=self.result.logz,

            model_file=self.model_file,
            model_pars=self.model_pars,

            free_pars=np.array(self.free_pars, dtype="U"),
            fix_pars=np.array(list(self.fix_pars.items()), dtype=object),
            custom_priors=custom_priors_serial,
            spectral_range=self.spectral_range,
            polydeg=self.polydeg,
            nlive=self.nlive,
            dlogz=self.dlogz,
            bands=self.bands,
            treat_as_upper_limits=np.array(self.treat_as_upper_limits, dtype="U"),
            best_models=np.array(getattr(self, "best_models", {}),dtype=object,),
        )

        if self.spec is not None:
            save_dict.update(
                dict(
                    obs_spec_wl=self.spec.wl,
                    obs_spec_flux=self.spec.flux,
                    obs_spec_flux_err=self.spec.flux_err,
                    obs_spec_res=self.spec.resolution,
                    obs_spec_wl_units=self.spec.header["WUNITS"],
                    obs_spec_flux_units=self.spec.header["FUNITS"],
                )
            )

        if self.phot is not None:
            save_dict.update(
                dict(
                    phot_band=self.bands,
                    phot_fluxes=[self.phot.data[key][0] for key in self.bands],
                    phot_errors=[self.phot.data[key][1] for key in self.bands],
                    phot_units=self.phot.header["UNITS"],
                )
            )

        np.savez_compressed(filename, **save_dict)

        print(f"Saved fit results to {filename}")








    def load(self, filename):
        """
        Load dynesty samples, metadata, input data, and pre-evaluated
        best-fit model arrays from a compressed NPZ file.
        """

        with np.load(filename, allow_pickle=True) as data:

            # -------------------------------------------------
            # Recreate dynesty Result-like object
            # -------------------------------------------------
            class Result:
                pass

            result = Result()
            result.samples = data["samples"]
            result.logl = data["logl"]
            result.logwt = data["logwt"]
            result.logz = data["logz"]

            self.result = result

            # -------------------------------------------------
            # Restore parameter information
            # -------------------------------------------------
            self.free_pars = data["free_pars"].tolist()
            self.fix_pars = dict(data["fix_pars"].tolist())

            # -------------------------------------------------
            # Restore custom priors
            # -------------------------------------------------
            if "custom_priors" in data.files:
                custom_priors = data["custom_priors"]

                if custom_priors.ndim == 0:
                    self.custom_priors = custom_priors.item()
                else:
                    self.custom_priors = custom_priors.tolist()
            else:
                self.custom_priors = {}

            # -------------------------------------------------
            # Restore metadata
            # -------------------------------------------------
            self.model_file = data["model_file"].item()

            model_pars = data["model_pars"]
            self.model_pars = (
                model_pars.tolist()
                if hasattr(model_pars, "tolist")
                else model_pars
            )

            self.spectral_range = (
                data["spectral_range"]
                if "spectral_range" in data.files
                else None
            )

            self.polydeg = (
                int(data["polydeg"])
                if "polydeg" in data.files
                else None
            )

            self.nlive = (
                int(data["nlive"])
                if "nlive" in data.files
                else None
            )

            self.dlogz = (
                float(data["dlogz"])
                if "dlogz" in data.files
                else None
            )

            self.bands = (
                data["bands"].tolist()
                if "bands" in data.files
                else []
            )

            self.treat_as_upper_limits = (
                data["treat_as_upper_limits"].tolist()
                if "treat_as_upper_limits" in data.files
                else []
            )

            # -------------------------------------------------
            # Restore observed photometry
            # -------------------------------------------------
            self.phot = None

            if "phot_fluxes" in data.files:

                phot_fluxes = data["phot_fluxes"]
                phot_errors = data["phot_errors"]
                phot_bands = data["phot_band"].tolist()

                phot_dict = {
                    band: (phot_fluxes[i], phot_errors[i])
                    for i, band in enumerate(phot_bands)
                }

                header = fits.Header()

                if "phot_units" in data.files:
                    header["UNITS"] = data["phot_units"].item()

                self.phot = PhotometryTable(
                    data=phot_dict,
                    header=header,
                )

            # -------------------------------------------------
            # Restore observed spectrum
            # -------------------------------------------------
            self.spec = None
            self.spec_crop = None

            if "obs_spec_wl" in data.files:

                header = fits.Header()
                header["WUNITS"] = data["obs_spec_wl_units"].item()
                header["FUNITS"] = data["obs_spec_flux_units"].item()

                self.spec = Spectrum(
                    wl=data["obs_spec_wl"],
                    flux=data["obs_spec_flux"],
                    flux_err=data["obs_spec_flux_err"],
                    resolution=data["obs_spec_res"],
                    header=header,
                )

            # -------------------------------------------------
            # Reconstruct posterior point estimates
            # -------------------------------------------------
            self.best_model_pars_MAP = self.best_model(method="MAP")
            self.best_model_pars_ML = self.best_model(method="ML")
            self.best_model_pars_median = self.best_model(
                method="median"
            )

            # -------------------------------------------------
            # Restore pre-evaluated model arrays for plotting
            # -------------------------------------------------
            if "best_models" in data.files:
                self.best_models = data["best_models"].item()
            else:
                # Compatibility with old saved retrievals.
                self.best_models = {}

            # Use stored parameter dictionaries when available.
            if "MAP" in self.best_models:
                self.best_model_pars_MAP = (
                    self.best_models["MAP"]["params"]
                )

            if "ML" in self.best_models:
                self.best_model_pars_ML = (
                    self.best_models["ML"]["params"]
                )

            if "median" in self.best_models:
                self.best_model_pars_median = (
                    self.best_models["median"]["params"]
                )

        print(f"Loaded fit results from {filename}")
        return self
