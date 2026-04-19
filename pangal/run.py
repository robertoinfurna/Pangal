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

        self.model_file = None
        self.bands = None
        self.spectral_range = None
        self.fix_pars = None
        self.custom_priors = None
        self.polydeg = None
        self.nlive = None
        self.dlogz = None

        # Processed data
        self.spec = None
        self.phot = None

        # Likelihood, priors, and sampler
        self.result = None




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
    


    
    def cornerplot(
            run,
            figsize,
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
        ):
            self=run
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
                    "redshift": r"redshift",
                    "luminosity_distance": r"luminosity distance",
                    "ln_spec_noise_scale": r"spectral noise scaling parameter",
                    "R_GAL": r"$R_\text{gal}$",
                    "V_DISC": r"$v_\text{disc}$",
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
            fig = plt.figure(figsize=figsize)

            # ---- corner plot ----
            fig = corner.corner(
                equal_samples,
                labels=plot_labels,
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
                if str(mark_point).lower() == "median":
                    pt = np.median(equal_samples, axis=0)
                else:
                    idx_best = int(np.argmax(self.result.logl))
                    pt = samples[idx_best][indices]

                for i in range(n):
                    for j in range(i):
                        ax = axes[i, j]
                        ax.axvline(pt[j], color=mark_color, lw=mark_lw)
                        ax.axhline(pt[i], color=mark_color, lw=mark_lw)

                if show_histograms:
                    for i, ax in enumerate(diag_axes):
                        ax.axvline(pt[i], color=mark_color, lw=mark_lw)


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


            # ---- save ----
            if filename is not None:
                if not str(filename).lower().endswith(".png"):
                    filename = f"{filename}.png"
                fig.savefig(filename, dpi=300, bbox_inches="tight")

            plt.show()




                
    def save(self, filename):
        """Save dynesty samples, priors, and run metadata to a compressed NPZ file."""

        if not hasattr(self, "result"):
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
            treat_as_upper_limits=np.array(self.treat_as_upper_limits, dtype="U")
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
        """Load dynesty samples, priors, and metadata from NPZ into this Run object."""

        with np.load(filename, allow_pickle=True) as data:

            # -------------------------------------------------
            # Recreate dynesty Result-like object
            # -------------------------------------------------
            class Result:
                pass

            r = Result()
            r.samples = data["samples"]
            r.logl    = data["logl"]
            r.logwt   = data["logwt"]
            r.logz    = data["logz"]
            self.result = r

            # -------------------------------------------------
            # Restore parameter info
            # -------------------------------------------------
            self.free_pars = list(data["free_pars"])
            self.fix_pars  = dict(data["fix_pars"].tolist())

            # -------------------------------------------------
            # Restore priors
            # -------------------------------------------------
            if "custom_priors" in data and data["custom_priors"] is not None:
                self.custom_priors = data["custom_priors"].tolist()
            else:
                self.custom_priors = None

            # -------------------------------------------------
            # Restore metadata
            # -------------------------------------------------
            self.spectral_range = data["spectral_range"] if "spectral_range" in data else None
            self.polydeg = int(data["polydeg"]) if "polydeg" in data else None
            self.nlive   = int(data["nlive"]) if "nlive" in data else None
            self.dlogz   = float(data["dlogz"]) if "dlogz" in data else None
            self.bands   = data["bands"].tolist() if "bands" in data else None
            self.treat_as_upper_limits = data["treat_as_upper_limits"].tolist() if "treat_as_upper_limits" in data else None

            # -------------------------------------------------
            # Restore observed photometry
            # -------------------------------------------------
            self.phot = None
            if "phot_fluxes" in data:
                phot_fluxes = data["phot_fluxes"]
                phot_errors = data["phot_errors"]
                bands = data["phot_band"].tolist()

                phot_dict = {
                    b: (phot_fluxes[i], phot_errors[i])
                    for i, b in enumerate(bands)
                }

                hdr = fits.Header()
                if "phot_units" in data:
                    hdr["UNITS"] = data["phot_units"].item()

                self.phot = PhotometryTable(data=phot_dict, header=hdr)

            # -------------------------------------------------
            # Restore observed spectrum
            # -------------------------------------------------
            self.spec = None
            if "obs_spec_wl" in data:
                hdr = fits.Header()
                hdr["WUNITS"] = data["obs_spec_wl_units"].item()
                hdr["FUNITS"] = data["obs_spec_flux_units"].item()

                self.spec = Spectrum(
                    wl=data["obs_spec_wl"],
                    flux=data["obs_spec_flux"],
                    flux_err=data["obs_spec_flux_err"],
                    resolution=data["obs_spec_res"],
                    header=hdr,
                )

            # -------------------------------------------------
            # Restore cropped observed spectrum
            # -------------------------------------------------
            self.spec = None
            if "obs_spec_wl" in data:
                hdr = fits.Header()
                hdr["WUNITS"] = data["obs_spec_wl_units"].item()
                hdr["FUNITS"] = data["obs_spec_flux_units"].item()

                self.spec = Spectrum(
                    wl=data["obs_spec_wl"],
                    flux=data["obs_spec_flux"],
                    flux_err=data["obs_spec_flux_err"],
                    resolution=data["obs_spec_res"],
                    header=hdr,
                )

            # ---------------------------------------------------
            # Restore pfitter
            # ---------------------------------------------------


            self.model_file = data["model_file"].item()
            self.model_pars = data["model_pars"]

        print(f"Loaded fit results from {filename}")
        return self