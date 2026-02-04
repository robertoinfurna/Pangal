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
        self.spec_crop = None
        self.phot = None

        # Likelihood, priors, and sampler
        self.result = None


    # FUNCTIONS IMPORTANT FOR FITTING


    def cornerplot(self, show_stats=True, cmap='viridis', alpha=0.7):
        """
        Make a corner plot from the nested sampling results stored in this Run.

        Parameters
        ----------
        show_stats : bool
            Add mean ± std titles on diagonal histograms.
        cmap : str
            Matplotlib colormap for filled contours.
        alpha : float
            Transparency of contour fills.
        """

        if not hasattr(self, "result"):
            raise ValueError("Run has no .result attribute. Did you run run_fit()?")

        # Extract samples
        samples = self.result.samples
        logwt   = self.result.logwt
        logz    = self.result.logz

        # Convert to equal-weight posterior samples
        weights = np.exp(logwt - logz[-1])
        equal_samples = resample_equal(samples, weights)

        # Labels for the parameters
        latex_labels = {
            "age": r"$\mathrm{Age~[Myr]}$",
            "tau_main": r"$\tau_\mathrm{main}\ \mathrm{[Myr]}$",
            "age_trunc": r"$Q_\mathrm{AGE}\ \mathrm{[Myr]}$",
            "tau_trunc": r"$\tau_Q\ \mathrm{[Myr]}$",
            "fesc": r"$f_\mathrm{esc}$",
            "ion_gas": r"$U_\mathrm{ion}$",
            "age_gas": r"$\mathrm{Age_{gas}}$",
            "av": r"$A_V$",
            "av_ext": r"$A_{V,\mathrm{ext}}$",
            "alpha": r"$\alpha$",
            "m_star": r"$\log M_\star$",
            "vel_sys": r"$v\ \mathrm{[km/s]}$",
            "sigma_vel": r"$\sigma_v\ \mathrm{[km/s]}$",
            "redshift": r'redshift',
            "luminosity_distance": r'luminosity distance',
            "ln_spec_noise_scale": r'spectral noise scaling parameter'
        }

        labels = [latex_labels.get(p, p) for p in self.free_pars]

        # Plot
        plt.close("all")
        fig = corner.corner(
            equal_samples,
            labels=labels,
            plot_contours=True,
            fill_contours=True,
            color="k",
            cmap=cmap,
            alpha=alpha
        )

        # Add stats on diagonal
        if show_stats:
            means = np.mean(equal_samples, axis=0)
            stds  = np.std(equal_samples, axis=0)

            n = len(self.free_pars)
            diag_axes = [fig.axes[i*(n+1)] for i in range(n)]

            for i, ax in enumerate(diag_axes):
                ax.set_title(f"{means[i]:.3f} ± {stds[i]:.3f}", fontsize=10, pad=12)

        plt.show()



    def best_model(self, method='MAP'):
        """
        Build a synthetic spectrum using the 'best' parameters from the fit.

        Parameters
        ----------
        method : str
            Which parameters to use:
            - 'MAP' : Maximum a posteriori
            - 'ML'  : Maximum likelihood
            - 'median' : Posterior median

        Returns
        best_params : dict
            Dictionary of parameter names and best-fit values
        """

        if not hasattr(self, 'result'):
            raise ValueError("No fit results found in this Run object.")

        samples = self.result.samples
        weights = np.exp(self.result.logwt - self.result.logz[-1])
        param_names = self.free_pars

        # --- Select best parameters ---
        if method == 'MAP':
            logpost = self.result.logl + np.log(weights + 1e-300)  # approximate posterior
            idx = np.argmax(logpost)
            best_vals = samples[idx]
        elif method == 'ML':
            idx = np.argmax(self.result.logl)
            best_vals = samples[idx]
        elif method == 'median':
            # weighted median
            best_vals = []
            for i in range(samples.shape[1]):
                sorted_idx = np.argsort(samples[:, i])
                cumsum = np.cumsum(weights[sorted_idx])
                cumsum /= cumsum[-1]
                best_vals.append(samples[sorted_idx][np.searchsorted(cumsum, 0.5), i])
            best_vals = np.array(best_vals)
        else:
            raise ValueError(f"Unknown method '{method}'")

        # --- Build parameter dictionary including fixed parameters ---
        best_params = {}

        # free parameters
        for name, val in zip(self.free_pars, best_vals):
            best_params[name] = val

        # fixed parameters
        for name, val in self.fix_pars.items():
            best_params[name] = val

        return best_params
    

    def cornerplot(
        self,
        parameters=None,
        show_stats=True,
        show_histograms=True,
        cmap=None,
        contour_color='k',
        alpha=0.7,
        mark_point=None,          # None, "median", "maxlogl"
        mark_color="crimson",
        mark_lw=2.0,
    ):
        if not hasattr(self, "result"):
            raise ValueError("Run has no .result attribute. Did you run run_fit()?")

        samples = self.result.samples
        logwt   = self.result.logwt
        logz    = self.result.logz

        weights = np.exp(logwt - logz[-1])
        equal_samples = resample_equal(samples, weights)

        latex_labels = {
            "age": r"$\mathrm{Age~[Myr]}$",
            "tau_main": r"$\tau_\mathrm{main}\ \mathrm{[Myr]}$",
            "TRUNCAGE": r"$Q_\mathrm{AGE}\ \mathrm{[Myr]}$",
            "TRUNCTAU": r"$\tau_Q\ \mathrm{[Myr]}$",
            "fesc": r"$f_\mathrm{esc}$",
            "ion_gas": r"$U_\mathrm{ion}$",
            "age_gas": r"$\mathrm{Age_{gas}}$",
            "av": r"$A_V$",
            "av_ext": r"$A_{V,\mathrm{ext}}$",
            "alpha": r"$\alpha$",
            "m_star": r"$\log M_\star$",
            "vel_sys": r"$v\ \mathrm{[km/s]}$",
            "sigma_vel": r"$\sigma_v\ \mathrm{[km/s]}$",
            "redshift": r"redshift",
            "luminosity_distance": r"luminosity distance",
            "ln_spec_noise_scale": r"spectral noise scaling parameter"
        }

        # ---- select parameters ----
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

        labels = [latex_labels.get(p, p) for p in selected_pars]

        # ---- make figure ----
        plt.close("all")

        if cmap is None:
            # ----- Standard white corner style -----
            fig = corner.corner(
                equal_samples,
                labels=labels,
                plot_contours=True,
                fill_contours=False,
                color=contour_color,
            )
        
        else:
            # ----- Colored filled contours -----
            fig = corner.corner(
                equal_samples,
                labels=labels,
                plot_contours=True,
                fill_contours=True,
                contourf_kwargs={
                    "cmap": cm.get_cmap(cmap),
                    "colors": None,
                    "alpha": alpha
                },
                contour_kwargs={"colors": contour_color},
            )

        n = len(selected_pars)

        # ---- optional: hide histograms (diagonal) ----
        diag_axes = [fig.axes[i*(n+1)] for i in range(n)]
        if not show_histograms:
            for ax in diag_axes:
                ax.set_visible(False)

        # ---- optional: stats ----
        if show_stats and show_histograms:
        
            if mark_point is None:
                vals = np.mean(equal_samples, axis=0)
                errs = np.std(equal_samples, axis=0)
                label = "mean"
        
            elif str(mark_point).lower() == "median":
                vals = np.median(equal_samples, axis=0)
                errs = np.std(equal_samples, axis=0)
                label = "median"
        
            elif str(mark_point).lower() in ("maxlogl", "maxlikelihood", "map"):
        
                if not hasattr(self.result, "logl"):
                    raise ValueError("No logl available for MAP.")
        
                idx_best = int(np.argmax(self.result.logl))
                pt_full = samples[idx_best]
                vals = pt_full[indices]
                errs = None
                label = "MAP"
        
            else:
                vals = np.mean(equal_samples, axis=0)
                errs = np.std(equal_samples, axis=0)
                label = str(mark_point)
        
            # ---- Apply titles ----
            for i, ax in enumerate(diag_axes):
        
                if errs is None:
                    ax.set_title(f"{vals[i]:.3f}", fontsize=10, pad=12) #{label} = 
                else:
                    ax.set_title(f"{vals[i]:.3f} ± {errs[i]:.3f}", #{label} = 
                                fontsize=10, pad=12)

        
        # =========================================================
        # Add a cross at a "best" point (median or max-likelihood)
        # =========================================================
        if mark_point is not None:
            mark_point = str(mark_point).lower()

            if mark_point == "median":
                pt = np.median(equal_samples, axis=0)

            elif mark_point in ("maxlogl", "maxlikelihood", "map"):
                # Prefer using the original samples + logl if available
                if not hasattr(self.result, "logl"):
                    raise ValueError(
                        "mark_point='maxlogl' requested but result has no .logl attribute. "
                        "Use mark_point='median' or store log-likelihoods in result.logl."
                    )
                idx_best = int(np.argmax(self.result.logl))
                pt_full = samples[idx_best]      # full parameter set
                pt = pt_full[indices]            # subset to selected parameters

            else:
                raise ValueError("mark_point must be None, 'median', or 'maxlogl'.")

            # Overplot crosses on 2D panels (lower triangle)
            # Axes are in row-major order; diag indices are i*(n+1).
            for i in range(n):
                for j in range(i):
                    ax = fig.axes[i*n + j]  # row i, col j
                    # Overplot crosshairs on 2D panels (lower triangle)
                    for i in range(n):
                        for j in range(i):
                            ax = fig.axes[i*n + j]  # row i, col j
                    
                            # vertical line at x = pt[j]
                            ax.axvline(
                                pt[j],
                                color=mark_color,
                                linewidth=mark_lw,
                                alpha=0.9,
                                zorder=10,
                            )
                    
                            # horizontal line at y = pt[i]
                            ax.axhline(
                                pt[i],
                                color=mark_color,
                                linewidth=mark_lw,
                                alpha=0.9,
                                zorder=10,
                            )
                            
            # Optional: mark on diagonal histograms with a vertical line
            if show_histograms:
                for i, ax in enumerate(diag_axes):
                    ax.axvline(pt[i], color=mark_color, linewidth=mark_lw)

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
                    w_spec=self.w_spec
                )
            )

        if self.spec_crop is not None:
            save_dict.update(
                dict(
                    obs_spec_crop_wl=self.spec_crop.wl,
                    obs_spec_crop_flux=self.spec_crop.flux,
                    obs_spec_crop_flux_err=self.spec_crop.flux_err,
                    obs_spec_crop_res=self.spec_crop.resolution,
                    obs_spec_crop_wl_units=self.spec_crop.header["WUNITS"],
                    obs_spec_crop_flux_units=self.spec_crop.header["FUNITS"],
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
                self.w_spec=data["w_spec"]

            # -------------------------------------------------
            # Restore cropped observed spectrum
            # -------------------------------------------------
            self.spec_crop = None
            if "obs_spec_crop_wl" in data:
                hdr = fits.Header()
                hdr["WUNITS"] = data["obs_spec_crop_wl_units"].item()
                hdr["FUNITS"] = data["obs_spec_crop_flux_units"].item()

                self.spec_crop = Spectrum(
                    wl=data["obs_spec_crop_wl"],
                    flux=data["obs_spec_crop_flux"],
                    flux_err=data["obs_spec_crop_flux_err"],
                    resolution=data["obs_spec_crop_res"],
                    header=hdr,
                )

            # ---------------------------------------------------
            # Restore pfitter
            # ---------------------------------------------------


            self.model_file = data["model_file"].item()
            self.model_pars = data["model_pars"]

        print(f"Loaded fit results from {filename}")
        return self