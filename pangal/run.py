import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from bisect import bisect_left
from scipy.special import erf, erfinv
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d #?
from scipy.signal import fftconvolve
import itertools
import os

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import corner

from .spectrum import Spectrum
from .filter import Filter, map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps

from .pfitter_utils import load_nebular_tables, load_dust_emission_models, dust_attenuation_curve, model_grid_interpolator

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
        self.spec = None
        self.phot = None
        self.bands = None
        self.spectral_range = None
        self.fix_pars = None
        self.custom_priors = None
        self.polymax = None
        self.nlive = None
        self.dlogz = None

        # Processed data
        self.wl_obs = None
        self.flux_obs_norm = None
        self.err_obs_norm = None
        self.obs_resolution_on_model_grid = None

        # Likelihood, priors, and sampler
        self.pfitter = None
        self.log_likelihood = None
        self.prior_transform = None
        self.sampler = None
        self.result = None


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
            "sigma_vel": r"$\sigma_v\ \mathrm{[km/s]}$"
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
        -------
        synth_spec : Spectrum
            Synthetic spectrum built with best parameters
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

        # --- Build synthetic spectrum ---
        synth_spec = self.pfitter.synthetic_spectrum(**best_params,
                                                    redshift=0,
                                                    dl=100,
                                                    observed_spectrum_resolution=getattr(self, 'obs_resolution_on_model_grid', None))

        return synth_spec, best_params
    
    #def summary_plot():
    #    synth_spec, best_params = 