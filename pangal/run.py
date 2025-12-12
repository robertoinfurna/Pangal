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
        self.polydeg = None
        self.nlive = None
        self.dlogz = None

        # Processed data
        self.obs_spec_wl = None, 
        self.obs_spec_flux = None,
        self.obs_spec_flux_err = None,
        #self.obs_resolution_on_model_grid = None

        # Likelihood, priors, and sampler
        self.pfitter = None
        self.log_likelihood = None
        self.prior_transform = None
        self.sampler = None
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
        synth_spec = self.pfitter.synthetic_spectrum(**best_params,)
                                                    #observed_spectrum_resolution=getattr(self, 'obs_resolution_on_model_grid', None))

        return synth_spec
        
        
 



    def top_models(self, n=50):

        if not hasattr(self, "result"):
            raise ValueError("No fit results available.")

        samples = self.result.samples
        logl = self.result.logl

        # Pick the indices of the N highest likelihood samples
        idx = np.argsort(logl)[-n:]      # last n are the largest
        idx = idx[::-1]                 # sort descending

        top_params = samples[idx]
        

        spectra = []

        for pars in top_params:
            param_dict = {}

            # free params
            for name, val in zip(self.free_pars, pars):
                param_dict[name] = val

            # fixed params
            for name, val in self.fix_pars.items():
                param_dict[name] = val

            spec = self.pfitter.synthetic_spectrum(
                **param_dict,
                observed_spectrum_resolution=getattr(self, "obs_resolution_on_model_grid", None)
            )
            spectra.append(spec)

        return spectra


    def fit_diagnostic(self,
                        spectra=[],

                        method='MAP',
                    
                        # plotting window
                        winf=None,
                        wsup=None,

                        # aestethics
                        figsize=(10,10),
                        color=None,
                        
                        # optionals
                        redshift=None,
                        show_H_lines=False,
                        show_all_spectral_lines=False,
                        show_atmospheric_lines=False,
                        show_filters=False,
                        
                        spec_legend_pars=None,
                        spec_legend_loc="upper left",
                        spec_legend_title=None,

                        phot_points=None,
                        phot_region=None,
                        synth_phot_points=None,
                        show_phot_legend=True,
                        winf_phot=1e2,
                        wsup_phot=1e7,
                        ymin=None,
                        ymax=None,
                    ):

        if self.spec:
        
            wl, flux_obs, flux_err, = self.obs_spec_wl, self.obs_spec_flux, self.obs_spec_flux_err
            header = fits.Header()
            header["WUNITS"]  = "A"
            header["FUNITS"]  = "erg/s/cm2/A"
            spectrum_obs = Spectrum(wl=wl,flux=flux_obs,flux_err=flux_err,header=header)
            
            model_spectra = [self.best_model(method=method)] + spectra 
            
            
            normalization_factor_list = []
            normalization_factor_smoothed_list = []
            models_comparable_to_obs = []
            residuals = []
            
            for synth_spec in model_spectra:
            
                normalization_factor, normalization_factor_smoothed, model = self.pfitter._adapt_model_spectrum_to_observed_spectrum(self,synth_spec)
            
                normalization_factor_list.append(normalization_factor)
                normalization_factor_smoothed_list.append(normalization_factor_smoothed)
            
                models_comparable_to_obs.append(model)
            
                res = (flux_obs - model) / flux_err
                residuals.append(res)
            
            
            fig, ax = plt.subplots(2,2,figsize=figsize)
            ax = ax.flatten()
            
            global_fmax = -np.inf
            global_fmin = +np.inf
            mask = (wl>winf) & (wl<wsup)
            
            global_norm_max = -np.inf
            global_norm_min = +np.inf
            global_res_max = -np.inf
            global_res_min = +np.inf
            
            for i in [0,2]:
                ax[i].errorbar(wl, flux_obs, yerr=flux_err, fmt='o', markersize=2,
                                color='black', ecolor='gray', elinewidth=1.0,
                                capsize=2, capthick=1.0, linestyle='-', lw=0.5,
                                label='observed')
            
                global_fmax = max(np.nanmax(flux_obs[mask]),global_fmax)
                global_fmin = min(np.nanmin(flux_obs[mask]),global_fmin)
            

            #for later
            chi2_models_spec = []
                
            for j,model in enumerate(model_spectra):
            
                # --- Color ---
                if isinstance(color, (list, tuple)) and len(color) > 0:
                    c = color[j % len(color)]
                elif color is None:
                    c = f"C{j % 10}" 
                else:
                    c = color
            
                # --- Label ---
                if isinstance(spec_legend_pars, str):
                    label = model.header[spec_legend_pars]
            
                elif isinstance(spec_legend_pars, dict):
                    label = ", ".join(
                        f"{spec_legend_pars[k][0]}={model.header[k]:.2f} {spec_legend_pars[k][1]}"
                        for k in spec_legend_pars
                        if k in model.header
                    )   
                else:
                    label = model.header.get("ID", getattr(model, "id", None)) or f"Spectrum {i + 1}"
            
                
                ax[0].plot(model.wl,model.flux,color=c,label=label)
            
                mask_m = (model.wl > winf) & (model.wl < wsup)
                global_fmax = max(global_fmax, np.nanmax(model.flux[mask_m]))
                global_fmin = min(global_fmin, np.nanmin(model.flux[mask_m]))
            
                ax[1].plot(wl,normalization_factor_list[j],color=c,alpha=0.3)
                ax[1].plot(wl,normalization_factor_smoothed_list[j],color=c,label=f'deg = {self.polydeg:.0f}')
                
                global_norm_max = max(np.nanmax(normalization_factor_list[j][mask]),global_norm_max)
                global_norm_min = min(np.nanmin(normalization_factor_list[j][mask]),global_norm_min)
                
                ax[2].plot(wl,models_comparable_to_obs[j],color=c)
            

                chi2 = -0.5  * np.nansum((residuals[j]**2 + np.log(flux_err**2) + np.log(2. * np.pi)))
                
                chi2_models_spec.append(chi2)
                ax[3].scatter(wl,residuals[j],s=10,alpha=1,color=c,label=f"$\chi_2=${chi2:.2f}")
                #ax[3].plot(wl,residuals[j],alpha=1,color=c,label=f"$\chi_2=${chi2:.2f}")
                
                global_res_max = max(np.nanmax(residuals[j][mask]),global_res_max)
                global_res_min = min(np.nanmin(residuals[j][mask]),global_res_min)
                
            
            for i in range(4):
                ax[i].set_xlim(winf,wsup)
            
            for i in [0,2]:
                ax[i].set_ylim(0.8*global_fmin,1.2*global_fmax)
            
            ax[1].set_ylim(0.8*global_norm_min,1.2*global_norm_max)
            ax[3].set_ylim(0.8*global_res_min,1.2*global_res_max)
            
            ax[3].axhline(y=0,color='black',lw=1)
            
            ax[0].legend() #title=spec_legend_pars
            #ax[2].legend() #title=spec_legend_pars
            ax[3].legend() #title=spec_legend_pars


            ax[0].set_title('Raw spectrum and model')
            ax[2].set_title('Raw spectrum and model fitted to observed spectrum continuum')
            ax[1].set_title('Polynomial multiplicative factor')
            ax[1].set_ylabel('$f_\\text{obs}/f_\\text{model}$')
            ax[3].set_title('Residuals')
            ax[3].set_ylabel('$(f_\\text{obs}-f_\\text{model})/\sigma_f$')
            
            plt.show()


        
        if self.phot:

            model_spectra[0].plot(

                spectra=model_spectra,
                per_wavelength=True,
                winf=winf_phot,
                wsup=wsup_phot,
                ymin=ymin,
                ymax=ymax,

                figsize=(20,7),
                color=color,

                # optionals
                redshift=None,
                show_H_lines=False,
                show_all_spectral_lines=False,
                show_atmospheric_lines=False,
                show_filters=False,
            
                phot_points=self.phot,
                phot_region=phot_region,
                synth_phot_points=None,
                spec_legend_pars=spec_legend_pars,
                show_phot_legend=True,
                show_spec_legend=True,
                spec_legend_loc="upper left",
                spec_legend_title=None,
            )

            bands = self.bands

            # This are the arrays!
            phot_points = np.array([self.phot.data[run.obj_id][b][0] for b in bands])
            phot_errors = np.array([self.phot.data[run.obj_id][b][1] for b in bands])
            upper_lims = (phot_points / phot_errors < 5).astype(int)
            
            for j,model in enumerate(model_spectra):

                print('For model number: ',j+1)

                model_phot_array = model.get_phot(for_likelihood=True, run=self)
                    
                if not np.all(np.isfinite(model_phot_array)):
                    print(-1e100)
                    break

                phot_lhood = 0
                for i in range(len(phot_points)):
                    if upper_lims[i] == 0:
                        contribution = -0.5 * (
                            (phot_points[i] - model_phot_array[i])**2 / phot_errors[i]**2
                            + np.log(phot_errors[i]**2)
                            + np.log(2. * np.pi)
                        ) 
                        phot_lhood += contribution
                        print(f"Band {bands[i]}: {contribution:.3f}")
                    else:
                        terf = 0.5 * (1 + erf((phot_points[i] - model_phot_array[i]) / 
                                            (np.sqrt(2.) * phot_errors[i])))
                        if terf <= 0:
                            return -1e100
                        print(f"Band {bands[i]}: {terf:.3f}")
                        phot_lhood += np.log(terf)
                print('TOTAL_PHOT: ',phot_lhood)
                if self.spec:
                    print('TOTAL_SPEC: ',chi2_models_spec[j])
                    print('TOTAL_PHOT+SPEC: ',phot_lhood + chi2_models_spec[j])
                print('\n')
                
                    

            



















    def save(self, filename):
        """Save dynesty samples, priors, and run metadata to a compressed NPZ file."""

        if not hasattr(self, "result"):
            raise ValueError("No fit results available. Run run_fit() first.")

        # Convert custom priors to a fully serializable form
        if self.custom_priors is not None:
            custom_priors_serial = np.array(self.custom_priors, dtype=object)
        else:
            custom_priors_serial = None

        np.savez_compressed(
            filename,
            samples=self.result.samples,
            logl=self.result.logl,
            logwt=self.result.logwt,
            logz=self.result.logz,

            # Parameter info
            free_pars=np.array(self.free_pars, dtype='U'),
            fix_pars=np.array(list(self.fix_pars.items()), dtype=object),

            # Priors and run settings
            custom_priors=custom_priors_serial,
            spectral_range=self.spectral_range,
            polydeg=self.polydeg,
            nlive=self.nlive,
            dlogz=self.dlogz,
            bands=self.bands,
        )

        print(f"Saved fit results to {filename}")


    def load(self, filename):
        """Load dynesty samples, priors, and metadata from NPZ into this Run object."""

        data = np.load(filename, allow_pickle=True)

        # Recreate dynesty Result-like object
        class Result: pass
        r = Result()
        r.samples = data['samples']
        r.logl    = data['logl']
        r.logwt   = data['logwt']
        r.logz    = data['logz']

        # Restore into self
        self.result = r
        self.free_pars = list(data['free_pars'])
        self.fix_pars  = dict(data['fix_pars'])

        print(self.free_pars)
        print(self.fix_pars)
        print(self.result)

        # Restore priors and other metadata
        self.custom_priors  = data.get('custom_priors', None)
        self.spectral_range = data.get('spectral_range', None)
        self.polydeg      = data.get('polydeg', None)
        self.nlive          = data.get('nlive', None)
        self.dlogz          = data.get('dlogz', None)
        self.bands          = data.get('bands', None)

        print(f"Loaded fit results from {filename}")

        return self