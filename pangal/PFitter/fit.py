import numpy as np
from bisect import bisect_left
from scipy.special import erf


from .base_classes import Image, Cube, Region, Point, Contours
from .filters import map_filter_names, nice_filter_names, default_plot_scale_lims, default_plot_units, default_cmaps


import numpy as np
from scipy.interpolate import RegularGridInterpolator
import itertools







# Loads emission lines models from file and adapts them to the cigale stellar template (same wavelength array, same resolution)
# manually building emission line spectra or injecting lines into a model spectrum
# Generates a cube

def load_nebular_tables(self,sps_model,emimetal,emimodel):
    
    wl = self.model_wl
    model_res = self.model_resolution

    wl_lyman = 912.
    ilyman = np.searchsorted(wl, wl_lyman, side='left') #wavelength just above Lyman limit
    lycont_wls = np.r_[wl[:ilyman], np.array([wl_lyman])]
    clyman_young = None #A list of two elements, first is for phot, the other for spec
    clyman_old   = None #A list of two elements, first is for phot, the other for spec

    # Emission line wavelengths have to be in VACUUM
    
    if emimodel == '2018':
        byler_bins = [382,9,7]
        byler_fname = 'nebular_Byler_mist_2018.lines'
        metallist = np.array([6.3245e-05,2.0000e-04,3.5565e-04,6.3245e-04,1.1246e-03,2.0000e-03,3.5565e-03,6.3245e-03,1.1246e-02,2.000e-02,3.5565e-02,6.3245e-02])
        metalstrg = np.array(['-2.5000e+00','-2.0000e+00','-1.7500e+00','-1.5000e+00','-1.2500e+00','-1.0000e+00','-7.5000e-01','-5.0000e-01','-2.5000e-01','0.0000e+00','2.5000e-01','5.0000e-01'])
    else:
        byler_bins = [128,10,7]
        byler_fname = 'nebular_Byler_mist_2017.lines'
        metallist = np.array([0.00020,0.00063246,0.00209426,0.00526054,0.00814761,0.01002374,0.01261915,0.01588656,0.02,0.02517851,0.03169786])
        metalstrg = np.array(['-1.9800e+00','-1.5000e+00','-9.8000e-01','-5.8000e-01','-3.9000e-01','-3.0000e-01','-2.0000e-01','-1.0000e-01','0.0000e+00','1.0000e-01','2.0000e-01'])
    
    emm_scales = np.zeros((byler_bins[2],byler_bins[1],byler_bins[0]), dtype=float)
    emm_wls    = np.zeros(byler_bins[0],        dtype=float)
    emm_ages   = np.zeros(byler_bins[1],        dtype=float)
    emm_ions   = np.zeros(byler_bins[2],        dtype=float)
    icnt = 0
    rline = 0
    iline = 0
    
    
    metind = np.argmin(np.abs(metallist-emimetal))  # Select the closest metallicity available
    print('INFO: Emission line metallicity requested {}, found {:5.4f}'.format(emimetal,metallist[metind]))
    emimetal = metallist[metind]

    # load emission lines tables
    with open('data/'+byler_fname,'r') as file:
        for line in file:
            if line[0] != '#':
                temp = (line.strip()).split(None)
                if not iline: #Read wave line
                    emm_wls[:] = np.array(temp, dtype=float)
                    iline = 1
                else:
                    if rline: #Read line fluxes
                        emm_scales[icnt%7,icnt//7,:] = np.array(temp, dtype=float)*3.839e33 #erg/s #output should be in erg/s/QHO
                        icnt += 1
                    if len(temp) == 3 and temp[0] == metalstrg[metind]:
                        rline = 1
                        emm_ages[icnt//7] = float(temp[1])/1e6
                        emm_ions[icnt%7]  = float(temp[2])
                    else:
                        rline = 0

    # Filters out non-physical or extremely high wavelengths ?? Why ??
    keep_scale = (emm_wls<1E5)        
    emm_scales = emm_scales[:,:,keep_scale]
    emm_wls    = emm_wls[keep_scale]

            
    # Convolve lines to model resolution
    # Emission lines are modeled as Gaussians centered at their rest wavelengths.
    # To add these lines realistically, you integrate the Gaussian emission line profile over each pixel's wavelength bin.
    # Simulates how nebular emission lines would appear in a spectrum observed at a given resolution and wavelength sampling
    
    dpix = np.diff(wl)
    wl_edges  = np.r_[np.array([wl[0]-dpix[0]/2.]), np.r_[wl[1:]-dpix/2., np.array([wl[-1]+dpix[-1]/2.])]]
    res_lines = np.interp(emm_wls, wl, model_res)/2.355
    

    emm_lines_all = np.zeros((len(emm_ions), len(emm_ages), len(wl)), dtype=float)

    for jj in range(len(emm_ions)):
        for ii in range(len(emm_ages)):
            this_scale = emm_scales[jj,ii,:]
            emm_lines_all[jj,ii,:] = np.sum(this_scale[:,None]*\
                np.diff(0.5*(1.+erf((wl_edges[None,:]-emm_wls[:,None])/\
                np.sqrt(2.*res_lines**2)[:,None])), axis=1)/np.diff(wl_edges), axis=0)

    return emm_ions, emm_ages, emimetal, emm_lines_all




def model_grid_interpolator(self, model_list, param_names, extra_pars_to_interpolate=None):
    """
    Build N-dimensional interpolators for young and old fluxes.

    Parameters:
        model_list (list): Flat list of Spectrum models, each with `.flux_young`, `.flux_old`, `.header`
        param_names (list of str): Names of parameters to define the grid (e.g. ['age', 'tau'])

    Returns:
        interp_flux_young: Callable f(**params) -> flux_young (shape: n_wl)
        interp_flux_old: Callable f(**params) -> flux_old (shape: n_wl)
        grid_axes: List of sorted arrays for each parameter axis
        wl: Wavelength array (from first model)
    """

    # Step 1: Build param space
    param_tuples = []
    param_to_model = {}
    for model in model_list:
        try:
            values = tuple(model.header[k] for k in param_names)
        except KeyError as e:
            raise KeyError(f"Model is missing parameter {e} in header.")
        param_tuples.append(values)
        param_to_model[values] = model

    # Step 2: Build grid axes
    grid_axes = [np.array(sorted(set(p[i] for p in param_tuples))) for i in range(len(param_names))]

    # Step 3: Allocate grids for young and old fluxes
    grid_shape = tuple(len(ax) for ax in grid_axes)
    n_wl = model_list[0].flux_young.size

    flux_young_grid = np.zeros(grid_shape + (n_wl,))
    flux_old_grid = np.zeros(grid_shape + (n_wl,))

    # Step 4: Fill the grids
    for idxs in np.ndindex(*grid_shape):
        key = tuple(grid_axes[i][idxs[i]] for i in range(len(grid_axes)))
        model = param_to_model[key]
        flux_young_grid[idxs] = model.flux_young
        flux_old_grid[idxs] = model.flux_old

    # Step 5: Build interpolators
    interp_young = RegularGridInterpolator(points=grid_axes, values=flux_young_grid, bounds_error=False, fill_value=None)
    interp_old = RegularGridInterpolator(points=grid_axes, values=flux_old_grid, bounds_error=False, fill_value=None)

    def interp_flux_young(**kwargs):
        x = [kwargs[name] for name in param_names]
        return interp_young(x)[0]

    def interp_flux_old(**kwargs):
        x = [kwargs[name] for name in param_names]
        return interp_old(x)[0]

    return interp_flux_young, interp_flux_old, grid_axes, model_list[0].wl