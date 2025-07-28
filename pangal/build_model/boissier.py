import numpy as np

from scipy.constants import c, parsec
import astropy.io.fits as fits
import itertools
from copy import deepcopy

import copy, glob, re, os
import tempfile
from tqdm import tqdm

from pcigale.sed import SED
from pcigale.sed import utils
from pcigale.sed_modules import get_module

from ..methods.base_classes import Spectrum



# Scaling relations

Rd_MW = 2.6                # kpc
Sigma_0_MW = 1150          # solar masses / pc2
Vc_MW = 220                # km/s
spin_MW = 0.03 
md_MW = 0.05
M_MW = 1e12                 # Msun
Md_MW = 0.05 * 1e12         # Msun

md = 0.05
jd = 0.05

def M(Vc):
    return  M_MW * (Vc / Vc_MW)**3

def Md(Vc,md):
    return  md * M(Vc)

def Rd(Vc,spin):
    return  Rd_MW * (spin / spin_MW) * (Vc / Vc_MW)

def Sigma_0(Vc,spin): 
    return Sigma_0_MW * (md / md_MW) * (spin / spin_MW)**-2 * (Vc / Vc_MW)

def surface_density(R,Vc,spin):
    return Sigma_0(Vc,spin) * np.exp(-R/Rd(Vc,spin))

def Vc_to_Md(Vc): return Md(Vc,md = 0.05)
def Md_to_Vc(Md): return Vc_MW * (Md / Md_MW)**(1/3)

# Velocity curves
from scipy.interpolate import interp1d
from scipy.special import iv as I, kv as K

G = 4.301e-6         # (kpc * (km/s)^2) / M_sun
rho0 = 0.015 * 1e9   # M_sun/kpc^3   0.008 - 0.015
Rc_MW = 5            # kpc -5 to 10

def Rc(Vc,spin):
    return Rc_MW * (spin/spin_MW) * (Vc/Vc_MW)

def rho_0(Vc,spin):
    return Vc**2 / (4*np.pi*G*Rc(Vc,spin)**2)


# Velocity profile

def velocity_profile(Vc, spin):

    R = np.linspace(0.03, 20, 500)

    # Disk component
    sigma_0 = Sigma_0(Vc, spin)
    rd = Rd(Vc, spin)
    y = R / (2*rd)
    Vd_squared = 4 * np.pi * G * (sigma_0 * 1e6) * rd * y**2 * (I(0, y) * K(0, y) - I(1, y) * K(1, y))
    
    # Halo component
    rho0_val = rho_0(Vc, spin)
    rc = Rc(Vc, spin)
    Vh_squared = 4 * np.pi * G * rho0_val * rc**2 * (1 - (rc / R) * np.arctan(R / rc))

    # Total rotation curve
    V_total = np.sqrt(Vd_squared + Vh_squared)

    V_interp = interp1d(R, V_total, kind='linear', bounds_error=False, fill_value='extrapolate')

    return V_interp













def boissier_sfh(V, R):
    """
    Load and return the closest Boissier unperturbed star formation history (SFH) model
    for a given circular velocity (V) and radius (R), based on precomputed grids.

    Parameters:
        V (float): Circular velocity in km/s.
        R (float): Galactocentric radius in kpc.

    Returns:
        selected_model (np.ndarray): Array containing the SFH data for the closest (V, R) pair.
    """

    # Directory containing the Boissier model files
    data_dir = "../Pangal/data/SFHs_Boissier/boissier_models_big_grid"

    # Match all files with the format DISKEVOL.RES_L0.05_V* (but not those with 'lignes')
    file_pattern = os.path.join(data_dir, "DISKEVOL.RES_L0.05_V*")
    vel_pattern = re.compile(r"_V(\d+)")  # Extract velocity value from filename

    # Collect (velocity, full_path) pairs
    velocity_files = []
    for full_path in glob.glob(file_pattern):
        fname = os.path.basename(full_path)
        if "lignes" in fname:
            continue  # Skip helper/diagnostic files
        m = vel_pattern.search(fname)
        if not m:
            continue
        velocity = int(m.group(1))
        velocity_files.append((velocity, full_path))

    # Sort all files by velocity
    velocity_files.sort(key=lambda vf: vf[0])

    # Parse the content of all matching files into a nested dictionary:
    # boissier_models[velocity][radius] = np.array of SFH rows
    boissier_models = {}
    for velocity, full_path in velocity_files:
        boissier_models[velocity] = {}
        current_radius = None
        boissier_models_by_radius = {}

        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("99999"):
                    continue  # Skip empty lines or separators

                # Header line indicating a new radius section
                if line.startswith("R ="):
                    parts = line.split()
                    try:
                        current_radius = float(parts[2])  # e.g. "R = 1.24 kpc"
                        boissier_models_by_radius[current_radius] = []
                    except (IndexError, ValueError):
                        current_radius = None
                    continue

                if current_radius is None:
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue  # Not enough columns

                try:
                    # Use columns 1 to 5 (skip column 0)
                    row = [float(x) for x in parts[1:6]]
                    boissier_models_by_radius[current_radius].append(row)
                except ValueError:
                    continue  # Skip malformed lines

        # Convert radius → list of rows into NumPy arrays
        for radius, rows in boissier_models_by_radius.items():
            boissier_models[velocity][radius] = np.array(rows)

    # Find the closest velocity match
    v_array = np.array(list(boissier_models.keys()))
    idx = np.argmin(np.abs(v_array - V))
    closest_v = v_array[idx]

    # Find the closest radius match at the chosen velocity
    r_array = np.array(list(boissier_models[closest_v].keys()))
    idx = np.argmin(np.abs(r_array - R))
    closest_r = r_array[idx]

    # Extract the matching SFH model
    selected_model = boissier_models[closest_v][closest_r]

    print(f"Selected unperturbed Boissier SFH model: V = {closest_v} km/s, R = {closest_r} kpc")

    return selected_model






def boissier_quenched(
    V,
    R,
    sfh_quenching_tau,
    sfh_quenching_age,
    metallicity=0.017,
    age=13500,
    stellar_library='cb19',
    filename=None,
    save_all_wavelengths=False
):
    """
    Generate quenched SFH galaxy spectra using Boissier unperturbed models.

    Parameters:
        V (float): Circular velocity to select unperurbed model.
        R (float): Radius to select unperturbed model.
        sfh_quenching_tau (float or list): Quenching tau value(s).
        sfh_quenching_age (float or list): Quenching age value(s).
        metallicity (float or list): Metallicity value(s).
        age (float or list): Total age(s) in Myr.
        stellar_library (str): Stellar population synthesis library ('cb19', 'bc03', etc.).
        filename (str): If provided, save spectra into a FITS file.
        save_all_wavelengths (bool): If False, wavelengths and resolution are only stored for the first model.

    Returns:
        models (list of Spectrum): List of Spectrum objects.
    """

    def ensure_list(x):
        return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

    sfh_quenching_tau = ensure_list(sfh_quenching_tau)
    sfh_quenching_age = ensure_list(sfh_quenching_age)
    metallicity = ensure_list(metallicity)
    age = ensure_list(age)
    
    print("Using stellar library: ",stellar_library)
    print(f"Metallicity values: {[met for met in metallicity]}")
    print(f"Ages (Myr): {[a for a in age]}")
    print(f"SF quenching ages (Myr): {[a for a in sfh_quenching_age]}")
    print(f"SF quenching tau (Myr): {[t for t in sfh_quenching_tau]}")

    combinations = list(itertools.product(sfh_quenching_age, sfh_quenching_tau, metallicity, age))
    print(f"Total models generated: {len(combinations)}")

    selected_boissier_sfh = boissier_sfh(V, R)

    # If filename is provided, it saves the models in fits format
    if filename:
        print("Saving to fits file: ",filename)
        hdu_list = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['NEXT'] = 0
        hdu_list.append(primary_hdu)

    models = []
    cnt = 0

    for index, (q_age, q_tau, Z, t_age) in tqdm(
        enumerate(combinations), total=len(combinations), desc="Processing combinations"
    ):
        # Interpolate SFH on proper time grid
        t_myr = selected_boissier_sfh[:, 0] * 1000.0
        sfr_myr = selected_boissier_sfh[:, 2]
        full_time_grid = np.arange(0, t_age + 1)
        interp = np.interp(full_time_grid, t_myr, sfr_myr, left=0.0, right=0.0)

        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_filename = tmp.name
        tmp.close()

        with open(tmp_filename, "w") as fout:
            for ti, si in zip(full_time_grid, interp):
                fout.write(f"{int(ti):d}   {si:.6e}\n")

        galaxy_sec = SED()
        sfh_module = get_module('sfhfromfile', filename=tmp_filename, sfr_column=1, age=t_age, normalise=True)
        ssp_module = get_module(stellar_library, imf=1, metallicity=Z)

        sfh_module.process(galaxy_sec)

        # Apply quenching and SSP
        quenching_module = get_module('sfh_quenching_exp', quenching_age=q_age, quenching_tau=q_tau)
        galaxy = deepcopy(galaxy_sec)
        quenching_module.process(galaxy)
        ssp_module.process(galaxy)

        wl = galaxy.wavelength_grid * 10  # to Å

        if stellar_library == 'bc03':
            resolution = wl / 300
            resolution[(wl >= 3200) & (wl <= 9500)] = 3.0
        elif stellar_library == 'cb19':
            resolution = np.full_like(wl, 2.0)
            resolution[(wl >= 912) & (wl <= 3540)] = 1.0
            resolution[(wl >= 3540) & (wl <= 7350)] = 2.5
            resolution[(wl >= 7350) & (wl <= 9400)] = 1.0
        else:
            resolution = wl / 300
            resolution[(wl >= 3200) & (wl <= 9500)] = 3.0

        lum_y = galaxy.luminosities['stellar.young']
        lum_o = galaxy.luminosities['stellar.old']

        flux_y = utils.luminosity_to_flux(lum_y, 10 * parsec) * 100  # erg/s/cm2/A
        flux_o = utils.luminosity_to_flux(lum_o, 10 * parsec) * 100

        total_flux = flux_y + flux_o

        spec = Spectrum(
            wl=wl if (cnt == 0 or save_all_wavelengths) else None,
            resolution=resolution if (cnt == 0 or save_all_wavelengths) else None,
            flux=total_flux
        )
        spec.flux_young = flux_y
        spec.flux_old = flux_o

        spec.header['sfh_model'] = 'exptrunc'
        spec.header['stellar_library'] = stellar_library
        spec.header['metallicity'] = float(Z)
        spec.header['truncage'] = float(q_age)
        spec.header['trunctau'] = float(q_tau)
        spec.header['model_age'] = float(t_age)

        models.append(spec)

        if filename and cnt == 0:
            fits_data = np.empty((len(wl), 2), dtype=np.float32)
            fits_data[:, 0] = flux_y
            fits_data[:, 1] = flux_o
            hdu_list[0].data = np.array([wl, resolution], dtype=np.float32)
            hdu_list[0].header['PAR1'] = 'trunctau'
            hdu_list[0].header['PAR2'] = 'truncage'
            hdu_list[0].header['SFHFUNC'] = 'exptrunc'
            hdu_list[0].header['NEXT'] = len(combinations)

            spectrum_hdu = fits.ImageHDU(fits_data)
            hdu_list.append(spectrum_hdu)

        os.remove(tmp_filename)
        cnt += 1

    if filename:
        hdu_list.writeto(filename, overwrite=True)

    return models
    
