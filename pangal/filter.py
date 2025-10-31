import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
module_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(module_dir, '.', 'data', 'allfilters.dat')


class Filter:

    def __init__(self, name, filter_file=data_file):
        self.name = name
        target_filter = map_filter_names[name]
        wavelengths = []
        transmissions = []
        reading = False

        with open(filter_file, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("#"):
                    current_filter = line[1:].strip()
                    reading = (current_filter == target_filter)
                    if not reading and wavelengths:
                        break  # Stop early once the desired filter is done
                elif reading:
                    try:
                        wl, tr = map(float, line.split())
                        wavelengths.append(wl)
                        transmissions.append(tr)
                    except ValueError:
                        continue  # Skip malformed lines

        if not wavelengths:
            raise ValueError(f"Filter data for '{name}' not found in file.")

        wavelengths = np.array(wavelengths)
        transmissions = np.array(transmissions)

        # in angstrom!
        self.transmission_curve = interp1d(
            wavelengths,
            transmissions,
            kind='linear',
            bounds_error=False,
            fill_value=(transmissions[0], transmissions[-1])
        )

        positive = transmissions > 0
        self.wavelength_range = (np.nanmin(wavelengths[positive]),
                                 np.nanmax(wavelengths[positive]))

    @property
    def pivot_wavelength(self):
        # Sample wavelength grid
        lam = np.linspace(self.wavelength_range[0], self.wavelength_range[1], 1000)
        T = self.transmission_curve(lam)
        
        # Calculate pivot wavelength
        numerator = np.trapz(lam * T, lam)
        denominator = np.trapz(T / lam, lam)
        lambda_pivot = np.sqrt(numerator / denominator)

        return lambda_pivot
        
    @property
    def effective_wavelength(self, F):
        """
        Compute the effective wavelength of the filter for a given source spectrum F(lambda).
         Parameters
        ----------
        F : callable
            Function returning the spectral flux density F(lambda) at wavelength lambda [same units as T(lambda)].
        Returns
        -------
        lambda_eff : float
            Effective wavelength in the same units as self.wavelength_range
        """
        # Sample wavelength grid
        lam = np.linspace(self.wavelength_range[0], self.wavelength_range[1], 1000)
        T = self.transmission_curve(lam)
        
        # Evaluate source spectrum
        F_lambda = F(lam)
        
        # Compute effective wavelength
        numerator = np.trapz(lam * T * F_lambda, lam)
        denominator = np.trapz(T * F_lambda, lam)
        
        lambda_eff = numerator / denominator
        return lambda_eff


    def plot(self, color='r'):
   
        wl_min, wl_max = self.wavelength_range[0] - 300, self.wavelength_range[1] + 300
        wavelengths = np.linspace(wl_min, wl_max, 1000)
        transmissions = self.transmission_curve(wavelengths)

        units = 'Å'
        w_eff = self.pivot_wavelength
        if wl_min > 8000:
            wavelengths = wavelengths/1e4
            w_eff = w_eff/1e4
            units = '$\mu m$'
            
        fig, ax = plt.subplots()
        ax.plot(wavelengths, transmissions, color=color)
        ax.set_xlabel(f"$\lambda$ [{units}]")
        ax.set_ylabel("Transmission")
        ax.set_ylim(0,1.2*max(transmissions))

        ax.text(
            0.02, 0.98,
            nice_filter_names[self.name],
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontweight='bold'
        )
        
        ax.text(
            0.02, 0.93,  # Move down slightly
            f'$\\lambda_{{\\rm eff}} =${w_eff:.2f} {units}',
            transform=ax.transAxes,
            ha='left',
            va='top'
        )
        
        plt.show()


def list_filters():
    for f in map_filter_names.keys():
        print(nice_filter_names[f]," ",'"',f,'"')

def list_hst_filters():
    for f in map_filter_names.keys():
        if "hst" in f:
            print(nice_filter_names[f]," ",f)


def plot_filters(filters):
    
    """
    Plot transmission curves of several filters present in self.data
    Default = None. All filters are considered

    Parameters
    ----------
    filters : list of filters
    """
 
    if filters is None:
        filters = list(self.images.keys())
    
    invalid_filters = [f for f in filters if f not in map_filter_names]
    if invalid_filters:
        raise ValueError(f"Invalid filters: {invalid_filters}. Must be in: {list(map_filter_names)}")

    filter_objs = [Filter(f) for f in filters]

    wl_min = min(f.wavelength_range[0] for f in filter_objs) - 300
    wl_max = max(f.wavelength_range[1] for f in filter_objs) + 300

    wavelengths = np.linspace(wl_min, wl_max, 100000)
    transmissions = [f_obj.transmission_curve(wavelengths) for f_obj in filter_objs]
   


    units = 'Å'
    if wl_max > 1e4:
        wavelengths /= 1e4
        units = r'$\mu m$'
        
    fig, ax = plt.subplots()
        
    for f,t in zip(filters,transmissions):
        ax.plot(wavelengths, t, label=nice_filter_names[f]) #color=color
    
    
    ax.set_xlabel(f"Wavelength [{units}]")
    ax.set_ylabel("Transmission")
    ax.legend()
    fig.tight_layout()
    

map_filter_names = {
    "johnson_v":"Johnson V filter",
    "johnson_u":"Johnson U filter",
    "johnson_b":"Johnson B",
    "buser_b2":"Buser's B2 filter (R. Buser)",
    "cousins_r":"Cousins R filter (Bessell 1983, PASP, 95,480)",
    "cousins_i":"Filter 85: Cousins I filter (Bessell 1983, PASP, 95,480). Use with V = 15.",
    "cfht_b":"DEEP/CFHT B-band from K-correct (Blanton)",
    "cfht_r":"DEEP/CFHT R-band from K-correct (Blanton)",
    "cfht_i":"DEEP/CFHT I-band from K-correct (Blanton)",
    "2mass_j":"2MASS J filter (total response w/atm)",
    "2mass_h":"2MASS H filter (total response w/atm)",
    "2mass_ks":"2MASS Ks filter (total response w/atm)",
    "sdss_u":"SDSS Camera u Response Function, airmass = 1.3 (June 2001)",
    "sdss_g":"SDSS Camera g Response Function, airmass = 1.3 (June 2001)",
    "sdss_r":"SDSS Camera r Response Function, airmass = 1.3 (June 2001)",
    "sdss_i":"SDSS Camera i Response Function, airmass = 1.3 (June 2001)",
    "sdss_z":"SDSS Camera z Response Function, airmass = 1.3 (June 2001)",
    "hst_wfpc2_f255w":"WFPC2 F255W",
    "hst_wfpc2_f300w":"WFPC2 F300W",
    "hst_wfpc2_f336w":"WFPC2 F336W",
    "hst_wfpc2_f439w":"WFPC2 F439W",
    "hst_wfpc2_f450w":"WFPC2 F450W",
    "hst_wfpc2_f555w":"WFPC2 F555W",
    "hst_wfpc2_f606w":"WFPC2 F606W",
    "hst_wfpc2_f702w":"WFPC2 F702W",
    "hst_wfpc2_f814w":"WFPC2 F814W",
    "hst_wfpc2_f850lp":"WFPC2 F850LP",
    "hst_acs_f435w":"WFC ACS F435W",
    "hst_acs_f475w":"WFC ACS filter F475W",
    "hst_acs_f555w":"WFC ACS filter F555W",
    "hst_acs_f606w":"WFC ACS filter F606W",
    "hst_acs_f625w":"WFC ACS filter F625W",
    "hst_acs_f775w":"WFC ACS filter F775W",
    "hst_acs_f814w":"WFC ACS filter F814W",
    "hst_acs_f850lp":"WFC ACS F850LP",
    "hst_wfc3_uvis_f218w":"HST WFC3 UVIS F218W",
    "hst_wfc3_uvis_f225w":"HST WFC3 UVIS F225W",
    "hst_wfc3_uvis_f275w":"HST WFC3 UVIS F275W",
    "hst_wfc3_uvis_f336w":"HST WFC3 UVIS F336W",
    "hst_wfc3_uvis_f390w":"HST WFC3 UVIS F390W",
    "hst_wfc3_uvis_f438w":"HST WFC3 UVIS F438W",
    "hst_wfc3_uvis_f475w":"HST WFC3 UVIS F475W",
    "hst_wfc3_uvis_f555w":"HST WFC3 UVIS F555W",
    "hst_wfc3_uvis_f606w":"HST WFC3 UVIS F606W",
    "hst_wfc3_uvis_f775w":"HST WFC3 UVIS F775W",
    "hst_wfc3_uvis_f814w":"HST WFC3 UVIS F814W",
    "hst_wfc3_uvis_f350lp":"HST WFC3 UVIS F350LP",
    "hst_wfc3_uvis_f600lp":"HST WFC3 UVIS F600LP",
    "hst_wfc3_uvis_f850lp":"HST WFC3 UVIS F850LP",
    "hst_wfc3_ir_f098m":"HST WFC3 IR F098M",
    "hst_wfc3_ir_f105w":"HST WFC3 IR F105W",
    "hst_wfc3_ir_f110w":"HST WFC3 IR F110W",
    "hst_wfc3_ir_f125w":"HST WFC3 IR F125W",
    "hst_wfc3_ir_f140w":"HST WFC3 IR F140W",
    "hst_wfc3_ir_f160w":"HST WFC3 IR F160W",
    "spitzer_irac_1":"IRAC Channel 1",
    "spitzer_irac_2":"IRAC Channel 2",
    "spitzer_irac_3":"IRAC Channel 3",
    "spitzer_irac_4":"IRAC CH4",
    "isaac_ks":"ISAAC Ks",
    "fors_u":"FORS U",
    "fors_b":"FORS B",
    "fors_v":"FORS V",
    "fors_r":"FORS R",
    "hst_nicmos_f110w":"NIC F110W",
    "hst_nicmos_f160w":"NIC F160W",
    "galex_fuv":"GALEX FUV",
    "galex_nuv":"GALEX NUV",
    "des_g":"DES g",
    "des_r":"DES r",
    "des_i":"DES i",
    "des_z":"DES z",
    "des_y":"DES Y",
    "wfcam_z":"WFCAM Z",
    "wfcam_y":"WFCAM Y",
    "wfcam_j":"WFCAM J",
    "wfcam_h":"WFCAM H",
    "wfcam_k":"WFCAM K",
    "steidel_un":"Steidel Un (via A. Shapley)",
    "steidel_g":"Steidel G (via A. Shapley)",
    "steidel_rs":"Steidel Rs (via A. Shapley)",
    "steidel_i":"Steidel I (via A. Shapley)",
    "cfht_megacam_us":"CFHT u* (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_gs":"CFHT g' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_rs":"CFHT r' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_is":"CFHT i' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_zs":"CFHT z' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "wise_w1":"WISE W1",
    "wise_w2":"WISE W2",
    "wise_w3":"WISE W3",
    "wise_w4":"WISE W4",
    "uvot_w2":"UVOT w2",
    "uvot_m2":"UVOT m2",
    "uvot_w1":"UVOT w1",
    "spitzer_mips_24um":"MIPS 24um",
    "spitzer_mips_70um":"MIPS 70um",
    "spitzer_mips_160um":"MIPS 160um",
    "scuba_450wb":"SCUBA 450WB",
    "scuba_850wb":"SCUBA 850WB",
    "herschel_pacs_70um":"Herschel PACS 70um",
    "herschel_pacs_100um":"Herschel PACS 100um",
    "herschel_pacs_160um":"Herschel PACS 160um",
    "herschel_spire_250um":"Herschel SPIRE 250um",
    "herschel_spire_350um":"Herschel SPIRE 350um",
    "herschel_spire_500um":"Herschel SPIRE 500um",
    "iras_12um":"IRAS 12um",
    "iras_25um":"IRAS 25um",
    "iras_60um":"IRAS 60um",
    "iras_100um":"IRAS 100um",
    "bessell_l":"Bessell & Brett (1988) L band",
    "bessell_l'":"Bessell & Brett (1988) L' band",
    "bessell_m":"Bessell & Brett (1988) M band",
    "stromgren_u":"Stromgren u",
    "stromgren_v":"Stromgren v",
    "stromgren_b":"Stromgren b",
    "stromgren_y":"Stromgren y",
    "idealized_1500a":"Idealized 1500A bandpass: rounded tophat centered at 1500A with 15% bandwidth, FWHM = 225A (MED)",
    "idealized_2300a":"Idealized 2300A bandpass: rounded tophat centered at 2300A with 15% bandwidth, FWHM = 345A (MED)",
    "idealized_2800a":"Idealized 2800A bandpass: rounded tophat centered at 2800A with 15% bandwidth, FWHM = 420A (MED)",
    "jwst_f070w":"JWST F070W",
    "jwst_f090w":"JWST F090W",
    "jwst_f115w":"JWST F115W",
    "jwst_f150w":"JWST F150W",
    "jwst_f200w":"JWST F200W",
    "jwst_f277w":"JWST F277W",
    "jwst_f356w":"JWST F356W",
    "jwst_f444w":"JWST F444W",
    "jwst_f150w2":"JWST F150W2",
    "jwst_f322w2":"JWST F322W2",
    "cfht_wircam_j":"CFHT WIRCAM J",
    "cfht_wircam_ks":"CFHT WIRCAM Ks",
    "vista_y":"VISTA Y",
    "vista_j":"VISTA J",
    "vista_h":"VISTA H",
    "vista_k":"VISTA K",
    "isaac_js":"ISAAC Js",
    "isaac_h":"ISAAC H",
    "nmbs_j1":"NMBS J1",
    "nmbs_j2":"NMBS J2",
    "nmbs_j3":"NMBS J3",
    "nmbs_h1":"NMBS H1",
    "nmbs_h2":"NMBS H2",
    "nmbs_ks":"NMBS Ks",
    "subaru_suprimecam_ia427":"Subaru SuprimeCam IA427 - 0320",
    "subaru_suprimecam_ia445":"Subaru SuprimeCam IA445 - 0321",
    "subaru_suprimecam_ia464":"Subaru SuprimeCam IA464 - 0322",
    "subaru_suprimecam_ia484":"Subaru SuprimeCam IA484 - 0323",
    "subaru_suprimecam_ia505":"Subaru SuprimeCam IA505 - 0324",
    "subaru_suprimecam_ia527":"Subaru SuprimeCam IA527 - 0325",
    "subaru_suprimecam_ia550":"Subaru SuprimeCam IA550 - 0326",
    "subaru_suprimecam_ia574":"Subaru SuprimeCam IA574 - 0327",
    "subaru_suprimecam_ia598":"Subaru SuprimeCam IA598 - 0328",
    "subaru_suprimecam_ia624":"Subaru SuprimeCam IA624 - 0329",
    "subaru_suprimecam_ia651":"Subaru SuprimeCam IA651 - 0330",
    "subaru_suprimecam_ia679":"Subaru SuprimeCam IA679 - 0331",
    "subaru_suprimecam_ia709":"Subaru SuprimeCam IA709 - 0332",
    "subaru_suprimecam_ia738":"Subaru SuprimeCam IA738 - 0333",
    "subaru_suprimecam_ia767":"Subaru SuprimeCam IA767 - 0334",
    "subaru_suprimecam_ia797":"Subaru SuprimeCam IA797 - 0335",
    "subaru_suprimecam_ia827":"Subaru SuprimeCam IA827 - 0336",
    "subaru_suprimecam_ia856":"Subaru SuprimeCam IA856 - 0337",
    "subaru_suprimecam_ia907":"Subaru SuprimeCam IA907 - 0338",
    "vimos_u":"VIMOS U",
    "vimos_r":"VIMOS R",
    "subaru_bj":"Subaru Bj",
    "subaru_vj":"Subaru Vj",
    "subaru_r+":"Subaru r+",
    "subaru_i+":"Subaru i+",
    "subaru_z+":"Subaru z+",
    "wfi_u38":"WFI U38",
    "wfi_b99":"WFI B99",
    "wfi_v":"WFI V",
    "wfi_rc":"WFI Rc",
    "wfi_ic":"WFI I",
    "cfht_wircam_h":"CFHT WIRCAM H",
    "wfi_i_eis":"WFI I EIS",
    "capak_subaru_b":"CAPAK_v2/B_subaru.res cosmos-b lambda_c= 4.4480e+03 AB-Vega=-0.112 w95=1035.4",
    "capak_subaru_v":"CAPAK_v2/V_subaru.res cosmos-v lambda_c= 5.4702e+03 AB-Vega=-0.000 w95=993.1",
    "capak_subaru_r":"CAPAK_v2/r_subaru.res cosmos-r lambda_c= 6.2755e+03 AB-Vega= 0.154 w95=1379.4",
    "capak_subaru_i":"CAPAK_v2/i_subaru.res cosmos-i lambda_c= 7.6712e+03 AB-Vega= 0.380 w95=1488.9",
    "capak_subaru_z":"CAPAK_v2/z_subaru.res cosmos-z lambda_c= 9.0282e+03 AB-Vega= 0.514 w95=1410.5",
    "lris_g_blue":"LRIS/g_blue_transmission.dat +atm lambda_c= 4.7508e+03 AB-Vega=-0.105 w95=940.4",
    "lris_rs_red":"LRIS/Rs_LRISred_transmission.dat +atm lambda_c= 6.8186e+03 AB-Vega= 0.247 w95=1461.1",
    "moircs_j":"MOIRCS/J277.txt +atm lambda_c= 1.2517e+04 AB-Vega= 0.913 w95=1571.2",
    "moircs_h":"MOIRCS/H117.txt +atm lambda_c= 1.6347e+04 AB-Vega= 1.348 w95=2686.4",
    "moircs_k":"MOIRCS/Ks_rot1707wedged120K.txt +atm lambda_c= 2.1577e+04 AB-Vega= 1.828 w95=3043.6",
    "mosaic_u":"NOAO/k1001bp_jul04.txt MOSAIC-U-2004+atm lambda_c= 3.5929e+03 AB-Vega= 0.842 w95=720.7",
    "maiz_u":"REST_FRAME/maiz-apellaniz_Johnson_U.res 2006AJ....131.1184M lambda_c= 3.5900e+03 AB-Vega= 0.769 w95=862.3",
    "maiz_b":"REST_FRAME/maiz-apellaniz_Johnson_B.res 2006AJ....131.1184M lambda_c= 4.3722e+03 AB-Vega=-0.106 w95=1321.3",
    "maiz_v":"REST_FRAME/maiz-apellaniz_Johnson_V.res 2006AJ....131.1184M lambda_c= 5.4794e+03 AB-Vega= 0.002 w95=1369.1",
    "vista_z":"VISTA_z band",
    "cfht_megacam_r":"CHFT megacam R vestige team",
    "cfht_megacam_ha":"CHFT megacam Ha vestige team",
    "muse_blue":"MUSE BLUE tophat",
    "muse_green":"MUSE GREEN tophat",
    "muse_red":"MUSE RED tophat",
    "muse_nir":"MUSE NIR tophat",
    "muse_white_aoe":"MUSE WHITE AOE",
    "uvit_fuv_f154w":"UVIT FUV F154W", #  (UVIT FUV BaF2 Filter, from Tandon et al. 2020, downloaded from SVO)
    "uvit_nuv_n245m":"UVIT NUV N245M", # (UVIT NUV B13 filter,  from Tandon et al. 2020, downloaded from SVO)
    "uvit_nuv_n263m":"UVIT NUV N263M", # (UVIT NUV B4 filter,   from Tandon et al. 2020, downloaded from SVO)
    "hawki_h":"HAWKI H",
    "hawki_ch4":"HAWKI CH4",
    "hawki_ks":"HAWKI Ks",
    "jwst_miri_f770w":"JWST MIRI F770W",
    "jwst_miri_f1000w":"JWST MIRI F1000W",
    "jwst_miri_f1280w":"JWST MIRI F1280W",
    "jwst_miri_f1800w":"JWST MIRI F1800W",
    "alma_band6_1p2mm":"ALMA Band 6 1p2mm",
}


nice_filter_names = {
    "NUV": 'NUV',
    "FUV": 'FUV',
    "johnson_v":"Johnson V filter",
    "johnson_u":"Johnson U filter",
    "johnson_b":"Johnson B",
    "buser_b2":"Buser's B2 filter (R. Buser)",
    "cousins_r":"Cousins R filter (Bessell 1983, PASP, 95,480)",
    "cousins_i":"Filter 85: Cousins I filter (Bessell 1983, PASP, 95,480). Use with V = 15.",
    "cfht_b":"DEEP/CFHT B-band from K-correct (Blanton)",
    "cfht_r":"DEEP/CFHT R-band from K-correct (Blanton)",
    "cfht_i":"DEEP/CFHT I-band from K-correct (Blanton)",
    "2mass_j":"2MASS J",
    "2mass_h":"2MASS H",
    "2mass_ks":"2MASS Ks",
    "sdss_u":"SDSS u",
    "sdss_g":"SDSS g",
    "sdss_r":"SDSS r",
    "sdss_i":"SDSS i",
    "sdss_z":"SDSS z",
    "hst_wfpc2_f255w":"HST WFPC2 F255W",
    "hst_wfpc2_f300w":"HST WFPC2 F300W",
    "hst_wfpc2_f336w":"HST WFPC2 F336W",
    "hst_wfpc2_f439w":"HST WFPC2 F439W",
    "hst_wfpc2_f450w":"HST WFPC2 F450W",
    "hst_wfpc2_f555w":"HST WFPC2 F555W",
    "hst_wfpc2_f606w":"HST WFPC2 F606W",
    "hst_wfpc2_f702w":"HST WFPC2 F702W",
    "hst_wfpc2_f814w":"HST WFPC2 F814W",
    "hst_wfpc2_f850lp":"HST WFPC2 F850LP",
    "hst_acs_f435w":"HST ACS F435W",
    "hst_acs_f475w":"HST ACS F475W",
    "hst_acs_f555w":"HST ACS F555W",
    "hst_acs_f606w":"HST ACS F606W",
    "hst_acs_f625w":"HST ACS F625W",
    "hst_acs_f775w":"HST ACS F775W",
    "hst_acs_f814w":"HST ACS F814W",
    "hst_acs_f850lp":"HST ACS F850LP",
    "hst_wfc3_uvis_f218w":"HST WFC3 UVIS F218W",
    "hst_wfc3_uvis_f225w":"HST WFC3 UVIS F225W",
    "hst_wfc3_uvis_f275w":"HST WFC3 UVIS F275W",
    "hst_wfc3_uvis_f336w":"HST WFC3 UVIS F336W",
    "hst_wfc3_uvis_f390w":"HST WFC3 UVIS F390W",
    "hst_wfc3_uvis_f438w":"HST WFC3 UVIS F438W",
    "hst_wfc3_uvis_f475w":"HST WFC3 UVIS F475W",
    "hst_wfc3_uvis_f555w":"HST WFC3 UVIS F555W",
    "hst_wfc3_uvis_f606w":"HST WFC3 UVIS F606W",
    "hst_wfc3_uvis_f775w":"HST WFC3 UVIS F775W",
    "hst_wfc3_uvis_f814w":"HST WFC3 UVIS F814W",
    "hst_wfc3_uvis_f350lp":"HST WFC3 UVIS F350LP",
    "hst_wfc3_uvis_f600lp":"HST WFC3 UVIS F600LP",
    "hst_wfc3_uvis_f850lp":"HST WFC3 UVIS F850LP",
    "hst_wfc3_ir_f098m":"HST WFC3 IR F098M",
    "hst_wfc3_ir_f105w":"HST WFC3 IR F105W",
    "hst_wfc3_ir_f110w":"HST WFC3 IR F110W",
    "hst_wfc3_ir_f125w":"HST WFC3 IR F125W",
    "hst_wfc3_ir_f140w":"HST WFC3 IR F140W",
    "hst_wfc3_ir_f160w":"HST WFC3 IR F160W",
    "spitzer_irac_1":"IRAC Channel 1 (3.6 μm)",
    "spitzer_irac_2":"IRAC Channel 2 (4.5 μm)",
    "spitzer_irac_3":"IRAC Channel 3 (5.8 μm)",
    "spitzer_irac_4":"IRAC Channel 4 (8.0 μm)",
    "isaac_ks":"ISAAC Ks",
    "fors_u":"FORS U",
    "fors_b":"FORS B",
    "fors_v":"FORS V",
    "fors_r":"FORS R",
    "hst_nicmos_f110w":"NIC F110W",
    "hst_nicmos_f160w":"NIC F160W",
    "galex_fuv":"GALEX FUV",
    "galex_nuv":"GALEX NUV",
    "des_g":"DES g",
    "des_r":"DES r",
    "des_i":"DES i",
    "des_z":"DES z",
    "des_y":"DES Y",
    "wfcam_z":"WFCAM Z",
    "wfcam_y":"WFCAM Y",
    "wfcam_j":"WFCAM J",
    "wfcam_h":"WFCAM H",
    "wfcam_k":"WFCAM K",
    "steidel_un":"Steidel Un (via A. Shapley)",
    "steidel_g":"Steidel G (via A. Shapley)",
    "steidel_rs":"Steidel Rs (via A. Shapley)",
    "steidel_i":"Steidel I (via A. Shapley)",
    "cfht_megacam_us":"CFHT u* (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_gs":"CFHT g' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_rs":"CFHT r' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_is":"CFHT i' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "cfht_megacam_zs":"CFHT z' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)",
    "wise_w1":"WISE Band 1 (3.4 μm)",
    "wise_w2":"WISE Band 2 (4.6 μm)",
    "wise_w3":"WISE Band 3 (12 μm)",
    "wise_w4":"WISE Band 4 (22 μm)",
    "uvot_w2":"UVOT w2",
    "uvot_m2":"UVOT m2",
    "uvot_w1":"UVOT w1",
    "spitzer_mips_24um":"MIPS 24 μm",
    "spitzer_mips_70um":"MIPS 70 μm",
    "spitzer_mips_160um":"MIPS 160 μm",
    "scuba_450wb":"SCUBA 450WB",
    "scuba_850wb":"SCUBA 850WB",
    "herschel_pacs_70um":"Herschel PACS 70 μm",
    "herschel_pacs_100um":"Herschel PACS 100 μm",
    "herschel_pacs_160um":"Herschel PACS 160 μm",
    "herschel_spire_250um":"Herschel SPIRE 250 μm",
    "herschel_spire_350um":"Herschel SPIRE 350 μm",
    "herschel_spire_500um":"Herschel SPIRE 500 μm",
    "iras_12um":"IRAS 12 μm",
    "iras_25um":"IRAS 25 μm",
    "iras_60um":"IRAS 60 μm",
    "iras_100um":"IRAS 100 μm",
    "bessell_l":"Bessell & Brett (1988) L band",
    "bessell_l'":"Bessell & Brett (1988) L' band",
    "bessell_m":"Bessell & Brett (1988) M band",
    "stromgren_u":"Stromgren u",
    "stromgren_v":"Stromgren v",
    "stromgren_b":"Stromgren b",
    "stromgren_y":"Stromgren y",
    "idealized_1500a":"Idealized 1500A bandpass: rounded tophat centered at 1500A with 15% bandwidth, FWHM = 225A (MED)",
    "idealized_2300a":"Idealized 2300A bandpass: rounded tophat centered at 2300A with 15% bandwidth, FWHM = 345A (MED)",
    "idealized_2800a":"Idealized 2800A bandpass: rounded tophat centered at 2800A with 15% bandwidth, FWHM = 420A (MED)",
    "jwst_f070w":"JWST F070W",
    "jwst_f090w":"JWST F090W",
    "jwst_f115w":"JWST F115W",
    "jwst_f150w":"JWST F150W",
    "jwst_f200w":"JWST F200W",
    "jwst_f277w":"JWST F277W",
    "jwst_f356w":"JWST F356W",
    "jwst_f444w":"JWST F444W",
    "jwst_f150w2":"JWST F150W2",
    "jwst_f322w2":"JWST F322W2",
    "cfht_wircam_j":"CFHT WIRCAM J",
    "cfht_wircam_ks":"CFHT WIRCAM Ks",
    "vista_y":"VISTA Y",
    "vista_j":"VISTA J",
    "vista_h":"VISTA H",
    "vista_k":"VISTA K",
    "isaac_js":"ISAAC Js",
    "isaac_h":"ISAAC H",
    "nmbs_j1":"NMBS J1",
    "nmbs_j2":"NMBS J2",
    "nmbs_j3":"NMBS J3",
    "nmbs_h1":"NMBS H1",
    "nmbs_h2":"NMBS H2",
    "nmbs_ks":"NMBS Ks",
    "subaru_suprimecam_ia427":"Subaru SuprimeCam IA427 - 0320",
    "subaru_suprimecam_ia445":"Subaru SuprimeCam IA445 - 0321",
    "subaru_suprimecam_ia464":"Subaru SuprimeCam IA464 - 0322",
    "subaru_suprimecam_ia484":"Subaru SuprimeCam IA484 - 0323",
    "subaru_suprimecam_ia505":"Subaru SuprimeCam IA505 - 0324",
    "subaru_suprimecam_ia527":"Subaru SuprimeCam IA527 - 0325",
    "subaru_suprimecam_ia550":"Subaru SuprimeCam IA550 - 0326",
    "subaru_suprimecam_ia574":"Subaru SuprimeCam IA574 - 0327",
    "subaru_suprimecam_ia598":"Subaru SuprimeCam IA598 - 0328",
    "subaru_suprimecam_ia624":"Subaru SuprimeCam IA624 - 0329",
    "subaru_suprimecam_ia651":"Subaru SuprimeCam IA651 - 0330",
    "subaru_suprimecam_ia679":"Subaru SuprimeCam IA679 - 0331",
    "subaru_suprimecam_ia709":"Subaru SuprimeCam IA709 - 0332",
    "subaru_suprimecam_ia738":"Subaru SuprimeCam IA738 - 0333",
    "subaru_suprimecam_ia767":"Subaru SuprimeCam IA767 - 0334",
    "subaru_suprimecam_ia797":"Subaru SuprimeCam IA797 - 0335",
    "subaru_suprimecam_ia827":"Subaru SuprimeCam IA827 - 0336",
    "subaru_suprimecam_ia856":"Subaru SuprimeCam IA856 - 0337",
    "subaru_suprimecam_ia907":"Subaru SuprimeCam IA907 - 0338",
    "vimos_u":"VIMOS U",
    "vimos_r":"VIMOS R",
    "subaru_bj":"Subaru Bj",
    "subaru_vj":"Subaru Vj",
    "subaru_r+":"Subaru r+",
    "subaru_i+":"Subaru i+",
    "subaru_z+":"Subaru z+",
    "wfi_u38":"WFI U38",
    "wfi_b99":"WFI B99",
    "wfi_v":"WFI V",
    "wfi_rc":"WFI Rc",
    "wfi_ic":"WFI I",
    "cfht_wircam_h":"CFHT WIRCAM H",
    "wfi_i_eis":"WFI I EIS",
    "capak_subaru_b":"CAPAK_v2/B_subaru.res cosmos-b lambda_c= 4.4480e+03 AB-Vega=-0.112 w95=1035.4",
    "capak_subaru_v":"CAPAK_v2/V_subaru.res cosmos-v lambda_c= 5.4702e+03 AB-Vega=-0.000 w95=993.1",
    "capak_subaru_r":"CAPAK_v2/r_subaru.res cosmos-r lambda_c= 6.2755e+03 AB-Vega= 0.154 w95=1379.4",
    "capak_subaru_i":"CAPAK_v2/i_subaru.res cosmos-i lambda_c= 7.6712e+03 AB-Vega= 0.380 w95=1488.9",
    "capak_subaru_z":"CAPAK_v2/z_subaru.res cosmos-z lambda_c= 9.0282e+03 AB-Vega= 0.514 w95=1410.5",
    "lris_g_blue":"LRIS/g_blue_transmission.dat +atm lambda_c= 4.7508e+03 AB-Vega=-0.105 w95=940.4",
    "lris_rs_red":"LRIS/Rs_LRISred_transmission.dat +atm lambda_c= 6.8186e+03 AB-Vega= 0.247 w95=1461.1",
    "moircs_j":"MOIRCS/J277.txt +atm lambda_c= 1.2517e+04 AB-Vega= 0.913 w95=1571.2",
    "moircs_h":"MOIRCS/H117.txt +atm lambda_c= 1.6347e+04 AB-Vega= 1.348 w95=2686.4",
    "moircs_k":"MOIRCS/Ks_rot1707wedged120K.txt +atm lambda_c= 2.1577e+04 AB-Vega= 1.828 w95=3043.6",
    "mosaic_u":"NOAO/k1001bp_jul04.txt MOSAIC-U-2004+atm lambda_c= 3.5929e+03 AB-Vega= 0.842 w95=720.7",
    "maiz_u":"REST_FRAME/maiz-apellaniz_Johnson_U.res 2006AJ....131.1184M lambda_c= 3.5900e+03 AB-Vega= 0.769 w95=862.3",
    "maiz_b":"REST_FRAME/maiz-apellaniz_Johnson_B.res 2006AJ....131.1184M lambda_c= 4.3722e+03 AB-Vega=-0.106 w95=1321.3",
    "maiz_v":"REST_FRAME/maiz-apellaniz_Johnson_V.res 2006AJ....131.1184M lambda_c= 5.4794e+03 AB-Vega= 0.002 w95=1369.1",
    "vista_z":"VISTA_z band",
    "cfht_megacam_r":"CHFT megacam R vestige team",
    "cfht_megacam_ha":"CHFT megacam Ha vestige team",
    "muse_blue":"MUSE BLUE",
    "muse_green":"MUSE GREEN",
    "muse_red":"MUSE RED",
    "muse_nir":"MUSE NIR",
    "muse_white_aoe":"MUSE WHITE AOE",
    "muse_Ha":"MUSE Hα",
    "muse_Hb":"MUSE Hβ",
    "muse_SII":"MUSE [S II]",
    "muse_OIII":"MUSE [O III] 5007",
    "uvit_fuv_f154w":"UVIT FUV F154W (BaF2)",
    "uvit_nuv_n245m":"UVIT NUV N245M (B13)",
    "uvit_nuv_n263m":"UVIT NUV N263M (B4)",
    "hawki_h":"HAWKI H",
    "hawki_ch4":"HAWKI CH4",
    "hawki_ks":"HAWKI Ks",
    "jwst_miri_f770w":"JWST MIRI F770W",
    "jwst_miri_f1000w":"JWST MIRI F1000W",
    "jwst_miri_f1280w":"JWST MIRI F1280W",
    "jwst_miri_f1800w":"JWST MIRI F1800W",
    "alma_band6_1p2mm":"ALMA Band 6 1p2mm",
}


default_cmaps = {
    'FUV': 'gnuplot', 'NUV': 'gnuplot2', 
    'galex_fuv': 'gnuplot', 'galex_nuv': 'gnuplot2', 
    'muse_blue': 'bone', 'muse_green': 'viridis','muse_red': 'hot', 'muse_nir': 'copper', 'muse_white_aoe': 'gray',
    'muse_Ha': 'plasma', 'muse_Hb': 'inferno', 'muse_SII': 'viridis', 
    **{band: 'gray' for band in map_filter_names if band.startswith('hst')},
    'sdss_u': 'Blues_r','sdss_g': 'Greens_r', 'sdss_r': 'Reds_r', 'sdss_i': 'YlOrBr_r', 'sdss_z': 'Greys_r',
    '2mass_j': 'PuRd_r', '2mass_h': 'BuPu_r', '2mass_ks': 'Oranges_r', 
    **{f'wise_w{band}': 'viridis' for band in ['1','2','3','4']},
    **{band: 'nipy_spectral' for band in map_filter_names if band.startswith('spitzer_irac')},
    'herschel_pacs_70um': 'Blues_r', 'herschel_pacs_100um': 'Greens_r', 'herschel_pacs_160um': 'Reds_r'
    }
    
default_filter_colors = { 
    'galex_fuv': 'violet', 'galex_nuv': 'darkblue', 
    'muse_blue': 'blue', 'muse_green': 'green', 'muse_red': 'red', 'muse_nir': 'gold', 'muse_white_aoe':'gray',
    'muse_Ha': 'orange', 'muse_Hb': 'purple', 'muse_sii': 'lime',
    **{band: 'purple' for band in map_filter_names if band.startswith('hst')},
    'sdss_u': 'midnightblue', 'sdss_g': 'cyan', 'sdss_r': 'orangered', 
    'sdss_i': 'red', 'sdss_z': 'darkred',
    '2mass_j': 'pink', '2mass_h': 'purple', '2mass_ks': 'darkviolet',
    **{f'wise_w{band}': 'brown' for band in ['1','2','3','4']},
    'spitzer_irac_1': 'teal', 'spitzer_irac_2': 'lightgreen', 
    'spitzer_irac_3': 'orange', 'spitzer_irac_4': 'darkred',
    'herschel_pacs_70um': 'lime', 'herschel_pacs_100um': 'lime', 'herschel_pacs_160um': 'lime'
}    

        
default_plot_units = {
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('galex')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('uvit')},
    'muse_blue': 'mag_arcsec2','muse_green': 'mag_arcsec2', 'muse_red': 'mag_arcsec2', 'muse_nir': 'mag_arcsec2', 'muse_white_aoe': 'mag_arcsec2',
    'muse_Ha': 'erg_s_cm2_arcsec2', 'muse_Hb': 'erg_s_cm2_arcsec2',
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('hst')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('sdss')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('2mass')},
    **{band: 'mag_arcsec2' for band in map_filter_names if band.startswith('wise')},
    **{band: 'MJy_sr' for band in map_filter_names if band.startswith('spitzer')},
    **{band: 'MJy_sr' for band in map_filter_names if band.startswith('herschel')},
}

default_plot_scale_lims = {
    'FUV': (22, 27), 'NUV': (22, 27), 
    'galex_fuv': (22, 27), 'galex_nuv': (22, 27), 
    'muse_blue': (20, 27),'muse_green': (20, 27), 'muse_red': (20, 27), 'muse_nir': (20, 27), 'muse_white_aoe': (20,27),
    'muse_Ha': (1e-19, 1e-14), 'muse_Hb': (1e-19, 1e-14),
    **{band: (19, 27) for band in map_filter_names if band.startswith('hst')},
    **{f'sdss_{band}': (20, 27) for band in ['u', 'g', 'r', 'i', 'z']},
    '2mass_j': (15, 22), '2mass_h': (15, 22), '2mass_ks': (15, 21), 
    **{f'wise_w{band}': (15, 22) for band in ['1','2','3','4']},
    'spitzer_irac_1': (6e-2, 10), 'spitzer_irac_2': (1e-1, 10), 'spitzer_irac_3': (1, 10), 'spitzer_irac_4': (5, 100),
    'herschel_pacs_70um': (10, 100), 'herschel_pacs_100um': (10, 100), 'herschel_pacs_160um': (10, 100)
}
         




