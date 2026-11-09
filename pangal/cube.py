import numpy as np
from dataclasses import dataclass, field

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

from .image import Image
from .spectrum import Spectrum
from .region import Region

from .data.spectral_lines import spectral_lines

@dataclass
class Cube:
    cube: np.ndarray=None                              # cube 
    var: np.ndarray=None                               # var 
    wl: np.array=None                                   # wavelength array
    resolution: np.array=None                                   # resolution array
    dw: float=None                                     # spectral channel width                     
    wcs: WCS=None                                       # Astropy WCS object
    dtheta_pix_deg: float=None                          # Pixel scale (degrees/pixel)
    area_pix_arcsec2: float=None                        # Pixel area (arcsec²)
    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    id: str = None

    @property
    def delta_lambda(self):
        """
        delta lambda (lambda): spectral resoolution
        """
        return self.wl / self.resolution  
    


    def tophat_map(self,wl_inf,wl_sup):
    
        channel_inf = np.digitize(wl_inf, self.wl) - 1
        channel_sup = np.digitize(wl_sup, self.wl) - 1
        bandwidth = wl_sup - wl_inf
        pivot_wavelength = (wl_sup + wl_inf)/2

        integrated_band = np.nansum(self.cube[channel_inf:channel_sup, :, :], axis=0)
        
        image = integrated_band * self.dw / bandwidth 

        ZP =  - 2.5 * np.log10(pivot_wavelength**2/2.998e18) - 48.60
        ZP_err = 0
        exptime = None

        image = Image(
            data=image,
            wcs=self.wcs,
            dtheta_pix_deg=self.dtheta_pix_deg,
            area_pix_arcsec2=self.area_pix_arcsec2,
            pivot_wavelength=pivot_wavelength,
            ZP=ZP,
            ZP_err=ZP_err,
            header=self.header,
            )
        
        return image



    def line_map(self, line, width, continuum_offset_1, continuum_offset_2, z=0,):
        """
        Extracts an emission line map from a datacube, subtracts continuum and applies extinction correction.
        
        Parameters
        ----------
        line : str
            Name of the spectral line (must be in spectral_lines dict).
        width : float
            Width of the integration window around the line (in Angstrom).
        continuum_offset_1, continuum_offset_2 : float
            Offsets for the two continuum sidebands (in Angstrom).
        z : float
            Redshift of the galaxy.
        pangal_object : Pangal
            If provided, appends the line maps as images to the pangal_object.images list
        """
        
        var = self.var
        wl = self.wl
    
        # Rest-frame and redshifted wavelength
        rest_wavelength = spectral_lines[line]
        obs_wavelength = rest_wavelength * (1 + z)
    
        # Convert to channel indices
        ch_inf = np.digitize(obs_wavelength - width / 2, wl) - 1
        ch_sup = np.digitize(obs_wavelength + width / 2, wl) - 1
    
        cont_inf_1 = np.digitize(obs_wavelength - continuum_offset_1, wl) - 1
        cont_sup_1 = np.digitize(obs_wavelength - continuum_offset_1 + width, wl) - 1
        cont_inf_2 = np.digitize(obs_wavelength + continuum_offset_2, wl) - 1
        cont_sup_2 = np.digitize(obs_wavelength + continuum_offset_2 + width, wl) - 1
    
        # Integrate flux and continuum
        line_cube = np.nansum(self.cube[ch_inf:ch_sup, :, :], axis=0)
        cont_cube_1 = np.nansum(self.cube[cont_inf_1:cont_sup_1, :, :], axis=0)
        cont_cube_2 = np.nansum(self.cube[cont_inf_2:cont_sup_2, :, :], axis=0)
        continuum = (cont_cube_1 + cont_cube_2) / 2
    
        # Optional extinction correction
        image_erg_s_cm2 = (line_cube - continuum) * self.dw 

        image = Image(
                data=image_erg_s_cm2,
                wcs=self.wcs,
                dtheta_pix_deg=self.dtheta_pix_deg,
                area_pix_arcsec2=self.area_pix_arcsec2,
                pivot_wavelength=obs_wavelength,
                header=self.header
            )
    
        return image
    


    
    def extract_spectra(self, regions, pangal_object=None):
        var = self.var
        resolution = self.resolution
        spectra_list = []

        # Air -> vacuum
        wl_vac = airtovac(self.wl)

        for region in regions:
            mask = region.project(self.cube[0, :, :], self.wcs).astype(bool)
            n_pixels = np.count_nonzero(mask)

            flux = np.nansum(self.cube[:, mask], axis=1)
            variance = np.nansum(var[:, mask], axis=1)
            error = np.sqrt(variance)

            hdr = fits.Header()
            hdr["SOURCE_ID"] = getattr(self, "id", "CUBE")
            hdr["REGION_ID"] = getattr(region, "id", "UNKNOWN")
            hdr["NPIXELS"]   = n_pixels
            hdr["FUNITS"]    = self.header.get("FUNITS", "UNKNOWN")
            hdr["WUNITS"]    = self.header.get("WUNITS", "UNKNOWN")
            hdr["WAVESYS"]   = "VACUUM"

            # copy so that the spectra don't share the same array
            spec = Spectrum(wl=wl_vac.copy(), resolution=resolution,
                            flux=flux, flux_err=error, header=hdr)
            spectra_list.append(spec)

            if pangal_object:
                base_name = f"{getattr(self, 'id', 'cube')}_spec_{getattr(region, 'id', 'region')}"
                spec_name = base_name
                i = 1
                while spec_name in pangal_object.spectra:
                    spec_name = f"{base_name}({i})"
                    i += 1
                pangal_object.spectra[spec_name] = spec

        return spectra_list
    

    
def vactoair(wl):
    """Convert vacuum wavelengths to air wavelengths using the conversion
    given by Morton (1991, ApJS, 77, 119).

    """
    wave2 = np.asarray(wl, dtype=float)**2
    fact = 1. + 2.735182e-4 + 131.4182/wave2 + 2.76249e8/(wave2*wave2)
    return wl/fact


def airtovac(wl):
    """Convert air wavelengths to vacuum wavelengths using the conversion
    given by Morton (1991, ApJS, 77, 119).

    """
    sigma2 = np.asarray(1E4/wl, dtype=float)**2
    
    fact = 1. + 6.4328e-5 + 2.949281e-2/(146.-sigma2) + 2.5540e-4/(41.-sigma2)
    fact[wl < 2000] = 1.0
    
    return wl*fact
