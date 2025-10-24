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
    cube: np.ndarray=None                              # cube in native units
    var: np.ndarray=None                               # var in native units
    units: float=None                                  # convert to erg/s/cm2/A
    wl: np.array=None                                   # wavelength array
    resolution: np.array=None                                   # resolution array
    dw: float=None                                     # spectral channel width                     
    wcs: WCS=None                                       # Astropy WCS object
    dtheta_pix_deg: float=None                          # Pixel scale (degrees/pixel)
    area_pix_arcsec2: float=None                        # Pixel area (arcsec²)
    id: str='cube'                                      # id

    header: fits.Header = field(default_factory=fits.Header)  # FITS header 
    
    @property
    def delta_lambda(self):
        """
        delta lambda (lambda): spectral resoolution
        """
        return self.wl / self.resolution  
    
    
    def line_map(self, line, width, continuum_offset_1, continuum_offset_2, z=0, pangal_object=None,):
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
        image_erg_s_cm2 = (line_cube - continuum) * self.dw * self.units  

        image = Image(
                image=image_erg_s_cm2,
                wcs=self.wcs,
                dtheta_pix_deg=self.dtheta_pix_deg,
                area_pix_arcsec2=self.area_pix_arcsec2,
                pivot_wavelength=obs_wavelength,
                header=self.header
            )
    
        if pangal_object:
            # Safe naming: avoid overwriting
            base_name = f"{self.id}_{line}"
            image_name = base_name
            i = 1
            while image_name in pangal_object.images:
                image_name = f"{base_name}({i})"
                i += 1
        
            # Save
            pangal_object.images[image_name] = image
        

        return image
            


    def extract_spectra(self, regions, pangal_object=None):
        """
        Extracts and stores spectra from defined spatial regions within data cube.

        Parameters
        ----------
        regions : list
            List of Region objects, each with a 'project' method 
        pangal_object : Pangal
            If provided, appends the line maps as images to the pangal_object.images list
        Returns
        -------
        list of spectra objects
        """
 
        var = self.var
        wl = self.wl
        resolution = self.resolution
        spectra_list = []

        for region in regions:
            # Generate a mask for the region using the WCS projection
            mask = region.project(self.cube[0, :, :], self.wcs).astype(bool)
            n_pixels = np.count_nonzero(mask)

            # Sum flux and variance over all pixels in the region
            flux = np.nansum(self.cube[:, mask], axis=1) / n_pixels * self.units
            variance = np.nansum(var[:, mask], axis=1) / n_pixels**2 * self.units**2
            error = np.sqrt(variance)
            
            spec = Spectrum(wl=wl,resolution=resolution,flux=flux,flux_err=error)

            spectra_list.append(spec)


            if pangal_object:
                # Safe naming: avoid overwriting
                base_name = f"{self.id}_spec_{region.id}"
                spec_name = base_name
                i = 1
                while spec_name in pangal_object.spectra:
                    spec_name = f"{base_name}({i})"
                    i += 1
            
                # Save
                pangal_object.spectra[spec_name] = spec

        return spectra_list
