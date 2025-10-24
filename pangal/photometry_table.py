import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from astropy.table import Table
from collections.abc import Iterable

from astropy.io import fits

from .filters import Filter, map_filter_names, nice_filter_names




@dataclass
class PhotometryTable:
    data: dict = field(default_factory=dict)                      # Dictionary of {band: flux}
    region: object = None                                         # Region of sky (define externally or with shapely/sregions)
    header: fits.Header = field(default_factory=fits.Header)      # FITS header (for WCS/area info)

    @property
    def to_ABmag(self):
        if 'mJy' in self.header['units']:
            for band in self.data.keys(): 
                self.data[band] = self.mJy_to_ABmag(self.data[band])

    def mJy_to_ABmag(self, flux_mJy):
        flux_Jy = flux_mJy * 1e-3
        with np.errstate(divide='ignore'):
            mag = -2.5 * np.log10(flux_Jy) + 8.90
        return mag

    """
    def mJy_to_erg_s_cm2_A(self, flux_mJy, wavelength):
        flux_Jy = flux_mJy * 1e-3  # Jy
        fnu = flux_Jy * u.Jy
        wav = wavelength * u.AA
        flam = fnu.to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(wav))
        return flam.value

    def MJy_sr_to_erg_s_cm2_A_arcsec2(self, flux_MJy_sr, wavelength):
        flux = flux_MJy_sr * u.MJy / u.sr
        wav = wavelength * u.AA
        flam = flux.to(u.erg / u.s / u.cm**2 / u.AA / u.arcsec**2, equivalencies=u.spectral_density(wav))
        return flam.value
    """

    @property
    def bands(self):
        print(self.data.keys())


    @property
    def df(self):
        df = pd.DataFrame.from_dict(self.data, orient='index', columns=['Value', 'Error'])
        df.index.name = 'Band'
        
        # Add SNR column
        df['SNR'] = df['Value'] / df['Error']
        
        #units = self.header.get('units', 'Unknown Units')
        return df


def print_photometric_tables(photometric_tables_list, nice_filters_name=None):
    """
    Combine multiple PhotometryTable objects into a single DataFrame.
    Each column corresponds to a region, each row to a band (renamed via nice_filters_name).
    Values are formatted as 'value ± error (SNR)'.
    Cells with SNR <= 5 are displayed in red.
    """
    import pandas as pd

    combined = {}
    snr_dict = {}

    for i, pt in enumerate(photometric_tables_list):
        # Region label
        if pt.region is not None and hasattr(pt.region, "name"):
            region_label = f"Region {pt.region.name}"
        else:
            region_label = f"Region {i+1}"

        # Copy the DataFrame
        df = pt.df.copy()

        # Rename bands if mapping provided
        df.index = [nice_filter_names.get(b, b) for b in df.index]

        # Format value ± error (SNR)
        formatted = df.apply(
            lambda row: f"{row['Value']:.3f} ± {row['Error']:.3f} ({row['SNR']:.1f})",
            axis=1
        )
        combined[region_label] = formatted

        # Keep SNR for styling
        snr_dict[region_label] = df['SNR']

    # Merge all into a DataFrame
    result = pd.DataFrame(combined)
    result.index.name = "Band"

    # Create a DataFrame with the same shape for SNR
    snr_df = pd.DataFrame(snr_dict, index=result.index)

    # Style: red text if SNR <= 5
    def highlight_low_snr(val, snr_val):
        if snr_val <= 5:
            return 'color: red'
        return ''

    df = result.style.apply(
        lambda col: [highlight_low_snr(v, s) for v, s in zip(col, snr_df[col.name])],
        axis=0
    )

    return df
    


# --- Functions to read and to write PhotometryTable to fits

def PhotometryTable_to_fits(filename, tables):

    if not isinstance(tables, Iterable) or isinstance(tables, dict):
        tables = [tables] 

    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())  # Empty primary HDU

    for i, table in enumerate(tables):
        # Convert photometry data to an astropy Table
        data_dict = table.data
        t = Table()
        for band, value in data_dict.items():
            t[band] = [value]  # Store as single-row table

        # Create a BinTableHDU from the Table
        hdu = fits.BinTableHDU(t)
        
        # Attach header from PhotometryTable (if any)
        if table.header:
            for key in table.header:
                try:
                    hdu.header[key] = table.header[key]
                except Exception:
                    pass  # Ignore any conflicts or FITS-invalid keywords

        # Give the extension a name
        hdu.name = f"PHOT_{i}" if len(tables) > 1 else "PHOTOMETRY"

        hdul.append(hdu)

    # Write to disk
    hdul.writeto(filename, overwrite=True)
    print(f"Written {len(tables)} photometry table(s) to {filename}")



def fits_to_PhotometryTable(filename):

    hdul = fits.open(filename)

    phot_tables = []
    for hdu in hdul[1:]:  # Skip primary HDU
        if not isinstance(hdu, fits.BinTableHDU):
            continue

        tbl = Table(hdu.data)
        if len(tbl) != 1:
            raise ValueError("Each photometry table should contain exactly one row.")

        # Extract data from the single row
        data_dict = {col: tbl[col][0] for col in tbl.colnames}
        phot_table = PhotometryTable(
            data=data_dict,
            header=hdu.header
        )
        phot_tables.append(phot_table)

    hdul.close()

    if len(phot_tables) == 1:
        return phot_tables[0]
    return phot_tables

