import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from astropy.table import Table
from collections.abc import Iterable

from astropy.io import fits

from .filter import Filter, map_filter_names, nice_filter_names



@dataclass
class PhotometryTable:
    """
    Multi-region photometry container.

    data:
        {
            region_name : {
                band_name : (flux, error)
            }
        }

    header:
        Global header — must include UNITS.
        For Fλ conversion, must contain:
            PIVOT_band = wavelength_Angstrom
    """
    data: dict = field(default_factory=dict)
    header: fits.Header = field(default_factory=fits.Header)

    # ==========================================================
    # --- Conversion helpers (no astropy.units)
    # ==========================================================

    @staticmethod
    def mJy_to_Jy(f_mJy):
        return f_mJy * 1e-3

    @staticmethod
    def Jy_to_ABmag(f_Jy):
        with np.errstate(divide="ignore"):
            return -2.5 * np.log10(f_Jy) + 8.90

    @staticmethod
    def ABmag_error(f_Jy, df_Jy):
        if df_Jy is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.085736 * (df_Jy / f_Jy)

    @staticmethod
    def mJy_to_flam(f_mJy, wavelength_A):
        c = 2.99792458e18  # Å/s
        fnu = f_mJy * 1e-26
        return fnu * c / (wavelength_A ** 2)

    # ==========================================================
    # --- Conversions (return NEW PhotometryTable)
    # ==========================================================

    def to_ABmag(self):
        """Convert ALL regions from mJy → AB mag."""
        if self.header.get("UNITS", "").lower() != "mjy":
            raise ValueError("Units must be mJy for ABmag conversion.")

        new_data = {}

        for region, region_dict in self.data.items():
            new_data[region] = {}

            for band, (flux, err) in region_dict.items():
                f_Jy = self.mJy_to_Jy(flux)
                df_Jy = self.mJy_to_Jy(err) if err is not None else None

                mag = self.Jy_to_ABmag(f_Jy)
                mag_err = self.ABmag_error(f_Jy, df_Jy)

                new_data[region][band] = (mag, mag_err)

        new_header = self.header.copy()
        new_header["UNITS"] = "ABMAG"

        return PhotometryTable(new_data, new_header)

    # ----------------------------------------------------------

    def to_flam(self):
        """Convert ALL regions from mJy → erg/s/cm²/Å."""
        if self.header.get("UNITS", "").lower() != "mjy":
            raise ValueError("Units must be mJy for Fλ conversion.")

        new_data = {}

        for region, region_dict in self.data.items():
            new_data[region] = {}

            for band, (flux, err) in region_dict.items():

                pivot_key = f"PIVOT_{band}"
                if pivot_key not in self.header:
                    raise KeyError(f"Missing pivot wavelength: {pivot_key}")

                wav = self.header[pivot_key]

                flam = self.mJy_to_flam(flux, wav)
                flam_err = self.mJy_to_flam(err, wav) if err is not None else None

                new_data[region][band] = (flam, flam_err)

        new_header = self.header.copy()
        new_header["UNITS"] = "erg/s/cm2/A"

        return PhotometryTable(new_data, new_header)



    # ==========================================================
    # --- Printing utilities
    # ==========================================================


    def print(self, region=None, nice_filter_names=None):

        if region: # prints single region
            if region not in self.data:
                raise KeyError(f"Region '{region}' not found.")

            region_data = self.data[region]
            df = pd.DataFrame.from_dict(region_data, orient="index",
                                        columns=["Value", "Error"])
            df.index.name = "Band"
            df["SNR"] = df["Value"] / df["Error"]
            return df
        
        # prints all regions

        combined = {}
        snr_map = {}

        for region, region_dict in self.data.items():
            df = pd.DataFrame.from_dict(region_dict, orient="index",
                                        columns=["Value", "Error"])
            df.index.name = "Band"
            df["SNR"] = df["Value"] / df["Error"]

            # rename filters if provided
            if nice_filter_names:
                df.index = [nice_filter_names.get(b, b) for b in df.index]

            formatted = df.apply(
                lambda row: f"{row['Value']:.3f} ± {row['Error']:.3f} ({row['SNR']:.1f})",
                axis=1
            )
            combined[region] = formatted
            snr_map[region] = df["SNR"]

        result = pd.DataFrame(combined)
        result.index.name = "Band"
        snr_df = pd.DataFrame(snr_map, index=result.index)

        def highlight(val, snr):
            return "color: red" if snr <= 5 else ""

        styled = result.style.apply(
            lambda col: [highlight(v, s) for v, s in zip(col, snr_df[col.name])],
            axis=0
        )

        return styled


















    def to_fits(self, filename, tables=None):
        """
        Export this PhotometryTable or a list of PhotometryTables to a FITS file.

        Parameters
        ----------
        filename : str
            Output FITS file name (overwritten if exists).
        tables : PhotometryTable or iterable of PhotometryTable, optional
            If provided, these tables will be written.
            If None, only this instance will be written.
        """

        if tables is None:
            tables = [self]    # save only this table
        elif (not isinstance(tables, Iterable)) or isinstance(tables, (dict, str)):
            tables = [tables]  # ensure list

        hdul = fits.HDUList([fits.PrimaryHDU()])

        for i, pt in enumerate(tables):

            # pt.data should be {band: (value, error)}
            data_dict = pt.data

            t = Table()
            bands  = []
            values = []
            errors = []
            snrs   = []

            for band, (value, error) in data_dict.items():
                bands.append(band)
                values.append(value)
                errors.append(error)
                snrs.append(value / error if error else None)

            t["BAND"]  = bands
            t["VALUE"] = values
            t["ERROR"] = errors
            t["SNR"]   = snrs

            hdu = fits.BinTableHDU(t)

            if hasattr(pt, "header") and pt.header:
                for key, val in pt.header.items():
                    try:
                        hdu.header[key] = val
                    except Exception:
                        pass  # skip invalid FITS keywords

            if hasattr(pt, "region") and pt.region and hasattr(pt.region, "name"):
                extname = f"PHOT_{pt.region.name}"
            else:
                extname = f"PHOT_{i}"

            hdu.name = extname[:68]  # FITS name limit

            hdul.append(hdu)
            
        hdul.writeto(filename, overwrite=True)
        print(f"Written {len(tables)} photometry table(s) to {filename}")


    def from_fits(self, filename):
        """
        Load photometry data from a FITS file.
        
        - If the FITS contains a single table, fills `self` and returns None.
        - If the FITS contains multiple tables, returns a list of PhotometryTable instances.
        """
        hdul = fits.open(filename)
        phot_tables = []

        for hdu in hdul[1:]:  # skip primary HDU
            if not isinstance(hdu, fits.BinTableHDU):
                continue

            tbl = Table(hdu.data)

            if len(tbl) != 1:
                raise ValueError(
                    f"Extension {hdu.name} contains {len(tbl)} rows. "
                    "Each photometry table should contain exactly one row."
                )

            row = tbl[0]

            # Reconstruct dictionary: band → (value, error)
            data = {band: (val, err) for band, val, err in zip(
                row["BAND"], row["VALUE"], row["ERROR"]
            )}

            # Create a new instance
            phot = PhotometryTable(data=data, header=dict(hdu.header))

            # Restore region name if present in extension name
            if hdu.name.startswith("PHOT_"):
                name = hdu.name[5:]
                if name:
                    try:
                        phot.region = type("RegionStub", (), {"name": name})()
                    except Exception:
                        pass

            phot_tables.append(phot)

        hdul.close()

        if len(phot_tables) == 1:
            # Fill self and return None
            self.data = phot_tables[0].data
            self.header = phot_tables[0].header
            if hasattr(phot_tables[0], "region"):
                self.region = phot_tables[0].region
            return None
        else:
            # Return list of new PhotometryTable instances
            return phot_tables




