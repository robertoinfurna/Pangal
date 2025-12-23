from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from .filter import nice_filter_names


@dataclass
class PhotometryTable:
    """
    Photometry for a SINGLE object.

    Invariant:
        data: dict[str, (float, float|None)]
              band -> (value, error)
    """
    data: Dict[str, Tuple[float, Optional[float]]] = field(default_factory=dict)
    header: fits.Header = field(default_factory=fits.Header)

    # ==========================================================
    # --- Validation
    # ==========================================================

    def validate(self):
        for band, val in self.data.items():
            if not isinstance(val, (tuple, list)) or len(val) < 1:
                raise ValueError(f"Band '{band}' must be (value, error)")

            value = float(val[0])
            error = float(val[1]) if len(val) > 1 and val[1] is not None else None

            if error is not None and error <= 0:
                raise ValueError(f"Non-positive error for band '{band}'")

            self.data[band] = (value, error)

    # ==========================================================
    # --- Units
    # ==========================================================

    def check_units(self):
        for key in ("UNITS", "BUNIT", "units", "bunit"):
            if key in self.header:
                return str(self.header[key]).lower()

        raise ValueError("Missing photometry units in FITS header")

    # ==========================================================
    # --- Conversions
    # ==========================================================

    @staticmethod
    def mJy_to_Jy(f_mJy):
        return np.asarray(f_mJy, dtype=float) * 1e-3

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

    # ----------------------------------------------------------

    def to_ABmag(self) -> "PhotometryTable":
        units = self.check_units()
        if units != "mjy":
            raise ValueError("ABmag conversion requires mJy units")

        new_data = {}
        for band, (flux, err) in self.data.items():
            f_Jy = self.mJy_to_Jy(flux)
            df_Jy = self.mJy_to_Jy(err) if err is not None else None

            mag = self.Jy_to_ABmag(f_Jy)
            mag_err = self.ABmag_error(f_Jy, df_Jy)

            new_data[band] = (float(mag), float(mag_err) if mag_err is not None else None)

        new_header = self.header.copy()
        new_header["UNITS"] = "ABMAG"

        return PhotometryTable(
            data=new_data,
            header=new_header,
        )

    # ----------------------------------------------------------

    def to_flam(self) -> "PhotometryTable":
        units = self.check_units()
        if units != "mjy":
            raise ValueError("Fλ conversion requires mJy units")

        new_data = {}
        for band, (flux, err) in self.data.items():
            pivot = self.header.get(f"PIVOT_{band}")
            if pivot is None:
                raise KeyError(f"Missing PIVOT_{band} in header")

            flam = self.mJy_to_flam(flux, pivot)
            flam_err = self.mJy_to_flam(err, pivot) if err is not None else None

            new_data[band] = (float(flam), float(flam_err) if flam_err else None)

        new_header = self.header.copy()
        new_header["UNITS"] = "erg/s/cm2/A"

        return PhotometryTable(
            data=new_data,
            header=new_header,
        )

    # ==========================================================
    # --- Display
    # ==========================================================

    def print(self):
        df = pd.DataFrame.from_dict(
            self.data, orient="index", columns=["Value", "Error"]
        )
        if "Error" in df:
            df["SNR"] = df["Value"] / df["Error"]
        return df

    # ==========================================================
    # --- FITS I/O (single object)
    # ==========================================================

    def to_fits(self, filename: str, overwrite: bool = True):
        self.validate()

        t = Table()
        t["BAND"] = list(self.data.keys())
        t["VALUE"] = [v for v, _ in self.data.values()]
        t["ERROR"] = [e for _, e in self.data.values()]

        hdu = fits.BinTableHDU(t, name="PHOTOMETRY")

        for k, v in self.header.items():
            try:
                hdu.header[k] = v
            except Exception:
                pass

        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            filename, overwrite=overwrite
        )

    @classmethod
    def from_fits(cls, filename: str) -> "PhotometryTable":
        """
        Load photometry for a SINGLE object from a FITS file.

        The FITS file must contain exactly one BinTableHDU with columns:
            BAND, VALUE, ERROR
        """
        hdul = fits.open(filename)

        # Find first BinTableHDU
        hdu = None
        for ext in hdul[1:]:
            if isinstance(ext, fits.BinTableHDU):
                hdu = ext
                break

        if hdu is None:
            hdul.close()
            raise ValueError("No BinTableHDU found in FITS file")

        tbl = Table(hdu.data)

        # Reconstruct data dict
        data = {}
        for band, val, err in zip(tbl["BAND"], tbl["VALUE"], tbl["ERROR"]):
            band = str(band)
            value = float(val)
            error = float(err) if err is not None else None
            data[band] = (value, error)

        # Rebuild header
        header = fits.Header(hdu.header)

        hdul.close()

        phot = cls(
            data=data,
            header=header,
        )

        phot.validate()
        return phot





@dataclass
class PhotometryCatalog:
    """
    Collection of PhotometryTable objects.

    Invariant:
        tables: dict[object_id -> PhotometryTable]
    """
    tables: Dict[str, "PhotometryTable"] = field(default_factory=dict)

    def __init__(self, tables: List["PhotometryTable"] = None):
        self.tables = {}
        keys = ["OBJECT", "object id", "region id", "Region ID", "ID", "id"]
        
        if tables:
            for table in tables:
                # Try to find a valid object_id from the header
                object_id = None
                for key in keys:
                    if key in table.header:
                        object_id = str(table.header[key])
                        break

                if object_id is None:
                    raise ValueError("Each PhotometryTable must have an object_id in the header")

                self.tables[object_id] = table
                

    # ==========================================================
    # --- Container behavior
    # ==========================================================

    def __len__(self):
        return len(self.tables)

    def __iter__(self):
        return iter(self.tables.values())

    def __getitem__(self, key):
        return self.tables[key]

    # ==========================================================
    # --- FITS I/O (many objects)
    # ==========================================================

    @classmethod
    def from_fits(cls, filename: str) -> "PhotometryCatalog":
        hdul = fits.open(filename)
        tables = []

        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU):
                continue

            tbl = Table(hdu.data)
            data = {
                band: (float(val), float(err) if err is not None else None)
                for band, val, err in zip(tbl["BAND"], tbl["VALUE"], tbl["ERROR"])
            }

            header = fits.Header(hdu.header)
            object_id = header.get("OBJECT", hdu.name)
            redshift = header.get("REDSHIFT")

            from_phot_table = __import__('__main__').PhotometryTable  # avoid circular import
            table = from_phot_table(
                data=data,
                header=header,
                object_id=object_id,
                redshift=redshift,
            )
            tables.append(table)

        hdul.close()
        return cls(tables)

    def to_fits(self, filename: str, overwrite: bool = True):
        hdul = fits.HDUList([fits.PrimaryHDU()])

        for obj_id, table in self.tables.items():
            t = Table()
            t["BAND"] = list(table.data.keys())
            t["VALUE"] = [v for v, _ in table.data.values()]
            t["ERROR"] = [e for _, e in table.data.values()]

            hdu = fits.BinTableHDU(t, name=f"PHOT_{obj_id}"[:68])

            for k, v in table.header.items():
                try:
                    hdu.header[k] = v
                except Exception:
                    pass

            hdu.header["OBJECT"] = obj_id
            if table.redshift is not None:
                hdu.header["REDSHIFT"] = table.redshift

            hdul.append(hdu)

        hdul.writeto(filename, overwrite=overwrite)

    # ==========================================================
    # --- Pretty-print
    # ==========================================================

    def print(self, nice_filter_names=None, snr_threshold=5):
        """
        Pretty print photometry for all objects in the catalog.
        Returns a pandas Styler:
            - Columns = object IDs
            - Rows = bands
            - Cells = "value ± error (SNR)"
            - Low-SNR highlighted in red
        """
        combined = {}
        snr_map = {}
        units = None

        for obj_id, table in self.tables.items():
            if units is None:
                units = table.check_units()

            df = pd.DataFrame.from_dict(
                table.data, orient="index", columns=["Value", "Error"]
            )
            df["SNR"] = df["Value"] / df["Error"]

            if nice_filter_names:
                df.index = [nice_filter_names.get(b, b) for b in df.index]

            formatted = df.apply(
                lambda row: f"{row['Value']:.3f} ± {row['Error']:.3f} ({row['SNR']:.1f})",
                axis=1
            )

            combined[obj_id] = formatted
            snr_map[obj_id] = df["SNR"]

        result = pd.DataFrame(combined)
        snr_df = pd.DataFrame(snr_map)

        # Highlight low-SNR cells
        def highlight(col):
            return [
                "color: red" if snr <= snr_threshold else ""
                for snr in snr_df[col.name]
            ]

        styled = result.style.apply(highlight, axis=0)
        styled.set_caption(f"Units: {units}")

        return styled




