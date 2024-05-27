"""Module containing functions to handle fits files.

List of functions and routines to load the data from the data folder and return
it as a pandas dataframe. This functions are extremely basic and each of them is
relative to the fits format.
"""

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy.typing as npt
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from engine.config_utils import check_file_exists

logger = logging.getLogger("engine.data_handling.read_fits")


@runtime_checkable
class BaseHDU(Protocol):
    """Protocol for the HDU class in astropy.io.fits."""

    header: dict
    data: npt.NDArray | fits.FITS_rec | Table | None


def get_header(
    file: str | Path,
    hdu: int = 0,
) -> dict:
    """Get the primary header of a fits file.

    Parameters
    ----------
    file : str or Path
        Path to the fits file.
    hdu : int
        HDU number to read.

    Returns
    -------
    dict
        Dictionary containing the primary header.

    """
    check_file_exists(file)

    try:
        with fits.open(file) as hdul:
            base_hdu: BaseHDU = hdul[hdu]  # type: ignore[attr-defined]
            header = dict(base_hdu.header)
    except Exception as e:
        logger.exception(
            "Error reading primary header from: %s",
            file,
            extra={
                "exception": e,
            },
        )
        raise

    logger.info("Successfully read primary header from: %s", file)
    return header


def get_table(
    file: str | Path,
    hdu: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Get a pandas dataframe from a fits file.

    Parameters
    ----------
    file : str or Path
        Path to the fits file.
    hdu : int
        HDU table number to read.
    columns : list[str], optional
        List of columns to read from the fits file.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing the data from the fits file.

    """
    check_file_exists(file)

    with fits.open(file) as hdul:
        base_hdu: BaseHDU = hdul[hdu]  # type: ignore[attr-defined]

        if isinstance(base_hdu, fits.PrimaryHDU):
            if columns is not None:
                logger.warning("Columns argument is ignored for PrimaryHDU.")
            table = Table(base_hdu.data)
            dataframe = table.to_pandas()
        else:
            if columns is None:
                columns = base_hdu.columns.names  # type: ignore[attr-defined]
            try:
                table = Table(base_hdu.data)
                table.keep_columns(columns)
                dataframe = table.to_pandas()
            except ValueError as e:
                table = Table(base_hdu.data)
                for col in table.columns:
                    if len(table[col].shape) > 1:  # type: ignore[attr-defined]
                        logger.critical("Column %s has shape %d", col, table[col].shape)  # type: ignore[attr-defined]
                logger.exception(
                    "Error reading data from: %s",
                    file,
                    extra={
                        "exception": e,
                    },
                )
                raise

    logger.info("Successfully read dataframe from: %s", file)
    return dataframe
