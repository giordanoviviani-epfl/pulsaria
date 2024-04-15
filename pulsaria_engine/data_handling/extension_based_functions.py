"""Extension-based functions to read data files.

It contains functions and routines to load the data from the data folder and
return it as a pandas dataframe.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger("pulsaria_engine.data_handling")


def _check_file_exists(file: str | Path) -> None:
    """Check if a file exists or raise an exception.

    Parameters
    ----------
    file : str or Path
        Path to the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    """
    file = Path(file)
    if not file.exists():
        message = f"File {file} does not exist."
        logger.error(message)
        raise FileNotFoundError(message)


def fits_get_primary_header(file: str | Path) -> dict:
    """Get the primary header of a fits file.

    Parameters
    ----------
    file : str or Path
        Path to the fits file.

    Returns
    -------
    dict
        Dictionary containing the primary header.

    """
    _check_file_exists(file)

    try:
        with fits.open(file) as hdul:
            header = dict(hdul[0].header)
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


def fits_get_dataframe(
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
    _check_file_exists(file)

    with fits.open(file) as hdul:
        if columns is None:
            columns = hdul[hdu].columns.names
        try:
            dataframe = pd.DataFrame(hdul[hdu].data, columns=columns)
        except ValueError as e:
            table = Table.read(hdul[hdu].data)
            for col in table.columns:
                if len(table[col].shape) > 1:
                    logger.critical("Column %s has shape %d", col, table[col].shape)
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


def yaml_get_data(file: str | Path) -> dict:
    """Read data from a yaml file.

    Parameters
    ----------
    file : str or Path
        Path to the yaml file.

    Returns
    -------
    dict
        Dictionary containing the data from the yaml file.

    """
    _check_file_exists(file)

    with Path(file).open("r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.exception(
                "Error reading data from: %s",
                file,
                extra={
                    "exception": e,
                },
            )
            raise

    logger.info("Successfully read data from: %s", file)
    return data
