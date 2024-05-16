"""Extension-based functions to read data files and to filter the results.

Read files:
List of functions and routines to load the data from the data folder and return
it as a pandas dataframe. This functions are extremely basic and each of them is
relative to a specific file format (fits, yaml, etc).

Filters:
List of functions to filter the data based on their structure (dict: header,
columns:dataframe, etc...).
"""

import logging
import operator
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy.typing as npt
import pandas as pd
import yaml
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger("pulsaria_engine.data_handling")


# Read files --------------------------------------------------------------------------
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


def fits_get_header(
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
    _check_file_exists(file)

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


# Filters -----------------------------------------------------------------------------
def filter_from_header(header: dict | None, filters: dict | None) -> bool:
    """Filter the data based on the header keys.

    Parameters
    ----------
    header : dict
        Dictionary containing the header keys and values.
    filters : dict
        Dictionary containing the filters to apply to the header.

    Returns
    -------
    bool
        True if the header passes the filters, False otherwise.

    Raises
    ------
    KeyError
        If a key in the filters is not found in the header.
    TypeError
        If one of the filters is not a dictionary.
    KeyError
        If one of the filters does not contain the keys "operator" and "value".

    """
    if filters is None or header is None:
        logger.info("Either filters or header is None. Returning True.")
        return True

    for key, header_filter in filters.items():
        if key not in header:
            logger.error("Key %s not found in header.", key)
            raise KeyError(key)

        if not isinstance(header_filter, dict):
            logger.error("Filters must be a dictionary.")
            raise TypeError(header_filter)

        if not {"operator", "value"}.issubset(set(header_filter)):
            logger.error("Filter must contain 'operator' and 'value' keys.")
            raise KeyError(header_filter)

        if isinstance(header_filter["operator"], str):
            header_filter["operator"] = [header_filter["operator"]]
            header_filter["value"] = [header_filter["value"]]

        for operator_str, value in zip(
            header_filter["operator"],
            header_filter["value"],
            strict=True,
        ):
            filter_operator = getattr(operator, operator_str)
            if not filter_operator(header[key], value):
                logger.info("Header key %s did not pass the filter.", key)
                return False

    logger.info("Header passed all filters.")
    return True


def filter_from_queries(
    data: pd.DataFrame | None,
    filters: dict | None,
) -> pd.DataFrame | None:
    """Filter the dataframe using queries based on the columns.

    Parameters
    ----------
    data : pd.DataFrame | None
        Dataframe to filter.
    filters : dict | None
        Dictionary containing the filters to apply to the dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with the filtered data. An empty dataframe is returned if no
        data passed the filters.

    Raises
    ------
    TypeError
        If a filter query is not a string.

    """
    if filters is None or data is None:
        logger.info("Either filters or header is None. Returning the dataframe.")
        return data

    filtered_data = data.copy()
    for key, query in filters.items():
        if not isinstance(query, str):
            logger.error("Filter query must be a string.")
            raise TypeError(query)

        logger.info("Filtering data based on query: %s", key)
        filtered_data = filtered_data.query(query)

    if filtered_data.empty:
        logger.info("No data passed the filters. Dataframe is empty")
    else:
        logger.info(
            "Dataframe was filtered successfully. Lines filtered: %d",
            len(data) - len(filtered_data),
        )
    return filtered_data


# Protocols ----------------------------------------------------------------------------
@runtime_checkable
class BaseHDU(Protocol):
    """Protocol for the HDU class in astropy.io.fits."""

    header: dict
    data: npt.NDArray | fits.FITS_rec | Table | None
