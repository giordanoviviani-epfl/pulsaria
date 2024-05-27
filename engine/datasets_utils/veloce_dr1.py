"""Module that contains the functions to read the Veloce DR1 dataset."""

import logging
from typing import Any

import pandas as pd

import engine.datasets_utils._read_fits as read_fits
from engine.datasets_utils._filters import filter_by_header, filter_by_queries
from engine.paths_reference import pulsaria_path

logger = logging.getLogger("engine.data_handling.veloce_dr1")


def read_target_rv(
    target: str,
    header_keys: list[str] | None = None,
    columns: list[str] | None = None,
    header_filters: dict | None = None,
    filters: dict | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Read the RV data for a target from the Veloce DR1 dataset.

    Parameters
    ----------
    target : str
        Name of the target to read the data.
    header_keys : list[str] | None, optional
        List of keys to extract from the header. Default is None, and all keys
        are returned.
    columns : list[str] | None, optional
        List of columns to extract from the data. Default is None, and all columns
        are returned.
    header_filters : dict | None, optional
        Dictionary with the header keys, operators and values to filter the data.
        Default is None, and no filter is applied.
    filters : dict | None, optional
        Dictionary of queries used to filter the data.
        Default is None, and no filter is applied.

    Returns
    -------
    header: dict
        Metadata of the target.
    data: pd.DataFrame
        Dataframe containing the data relative to the target.

    """
    path_to_file = pulsaria_path.data / "veloce_dr1" / "FitsFiles"
    file = path_to_file / (target.replace(" ", "_") + ".fits")
    reference = "veloce_dr1"
    all_columns = ["RV", "RV_ERR", "BJD", "SOURCE", "UNIQUE_ID", "SN_60", "MASK"]
    rename_columns = {key: key.lower() for key in all_columns}

    # HEADER
    header = read_fits.get_header(file)
    # Filter target based on header keys
    if not filter_by_header(header, header_filters):
        return {}, pd.DataFrame()

    header_to_return = (
        {key.lower(): header[key] for key in header_keys} if header_keys else header
    )

    # DATA
    data = read_fits.get_table(file, hdu=1, columns=all_columns)
    rename_data = data.rename(columns=rename_columns)
    # Filter the data based on the filters
    filtered_data = filter_by_queries(rename_data, filters)
    filtered_data = filtered_data if filtered_data is not None else pd.DataFrame()
    if not filtered_data.empty:
        filtered_data["target"] = target
        filtered_data["reference"] = reference
        logger.info("Added reference and target columns to the data: %s", reference)
        if columns:
            filtered_data = filtered_data[columns]
            logger.info("Filter columns: %s", columns)

    return header_to_return, filtered_data
