"""Module that allows to read data given a dataset configuration file."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import engine.config_utils as conf_util
import engine.measurements.read_filter_data_funcs as rfdf
from engine.measurements.class_measurements import Measurements

logger = logging.getLogger("pulsaria_engine.data_handling")


class MeasurementsReader:
    """Class use to read the data and return a Measurements object."""

    def __init__(
        self,
        config_file: str | Path,
        method: str,
    ) -> None:
        """Initialize the DataReader object.

        Parameters
        ----------
        config_file : str | Path
            Path to the configuration file.
        method : str
            Name of the method within the configuration file to read the data.

        Raises
        ------
        NotImplementedError
            Raised if the method is not implemented in the configuration file.
        KeyError
            Raised if the method does not contain the Measurements key.

        """
        # Read configuration file
        self.config_file = config_file
        self.config = rfdf.yaml_get_data(config_file)

        # Check method is implemented
        if method not in self.config:
            message = f"Method {method} is not implemented in the configuration file."
            logger.error(message)
            raise NotImplementedError(method)

        self.reading_method: dict = self.config[method]

        # Check if the method contains a Measurements key
        if "Measurements" not in self.reading_method:
            message = "The method does not contain a Measurements key."
            logger.error(message)
            raise KeyError(message)

        self.measurements_config: list = self.reading_method["Measurements"]
        logger.info("MeasurementsReader object created.")

    def get_data(self, target: str | list) -> Measurements:
        """Read the measurements of a given a target or a list of targets.

        Parameters
        ----------
        target : str | list
            Name of the target or a list of targets of which data will be retrieved.

        Returns
        -------
        Measurements
            Object containing the measurements of the target or targets.

        """
        multiple_targets = False
        header_all: dict[str, Any] = {}
        data_all: pd.DataFrame = pd.DataFrame()
        for i, measurements_elem in enumerate(self.measurements_config):
            dataset, dataset_config = next(iter(measurements_elem.items()))
            logger.info("Reading measurements from dataset: %s", dataset)
            reading_function = conf_util.resolve(dataset_config.get("reading_function"))
            if not callable(reading_function):
                message = "The reading function is not callable."
                logger.error(message)
                raise TypeError(message)

            reading_config = {
                k: v for k, v in dataset_config.items() if k != "reading_function"
            }
            header, data, multiple_targets = read_targets_measurements_general(
                target,
                reading_function,
                **reading_config,
            )

            if i == 0:
                header_all, data_all = header, data
            elif multiple_targets:
                for target_name, value_dict in header.items():
                    header_all.setdefault(target_name, {}).update(value_dict)
                data_all = pd.concat([data_all, data], ignore_index=True)
            else:
                header_all.update(header)
                data_all = pd.concat([data_all, data], ignore_index=True)

        logger.warning("Dataframe is empty") if data_all.empty else None
        logger.warning("Header is empty") if not header_all else None
        return Measurements(
            metadata=header_all,
            data=data_all,
            multi_target=multiple_targets,
        )


# General/Multiple targets functions ---------------------------------------------------
def read_targets_measurements_general(
    target: str | list | set | np.ndarray,
    read_target_func: Callable[[Any], tuple[dict[str, Any], pd.DataFrame]],
    **kwargs: dict[str, Any],
) -> tuple[dict[str, Any] | dict[str, dict[Any, Any]], pd.DataFrame, bool]:
    """Read the measurements of a target or a list of targets.

    This function generalizes the inputted :function:`read_target_func` so that
    it can read a single target or multiple targets.

    Parameters
    ----------
    target : str | list
        Name of the target or a list of targets of which data will be retrieved.
    read_target_func : Callable[[Any], tuple[dict, pd.DataFrame]]
        Function that reads the data for a single target. This function must
        return a header (dict) and data in tabular format (pd.DataFrame).
    kwargs : dict
        Additional keyword arguments to pass to the :function:`read_target_func`.

    Returns
    -------
    header: dict
        Metadata of the target or targets.
    data: pd.DataFrame
        Dataframe containing the measurements of the target or targets.
    multiple_targets: bool
        Boolean indicating if the target is a single target or multiple targets.
        True if multiple targets, False otherwise.

    """
    multiple_targets = False
    if isinstance(target, str):
        header, data = read_target_func(target, **kwargs)
        return header, data, multiple_targets

    if not isinstance(target, list | set | np.ndarray):
        msg = (
            f"Invalid type for target (valid: str|list|set|np.ndarray): {type(target)}."
        )
        logger.error(msg)
        raise TypeError(msg)

    multiple_targets = True
    multiple_header, multiple_data = {}, []
    # Loop over the targets
    for single_target in target:
        header_target, data_target = read_target_func(single_target, **kwargs)
        if header_target:
            multiple_header[single_target] = header_target
            multiple_data.append(data_target)

    multiple_targets_data = pd.concat(multiple_data, ignore_index=True, sort=False)
    try:
        multiple_targets_data = multiple_targets_data.sort_values(
            by="target",
            ignore_index=True,
        ).reset_index(drop=True)
    except KeyError:
        msg = "`target` key not found in the dataframe. No sorting applied."
        logger.info(msg)

    return multiple_header, multiple_targets_data, multiple_targets
