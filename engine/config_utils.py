"""Module for configuration utilities."""

import logging
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import yaml

logger = logging.getLogger("engine.config_utils")


def resolve(name: str) -> ModuleType | type | Callable | object:
    """Resolve a dotted name to a global object.

    Function is taken by the logging module:
    https://github.com/python/cpython/blob/main/Lib/logging/config.py#L94
    """
    split_name = name.split(".")
    used = split_name.pop(0)
    found = None
    try:
        found = __import__(used)
        for n in split_name:
            used = used + "." + n
            try:
                found = getattr(found, n)
            except AttributeError:
                __import__(used)
                found = getattr(found, n)
    except (AttributeError, ModuleNotFoundError):
        split_name = name.split(".")
        for i, n in enumerate(split_name):
            if i == 0 and (n not in globals()):
                raise
            found = globals()[n] if i == 0 else getattr(found, n)

    return found


def check_file_exists(file: str | Path) -> None:
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


def read_yaml(file: str | Path) -> dict:
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
    check_file_exists(file)

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


# Exception ---------------------------------------------------------------------------
class ConfigError(Exception):
    """Exception raised for errors in the configuration data."""

    def __init__(self, config: dict, extra: dict | None = None) -> None:
        """Exception raised for errors in the configuration data.

        Parameters
        ----------
        config : dict
            Configuration dict used.
        extra : dict | None, optional
            Extra information about the error. Default is None.

        """
        self.extra = extra or {}
        self.extra.update({"config": config})
        logger.error("Configuration error.", extra=self.extra)
        super().__init__(self.extra)
