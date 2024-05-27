"""Import all modules in the package."""

import logging
import sys
from pathlib import Path

from IPython.core.getipython import get_ipython

sys.path.append(str(Path(__file__).parent))

import measurements
import model
from logging_setup import configure_logging
from math_functions import ConstantFunction, FourierSeries
from measurements import MeasurementsReader
from model import FitModel, FitModelFactory


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        return True  # Jupyter notebook or qtconsole
    if shell == "TerminalInteractiveShell":
        return False  # Terminal running IPython
    return False  # Other type (?)


logger = logging.getLogger("pulsaria_engine")  # __name__ is a common choice
configure_logging()

# Check the environment
if is_notebook():
    logger.info("Running in a Jupyter notebook.")
else:
    logger.info("Running in a Python script or shell.")
__all__ = [
    "logger",
    "measurements",
    "MeasurementsReader",
    "model",
    "FourierSeries",
    "FitModel",
    "FitModelFactory",
    "ConstantFunction",
]
