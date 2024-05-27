"""Datasets specific utilities.

Use to read and conform the data from the datasets so that they can be used by
the engine.


_read_*:
    Modules contiaing functions and routines to load the data from the data
    folder and return it as a pandas dataframe. This functions are extremely
    basic and each of them is relative to a specific file format (fits, yaml,
    etc).

_filters:
    Module containing functions to filter the data based on their content and
    structure.

*:
    Modules contaning functions to read the data from the datasets and return
    it as a pandas dataframe. These functions are specific to the dataset.
    The module name shoudl follow the pattern: <dataset_name>.py

"""

from engine.datasets_utils import veloce_dr1

__all__ = [
    "veloce_dr1",
]
