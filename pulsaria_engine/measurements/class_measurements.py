"""Module that define the Measurement class and its subclasses."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Measurements:
    """Class to store the measurements of a dataset."""

    metadata: dict
    data: pd.DataFrame
    multi_target: bool = False
