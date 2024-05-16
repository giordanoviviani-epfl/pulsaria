"""Model abstract class.

This module contains the abstract class Model, which is the base class for all
models in the Pulsaria Engine.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Model(Protocol):
    """Abstract class for models in the Pulsaria Engine.

    This class is the base class for all models in the Pulsaria Engine. It
    defines the interface that all models must implement.
    """

    model_identifier: str
    necessary_metadata: list
    computing_function: Callable
    bool_parametrization: bool
    parametrization: str
    coefficients: float | npt.NDArray
    coefficients_errors: float | npt.NDArray
    metadata: dict

    # Abstract methods ----------------------------------------------------------------
    # def __init__(self) -> None:
    #     """Initialize the model.

    #     This method initializes the model. It should be called by the
    #     constructor of any subclass.
    #     """
    #     ...

    def compute(
        self,
        x: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """Compute the model.

        This method computes the model for the given input arguments. It should
        be implemented by all subclasses.
        """
        ...

    def formula(self) -> str:
        """Return the formula of the model.

        This method returns the formula of the model. It should be implemented
        by all subclasses.
        """
        ...
