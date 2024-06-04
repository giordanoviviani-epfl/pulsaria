"""Protocol for mathematical function.

This module contains the protocol MathFunction, which is the base class for all
mathematical expression of models in pulsaria's engine.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MathFunction(Protocol):
    """Abstract class for mathematical functions.

    This class is the base class for mathematical functions. It
    defines the interface that all math function must implement.
    """

    id_: str
    required_meta: list
    compute_func: Callable
    has_param: bool
    param_form: str
    coeffs: npt.NDArray
    coeffs_err: npt.NDArray
    meta: dict

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
        """Compute the function for a given input.

        This method computes the function for the given input arguments. It should
        be implemented by all subclasses.
        """
        ...

    def formula(self) -> str:
        """Return the expression of the mathematical function.

        This method returns the mathematical expression. It should be implemented
        by all subclasses.
        """
        ...
