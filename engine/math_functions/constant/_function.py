"""Constant model function."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from engine.math_functions.constant._toolkit import constant_function

logger = logging.getLogger("engine.math_functions.constant")


class ConstantFunction:
    """Constant function.

    This class represents an Offset model.
    The offset can be a float or an array of floats.

    """

    def __init__(
        self,
        offset: float | npt.NDArray[np.float64] | None = None,
        offset_err: float | npt.NDArray[np.float64] | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        """Initialize ConstantFunction."""
        self.identifier = "offset"
        self.necessary_metadata = []
        self.computing_function = constant_function
        self.bool_parametrization = False
        self.parametrization = "None"
        self.coefficients = offset
        self.coefficients_errors = offset_err
        self.metadata = metadata or {}

    @property
    def offset(self) -> float | npt.NDArray[np.float64] | None:
        """Return the offset value."""
        return self.coefficients

    @property
    def offset_error(self) -> float | npt.NDArray[np.float64] | None:
        """Return the offset error."""
        return self.coefficients_errors

    def compute(
        self,
        x: float | npt.NDArray[np.float64],
    ) -> float | npt.NDArray[np.float64]:
        """Add the constant to the input value/s.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
            Input value/s.

        """
        if self.coefficients is None:
            msg = "constant value is None."
            logger.error(msg)
            raise ValueError(msg)
        return self.computing_function(self.coefficients, x)

    def formula(self) -> str:
        """Constant formula."""
        return "constant"
