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
        self.id_ = "offset"
        self.required_meta = []
        self.compute_func = constant_function
        self.has_param = False
        self.param_form = "None"
        self.coeffs = np.array(offset)
        self.coeffs_err = np.array(offset_err)
        self.meta = metadata or {}

    @property
    def offset(self) -> float | npt.NDArray[np.float64] | None:
        """Return the offset value."""
        return self.coeffs

    @property
    def offset_error(self) -> float | npt.NDArray[np.float64] | None:
        """Return the offset error."""
        return self.coeffs_err

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
        if self.coeffs is None:
            msg = "constant value is None."
            logger.error(msg)
            raise ValueError(msg)
        return self.compute_func(self.coeffs, x)

    def formula(self) -> str:
        """Constant formula."""
        return "constant"
