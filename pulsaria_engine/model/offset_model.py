"""Offset Model.

In this module, the Offset model is defined. This class is a subclass of the
Model class and represents a float constant that moves the model on the y-axis.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("pulsaria_engine.model.Offset")


def offset_function(
    offset: float | npt.NDArray[np.float64],
    x: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Offset function.

    Function that adds an offset to the input value/s.

    Parameters
    ----------
    offset : float | npt.NDArray[np.float64]
        Offset value to add to the input value/s.
    x : float | npt.NDArray[np.float64]
        Input value/s to add the offset to.

    Returns
    -------
    float | npt.NDArray[np.float64]
        Result of adding the offset to the input value/s.

    """
    return offset + x


class Offset:
    """Offset model.

    This class represents an Offset model.
    The offset can be a float or an array of floats.

    """

    def __init__(
        self,
        offset: float | npt.NDArray[np.float64] | None = None,
        offset_err: float | npt.NDArray[np.float64] | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        """Initialize the Offset model."""
        self.model_identifier = "offset"
        self.necessary_metadata = []
        self.computing_function = offset_function
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
        """Compute the model.

        This method computes the model for the given input arguments. It should
        be implemented by all subclasses.

        Parameters
        ----------
        x : float
            Input value to compute the model.

        """
        if self.coefficients is None:
            msg = "Offset value is None."
            logger.error(msg)
            raise ValueError(msg)
        return self.computing_function(self.coefficients, x)

    def formula(self) -> str:
        """Return the formula of the model.

        This method returns the formula of the model. It should be implemented
        by all subclasses.

        """
        return "offset"
