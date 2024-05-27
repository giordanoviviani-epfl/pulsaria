"""Constant function toolkit."""

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("engine.math_functions.constant")


def constant_function(
    value: float | npt.NDArray[np.float64],
    x: float | npt.NDArray[np.float64],
) -> float | npt.NDArray[np.float64]:
    """Constant function.

    Function that adds an offset to the input value/s.

    Parameters
    ----------
    value : float | npt.NDArray[np.float64]
        Constant value to add to the input value/s.
    x : float | npt.NDArray[np.float64]
        Input value/s to add the offset to.

    Returns
    -------
    float | npt.NDArray[np.float64]
        Result of adding the constant to the input value/s.

    """
    return value + x
