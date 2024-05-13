"""Model abstract class.

This module contains the abstract class Model, which is the base class for all
models in the Pulsaria Engine.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


class Model(ABC):
    """Abstract class for models in the Pulsaria Engine.

    This class is the base class for all models in the Pulsaria Engine. It
    defines the interface that all models must implement.
    """

    def __init__(self) -> None:
        """Initialize the model.

        This method initializes the model. It should be called by the
        constructor of any subclass.
        """
        self._model_identifier: str = ""
        self._necessary_metadata: list = []
        self._computing_function: Callable = None
        self._bool_parametrization: bool = False
        self._parametrization: str = None
        self._coefficients: npt.ArrayLike = []
        self._coefficients_errors: npt.ArrayLike = []
        self._metadata: dict = {}

    # Abstract attributes -------------------------------------------------------------
    @property
    def model_identifier(self) -> str:
        """The identifier of the model."""
        if self._model_identifier == "":
            self.model_identifier = self.__class__.__name__
        return self._model_identifier

    @model_identifier.setter
    def model_identifier(self, value: str) -> None:
        """Set the identifier of the model."""
        if not isinstance(value, str):
            msg = "model_identifier must be a string."
            raise TypeError(msg)
        self._model_identifier = value

    @property
    def necessary_metadata(self) -> list:
        """The metadata necessary for the model."""
        return self._necessary_metadata

    @necessary_metadata.setter
    def necessary_metadata(self, value: list) -> None:
        """Set the metadata necessary for the model."""
        if not isinstance(value, list):
            msg = "necessary_metadata must be a list."
            raise TypeError(msg)
        self._necessary_metadata = value

    @property
    def computing_function(self) -> Callable[[Any], npt.ArrayLike]:
        """The function that computes the model."""
        return self._computing_function

    @computing_function.setter
    def computing_function(self, value: Callable[[Any], npt.ArrayLike]) -> None:
        """Set the function that computes the model."""
        if not callable(value):
            msg = "computing_function must be a callable function."
            raise TypeError(msg)
        self._computing_function = value

    @property
    def bool_parametrization(self) -> bool:
        """Boolean indicating if the model has differnt possible parametrization."""
        return self._bool_parametrization

    @bool_parametrization.setter
    def bool_parametrization(self, value: bool) -> None:
        """Set the boolean indicating if the model has different parametrizations."""
        if not isinstance(value, bool):
            msg = "bool_parametrization must be a boolean."
            raise TypeError(msg)
        self._bool_parametrization = value

    @property
    def parametrization(self) -> str | None:
        """The parametrization of the model."""
        return self._parametrization

    @parametrization.setter
    def parametrization(self, value: str) -> None:
        """Set the parametrization of the model."""
        if not isinstance(value, str):
            msg = "parametrization must be a string."
            raise TypeError(msg)
        self._parametrization = value

    @property
    def coefficients(self) -> list:
        """Return the coefficients of the Fourier series."""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: npt.ArrayLike) -> None:
        """Set the coefficients of the Fourier series."""
        if not isinstance(value, list | float | int | np.ndarray):
            msg = "The coefficients must be a int, float, list or numpy array."
            raise TypeError(msg)

        if isinstance(value, list):
            value = np.array(value)

        self._coefficients = value

    @property
    def coefficients_errors(self) -> list:
        """The errors of the coefficients of the model."""
        return self._coefficients_errors

    @coefficients_errors.setter
    def coefficients_errors(self, value: npt.ArrayLike) -> None:
        """Set the errors of the coefficients of the model."""
        if not isinstance(value, list | float | int | np.ndarray):
            msg = "The coefficients_errors must be a int, float, list or numpy array."
            raise TypeError(msg)
        if isinstance(value, list):
            value = np.array(value)

        self._coefficients_errors = value

    @property
    def metadata(self) -> dict:
        """The metadata of the model."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict) -> None:
        """Set the metadata of the model."""
        if not isinstance(value, dict):
            msg = "metadata must be a dictionary."
            raise TypeError(msg)
        self._metadata = value

    # Abstract methods ----------------------------------------------------------------
    @abstractmethod
    def compute(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Compute the model.

        This method computes the model for the given input arguments. It should
        be implemented by all subclasses.
        """

    @abstractmethod
    def formula(self) -> str:
        """Return the formula of the model.

        This method returns the formula of the model. It should be implemented
        by all subclasses.
        """
