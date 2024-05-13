"""Fourier Series Model.

In this module, the FourierSeries class is defined. This class is a subclass
of the Model class and represents a Fourier series model.
"""

import logging
from typing import Any

import numpy.typing as npt

from pulsaria_engine.model.abstract_class_model import Model
from pulsaria_engine.model.fourier_series_toolkit import (
    FOURIER_SERIES_PARAMETRIZATIONS,
    change_parametrization_fs_coeff,
    change_parametrization_fs_cov_matrix,
    formula_fourier_series,
    fourier_series,
)

logger = logging.getLogger("pulsaria_engine.model.FourierSeries")


class FourierSeries(Model):
    """Fourier series model.

    This class represents a Fourier series model. It is a subclass of the Model
    class and implements the interface defined by that class.
    """

    def __init__(
        self,
        parametrization: str,
        coeff: npt.ArrayLike | None = None,
        coeff_err: npt.ArrayLike | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        """Initialize the Fourier series model."""
        super().__init__()
        self.model_identifier = "FS"
        self.necessary_metadata = ["P", "E"]
        self.computing_function = fourier_series
        self.bool_parametrization = True
        self.parametrization = parametrization
        self.coefficients = coeff if coeff is not None else []
        self.coefficients_errors = coeff_err if coeff_err is not None else []
        self.metadata = metadata if metadata is not None else {}

        self._memory: dict | None = None
        self.possible_parametrizations = FOURIER_SERIES_PARAMETRIZATIONS
        self.P = self.metadata.get("P", None)
        self.E = self.metadata.get("E", None)

    # Properties ----------------------------------------------------------------------
    @property
    def n_harmonics(self) -> int:
        """Return the number of harmonics in the Fourier series."""
        return len(self.coefficients) // 2

    # Required methods ----------------------------------------------------------------
    def compute(
        self,
        x: npt.ArrayLike,
        n_harmonics: int | None = None,
        **kwargs: dict[str, Any],
    ) -> npt.ArrayLike:
        """Compute the Fourier series model for the given x values.

        Parameters
        ----------
        x : npt.ArrayLike
            Input values to compute the model.
        n_harmonics : int | None
            Number of harmonics to consider.
        kwargs : dict[str, Any]
            Additional keyword arguments.
            At the moment, the only additional keywords are:
            - input_type : str
                Type of input to compute the model. It can be 'bjd' or 'phase'.
            - P : float
                Pulsation period. Used only if input_type is 'bjd'.
            - E : float
                Epoch. Used only if input_type is 'bjd'.

        Returns
        -------
        npt.ArrayLike
            The values of the Fourier series model.

        """
        input_type = kwargs.get("input_type", None)
        if input_type is None:
            logger.info("The input_type is undefined. Using 'bjd' as default.")
            input_type = "bjd"
        match input_type:
            case "bjd":
                P = kwargs.get("P", self.P)  # noqa: N806
                E = kwargs.get("E", self.E)  # noqa: N806
                return self.compute_at_bjd(x, P, E, n_harmonics)
            case "phase":
                return self.compute_at_phase(x, n_harmonics)
            case _:
                msg = f"Input type not recognized: {input_type}\n"
                msg += "The input_type must be 'bjd' or 'phase'."
                logger.error(msg)
                raise ValueError(msg)

    def compute_at_phase(
        self,
        x: npt.ArrayLike,
        n_harmonics: int | None,
    ) -> npt.ArrayLike:
        """Compute the Fourier series model for the given phase values.

        Parameters
        ----------
        x : npt.ArrayLike
            Phase values.
        n_harmonics : int | None
            Number of harmonics to consider.

        Returns
        -------
        npt.ArrayLike
            The values of the Fourier series model.

        """
        return self.computing_function(
            self.coefficients,
            x,
            self.parametrization,
            n_harmonics,
        )

    def compute_at_bjd(
        self,
        x: npt.ArrayLike,
        P: float,  # noqa: N803
        E: float,  # noqa: N803
        n_harmonics: int | None,
    ) -> npt.ArrayLike:
        """Compute the Fourier series model for the given BJD values.

        Parameters
        ----------
        x : npt.ArrayLike
            BJD values.
        P : float
            Pulsation period.
        E : float
            Epoch.
        n_harmonics : int | None
            Number of harmonics to consider.

        Returns
        -------
        npt.ArrayLike
            The values of the Fourier series model.

        """
        phase = self.calculate_phase(x, P, E)
        return self.compute_at_phase(phase, n_harmonics)

    def formula(self, n_harmonics: int | str | None = None) -> str:
        """Return the formula of the Fourier series model.

        Parameters
        ----------
        n_harmonics : int | str | None
            Number of harmonics to consider. If None, the formula will use the
            number of harmonics of the model.

        Returns
        -------
        str
            The formula of the Fourier series model.

        """
        if n_harmonics is None:
            n_harmonics = self.n_harmonics
            if n_harmonics == 0:
                n_harmonics = "N"
        return formula_fourier_series(self.parametrization, n_harmonics)

    # Specific methods ----------------------------------------------------------------
    def calculate_phase(
        self,
        x: npt.ArrayLike,
        P: float | None = None,  # noqa: N803
        E: float | None = None,  # noqa: N803
    ) -> npt.ArrayLike:
        """Calculate the phase for the given x values."""
        E = E if E is not None else self.E  # noqa: N806
        P = P if P is not None else self.P  # noqa: N806

        if E is None or P is None:
            msg = "Both E (%s) and P (%s) must be provided."
            logger.error(msg, E, P)
            raise ValueError(msg % (E, P))

        return (x - E) / P % 1.0

    def change_parametrization(self, new_parametrization: str) -> None:
        """Change the parametrization of the coefficients.

        Parameters
        ----------
        new_parametrization : str
            Parametrization of the new coefficients.

        """
        cov_matrix = None
        if (
            "cov_matrix" in self.metadata
            and self.parametrization in self.metadata["cov_matrix"]
        ):
            cov_matrix = self.metadata["cov_matrix"][self.parametrization]
            try:
                new_cov_matrix = change_parametrization_fs_cov_matrix(
                    cov_matrix,
                    self.parametrization,
                    new_parametrization,
                )
                self.metadata["cov_matrix"][new_parametrization] = new_cov_matrix
            except NotImplementedError:
                msg = "Covariance matrix transformation %s -> %s is not implemented."
                logger.info(msg, self.parametrization, new_parametrization)
                new_cov_matrix = None

        new_param_values = change_parametrization_fs_coeff(
            self.coefficients,
            self.coefficients_errors,
            self.parametrization,
            new_parametrization,
            cov_matrix,
        )
        self.coefficients = new_param_values[0]
        self.coefficients_errors = new_param_values[1]
        self.parametrization = new_parametrization