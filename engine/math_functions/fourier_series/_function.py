"""Fourier Series.

In this module, the FourierSeries class is defined. This class is a subclass
(follows protocol) of MathFunction and it represents a Fourier series model.
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from engine.math_functions.fourier_series._toolkit import (
    FOURIER_SERIES_PARAMETRIZATIONS,
    change_parametrization_fs_coeff,
    change_parametrization_fs_cov_matrix,
    formula_fourier_series,
    fourier_series,
)

logger = logging.getLogger("engine.math_functions.fourier_series")


class FourierSeries:
    """Fourier series model.

    This class represents a Fourier series model. It is a subclass of the Model
    class and implements the interface defined by that class.
    """

    def __init__(
        self,
        parametrization: str,
        coeff: npt.NDArray[np.float64] | None = None,
        coeff_err: npt.NDArray[np.float64] | None = None,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        """Initialize the Fourier series model."""
        self.id_ = "FS"
        self.required_meta = ["P", "E"]
        self.compute_func = fourier_series
        self.has_param = True
        self.param_form = parametrization
        self.coeffs = coeff or np.array([], dtype=np.float64)
        self.coeffs_err = coeff_err or np.array([], dtype=np.float64)
        self.meta = metadata if metadata is not None else {}

        self._memory: dict | None = None
        self.param_variants = FOURIER_SERIES_PARAMETRIZATIONS

    # Properties ----------------------------------------------------------------------
    @property
    def n_harmonics(self) -> int:
        """Return the number of harmonics in the Fourier series."""
        return len(np.asarray(self.coeffs)) // 2

    @property
    def P(self) -> float | None:  # noqa: N802
        """Return the pulsation period."""
        return self.meta.get("P", None)

    @P.setter
    def P(self, value: float | None) -> None:  # noqa: N802
        """Set the pulsation period."""
        self.meta["P"] = value

    @property
    def E(self) -> float | None:  # noqa: N802
        """Return the epoch."""
        return self.meta.get("E", None)

    @E.setter
    def E(self, value: float | None) -> None:  # noqa: N802
        """Set the epoch."""
        self.meta["E"] = value

    # Required methods ----------------------------------------------------------------
    def compute(
        self,
        x: float | npt.NDArray[np.float64],
        n_harmonics: int | None = None,
        **kwargs: dict[str, Any],
    ) -> float | npt.NDArray[np.float64]:
        """Compute the Fourier series model for the given x values.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
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
        float | npt.NDArray[np.float64]
            The values of the Fourier series model.

        """
        input_type = kwargs.get("input_type", None)
        if input_type is None:
            logger.info("The input_type is undefined. Using 'bjd' as default.")
            input_type = "bjd"
        match input_type:
            case "bjd":
                period = kwargs.get("P", self.P)
                epoch = kwargs.get("E", self.E)
                period_type_check = isinstance(period, (float | None))
                epoch_type_check = isinstance(epoch, (float | None))
                if not period_type_check or not epoch_type_check:
                    msg = "Both P (%s) and E (%s) must be either a number or None."
                    logger.error(msg, period, epoch)
                    raise TypeError(msg % (period, epoch))
                return self.compute_at_bjd(x, period, epoch, n_harmonics)
            case "phase":
                return self.compute_at_phase(x, n_harmonics)
            case _:
                msg = f"Input type not recognized: {input_type}\n"
                msg += "The input_type must be 'bjd' or 'phase'."
                logger.error(msg)
                raise ValueError(msg)

    def compute_at_phase(
        self,
        x: float | npt.NDArray[np.float64],
        n_harmonics: int | None,
    ) -> float | npt.NDArray[np.float64]:
        """Compute the Fourier series model for the given phase values.

        Parameters
        ----------
        x : float | npt.NDArray[np.number]
            Phase values.
        n_harmonics : int | None
            Number of harmonics to consider.

        Returns
        -------
        float | npt.NDArray[np.float64]
            The values of the Fourier series model.

        """
        if self.coeffs is None:
            msg = "The coefficients of the Fourier series model are not defined."
            logger.error(msg)
            raise ValueError(msg)
        return self.compute_func(
            self.coeffs,
            x,
            self.param_form,
            n_harmonics,
        )

    def compute_at_bjd(
        self,
        x: float | npt.NDArray[np.float64],
        P: float | None,  # noqa: N803
        E: float | None,  # noqa: N803
        n_harmonics: int | None,
    ) -> float | npt.NDArray[np.float64]:
        """Compute the Fourier series model for the given BJD values.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
            BJD values.
        P : float
            Pulsation period.
        E : float
            Epoch.
        n_harmonics : int | None
            Number of harmonics to consider.

        Returns
        -------
        float | npt.NDArray[np.float64]
            The values of the Fourier series model.

        """
        phase = self.calculate_phase(x, P, E)
        return self.compute_at_phase(phase, n_harmonics)

    def formula(self) -> str:
        """Return the formula of the Fourier series model.

        Returns
        -------
        str
            The formula of the Fourier series model.

        """
        n_harmonics = self.n_harmonics
        if n_harmonics == 0:
            n_harmonics = "N"
        return formula_fourier_series(self.param_form, n_harmonics)

    # Specific methods ----------------------------------------------------------------
    def calculate_phase(
        self,
        x: float | npt.NDArray[np.float64],
        P: float | None = None,  # noqa: N803
        E: float | None = None,  # noqa: N803
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the phase for the given x values."""
        E = E or self.E  # noqa: N806
        P = P or self.P  # noqa: N806

        if E is None or P is None:
            msg = "Both E (%s) and P (%s) must be provided."
            logger.error(msg, E, P)
            raise ValueError(msg % (E, P))

        return np.remainder(np.subtract(x, E) / P, 1.0)

    def change_parametrization(self, new_parametrization: str) -> None:
        """Change the parametrization of the coefficients.

        Parameters
        ----------
        new_parametrization : str
            Parametrization of the new coefficients.

        """
        cov_matrix = None
        if "cov_matrix" in self.meta and self.param_form in self.meta["cov_matrix"]:
            cov_matrix = self.meta["cov_matrix"][self.param_form]
            try:
                new_cov_matrix = change_parametrization_fs_cov_matrix(
                    cov_matrix,
                    self.param_form,
                    new_parametrization,
                )
                self.meta["cov_matrix"][new_parametrization] = new_cov_matrix
            except NotImplementedError:
                msg = "Covariance matrix transformation %s -> %s is not implemented."
                logger.info(msg, self.param_form, new_parametrization)
                new_cov_matrix = None

        new_param_values = change_parametrization_fs_coeff(
            self.coeffs,
            self.coeffs_err,
            self.param_form,
            new_parametrization,
            cov_matrix,
        )
        self.coeffs = new_param_values[0]
        self.coeffs_err = new_param_values[1]
        self.param_form = new_parametrization
