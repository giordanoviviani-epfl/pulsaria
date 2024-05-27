"""FitModel class for applying models to data.

This model contains the FitModel class, which generalize the application of models
to data. Allowing to combine different models and fit them to the datapoints.
"""

import logging
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

import engine.config_utils as conf_util
from engine.math_functions import MathFunction

logger = logging.getLogger("pulsaria_engine.model.FitModel")


class FitModel:
    """FitModel class."""

    def __init__(
        self,
        models: MathFunction | list[MathFunction],
        operators: list[str] | None = None,
        common_metadata: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the FitModel object.

        Parameters
        ----------
        models: list[MathFunction]
            List of models to be fitted to the data.
        operators: list[str] | None
            List of operators to be applied between the models.
            Order is relevant.
        common_metadata: dict[str, list[str]] | None
            Common metadata to be used in the models.
            It must be a dictionary with the necessary metadata keys as keys and a list
            of model identifiers as values.

        """
        self.models = [models] if isinstance(models, MathFunction) else models
        _check_models(self.models)

        self.operators_symbols = operators
        self.operators = _get_operators(operators)
        _check_number_operators(self.models, self.operators)

        _set_models_as_attr(self)

        self.common_metadata = common_metadata
        _check_common_metadata(
            self.models,
            self.necessary_metadata,
            self.common_metadata,
        )

    @property
    def numodels(self) -> int:
        """Number of models in use."""
        return len(self.models)

    @property
    def bool_single_model(self) -> bool:
        """Boolean indicating if only one MathFunction is used."""
        match self.numodels:
            case 1:
                return True
            case _:
                return False

    @property
    def necessary_metadata(self) -> dict[str, list[str]]:
        """Necessary metadata for the models."""
        metadata = {}
        for model in self.models:
            metadata[model.identifier] = model.necessary_metadata
        return metadata

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Concatenated coefficients of the models."""
        coeff_list = [model.coefficients for model in self.models]
        return np.concatenate(coeff_list)

    @property
    def coefficients_errors(self) -> npt.NDArray[np.float64]:
        """Concatenated coefficients errors of the models."""
        coeff_err_list = [model.coefficients_errors for model in self.models]
        return np.concatenate(coeff_err_list)

    def compute(
        self,
        x: float | npt.NDArray[np.float64],
        **metadata: dict[str, Any],
    ) -> float | npt.NDArray[np.float64]:
        """Compute the model for the given x values.

        Parameters
        ----------
        x: float | npt.NDArray[np.float64]
            Values to compute the model.
        metadata: dict[str, float | npt.NDArray[np.float64]]
            Metadata to be used in the computation.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Model computed for the given x values.

        Raises
        ------
        MetadataError
            If the metadata is incorrect. In particular, if the keys are not the model
            identifiers or if the keys are not in the necessary metadata of the models.

        """
        metadata = _setup_metadata_for_compute(
            metadata,
            self.common_metadata,
            self.models,
            self.necessary_metadata,
        )

        # Single MathFunction
        if self.bool_single_model:
            if metadata:
                if self.models[0].identifier in metadata:
                    metadata = metadata[self.models[0].identifier]
                logger.debug("Metadata: %s", metadata)
                return self.models[0].compute(x, **metadata)
            return self.models[0].compute(x)

        # Combination of Models

        results_list: list[float | npt.NDArray[np.float64]] = []
        for model in self.models:
            model_id = model.identifier
            if metadata and model_id in metadata:
                model_values = model.compute(x, **metadata[model_id])
            else:
                model_values = model.compute(x)
            results_list.append(model_values)

        # Apply operators
        result: float | npt.NDArray[np.float64] = 0
        for i, model_values in enumerate(results_list):
            if i == 0:
                result = model_values
            else:
                if self.operators is None:
                    msg = "Operators is None but more than one model is used."
                    raise ValueError(msg)
                result = self.operators[i - 1](result, model_values)

        return result

    def formula(self) -> str:
        """Return the formula of the model.

        Returns
        -------
        str
            Formula of the model.

        """
        if self.numodels == 1:
            return self.models[0].formula()

        formula = ""
        if self.operators_symbols is None:
            msg = "Operators is None but more than one model is used."
            raise ValueError(msg)
        for i, model in enumerate(self.models):
            if i == 0:
                formula += model.formula()
            else:
                formula += f" {self.operators_symbols[i - 1]} {model.formula()}"
        return formula


class FitModelFactory:
    """FitModelFactory class.

    This class is used to configure and build the FitModel object.
    It is useful when a configuration file is used to define a model of qwhich
    we need multiple instances.
    """

    def __init__(self) -> None:
        """Initialize the FitModelFactory object."""
        self._bool_configured = False
        self.models = None
        self.operators = None
        self.common_metadata = None
        self._config_file = None

    @property
    def bool_configured(self) -> bool:
        """Boolean indicating if the object is configured."""
        return self._bool_configured

    @bool_configured.setter
    def bool_configured(self, value: bool) -> None:
        if not isinstance(value, bool):
            msg = "bool_configured must be a boolean."
            raise TypeError(msg)
        self._bool_configured = value

    def check_configured(self) -> None:
        """Check if the FitModelFactory object is configured."""
        if not self.bool_configured:
            msg = "FitModelFactory object not configured."
            logger.warning(msg)

    def configure(
        self,
        models: list[MathFunction],
        operators: list[str],
        common_metadata: dict[str, list[str]] | None,
    ) -> None:
        """Configure the FitModelFactory object.

        Parameters
        ----------
        models: list[MathFunction]
            List of models to be fitted to the data.
        operators: list[str]
            List of operators to be applied between the models.
        common_metadata: dict[str, list[str]] | None
            Common metadata to be used in the models.

        """
        # Models
        _check_models(models)
        self.models = models

        # Operators
        _operators = _get_operators(operators)
        _check_number_operators(self.models, _operators)
        self.operators = operators

        # Common metadata
        self.common_metadata = common_metadata
        self.bool_configured = True
        logger.info("FitModelFactory object correctly configured.")

    def configure_from_file(self, config_file: str | Path, method: str) -> None:
        """Configure the FitModelFactory object from a configuration file.

        Parameters
        ----------
        config_file: str | Path
            Path to the configuration file.
        method: str
            Method to be used from the configuration file.

        """
        self._config_file = config_file
        config = conf_util.read_yaml(config_file)

        # Check method is implemented
        if method not in config:
            message = f"Method {method} is not implemented in the configuration file."
            logger.error(message)
            raise NotImplementedError(method)

        building_method = config[method]
        if "FitModel" not in building_method:
            message = "The method does not contain a FitModel key."
            logger.error(message)
            raise KeyError(message)

        fit_model_config = building_method["FitModel"]

        models = []
        for model in fit_model_config["models"]:
            if not isinstance(model, dict) or "model" not in model:
                msg = "Each model must be a dictionary with a 'model' key."
                logger.error(msg)
                raise ValueError(msg)

            model_name = model.pop("model")
            class_model = conf_util.resolve(model_name)
            if not isinstance(class_model, type):
                msg = "MathFunction %s not found in the available models."
                logger.error(msg, model_name)
                raise TypeError(msg % model_name)
            model_instance = class_model(**model)
            models.append(model_instance)

        operators = fit_model_config.get("operators", None)
        common_metadata = fit_model_config.get("common_metadata", None)
        self.configure(models, operators, common_metadata)

    def summarize(self) -> str | None:
        """Summary of the FitModelFactory object.

        Returns
        -------
        summary: str | None
            Summary of the FitModelFactory object.
            If the factory is not configured, it returns None.

        """
        if not self.bool_configured:
            msg = "FitModelFactory object not configured."
            logger.warning(msg)
            return None

        summary = "Summary of the FitModelFactory object:\n"
        if self._config_file is not None:
            summary += f"Configuration file: {self._config_file}\n"
        summary += "todo...."
        return summary

    def build(self) -> FitModel:
        """Build the FitModel object.

        Returns
        -------
        FitModel
            FitModel object.
            If the factory is not configured, it raises a ValueError.

        Raises
        ------
        ValueError
            If FitModelFactory object is not configured.

        """
        if self.models is None:
            if self.bool_configured:
                msg = "FitModelFactory object is configured but models is empty."
            else:
                msg = "FitModelFactory object not configured."
            logger.error(msg)
            raise ValueError(msg)

        return FitModel(self.models, self.operators, self.common_metadata)


# Utility functions -------------------------------------------------------------------
def _check_models(models: list[MathFunction]) -> None:
    """Check if models are subclasses of the MathFunction class.

    Parameters
    ----------
    models: list[MathFunction]
        List of models to be fitted to the data.

    Raises
    ------
    TypeError
        If the models list is not a list of MathFunction instances.

    """
    if not all(isinstance(model, MathFunction) for model in models):
        msg = "Models must be a list of MathFunction instances."
        logger.error(msg)
        raise TypeError(msg)


def _get_operators(
    operators: list[str] | None,
) -> (
    None
    | list[
        Callable[
            [float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]],
            float | npt.NDArray[np.float64],
        ]
    ]
):
    """Match the operators symbols with the corresponding functions.

    Useful to return the list of operators to be applied between the models. If
    no operators are provided, it returns None.

    Parameters
    ----------
    operators: list[str]
        List of operators to be applied between the models.

    Returns
    -------
    list[Callable[[Any, Any], Any]] or None
        List of operators to be applied between the models.

    Raises
    ------
    ValueError
        If the operators list contains invalid operators.
    TypeError
        If the operators list is not a list of strings.

    """
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "%": operator.mod,
        "^": operator.xor,
    }

    if operators is None:
        return None

    if not (
        isinstance(operators, list) and all(isinstance(op, str) for op in operators)
    ):
        msg = "Operators must be a list of strings."
        logger.error(msg)
        raise TypeError(msg)

    if not set(operators).issubset(ops.keys()):
        msg = "Invalid operator in the list: %s"
        logger.error(msg, set(operators) - set(ops.keys()))
        raise ValueError(msg % (set(operators) - set(ops.keys())))

    return [ops[op] for op in operators]


def _check_number_operators(
    models: list[MathFunction],
    operators: list[Callable[[Any, Any], Any]] | None,
) -> None:
    """Check if the number of operators is correct for the given number of models.

    Parameters
    ----------
    models: list[MathFunction]
        List of models to be fitted to the data.
    operators: list[Callable[[Any, Any], Any]]
        List of operators to be applied between the models.

    Raises
    ------
    ValueError
        If the number of operators is not correct for the number of models.

    """
    logger.debug("Checking number of operators...")
    if operators is None:
        if len(models) == 1:
            return
        msg = "Operators is None but more than one model is used."
        logger.error(msg)
        raise ValueError(msg)

    if len(models) == 1:
        msg = "No operators are needed when only one model is used."
        logger.error(msg, extra={"numodels": len(models), "operators": operators})
        raise ValueError(msg)

    if len(models) - 1 != len(operators):
        msg = "The number of operators must be one less than the number of models."
        logger.error(msg)
        raise ValueError(msg)


def _set_models_as_attr(obj: FitModel) -> None:
    """Set the models as attributes of the object.

    It sets the models as attributes of the object, using the class name
    and the model identifier as the attribute names.

    Parameters
    ----------
    obj : FitModel
        FitModel object.

    """
    logger.debug("Setting Models as attributes...")
    for model in obj.models:
        setattr(obj, model.__class__.__name__, model)
        setattr(obj, model.identifier, model)
    logger.info("Models set as attributes.")


def _check_keys_subset_models_ids(
    keys: list[str],
    models: list[MathFunction],
) -> None:
    """Check if the keys are model identifiers.

    Parameters
    ----------
    keys: list[str]
        List of keys to be checked.
    models: list[MathFunction]
        List of models to be fitted to the data.

    Raises
    ------
    ValueError
        If the keys are not the model identifiers.

    """
    if not set(keys).issubset({model.identifier for model in models}):
        msg = "Keys must be the model identifiers."
        logger.error(msg)
        raise ValueError(msg)


def _check_common_metadata(
    models: list[MathFunction],
    necessary_metadata: dict[str, list[str]],
    common_metadata: dict[str, list[str]] | None,
) -> None:
    """Check if the common metadata is correct.

    Check if the common_metadata is relative to only the models considered and
    if the keys are in the necessary_metadata of the specified subset of models.

    Parameters
    ----------
    models : list[MathFunction]
        List of Models.
    necessary_metadata : dict[str, list[str]]
        Necessary metadata for the models.
    common_metadata : dict[str, list[str]] | None
        Common metadata to be checked.

    Raises
    ------
    TypeError
        If the common_metadata is not a dictionary or None.
    TypeError
        If the common_metadata values are not lists.
    ValueError
        If the common_metadata key is not in the necessary_metadata of the model.

    """
    if common_metadata is None:
        return

    if len(models) == 1 and common_metadata is not None:
        raise CommonMetadataError(common_metadata, extra={"numodels": len(models)})

    if not isinstance(common_metadata, dict):
        raise CommonMetadataError(common_metadata)

    for key, model_ids in common_metadata.items():
        extra = {"key": key, "model_ids": model_ids}
        if not isinstance(model_ids, list):
            raise CommonMetadataError(common_metadata, extra=extra)

        try:
            _check_keys_subset_models_ids(model_ids, models)
        except ValueError as exc:
            raise CommonMetadataError(common_metadata, extra=extra) from exc

        for model_id in model_ids:
            if key not in necessary_metadata[model_id]:
                raise CommonMetadataError(common_metadata, extra=extra)


def _setup_metadata_for_compute(
    metadata: dict[str, dict[str, Any]],
    common_metadata: dict[str, list[str]] | None,
    models: list[MathFunction],
    necessary_metadata: dict[str, list[str]],
) -> dict[str, Any]:
    """Set the metadata for the compute method.

    Check if the metadata is correct and broadcast the input to all models that share
    a common key (common_metadata).

    Parameters
    ----------
    metadata : dict[str, dict[str, Any]]
        Input metadata to be checked and broadcasted.
    common_metadata : dict[str, list[str]] | None
        Common metadata shared among models.
    models : list[MathFunction]
        List of models.
    necessary_metadata : dict[str, list[str]]
        Necessary metadata for the models.

    Returns
    -------
    dict[str, Any]
        Updated metadata, in which all models have the same value for the common
        metadata keys.

    Raises
    ------
    MetadataError
        If the input metadata is incorrect. Like, if the keys are not the model
        identifiers or if there are multiple values for a single common metadata key.

    """
    # Check that the provided metadata is correct.
    try:
        _check_keys_subset_models_ids(list(metadata.keys()), models)
    except ValueError as exc:
        raise MetadataError(extra={"metadata_keys": metadata.keys()}) from exc

    for model_id, model_metadata in metadata.items():
        for key in model_metadata:
            if key not in necessary_metadata[model_id]:
                raise MetadataError(
                    extra={
                        "no_necessary_key": key,
                        "model_id": model_id,
                    },
                )

    # If common_metadata None or empty, return the metadata as is.
    if common_metadata is None:
        return metadata

    # If common_metadata is not None, check if a common value can be used.
    return _broadcast_metadata_to_common_metadata(
        metadata,
        common_metadata,
        models,
    )


def _broadcast_metadata_to_common_metadata(
    metadata: dict[str, dict[str, Any]],
    common_metadata: dict[str, list[str]],
    models: list[MathFunction],
) -> dict[str, dict[str, Any]]:
    """Check and broadcast the specified metadata to all models that share it.

    Parameters
    ----------
    metadata : dict[str, dict[str, Any]]
        Input metadata to be checked and broadcasted.
    common_metadata : dict[str, list[str]]
        Common metadata shared among models.
    models : list[MathFunction]
        List of models.

    Returns
    -------
    dict[str, dict[str, Any]]
        Updated metadata, in which all models have the same value for the common
        metadata keys.

    Raises
    ------
    MetadataError
        If the input metadata is incorrect. Like, if the keys are not the model
        identifiers or if there are multiple values for a single common metadata key.

    """
    for key, common_model_ids in common_metadata.items():
        specified_in_metadata = [
            model_id
            for model_id in common_model_ids
            if model_id in metadata and key in metadata[model_id]
        ]
        match len(specified_in_metadata):
            case 1:
                default_value = metadata[specified_in_metadata[0]][key]

            case 0:
                logger.info("No specified value for common metadata key: %s", key)
                logger.info("Search for a default value in the models metadata...")
                values = [
                    getattr(model, key)
                    for model_id in common_model_ids
                    for model in models
                    if model_id == model.identifier and getattr(model, key) is not None
                ]

                empty_list = not values
                same_value = len(set(values)) == 1
                all_close = False
                if (
                    not empty_list
                    and not same_value
                    and len(values) > 1
                    and isinstance(values[0], int | float)
                ):
                    all_close = all(np.isclose(values[0], values[1:]))

                if empty_list or not same_value or not all_close:
                    extra = {
                        "common_metadata": common_metadata,
                        "key": key,
                        "found_values": values,
                        "note": "No default value found.",
                    }
                    raise MetadataError(extra=extra)

                default_value = values[0]

            case _:
                values = [metadata[model_id][key] for model_id in specified_in_metadata]
                same_value = len(set(values)) == 1
                all_close = False
                if not same_value and isinstance(values[0], int | float):
                    all_close = all(np.isclose(values[0], values[1:]))
                if not same_value or not all_close:
                    extra = {
                        "common_metadata": common_metadata,
                        "key": key,
                        "specified_in_metadata": specified_in_metadata,
                        "values": values,
                        "note": "Specified values are not the same.",
                    }
                    raise MetadataError(extra=extra)
                default_value = values[0]

        msg = f"Default value for common metadata key {key}: {default_value}"
        logger.info(msg)
        for model_id in common_model_ids:
            metadata[model_id].update({key: default_value})

    return metadata


# Exceptions --------------------------------------------------------------------------
class MetadataError(Exception):
    """MetadataError class."""

    def __init__(
        self,
        extra: dict | None = None,
    ) -> None:
        """Initialize the MetadataError object.

        Parameters
        ----------
        extra: dict | None
            Dictionary with extra information.
            It will be added to the log message.

        """
        extra = extra or {}
        message = "The input metadata is incorrect."
        logger.error(message, extra=extra)
        for key, value in extra.items():
            message += f"\n{key}: {value}"
        super().__init__(message)


class CommonMetadataError(Exception):
    """CommonMetadataError class.

    Class that inherits from Exception and is raised when the common metadata is
    incorrect.
    """

    def __init__(
        self,
        common_metadata: dict | None,
        extra: dict | None = None,
    ) -> None:
        """Initialize the CommonMetadataError object.

        Parameters
        ----------
        common_metadata: dict | None
            Common metadata.
        extra: dict | None
            Dictionary with extra information.
            It will be added to the log message.

        """
        message = (
            "Common metadata should be None or a dictionary of lists. "
            "\nThe list should contain only keys present in the necessary "
            "metadata of the different models."
            "\nKeys of the dictionary should be model identifiers."
        )
        extra = extra or {}
        extra.update({"common_metadata": common_metadata})
        logger.error(message, extra=extra)
        message += f"\nCommon metadata: {common_metadata}"
        super().__init__(message)
