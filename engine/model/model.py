"""Model class for applying mathematical functions to data.

This model contains the Model class, which generalize the application of
mathematical functions to data. Allowing to combine different models and fit
them to the datapoints.
"""

import logging
from string import ascii_lowercase

import numexpr as ne
import numpy as np
import numpy.typing as npt

from engine.math_functions import MathFunction

logger = logging.getLogger("engine.model")


class Model:
    """Model class for applying mathematical functions to data."""

    def __init__(
        self,
        math_funcs: MathFunction | list[MathFunction],
        math_expression: str | None = None,
        shared_meta: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize the Model class.

        Parameters
        ----------
        math_funcs : MathFunction | list[MathFunction]
            Mathematical functions to apply to the data. It can be a single
            function or a list of functions.
        math_expression : str | None, optional
            Mathematical expression that represents the model. Default is None.
            The expression will be evaluated by the numexpr library to compute
            the final result.
        shared_meta : dict[str, list[str]] | None, optional
            Dictionary with the shared metadata between the functions. Default
            is None, and no shared metadata is used.

        """
        self.submodels = check_and_format_submodels(math_funcs)
        self.shared_meta = check_and_format_shared_meta(shared_meta, self.submodels)

        if math_expression is None:
            math_expression = " + ".join(ascii_lowercase[: self.nusubmodels])
        self.math_expression = math_expression

    @property
    def nusubmodels(self) -> int:
        """Return the number of sub-models in the Model.

        Returns
        -------
        int
            Number of sub-models.

        """
        return len(self.submodels)

    @property
    def is_single_model(self) -> bool:
        """Boolean indicating if the model has only one sub-model."""
        return self.nusubmodels == 1

    @property
    def required_meta(self) -> dict[str, list[str]]:
        """Return the required metadata for the Model."""
        return {model.id_: model.required_meta for model in self.submodels}

    @property
    def coeffs(self) -> npt.NDArray[np.float64]:
        """Concatenated coefficients of the sub-models."""
        return np.concatenate([model.coeffs for model in self.submodels])

    @property
    def coeffs_err(self) -> npt.NDArray[np.float64]:
        """Concatenated coefficients' errors of the sub-models."""
        return np.concatenate([model.coeffs_err for model in self.submodels])

    def compute(
        self,
        x: float | npt.NDArray[np.float64],
        **meta: dict[str, float | npt.NDArray[np.float64]],
    ) -> float | npt.NDArray[np.float64]:
        """Compute the Model for the given x values.

        Parameters
        ----------
        x : float | npt.NDArray[np.float64]
            Input values to compute the model.
        meta : dict[str, float | npt.NDArray[np.float64]]
            Metadata to be used in the computation.

        Returns
        -------
        float | npt.NDArray[np.float64]
            Model computed for the input values.

        """
        logger.info("Computing model")
        computing_meta = set_meta_for_computation(
            meta,
            self.shared_meta,
            self.submodels,
        )

        result_dict = {}
        for id_letter, submodel in zip(ascii_lowercase, self.submodels, strict=False):
            result_dict[id_letter] = submodel.compute(x, **computing_meta[submodel.id_])

        return ne.evaluate(self.math_expression, local_dict=result_dict)

    def formula(self) -> str:
        """Return the mathematical expression of the Model.

        Returns
        -------
        str
            Mathematical expression.

        """
        math_exp = (" ".join(self.math_expression)).replace("  ", " ")
        for i, model in enumerate(self.submodels):
            math_exp.replace(f" {ascii_lowercase[i]} ", f" {model.formula()} ")
        return math_exp

    def explicit_math_expression(self) -> str:
        """Explain the math expression.

        Return the math expression and spcificies which model is corresponding
        with each letter.

        """
        message = f"Formula: {self.math_expression}\n"
        message += "Terms:\n"
        for letter_id, submodel in zip(ascii_lowercase, self.submodels, strict=False):
            message += f"\t- {letter_id}: {submodel.id_}"
        return message


# Utility functions -------------------------------------------------------------------
def check_and_format_submodels(
    submodels: MathFunction | list[MathFunction],
) -> list[MathFunction]:
    """Check and format the input sub-models.

    Check if the input sub-models are valid and in case format them to a list
    of MathFunction.

    Parameters
    ----------
    submodels: Any
        Should be the list of submodels used in the Model.

    Raises
    ------
    TypeError
        If the models list is not a list of MathFunction instances.

    """
    if isinstance(submodels, MathFunction):
        submodels = [submodels]

    if not all(isinstance(model, MathFunction) for model in submodels):
        msg = "Sub-models must be a list of MathFunction instances."
        logger.error(msg)
        raise TypeError(msg)

    # Numerate ids if more than one instance of the same MathFunction
    submodels_ids = [model.id_ for model in submodels]
    id_count = {id_: 0 for id_ in set(submodels_ids)}

    for model in submodels:
        if submodels_ids.count(model.id_) > 1:
            model.id_ = f"{model.id_}_{id_count[model.id_]}"
            id_count[model.id_] += 1
            logger.info("Changed submodel id: %s", model.id_)

    return submodels


def check_keys_are_submodels(
    keys: list[str],
    submodels: list[MathFunction],
) -> None:
    """Check if the keys are valid sub-models of the Model.

    Check if the keys are valid ids of sub-models in the list of sub-models.

    Parameters
    ----------
    keys : list[str]
        List of keys to check.
    submodels : list[MathFunction]
        List of sub-models.

    Raises
    ------
    ValueError
        If any of the keys is not a valid sub-model.

    """
    submodels_ids = [model.id_ for model in submodels]

    for key in keys:
        if key not in submodels_ids:
            msg = f"Key {key} is not a valid sub-model."
            logger.error(msg)
            raise ValueError(msg)


def check_and_format_shared_meta(
    shared_meta: dict[str, list[str]] | None,
    submodels: list[MathFunction],
) -> dict[str, list[str]]:
    """Check and format the shared metadata.

    Check if the shared metadata is valid and format it to a dictionary.

    Parameters
    ----------
    shared_meta : dict[str, list[str]] | None
        Shared metadata between the sub-models.
    submodels : list[MathFunction]
        List of sub-models.

    Returns
    -------
    dict[str, list[str]]
        The shared metadata.

    Raises
    ------
    SharedMetaError
        If the shared metadata is not valid:
            - it is not a dictionary.
            - it is not empty for a single model.
            - the values are not lists.
            - the keys are not valid sub-models.
            - the keys are not in the required metadata of the sub-models.

    """
    if shared_meta is None:
        return {}

    # shared_meta must be a dictionary
    if not isinstance(shared_meta, dict):
        raise SharedMetaError(shared_meta, {"error": "Should be a dictionary."})

    # shared_meta must be empty for a single model
    nusubmodels = len(submodels)
    if nusubmodels == 1 and shared_meta != {}:
        extra = {"error": "Should be empty for a single model."}
        raise SharedMetaError(shared_meta, extra)

    # check shared_meta has valid format
    for key, model_ids in shared_meta.items():
        extra = {"key": key, "model_ids": model_ids}
        if not isinstance(model_ids, list):
            extra.update({"error": "Values should be lists."})
            raise SharedMetaError(shared_meta, extra)

        try:
            check_keys_are_submodels(model_ids, submodels)
        except ValueError as exc:
            extra.update({"error": str(exc)})
            raise SharedMetaError(shared_meta, extra) from exc

        for model_id in model_ids:
            model = next(model for model in submodels if model.id_ == model_id)
            if key not in model.required_meta:
                error_msg = f"Key {key} not in model {model_id} required metadata."
                extra.update({"error": error_msg})
                raise SharedMetaError(shared_meta, extra)

    return shared_meta


def broadcast_meta(
    meta: dict[str, dict[str, float | npt.NDArray[np.float64]]],
    shared_meta: dict[str, list[str]],
    submodels: list[MathFunction],
) -> dict[str, dict[str, float | npt.NDArray[np.float64]]]:
    """Broadcast the metadata to the sub-models based on the shared metadata.

    Parameters
    ----------
    meta : dict[str, dict[str, float | npt.NDArray[np.float64]]]
        Input metadata to broadcast.
    shared_meta : dict[str, list[str]]
        Shared metadata between the sub-models.
    submodels : list[MathFunction]
        List of sub-models.

    Returns
    -------
    dict[str, dict[str, float | npt.NDArray[np.float64]]]
        Updated metadata. All models have the same value for the common metadata keys.

    """
    for shared_key in shared_meta:
        specified_in_meta = {
            model_id: meta[model_id][shared_key]
            for model_id in shared_meta[shared_key]
            if model_id in meta and shared_key in meta[model_id]
        }

        if len(specified_in_meta) == 0:
            logger.info("No specified value for common metadata key: %s", shared_key)
            logger.info("Search for a default value in the models metadata...")
            specified_in_meta = {
                model_id: getattr(model, shared_key)
                for model_id in shared_meta[shared_key]
                for model in submodels
                if model_id == model.id_ and getattr(model, shared_key) is not None
            }

        values_specified = np.array(list(specified_in_meta.values()))

        empty_list = not specified_in_meta
        same_values = len(set(specified_in_meta)) == 1
        all_close = np.allclose(values_specified, values_specified[0])

        if empty_list or not same_values or not all_close:
            msg = f"No default value found for shared meta key: {shared_key}"
            extra = {
                "error": msg,
                "shared_key": shared_key,
                "meta": meta,
                "specified_values": specified_in_meta,
            }
            logger.error(msg, extra=extra)
            raise MetaError(meta, extra)

        shared_key_default_value = values_specified[0]
        logger.info(
            "One default value found for common metadata key: %s = %d",
            shared_key,
            shared_key_default_value,
        )
        for model_id in shared_meta[shared_key]:
            meta[model_id].update({shared_key: shared_key_default_value})

    return meta


def set_meta_for_computation(
    meta: dict[str, dict[str, float | npt.NDArray[np.float64]]],
    shared_meta: dict[str, list[str]],
    submodels: list[MathFunction],
) -> dict[str, dict[str, float | npt.NDArray[np.float64]]]:
    """Set and conform metadata for the `compute` method.

    First it checks that the input metadta has the correct structure and type.
    Successively it broadcast the shared keys among submodels, ensuring that the
    same value is used for all models.

    Parameters
    ----------
    meta : dict[str, dict[str, float  |  npt.NDArray[np.float64]]]
        Input metadata.
    shared_meta : dict[str, list[str]]
        Dict containing the key of the shared metadata among models.
    submodels : list[MathFunction]
        Submodels of the Model

    Returns
    -------
    dict[str, dict[str, float | npt.NDArray[np.float64]]]
        Updated meta, ready to be used to compute the model.

    Raises
    ------
    MetaError
        If `meta` is not a dict.
        If `meta` keys are not only submodels' ids.
        If `meta` does not contain only required keys.

    """
    # check meta is a dict
    if not isinstance(meta, dict):
        raise MetaError(meta, extra={"type": type(meta), "note": "Not a dict"})

    # check input meta is relative to the present submodels
    try:
        check_keys_are_submodels(list(meta.keys()), submodels)
    except ValueError as exc:
        raise MetaError(meta, extra={"note": "Keys are not all necessary."}) from exc

    # check that for each submodel only the required meta keys are specified
    required_meta = {model.id_: model.required_meta for model in submodels}
    for submodel_id, submodel_meta in meta.items():
        if set(submodel_meta.keys()).issubset(required_meta[submodel_id]):
            extra = {
                "submodel_id": submodel_id,
                "required_keys": required_meta[submodel_id],
            }
            raise MetaError(meta, extra=extra)

    return broadcast_meta(meta, shared_meta, submodels)


# Exceptions --------------------------------------------------------------------------
class SharedMetaError(Exception):
    """Exception raised for errors in the shared metadata."""

    def __init__(self, shared_meta: dict, extra: dict | None = None) -> None:
        """Initialize the SharedMetaError class.

        Parameters
        ----------
        shared_meta : dict
            Shared metadata between the sub-models.
        extra : dict | None, optional
            Extra information about the error. Default is None.

        """
        self.extra = extra or {}
        self.extra.update({"shared_meta": shared_meta})
        logger.error("Shared metadata error.", extra=self.extra)
        super().__init__(self.extra)


class MetaError(Exception):
    """Exception raised for errors in the metadata."""

    def __init__(self, meta: dict, extra: dict | None = None) -> None:
        """Initialize the MetaError class.

        Parameters
        ----------
        meta : dict
            Metadata.
        extra : dict | None, optional
            Extra information about the error. Default is None.

        """
        self.extra = extra or {}
        self.extra.update({"meta": meta})
        logger.error("Metadata error.", extra=self.extra)
        super().__init__(self.extra)
