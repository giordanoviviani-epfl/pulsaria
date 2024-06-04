"""Module that defines the ModelFactory class.

The ModelFactory class is used to configure and generates multiple Model
objects.
"""

import logging
from pathlib import Path

from engine import config_utils
from engine.math_functions import MathFunction
from engine.model.model import Model, check_and_format_submodels

logger = logging.getLogger("engine.model.factory")


class ModelFactory:
    """ModelFactory class."""

    def __init__(self) -> None:
        """Initialize the ModelFactory class."""
        self.is_configured = False
        self.config_file = None
        self.config = {}
        self.submodels: list[MathFunction]
        self.shared_meta = {}
        self.math_expression = ""

    def configure(
        self,
        math_funcs: list[MathFunction],
        math_expression: str | None = None,
        shared_meta: dict[str, list[str]] | None = None,
    ) -> None:
        """Configure the ModelFactory.

        Parameters
        ----------
        math_funcs : list[MathFunction]
            List of math functions.
        math_expression : str, optional
            Math expression to be used in the model, by default None.
        shared_meta : dict[str, list[str]], optional
            Shared metadata for all the sub-models, by default None.

        """
        self.math_expression = math_expression
        self.shared_meta = shared_meta
        self.submodels = check_and_format_submodels(math_funcs)

        self.is_configured = True
        logger.info("ModelFactory is configured.")

    def configure_from_file(self, config_file: str | Path, method: str) -> None:
        """Configure the ModelFactory from a configuration file.

        Parameters
        ----------
        config_file : str or Path
            Path to the configuration file.
        method : str
            Method to be used for the configuration.

        """
        self.config_file = Path(config_file)
        config_settings = config_utils.read_yaml(config_file)

        # Check method is implemented
        if method not in config_settings:
            message = f"Method {method} is not implemented in the configuration file."
            logger.error(message)
            raise NotImplementedError(method)

        building_method = config_settings[method]

        # Check if the method contains the ModelFactory key
        key = "Model"
        if key not in building_method:
            message = f"The method does not contain a {key} key."
            logger.error(message)
            raise KeyError(message)

        self.config = building_method[key].copy()

        # Check match_funcs are present and of the correct format
        if "math_funcs" not in self.config:
            raise config_utils.ConfigError(
                self.config,
                extra={"note": "Key `math_funcs` is missing."},
            )
        if not isinstance(self.config["math_funcs"], list):
            raise config_utils.ConfigError(
                self.config,
                extra={"note": "math_funcs is not a list"},
            )
        for item in self.config["math_funcs"]:
            if "math_func" not in item:
                raise config_utils.ConfigError(
                    self.config,
                    extra={
                        "item": item,
                        "note": "`math_funcs` item does not contain `math_func` key.",
                    },
                )

        # Get configuration
        math_funcs_list = []
        for math_func_dict in self.config["math_funcs"]:
            math_func_name = math_func_dict["math_func"]
            math_func_config = {
                key: item for key, item in math_func_dict.items() if key != "math_func"
            }
            math_func_class = config_utils.resolve(math_func_name)
            if not isinstance(math_func_class, type):
                raise config_utils.ConfigError(
                    self.config,
                    extra={
                        "note": f"Math function {math_func_name} is not a class.",
                        "type": type(math_func_class),
                    },
                )
            math_func_instance = math_func_class(**math_func_config)
            math_funcs_list.append(math_func_instance)

        math_expression = self.config.get("math_expression", None)
        shared_meta = self.config.get("shared_meta", None)
        self.configure(math_funcs_list, math_expression, shared_meta)

    def build(self) -> Model:
        """Build the Model object.

        Returns
        -------
        Model
            Model object.

        Raises
        ------
        ValueError
            If ModelFactory object is not configured.

        """
        if not self.is_configured:
            message = "ModelFactory is not configured."
            logger.error(message)
            raise ValueError(message)

        m = Model(self.submodels, self.math_expression, self.shared_meta)
        logger.info("ModelFactory: Model is built.")
        return m
