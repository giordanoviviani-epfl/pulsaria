"""Import logger and its configuration."""

import logging

from logging_setup.logger_setup import configure_logging

__all__ = ["logger", "configure_logging"]
logger = logging.getLogger("pulsaria")  # __name__ is a common choice
configure_logging()
