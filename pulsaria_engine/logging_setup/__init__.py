"""Import logger and its configuration."""

import logging

from .logger_setup import configure_logging

logger = logging.getLogger("pulsaria")  # __name__ is a common choice
configure_logging()

__all__ = ["configure_logging"]
