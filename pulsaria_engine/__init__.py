"""Import all modules in the package."""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import data_handling
from logging_setup import configure_logging

logger = logging.getLogger("pulsaria_engine")  # __name__ is a common choice
configure_logging()


__all__ = ["logger", "data_handling"]
