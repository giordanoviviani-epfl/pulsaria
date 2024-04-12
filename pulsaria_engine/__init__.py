"""Import all modules in the package."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from logging_setup import logger

__all__ = ["logger"]
