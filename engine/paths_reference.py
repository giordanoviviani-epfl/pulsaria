"""Module that declares the useful paths for the project."""

from pathlib import Path


class PathReference:
    """Class that declares the useful paths for the project."""

    def __init__(self) -> None:
        """Define useful path variables for the project as attributes.

        The paths are defined relative to the path of this file, and they are
        independent of the current working directory.
        """
        path_reference_file = Path(__file__)

        self.root = path_reference_file.parent.parent
        self.engine = path_reference_file.parent
        self.data = self.root / "data"


pulsaria_path = PathReference()
