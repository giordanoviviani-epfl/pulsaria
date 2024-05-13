"""Module for configuration utilities."""

from collections.abc import Callable


def resolve(name: str) -> Callable:
    """Resolve a dotted name to a global object.

    Function is taken by the logging module:
    https://github.com/python/cpython/blob/main/Lib/logging/config.py#L94
    """
    split_name = name.split(".")
    used = split_name.pop(0)
    try:
        found = __import__(used)
        for n in split_name:
            used = used + "." + n
            try:
                found = getattr(found, n)
            except AttributeError:
                __import__(used)
                found = getattr(found, n)
    except (AttributeError, ModuleNotFoundError):
        split_name = name.split(".")
        for i, n in enumerate(split_name):
            if i == 0 and (n not in globals()):
                raise
            found = globals()[n] if i == 0 else getattr(found, n)

    return found
