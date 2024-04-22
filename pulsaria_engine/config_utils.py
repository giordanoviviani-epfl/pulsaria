"""Module for configuration utilities."""

from collections.abc import Callable


def resolve(name: str) -> Callable:
    """Resolve a dotted name to a global object.

    Function is taken by the logging module:
    https://github.com/python/cpython/blob/main/Lib/logging/config.py#L94
    """
    name = name.split(".")
    used = name.pop(0)
    try:
        found = __import__(used)
        for n in name:
            used = used + "." + n
            try:
                found = getattr(found, n)
            except AttributeError:
                __import__(used)
                found = getattr(found, n)
    except (AttributeError, ModuleNotFoundError):
        for i, n in enumerate(name):
            if i == 0 and (n not in globals()):
                raise
            found = globals()[n] if i == 0 else getattr(found, n)

    return found
