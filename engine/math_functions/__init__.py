"""Math functions to define models and perform calculations.

This directory contains the mathematical functions that are used to define the
models and perform calculations.

math_function_protocol.py:
    Define the protocol class MathFunction, which is the base class for all
    mathematical funciton of models in pulsaria's engine.

Subdirectories:
    Each subdirectory contains a specific type of mathematical function. The
    subdirectory name should be descriptive of the type of function it contains.
    The subdirectory should contain the following files:

    __init__.py:
        A file that contains the import statements for the functions in the
        subdirectory. The __all__ variable should be defined to include all the
        functions in the subdirectory.

    _function.py:
        The file that contains the class definition for the mathematical
        function. The class should implement the MathFunction protocol.

    _toolkit.py:
        The file that contains the toolkit class for the mathematical function.
        The toolkit class should contain all the functions that are used to
        manipulate the function class.

    Optional files:
        _tests.py:
            The file that contains the tests for the mathematical function. The tests
            should test all the methods of the function class.

        _examples.py:
            The file that contains examples of how to use the mathematical function.
            The examples should demonstrate how to create an instance of the function
            class and how to use the toolkit class to manipulate the function.

        _docs.md:
            The file that contains the documentation for the mathematical function.
            The documentation should describe the function, its parameters, and its
            methods.

        _notes.md:
            The file that contains notes on the mathematical function. The notes
            should contain any information that is relevant to the function but does
            not fit in the documentation.

        _references.bib:
            The file that contains the references for the mathematical function. The
            references should include any papers, books, or websites that are
            relevant to the function.
"""

from engine.math_functions.constant import ConstantFunction
from engine.math_functions.fourier_series import FourierSeries
from engine.math_functions.math_function_protocol import MathFunction

__all__ = [
    "ConstantFunction",
    "FourierSeries",
    "MathFunction",
]
