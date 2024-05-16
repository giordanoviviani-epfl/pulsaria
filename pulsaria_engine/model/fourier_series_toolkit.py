"""Fourier Series Toolkit.

Contains fourier series function with all its parametrizations, and multiple
other functions to statistics and carry specific operations.
"""

import logging
from itertools import batched, permutations

import numpy as np
import numpy.typing as npt

logger = logging.getLogger("pulsaria_engine.model.fs_toolkit")

FOURIER_SERIES_PARAMETRIZATIONS = [
    "cos-sin",
    "sin-cos",
    "amp-phase",
    "amp-phase+",
    "exp",
]


def fourier_series(
    coeff: npt.NDArray[np.float64],
    phase: float | npt.NDArray[np.float64],
    parametrization: str | None = None,
    n_harmonics: int | None = None,
) -> float | npt.NDArray[np.float64]:
    """Compute a Fourier series.

    Given the coefficients of the Fourier series and the phase, compute the
    model values.

    Parameters
    ----------
    coeff : npt.NDArray[np.float64]
        Coefficients of the Fourier series.
    phase : float or npt.NDArray[np.float64]
        Phase at which compute the Fourier series.
    parametrization : str or None
        Parametrization/Form of the Fourier series coefficients (:param:`coeff`)
        inputted. If None (default), the function will assume that the
        coefficients are in the sin-cos form.
    n_harmonics : int or None
        Number of harmonics. It is used to check the input shape of the
        coefficients or to trim the Fourier series to the specified number of
        harmonics/terms. Default is None, and no checks or trims are performed.

    Returns
    -------
    float or npt.NDArray[np.float64]
        Value of the Fourier series.

    Raises
    ------
    ValueError
        If the number of coefficients is not even.
        If the number of harmonics is less than 1 or greater than half of the
        coefficients.
        If the parametrization is not valid.
        If the Fourier series has complex values.
    NotImplementedError
        If the parametrization is not implemented.

    """
    _check_coefficients_even(coeff)
    n_harmonics = _check_harmonics_fourier_series(coeff, n_harmonics)
    coeff_trimmed: npt.NDArray[np.float64] = coeff[: 2 * n_harmonics].copy()

    parametrization = parametrization if parametrization else "cos-sin"
    _check_parametrization(parametrization)

    fs_values: float | npt.NDArray = 0
    iter_coeff = enumerate(batched(coeff_trimmed, 2), 1)

    match parametrization:
        case "cos-sin":
            for n, (an, bn) in iter_coeff:
                factor = 2 * np.pi * n
                fs_values += an * np.cos(factor * phase) + bn * np.sin(factor * phase)
        case "sin-cos":
            for n, (an, bn) in iter_coeff:
                factor = 2 * np.pi * n
                fs_values += an * np.sin(factor * phase) + bn * np.cos(factor * phase)
        case "amp-phase":
            for n, (An, phin) in iter_coeff:  # noqa: N806
                factor = 2 * np.pi * n
                fs_values += An * np.cos(factor * phase - phin)
        case "amp-phase+":
            for n, (An, phin) in iter_coeff:  # noqa: N806
                factor = 2 * np.pi * n
                fs_values += An * np.cos(factor * phase + phin)
        case "exp":
            for n, (pos_coeff, neg_coeff) in iter_coeff:
                complex_factor = 1j * 2 * np.pi * n
                fs_values += pos_coeff * np.exp(complex_factor * phase)
                fs_values += neg_coeff * np.exp(-complex_factor * phase)
            # Check that there is no imaginary part in the Fourier series
            if not all(np.isclose(np.imag(fs_values), 0)):
                msg = "Complex value found in the Fourier series."
                raise ValueError(msg)
            fs_values = np.real(fs_values)
        case _:
            msg = f"Parametrization {parametrization} not implemented."
            raise NotImplementedError(msg)

    return fs_values


def change_parametrization_fs_coeff(
    coeff: npt.NDArray[np.float64],
    coeff_errors: npt.NDArray[np.float64],
    old_parametrization: str,
    new_parametrization: str,
    cov_matrix: np.ndarray | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Change the parametrization of the Fourier series coefficients.

    Parameters
    ----------
    coeff : npt.NDArray[np.float64]
        Coefficients of the Fourier series.
    coeff_errors : npt.NDArray[np.float64]
        Relative uncertainties of the Fourier series coefficients.
    old_parametrization : str
        Old parametrization/form of the Fourier series coefficients.
    new_parametrization : str
        New parametrization/form of the Fourier series coefficients.
    cov_matrix : npt.NDArray[np.float64] or None
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series with the new parametrization.
    coeff_errors: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients with the new parametrization.

    """
    _check_parametrization(old_parametrization)
    _check_parametrization(new_parametrization)

    if old_parametrization == new_parametrization:
        msg = "Old and new parametrizations are the same: %s. No action taken."
        logger.info(msg, old_parametrization)
        return coeff, coeff_errors

    # Change from whatever parametrization to cos-sin
    to_cos_sin = {
        "sin-cos": _switch_cos_sin_and_sin_cos,
        "amp-phase": _change_amp_phase_to_cos_sin,
        "amp-phase+": _change_amp_phase_plus_to_cos_sin,
        "exp": _change_exp_to_cos_sin,
    }

    to_whatever = {
        "sin-cos": _switch_cos_sin_and_sin_cos,
        "amp-phase": _change_cos_sin_to_amp_phase,
        "amp-phase+": _change_cos_sin_to_amp_phase_plus,
        "exp": _change_cos_sin_to_exp,
    }

    if old_parametrization != "cos-sin":
        func_to_cos_sin = to_cos_sin[old_parametrization]
        if old_parametrization in ["amp-phase", "amp-phase+"]:
            coeff, coeff_errors = func_to_cos_sin(coeff, coeff_errors, cov_matrix)
        else:
            coeff, coeff_errors = func_to_cos_sin(coeff, coeff_errors)

    if new_parametrization != "cos-sin":
        func_to_whatever = to_whatever[new_parametrization]
        if new_parametrization in ["amp-phase", "amp-phase+"]:
            coeff, coeff_errors = func_to_whatever(coeff, coeff_errors, cov_matrix)
        else:
            coeff, coeff_errors = func_to_whatever(coeff, coeff_errors)

    return coeff, coeff_errors


def change_parametrization_fs_cov_matrix(
    cov_matrix: npt.NDArray[np.float64],
    old_parametrization: str,
    new_parametrization: str,
) -> npt.NDArray[np.float64]:
    """Permutation of the covariance matrix.

    Permute the covariance matrix of the Fourier series coefficients from the
    old parametrization to the new parametrization.
    Currently implemented permutations are:
    - cos-sin to sin-cos
    - sin-cos to cos-sin

    Parameters
    ----------
    cov_matrix : 2D np.ndarray
        Covariance matrix of the Fourier series coefficients in the sin-cos or
        cos-sin form
    old_parametrization : str
        Parametrization of the inputted covariance matrix. Can be either
        'cos-sin' or 'sin-cos'.
    new_parametrization : str
        Parametrization of the resulting covariance matrix. Can be either
        'cos-sin' or 'sin-cos'.

    Returns
    -------
    cov_matrix: np.ndarray
        Covariance matrix of the Fourier series coefficients in the sin-cos or
        cos-sin form.


    Raises
    ------
    NotImplementedError
        If the permutation of the covariance matrix for the parametrizations
        is not implemented.

    """
    _check_parametrization(old_parametrization)
    _check_parametrization(new_parametrization)

    tuple_permutations = (old_parametrization, new_parametrization)
    if tuple_permutations in permutations(["cos-sin", "sin-cos"]):
        p = _permutation_matrix_swap_adjacent(len(cov_matrix))
        return p @ cov_matrix @ p.T

    msg = (
        "Permutation of the covariance matrix for "
        "the parametrizations %s not implemented."
    )
    raise NotImplementedError(msg % tuple_permutations)


def formula_fourier_series(
    parametrization: str,
    n_harmonics: int | str,
) -> str:
    """Return the formula of the Fourier series.

    Latex string of the Fourier series formula for the given parametrization
    and number of harmonics.

    Parameters
    ----------
    parametrization : str
        Parametrization of the Fourier series.
    n_harmonics : int or str
        Number of harmonics to show the formula for. It can be an integer or
        str.

    Returns
    -------
    str
        Formula of the Fourier series.

    Raises
    ------
    NotImplementedError
        If the parametrization is not implemented.

    Examples
    --------
    >>> formula_fourier_series("cos-sin", 3)
    '$\\\\sum_{n=1}^3 (a_n \\\\cos(2\\\\pi n x) + b_n \\\\sin(2\\\\pi n x))$'

    """  # noqa: D301
    match parametrization:
        case "cos-sin":
            return (
                rf"$\sum_{{n=1}}^{n_harmonics} (a_n \cos(2\pi n x) "
                rf"+ b_n \sin(2\pi n x))$"
            )
        case "sin-cos":
            return (
                rf"$\sum_{{n=1}}^{n_harmonics} (a_n \sin(2\pi n x) "
                rf"+ b_n \cos(2\pi n x))$"
            )
        case "amp-phase":
            return rf"$\sum_{{n=1}}^{n_harmonics} A_n \cos(2\pi n x - \phi_n)$"
        case "amp-phase+":
            return rf"$\sum_{{n=1}}^{n_harmonics} A_n \cos(2\pi n x + \phi_n)$"
        case "exp":
            return rf"$\sum_{{n=-{n_harmonics}}}^{n_harmonics} C_n e^{{i2\pi n x}}$"
        case _:
            msg = "Parametrization %s not implemented."
            raise NotImplementedError(msg % parametrization)


# Utility functions -------------------------------------------------------------------
def _check_coefficients_even(coeff: npt.NDArray) -> None:
    """Check if the number of coefficients is even.

    Parameters
    ----------
    coeff : npt.NDArray
        Coefficients of the Fourier series.

    Raises
    ------
    ValueError
        If the number of coefficients is not even.

    """
    if len(coeff) % 2 != 0:
        msg = "Number of coefficients must be even."
        logger.error(msg)
        raise ValueError(msg)


def _check_harmonics_fourier_series(coeff: npt.NDArray, n_harmonics: int | None) -> int:
    if n_harmonics:
        n_harmonics = int(n_harmonics)
        if n_harmonics < 1:
            msg = "Number of harmonics must be greater than 0."
            raise ValueError(msg)
        if n_harmonics > (len(coeff) // 2):
            msg = "Number of harmonics must be <= to half of the coefficients."
            raise ValueError(msg)
    else:
        n_harmonics = len(coeff) // 2
    logger.info("Number of harmonics: %s", n_harmonics)
    return n_harmonics


def _check_parametrization(parametrization: str) -> None:
    """Check if the parametrization is valid.

    Parameters
    ----------
    parametrization : str
        Parametrization of the Fourier series.

    Raises
    ------
    ValueError
        If the parametrization is not valid.

    """
    if parametrization not in FOURIER_SERIES_PARAMETRIZATIONS:
        msg = "Invalid parametrization: %s. Choose one of: %s"
        logger.error(msg, parametrization, FOURIER_SERIES_PARAMETRIZATIONS)
        raise ValueError(msg % (parametrization, FOURIER_SERIES_PARAMETRIZATIONS))


def _permutation_matrix_swap_adjacent(n: int) -> npt.NDArray[np.float64]:
    p = np.zeros((n, n))
    p[range(0, n, 2), range(1, n, 2)] = 1  # Set 1s in even rows, odd columns
    p[range(1, n, 2), range(0, n, 2)] = 1  # Set 1s in odd rows, even columns
    return p


# Parametrization functions -----------------------------------------------------------
def _switch_cos_sin_and_sin_cos(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Switch the coefficients from cos-sin to sin-cos form or viceversa.

    The function simply swaps the adjacent coefficients of the Fourier series
    from the cos-sin to the sin-cos form or viceversa.
    If the covariance matrix is provided, it is also transformed.

    Parameters
    ----------
    coeff : npt.NDArray
        Coefficients of the Fourier series in the cos-sin or sin-cos form.
    coeff_err : npt.NDArray
        Uncertainties of the Fourier series coefficients.
    cov_matrix : npt.NDArray or None
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    new_coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the sin-cos or cos-sin form.
    new_coeff_errors: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients.


    Examples
    --------
    >>> coeff = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> coeff_err = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    >>> new_coeff, new_coeff_errors = _switch_cos_sin_and_sin_cos(coeff, coeff_err)
    >>> new_coeff
    array([2., 1., 4., 3., 6., 5., 8., 7.])
    >>> new_coeff_errors
    array([0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7])

    Using the function twice should return the original coefficients:
    >>> old_coeff, old_coeff_errors = _switch_cos_sin_and_sin_cos(new_coeff,
    ...                                                           new_coeff_errors)
    >>> all(old_coeff == coeff)
    True
    >>> all(old_coeff_errors == coeff_err)
    True

    """
    _check_coefficients_even(coeff)

    p_matrix = _permutation_matrix_swap_adjacent(len(coeff))
    coeff_matrix, coeff_err_matrix = np.diag(coeff), np.diag(coeff_err)

    new_coeff = np.diag(p_matrix @ coeff_matrix @ p_matrix.T)
    new_coeff_errors = np.diag(p_matrix @ coeff_err_matrix @ p_matrix.T)

    return new_coeff, new_coeff_errors


def _change_cos_sin_to_amp_phase(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
    cov_matrix: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from cos-sin to amp-phase form.

    Change the parametrization of the Fourier series coefficients from
    the cos-sin to the amp-phase form.

    Parameters
    ----------
    coeff : npt.NDArray
        Coefficients of the Fourier series in the cos-sin form.
    coeff_err : npt.NDArray
        Uncertainties of the Fourier series coefficients.
    cov_matrix : npt.NDArray or None
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    new_coeff: npt.NDArray
        Coefficients of the Fourier series in the amp-phase form.
    new_coeff_err: npt.NDArray
        Uncertainties of the Fourier series coefficients in the amp-phase form.

    """
    _check_coefficients_even(coeff)
    new_coeff = np.zeros_like(coeff)
    new_coeff_errors = np.zeros_like(coeff_err)

    coeff_and_errors_per_n = batched(zip(coeff, coeff_err, strict=True), 2)
    for n, ((an, e_an), (bn, e_bn)) in enumerate(coeff_and_errors_per_n, 1):
        An = np.sqrt(an**2 + bn**2)  # noqa: N806
        phin = np.arctan2(bn, an)

        # See propagation of errors in:
        # 1. Petersen 1986 - bibcode: 1986A&A...170...59P
        # 2. https://math.stackexchange.com/questions/4474757
        e_An_2 = (an**2 * e_an**2 + bn**2 * e_bn**2) / An**2  # noqa: N806
        e_phin_2 = (bn**2 * e_an**2 + an**2 * e_bn**2) / An**4
        if cov_matrix:
            e_An_2 += 2 * an * bn * cov_matrix[2 * n - 2, 2 * n - 1] / An**2  # noqa: N806
            e_phin_2 -= 2 * an * bn * cov_matrix[2 * n - 2, 2 * n - 1] / An**4

        new_coeff[2 * n - 2] = An
        new_coeff[2 * n - 1] = phin
        new_coeff_errors[2 * n - 2] = np.sqrt(e_An_2)
        new_coeff_errors[2 * n - 1] = np.sqrt(e_phin_2)

    return new_coeff, new_coeff_errors


def _change_amp_phase_to_cos_sin(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
    cov_matrix: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from amp-phase to cos-sin form.

    Change the parametrization of the Fourier series coefficients from
    the amp-phase to the cos-sin form.

    Parameters
    ----------
    coeff : np.array
        Coefficients of the Fourier series in the amp-phase form.
    coeff_err : np.array
        Uncertainties of the Fourier series coefficients.
    cov_matrix : np.ndarray | None, optional
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    new_coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the cos-sin form.
    new_coeff_errors: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients in the cos-sin form.

    """
    _check_coefficients_even(coeff)
    new_coeff = np.zeros_like(coeff)
    new_coeff_errors = np.zeros_like(coeff_err)
    coeff_and_errors_per_n = batched(zip(coeff, coeff_err, strict=True), 2)

    for n, ((An, e_An), (phin, e_phin)) in enumerate(coeff_and_errors_per_n, 1):  # noqa: N806
        an = An * np.cos(phin)
        bn = An * np.sin(phin)

        sin_phin, cos_phin = np.sin(phin), np.cos(phin)
        e_an_2 = (cos_phin * e_An) ** 2 + (An * sin_phin * e_phin) ** 2
        e_bn_2 = (sin_phin * e_An) ** 2 + (An * cos_phin * e_phin) ** 2
        if cov_matrix:
            e_an_2 -= 2 * An * cos_phin * sin_phin * cov_matrix[2 * n - 2, 2 * n - 1]
            e_bn_2 += 2 * An * cos_phin * sin_phin * cov_matrix[2 * n - 2, 2 * n - 1]

        new_coeff[2 * n - 2] = an
        new_coeff[2 * n - 1] = bn
        new_coeff_errors[2 * n - 2] = np.sqrt(e_an_2)
        new_coeff_errors[2 * n - 1] = np.sqrt(e_bn_2)

    return new_coeff, new_coeff_errors


def _change_cos_sin_to_amp_phase_plus(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
    cov_matrix: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from cos-sin to amp-phase (with +) form.

    Change the parametrization of the Fourier series coefficients from
    the cos-sin to the amp-phase form.

    Parameters
    ----------
    coeff : np.array
        Coefficients of the Fourier series in the cos-sin form.
    coeff_err : np.array
        Uncertainties of the Fourier series coefficients.
    cov_matrix : np.array or None
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    new_coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the amp-phase form.
    new_coeff_err: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients in the amp-phase form.

    """
    new_coeff, new_coeff_errors = _change_cos_sin_to_amp_phase(
        coeff,
        coeff_err,
        cov_matrix,
    )

    # It should be phin = np.arctan2(-bn, an), but currently the function
    # returned phin = np.arctan2(bn, an).
    # We can use the property atan(-x) = -atan(x) to change the sign of phin.
    # So, we need to change the sign of phin.
    new_coeff[1::2] *= -1

    # Note uncertainties are the same.
    return new_coeff, new_coeff_errors


def _change_amp_phase_plus_to_cos_sin(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
    cov_matrix: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from cos-sin to amp-phase (with +) form.

    Change the parametrization of the Fourier series coefficients from
    the cos-sin to the amp-phase form.

    Parameters
    ----------
    coeff : npt.NDArray[np.float64]
        Coefficients of the Fourier series in the cos-sin form.
    coeff_err : npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients.
    cov_matrix : npt.NDArray[np.float64] or None
        Covariance matrix of the Fourier series coefficients. Default is None.

    Returns
    -------
    new_coeff_err: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the amp-phase+ form.
    new_coeff_err: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients in the amp-phase form.

    """
    new_coeff, new_coeff_errors = _change_amp_phase_to_cos_sin(
        coeff,
        coeff_err,
        cov_matrix,
    )

    # Since bn = An * sin(phin), but should be bn = - An * sin(phin),
    # we need to change the sign of bn.
    new_coeff[1::2] *= -1

    # Note uncertainties are the same.
    return new_coeff, new_coeff_errors


def _change_cos_sin_to_exp(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from cos-sin to exp form.

    Change the parametrization of the Fourier series coefficients from
    the cos-sin to the exp form.

    Parameters
    ----------
    coeff : npt.NDArray[np.float64]
        Coefficients of the Fourier series in the cos-sin form.
    coeff_err : npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients.

    Returns
    -------
    new_coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the exp form.
    new_coeff_err: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients in the exp form.

    Examples
    --------
    >>> coeff = np.random.rand(8)
    >>> coeff_err = np.random.rand(8)
    >>> new_coeff, new_coeff_errors = _change_cos_sin_to_exp(coeff, coeff_err)
    >>> new_coeff.shape == coeff.shape
    True
    >>> check_conjugate = []
    >>> for n, (pos_coeff, neg_coeff) in enumerate(batched(new_coeff, 2), 1):
    ...     check_conjugate.append(np.isclose(pos_coeff, np.conjugate(neg_coeff)))
    >>> all(check_conjugate)
    True

    """
    _check_coefficients_even(coeff)
    new_coeff = np.zeros_like(coeff, dtype=complex)
    new_coeff_errors = np.zeros_like(coeff_err, dtype=complex)

    coeff_and_errors_per_n = batched(zip(coeff, coeff_err, strict=True), 2)
    for n, ((an, e_an), (bn, e_bn)) in enumerate(coeff_and_errors_per_n, 1):
        Cn = (an - 1j * bn) / 2  # noqa: N806
        Cn_neg = np.conjugate(Cn)  # noqa: N806

        # Note: Error propagation for compelx values is not well-defined.
        e_Cn_real = e_an / 2  # noqa: N806
        e_Cn_imag = e_bn / 2  # noqa: N806
        e_Cn = e_Cn_neg = complex(e_Cn_real, e_Cn_imag)  # noqa: N806

        new_coeff[2 * n - 2] = Cn
        new_coeff[2 * n - 1] = Cn_neg
        new_coeff_errors[2 * n - 2] = e_Cn
        new_coeff_errors[2 * n - 1] = e_Cn_neg

    return new_coeff, new_coeff_errors


def _change_exp_to_cos_sin(
    coeff: npt.NDArray,
    coeff_err: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Change FS from exp to cos-sin form.

    Change the parametrization of the Fourier series coefficients from
    the exp to the cos-sin form.

    Parameters
    ----------
    coeff : npt.NDArray[np.float64]
        Coefficients of the Fourier series in the exp form.
    coeff_err : npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients.

    Returns
    -------
    new_coeff: npt.NDArray[np.float64]
        Coefficients of the Fourier series in the cos-sin form.
    new_coeff_err: npt.NDArray[np.float64]
        Uncertainties of the Fourier series coefficients in the cos-sin form.

    """
    _check_coefficients_even(coeff)
    new_coeff = np.zeros_like(coeff, dtype=float)
    new_coeff_errors = np.zeros_like(coeff_err, dtype=float)

    coeff_and_errors_per_n = batched(zip(coeff, coeff_err, strict=True), 2)
    for n, ((Cn, e_Cn), (Cn_neg, _)) in enumerate(coeff_and_errors_per_n, 1):  # noqa: N806
        an = np.real(Cn + Cn_neg)
        bn = np.real(1j * (Cn - Cn_neg))

        e_an = 2 * np.real(e_Cn)
        e_bn = 2 * np.imag(e_Cn)

        new_coeff[2 * n - 2] = an
        new_coeff[2 * n - 1] = bn
        new_coeff_errors[2 * n - 2] = e_an
        new_coeff_errors[2 * n - 1] = e_bn

    return new_coeff, new_coeff_errors


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
