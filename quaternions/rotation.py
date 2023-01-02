import numpy as np
import pandas as pd

from quaternions.utils import AXES, ZERO_VECTOR, X, Y, Z


def transform_cartesian_to_spherical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion from cartesian coordinates :math:`(x, y, z)`
    to spherical coordinates :math:`(r, \\theta, \\phi)`\\.

    Obtains the **radius** :math:`r = \\sqrt{x^2 + y^2 + z^2}`\\,
    the **inclination** (polar angle, zenith, geographic latitude) :math:`\\theta` defined by the formula
    :math:`cos(\\theta) = \\frac{z}{r}` and the **azimuth** (geographic longitude) :math:`\\phi`
    defined by the formula :math:`tan(\\phi) = \\frac{y}{x}`\\.

    The convention here is to measure the inclination from north to south
    and the azimuth starting from the X axis, in counterclockwise (
    positive) orientation

    Parameters
    ----------
    data
        a data frame with 3 columns, containing cartesian coordinates

    Returns
    -------
    spherical_df
        a data frame containing radius, inclination and azimuth

    """  # noqa

    r = data.pow(2).sum(1).apply(np.sqrt)

    x, y, z = (data.iloc[:, c] for c in range(3))

    theta = np.arccos(z / r)

    phi = np.arctan2(y, x)

    return pd.DataFrame(
        {
            "r": r,
            "theta": theta,
            "phi": phi,
        },
        index=data.index,
    )


def transform_spherical_to_cartesian(data: pd.DataFrame) -> pd.DataFrame:
    """
    Conversion from spherical coordinates :math:`(r, \\theta, \\phi)`
    to cartesian coordinates :math:`(x, y, z)`\\.

    Obtains the cartesian coordinates :math:`(x, y, z)` by the formulae:

    :math:`\\begin{equation} \\begin{split}
    x & = & r sin(\\theta) cos(\\phi) \\\\
    y & = & r sin(\\theta) sin(\\phi) \\\\
    z & = & r cos(\\theta)
    \\end{split} \\end{equation}`

    See :func:`cartesian_to_spherical` for the angles conventions.

    Parameters
    ----------
    data
        a data frame with 3 columns, containing spherical coordinates

    Returns
    -------
    cartesian_df
        a data frame containing cartesian coordinates

    """  # noqa
    r, theta, phi = (data.iloc[:, c] for c in range(3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
        },
        index=data.index,
    )


def euler_rotation(axis: str, angle: float) -> np.ndarray:
    """Return the (extrinsic) Euler rotation matrix about
    the principal `axis` by `angle`.

    Parameters
    ----------
    axis
        One of the principal axes `X`, `Y`, `Z`.
        In aircraft terminology, the `Z` axis is aka `yaw` or `head`,
        the `Y` axis is aka `pitch` and the `X` axis is aka `roll`
    angle
        The angle of rotation

    Returns
    -------
    rot_mat
        The :math:`3 \\times 3` matrix
        representing the rotation
        about the `axis` by `angle`

    Raises
    ------
    ValueError
        if `axis` is not one of `AXES`

    """

    rotation = np.eye(3)

    if axis not in AXES:
        raise ValueError(f"{axis} must be one of {AXES}")

    # fmt: off
    if axis == X:
        rotation = np.array(
            [
                [            1.0,            0.0,            0.0 ], # noqa 
                [            0.0,  np.cos(angle), -np.sin(angle) ], # noqa 
                [            0.0,  np.sin(angle),  np.cos(angle) ], # noqa 
            ]
        )

    if axis == Y:
        rotation = np.array(
            [
                [  np.cos(angle),            0.0,  np.sin(angle) ], # noqa 
                [            0.0,            1.0,            0.0 ], # noqa 
                [ -np.sin(angle),            0.0,  np.cos(angle) ], # noqa 
            ]
        )

    if axis == Z:
        rotation = np.array(
            [
                [  np.cos(angle), -np.sin(angle),             0.0], # noqa 
                [  np.sin(angle),  np.cos(angle),             0.0], # noqa 
                [            0.0,            0.0,             1.0], # noqa 
            ]
        )
    # fmt: on

    return rotation


def cross_product_matrix(vector: np.ndarray) -> np.ndarray:
    """
    Get the skew-symmetric matrix for vector cross product operation.
    Derived from natural Lie algebra isomorphism :math:`R^{3} ~ so(3)`\\.

    Parameters
    ----------
    vector
        a 3 dim vector

    Returns
    -------
    Omega
        skew symmetric matrix

    """

    Omega = np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )

    return Omega


def rotation_axis_angle(axis: np.ndarray, phi: float) -> np.ndarray:
    """
    Calculate rotation matrix around given axis by given angle,
    in positive (counterclockwise) orientation.
    Also known as **Rodrigues' formula**:

    :math:`\\begin{equation}
    R = exp(\\phi \\Omega) \\\\
    R = id + sin(\\phi)\\Omega + (1 - cos(\\phi))\\Omega^2
    \\end{equation}`

    where :math:`\\Omega` is the skew-symmetric
    matrix in :math:`\\mathfrak{so}(3)` which corresponds to
    :math:`\\frac{\\vec{x}}{ \\|\\vec{x}\\|}`.
    See Also :func:`cross_product_matrix`

    Parameters
    ----------
    axis
        the rotation axis
    phi
        the rotation angle

    Returns
    -------
    R:
        Rotation matrix around axis :math:`\\frac{\\vec{axis}}
        { \\|\\vec{axis}\\|}` by angle :math:`\\phi`

    Raises
    ------
    ArithmeticError
        if the axis vector is ~ zero

    """  # noqa
    if np.allclose(axis, ZERO_VECTOR):
        raise ArithmeticError("Axis vector almost zero")

    axis = axis / np.sqrt(sum(axis**2))
    Omega = cross_product_matrix(axis)

    R = (
        np.identity(3)
        + np.sin(phi) * Omega
        + (1 - np.cos(phi)) * Omega @ Omega
    )

    return R
