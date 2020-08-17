import logging
import math
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from numba import jit

from quaternions.types import POINT, SCALAR, T_PAIR, T_POINT, VECTOR

ZERO_VECTOR = np.zeros(3)

LOGGER = logging.getLogger("quaternions")


@jit
def mul_two_points(p: T_POINT, q: T_POINT) -> T_POINT:
    """ This is the quaternion multproduct of two 4D `T_POINT`s

    Parameters
    ----------
    p
        left (4D) point
    q
        right (4D) point

    Returns
    -------
    prod
        the quaternion product of p and q

    """

    s = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    x = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2]
    y = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1]
    z = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]

    return np.array([s, x, y, z])


@jit
def rot_matrix_from_point(q: T_POINT) -> np.ndarray:
    """ Yields the rotation matrix for a point representing a unit quaternion

    Parameters
    ----------
    q
        (4D) point

    Returns
    -------
    mat
        the rotation matrix 

    """

    w, x, y, z = q

    row0 = [
        w ** 2 + x ** 2 - y ** 2 - z ** 2,
        2 * (x * y - w * z),
        2 * (x * z + w * y),
    ]
    row1 = [
        2 * (x * y + w * z),
        w ** 2 - x ** 2 + y ** 2 - z ** 2,
        2 * (y * z - w * x),
    ]
    row2 = [
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        w ** 2 - x ** 2 - y ** 2 + z ** 2,
    ]

    return np.array([row0, row1, row2])


def matrix_from_point(p: T_POINT) -> np.ndarray:
    matrix = np.array(
        [[p[0] + p[1] * 1j, -p[2] - p[3] * 1j], [p[2] - p[3] * 1j, p[0] - p[1] * 1j]]
    )
    return matrix


def is_list(o: Any) -> bool:
    return isinstance(o, List)


def is_tuple(o: Any) -> bool:
    return isinstance(o, Tuple)


def _is_2d_array(a: Any) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2


def _is_vector_or_point(o: Any, t: Any, dim: int) -> bool:
    if not isinstance(o, t):
        return False
    if isinstance(o, np.ndarray) and o.ndim != 1:
        return False
    if len(o) != dim:
        return False

    return all(is_scalar(e) for e in o)


def is_scalar(o: Any) -> bool:
    return isinstance(o, SCALAR)


def is_vector(o: Any) -> bool:
    return _is_vector_or_point(o, VECTOR, 3)


def is_point(o: Any) -> bool:
    return _is_vector_or_point(o, POINT, 4)


def is_pair(p: Any) -> bool:
    if not (is_tuple(p) and len(p) == 2):
        return False
    s, v = p
    return is_scalar(s) and is_vector(v)


def is_vector_array(a: Any) -> bool:
    if not (_is_2d_array(a) and a.shape[-1] == 3):
        return False
    return all(is_vector(e) for e in a)


def is_point_array(a: Any) -> bool:
    if not (_is_2d_array(a) and a.shape[-1] == 4):
        return False
    return all(is_point(e) for e in a)


def is_vector_list(l: Any) -> bool:
    return is_list(l) and all(is_vector(e) for e in l)


def is_point_list(l: Any) -> bool:
    return is_list(l) and all(is_point(e) for e in l)


def is_pair_list(l: Any) -> bool:
    return is_list(l) and all(is_pair(e) for e in l)


def pair_to_point(pair: T_PAIR) -> T_POINT:
    s, v = pair
    return [s] + list(v)


def euler_arcsin_numerical_stability(rot_mat_elem: float) -> None:
    sign = np.sign(rot_mat_elem)

    if math.isclose(rot_mat_elem, sign, rel_tol=1e-7):
        LOGGER.warning(
            f"setting rotation matrix element "
            "{rot_mat_elem} to {sign} for numerical "
            "stability of arcsin"
        )
        rot_mat_elem = sign

    return rot_mat_elem


def euler_angles_from_rot_matrix(
    *args: float, swap_sign_1: bool = False, swap_sign_2: bool = False
) -> Tuple[float, float, float]:
    """
    From Philip J. Schneider, David H. Eberly:
    Geometric Tools for Computer Graphics.
    2003, Morgan Kaufmann Publishers
    """
    sin_angle2 = euler_arcsin_numerical_stability(args[0])
    angle2 = np.arcsin(sin_angle2)

    if angle2 < np.pi / 2:
        if angle2 > -np.pi / 2:
            angle3 = np.arctan2(args[1], args[2])
            angle1 = np.arctan2(args[3], args[4])
        else:
            # solution not unique
            angle3 = np.arctan2(args[5], args[6])
            if swap_sign_1:
                angle3 = -angle3
            angle1 = 0.0
    else:
        # solution not unique
        angle3 = np.arctan2(args[7], args[8])
        if swap_sign_2:
            angle3 = -angle3
        angle1 = 0.0

    return angle3, angle2, angle1
