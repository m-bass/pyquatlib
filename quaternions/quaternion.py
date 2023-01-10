"""
This is a custom implementation of quaternions.

Author: Marco Bosatta

-----------------------------------

Data structures that may be used to construct Quaternions
---------------------------------------------------------

SCALAR
VECTOR              list of  ;  np.array of
POINT               list of  ;  np.array of
PAIR                list of


    Type of quaternions, can be :class:`Quaternion` or a list of them

"""

import math
from typing import Any, List, Tuple, Union, cast

import numpy as np

from quaternions.types import T_PAIR, T_SCALAR, T_VECTOR
from quaternions.utils import (
    AXES,
    ZERO_VECTOR,
    euler_angles_from_rot_matrix,
    is_list,
    is_pair,
    is_pair_list,
    is_point,
    is_point_array,
    is_point_list,
    is_scalar,
    is_tuple,
    is_vector,
    is_vector_array,
    is_vector_list,
    matrix_from_point,
    mul_two_points,
    pair_to_point,
    rot_matrix_from_point,
)

ALL_QUAT = Union["Quaternion", List["Quaternion"]]

ALL_DQUAT = Union["DualQuaternion", List["DualQuaternion"]]

# map from numpy ufunc to binary operation
_UFUNC_TO_BINOP = {
    "add": lambda p, q: p + q,
    "subtract": lambda p, q: p - q,
    "multiply": lambda p, q: p * q,
    "divide": lambda p, q: p / q,
    # for backward compatibility:
    "true_divide": lambda p, q: p / q,
}


class Quaternion(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A quaternion is represented internally by a 4D `point`,
    with scalar part at index 0 and vector part at index 1 to 3:

    :math:`\\begin{equation}
    point = [s, x, y, z]
    \\end{equation}`

    """

    def __init__(self, obj: Any) -> None:
        """A Quaternion can be constructed:

        * from a scalar

        * from a (3D) vector

        * from a (4D) point

        * from a (scalar, vector) pair

        * from a quaternion

        The internal representation is stored in self.point

        """

        if is_scalar(obj):
            self._point = np.array([obj, 0, 0, 0])
        elif is_vector(obj):
            self._point = np.concatenate([[0], obj])
        elif is_point(obj):
            self._point = np.array(obj)
        elif is_pair(obj):
            self._point = np.array(pair_to_point(obj))
        elif is_quaternion(obj):
            self._point = obj.point
        else:
            raise TypeError(f"Cannot construct a Quaternion from {obj}")

    ########################################
    #    properties
    ########################################

    @property
    def point(self) -> np.ndarray:
        return self._point

    @property
    def scalar_part(self) -> T_SCALAR:
        return self.point[0]

    @property
    def vector_part(self) -> np.ndarray:
        return self.point[1:]

    @property
    def pair(self) -> T_PAIR:
        return self.scalar_part, self.vector_part

    @property
    def complex_matrix(self) -> np.ndarray:
        """ Quaternion representation as a complex matrix
        (in :math:`SU(2)` for unit quaternions):
        :math:`\\begin{equation}
        q = \\begin{bmatrix} s \\ x \\ y \\ z \\end{bmatrix}
        = \\begin{bmatrix} a_{1} \\ a_{2} \\ b_{1} \\ b_{2} \\end{bmatrix}
        \\longmapsto
        \\begin{pmatrix}
        \\alpha & - \\beta \\\\
        \\overline{\\beta} & \\overline{\\alpha}
        \\end{pmatrix}
        \\end{equation}`
        , where :math:`\\alpha = a_{1} + i a_{2}` and :math:`\\beta
        = b_{1} + i b_{2}`

        """
        return matrix_from_point(self.point)

    @property
    def is_scalar(self) -> bool:
        """
        Check if this quaternion has only a scalar part

        Returns
        -------
        is_vector
            True if the vector part vanishes

        """
        return np.allclose(self.vector_part, ZERO_VECTOR)

    @property
    def is_vector(self) -> bool:
        """
        Check if this quaternion has only a vector part

        Returns
        -------
        is_vector
            True if the scalar part vanishes

        """
        return math.isclose(self.scalar_part, 0.0)

    @property
    def is_unit(self) -> bool:
        """
        Check if this is a `unit` quaternion.
        Unit quaternions are isomorphic to the group :math:`SO(3)`
        of 3D rotations.

        Returns
        -------
        is_unit
            True if this is a `unit` quaternion, False otherwise

        """
        return math.isclose(self.squared_norm, 1.0)

    @property
    def is_zero(self) -> bool:
        return self.is_scalar and self.is_vector

    @property
    def conjugate(self) -> "Quaternion":
        """
        The conjugate quaternion

        Returns
        -------
        conj
            The conjugate quaternion

        """
        conj = Quaternion((self.scalar_part, -self.vector_part))
        return conj

    @property
    def inverse(self) -> "Quaternion":
        if self.is_zero:
            raise ArithmeticError("zero is not invertible")

        ip = pair_to_point((self.scalar_part, -self.vector_part))
        ip /= self.squared_norm

        return Quaternion(ip)

    @property
    def norm(self):
        return np.sqrt(self.squared_norm)

    @property
    def squared_norm(self):
        return sum(self.point**2)

    ##############################
    # OPERATOR OVERLOADING
    ##############################

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: ({self.pair})\n"

    def __str__(self) -> str:
        return f"({self.pair})"

    def __eq__(self, other: Any) -> bool:
        if not is_quaternion(other):
            return False

        return np.allclose(self.point, other.point)

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self):
        return hash(tuple(self.point))

    ##############################
    # unary ops
    ##############################

    # positive value: +q
    def __pos__(self) -> "Quaternion":
        return self

    # negative value: -q
    def __neg__(self) -> "Quaternion":
        return Quaternion(-self.point)

    # conjugate: ~q
    def __invert__(self) -> "Quaternion":
        return self.conjugate

    # absolute value abs(q)
    def __abs__(self) -> float:
        return self.norm

    ##############################
    # binary ops
    ##############################

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> ALL_QUAT:
        """
        Make sure that we can do binary arithmetic operations
        using compatible data structures as left operand and
        Quaternions as right operand.
        See Also :class:`numpy.lib.mixins.NDArrayOperatorsMixin`
        """

        lop = inputs[0]
        lop = self.to_quaternion(lop)

        binop = _UFUNC_TO_BINOP.get(ufunc.__name__)
        if binop is None:
            raise NotImplementedError(
                f"numpy ufunc {ufunc.__name__} is not " "supported"
            )

        if is_list(lop):
            result = [binop(o, self) for o in lop]
        else:
            result = binop(lop, self)

        return result

    # helper function for multiplication
    def _mult_2_quaternions(
        self, p: "Quaternion", q: "Quaternion"
    ) -> "Quaternion":
        point_prod = mul_two_points(p.point, q.point)

        return Quaternion(point_prod)

    def __add__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [Quaternion(self.point + o.point) for o in other]
        else:
            return Quaternion(self.point + other.point)

    def __iadd__(self, other: Any) -> ALL_QUAT:
        return self + other

    def __radd__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [o + self for o in other]
        else:
            return other + self

    def __sub__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [Quaternion(self.point - o.point) for o in other]
        else:
            return Quaternion(self.point - other.point)

    def __isub__(self, other: Any) -> ALL_QUAT:
        return self - other

    def __rsub__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [o - self for o in other]
        else:
            return other - self

    def __mul__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [self._mult_2_quaternions(self, o) for o in other]
        else:
            return self._mult_2_quaternions(self, other)

    def __imul__(self, other: Any) -> "Quaternion":
        return self * other

    def __rmul__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            return [o * self for o in other]
        else:
            return other * self

    def __truediv__(self, other: Any) -> ALL_QUAT:
        other = self.to_quaternion(other)

        if is_list(other):
            if any(o.is_zero for o in other):
                raise ArithmeticError("cannot divide by zero")
            return [self._mult_2_quaternions(self, o.inverse) for o in other]
        else:
            if other.is_zero:
                raise ArithmeticError("cannot divide by zero")
            return self._mult_2_quaternions(self, other.inverse)

    def __itruediv__(self, other: Any) -> "Quaternion":
        return self / other

    def __rtruediv__(self, other: Any) -> ALL_QUAT:
        if self.is_zero:
            raise ArithmeticError("cannot divide by zero")

        other = self.to_quaternion(other)

        if is_list(other):
            return [o / self for o in other]
        else:
            return other / self

    ########################################
    #    instance methods
    ########################################

    def group_conjugate(self, argument: Any) -> ALL_QUAT:
        """
        This is the conjugation operation from a group theoretic perspective.

        Parameters
        ----------
        argument
            a (list, array) of quaternions, scalars or vectors

        Returns
        -------
        result
            :math:`q \\cdot v \\cdot q^{-1}` of arguments v

        """
        result = self * argument * self.inverse
        return result

    def euler_angles(
        self, axis3: str, axis2: str, axis1: str
    ) -> Tuple[float, float, float]:
        """
        Return the 3 Euler angles corresponding to the factorisation
        of the rotation represented by this quaternion into
        :math:`R_{axis3} \\circ R_{axis2} \\circ R_{axis1}`

        Parameters
        ----------
        axis3
            The axis of the 3rd Euler rotation
        axis2
            The axis of the second Euler rotation
        axis1
            The axis of the first Euler rotation

        Returns
        -------
        angle3
            The angle of the Euler rotation around axis3
        angle2
            The angle of the Euler rotation around axis2
        angle1
            The angle of the Euler rotation around axis1

        Raises
        ------
        ValueError
            if the sequence of `axis3`, `axis2`, `axis1`
            is not a permutation of `x`, `y`, `z`
        ArithmeticError
            if this is not a unit quaternion

        """

        axes = [axis3, axis2, axis1]

        if not set(axes) == AXES:
            raise ValueError(
                f"{axis3}, {axis2}, {axis1} must be a permutation of {AXES}"
            )

        if not self.is_unit:
            raise ArithmeticError("This is not a unit quaternion")

        rm = self.rot_matrix()

        if axes == ["x", "y", "z"]:
            angles = euler_angles_from_rot_matrix(
                rm[0, 2],
                -rm[1, 2],
                rm[2, 2],
                -rm[0, 1],
                rm[0, 0],
                rm[1, 0],
                rm[1, 1],
                rm[1, 0],
                rm[1, 1],
                swap_sign_1=True,
            )

        elif axes == ["x", "z", "y"]:
            angles = euler_angles_from_rot_matrix(
                -rm[0, 1],
                rm[2, 1],
                rm[1, 1],
                rm[0, 2],
                rm[0, 0],
                -rm[2, 0],
                rm[2, 2],
                rm[2, 0],
                rm[2, 2],
            )

        elif axes == ["y", "x", "z"]:
            angles = euler_angles_from_rot_matrix(
                -rm[1, 2],
                rm[0, 2],
                rm[2, 2],
                rm[1, 0],
                rm[1, 1],
                -rm[0, 1],
                rm[0, 0],
                rm[0, 1],
                rm[0, 0],
            )

        elif axes == ["y", "z", "x"]:
            angles = euler_angles_from_rot_matrix(
                rm[1, 0],
                -rm[2, 0],
                rm[0, 0],
                -rm[1, 2],
                rm[1, 1],
                rm[2, 1],
                rm[2, 2],
                rm[2, 1],
                rm[2, 2],
                swap_sign_1=True,
            )

        elif axes == ["z", "x", "y"]:
            angles = euler_angles_from_rot_matrix(
                rm[2, 1],
                -rm[0, 1],
                rm[1, 1],
                -rm[2, 0],
                rm[2, 2],
                rm[0, 2],
                rm[0, 0],
                rm[0, 2],
                rm[0, 0],
                swap_sign_1=True,
            )

        elif axes == ["z", "y", "x"]:
            angles = euler_angles_from_rot_matrix(
                -rm[2, 0],
                rm[1, 0],
                rm[0, 0],
                rm[2, 1],
                rm[2, 2],
                -rm[0, 1],
                -rm[0, 2],
                rm[0, 1],
                rm[0, 2],
                swap_sign_2=True,
            )

        else:
            # we should never reach this code
            raise NotImplementedError()

        return angles

    def rot_matrix(self) -> np.ndarray:
        """
        For a unit quaternion, return the corresponding rotation matrix.

        Returns
        -------
        rot_matrix
            the rotation matrix

        Raises
        ------
        ArithmeticError
            if this is not a unit quaternion

        """
        if not self.is_unit:
            raise ArithmeticError("This is not a unit quaternion")

        return rot_matrix_from_point(self.point)

    def axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        For a unit quaternion, return the axis and angle of the
        corresponding rotation matrix.

        Returns
        -------
        axis, angle
            the normalized axis and the angle of rotation

        Raises
        ------
        ArithmeticError
            if this is not a unit quaternion

        """
        if not self.is_unit:
            raise ArithmeticError("This is not a unit quaternion")

        if self.is_scalar:
            # represents the identity element: angle is zero, and
            #    we could return any unit axis
            return np.array([1, 0, 0]), 0.0

        angle = 2 * np.arccos(self.scalar_part)
        axis = self.vector_part / np.sqrt(1 - self.scalar_part**2)
        return axis, angle

    ########################################
    #    class methods
    ########################################

    @classmethod
    def to_quaternion(cls, obj: Any) -> ALL_QUAT:
        """

        Parameters
        ----------
        obj

        Returns
        -------

        """

        # it's already of type `ALL_QUAT`
        if is_quaternion_type(obj):
            return obj

        # it's a list
        if (
            is_vector_list(obj)
            or is_vector_array(obj)
            or is_point_list(obj)
            or is_point_array(obj)
            or is_pair_list(obj)
        ):
            return [cls(p) for p in obj]

        # one element
        return cls(obj)

    @classmethod
    def rotation_quat_from_axis_angle(
        cls,
        axis: np.ndarray,
        angle: T_SCALAR,
    ) -> "Quaternion":
        """
        Create a rotation quaternion from axis and angle.
        The resulting unit quaternion :math:`r` defines a rotation
        around `axis` by `angle`.
        """

        if not is_vector(axis):
            raise ValueError(f"axis {axis} needs to be a vector")
        if not is_scalar(angle):
            raise ValueError(f"angle {angle} needs to be a scalar")
        if np.allclose(axis, ZERO_VECTOR):
            raise ArithmeticError(f"axis {axis} is almost zero")

        v = axis / np.sqrt(sum(a**2 for a in axis))

        s = np.cos(angle / 2)
        u = np.sin(angle / 2) * v

        return Quaternion((s, u))

    @classmethod
    def rotate_by_axis_angle(
        cls, axis: np.ndarray, angle: T_SCALAR, vectors: T_VECTOR
    ) -> T_VECTOR:
        """
        Rotate a bunch of vectors by axis, angle
        """

        if not (
            is_vector(vectors)
            or is_vector_list(vectors)
            or is_vector_array(vectors)
        ):
            raise ValueError("wrong data type for vectors")

        q = cls.rotation_quat_from_axis_angle(axis, angle)
        rotated = q.group_conjugate(vectors)

        if is_list(rotated):
            rotated = cast(List["Quaternion"], rotated)
            return [q.vector_part for q in rotated]  # type: ignore
        else:
            rotated = cast("Quaternion", rotated)
            return rotated.vector_part


def is_quaternion(o: Any) -> bool:
    return isinstance(o, Quaternion)


def is_quaternion_list(lst: Any) -> bool:
    return is_list(lst) and all(is_quaternion(e) for e in lst)


def is_quaternion_type(obj: Any) -> bool:
    return is_quaternion(obj) or is_quaternion_list(obj)


class DualQuaternion(np.lib.mixins.NDArrayOperatorsMixin):
    """
    A dual quaternion is represented internally by two quaternions
    `p` and `q`:

    :math:`\\begin{equation}
    \\sigma = p + \\epsilon q
    \\end{equation}`

    where :math:`\\epsilon` is the ``dual unit`` obeying the rule
    :math:`\\epsilon^2 = 0`.

    """

    def __init__(self, *args: Any) -> None:
        errmsg = f"Cannot construct a dual quaternion from {args}"

        if len(args) == 1:
            (t,) = args
            if not (is_tuple(t) and all(is_quaternion(a) for a in t)):
                raise TypeError(errmsg)
            self._quats = t

        elif len(args) == 2:
            if not all(is_quaternion(a) for a in args):
                raise TypeError(errmsg)
            self._quats = tuple(args)

        else:
            raise TypeError(errmsg)

    ########################################
    #    properties
    ########################################

    @property
    def quats(self) -> Tuple["Quaternion", "Quaternion"]:
        return self._quats

    @property
    def is_zero(self) -> bool:
        return all(x.is_zero for x in self.quats)

    @property
    def conjugate(self) -> "DualQuaternion":
        """
        The conjugate dual quaternion
        :math:`\\sigma^* = p^* + \\epsilon q^*`

        Returns
        -------
        conj
            The conjugate dual quaternion

        """
        p, q = self.quats
        return DualQuaternion(p.conjugate, q.conjugate)  # type:ignore

    @property
    def inverse(self) -> "DualQuaternion":
        """The inverse of :math:`\\sigma = p + \\epsilon q`
        exists if :math:`p` is invertible and
        is :math:`\\sigma^{-1} = p^{-1} - \\epsilon p^{-1} q p^{-1}`
        """

        p, q = self.quats

        if p.is_zero:
            raise ArithmeticError(f"{self} is not invertible")

        ip = p.inverse
        iq = -ip * q * ip

        return DualQuaternion(ip, iq)  # type:ignore

    @property
    def double_conjugate(self) -> "DualQuaternion":
        """
        The double conjugate dual quaternion
        :math:`\\sigma^* = p^* - \\epsilon q^*`

        Returns
        -------
        conj
            The double conjugate dual quaternion

        """
        p, q = self.quats
        return DualQuaternion(p.conjugate, -q.conjugate)  # type:ignore

    @property
    def is_unit(self) -> bool:
        norm = self * self.conjugate
        p, q = norm.quats  # type:ignore
        return p == Quaternion(1) and q.is_zero

    ##############################
    # OPERATOR OVERLOADING
    ##############################

    def __repr__(self) -> str:
        p, q = self.quats
        return f"{self.__class__.__name__}: ({p}; {q})\n"

    def __str__(self) -> str:
        p, q = self.quats
        return f"({p}; {q})"

    def __eq__(self, other: Any) -> bool:
        if not is_dual_quaternion(other):
            return False

        return all(x == y for x, y in zip(self.quats, other.quats))

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self):
        return hash(self.quats)

    ##############################
    # binary ops
    ##############################

    # helper function for multiplication
    def _mult_2_dual_quaternions(
        self, a: "DualQuaternion", b: "DualQuaternion"
    ) -> "DualQuaternion":
        p, q = a.quats
        po, qo = b.quats
        return DualQuaternion(p * po, p * qo + q * po)  # type:ignore

    def __add__(self, other: ALL_DQUAT) -> ALL_DQUAT:
        if is_list(other):
            other = cast(List["DualQuaternion"], other)
            return [
                DualQuaternion(  # type:ignore
                    tuple(x + y for x, y in zip(self.quats, o.quats))
                )
                for o in other
            ]
        else:
            return DualQuaternion(  # type:ignore
                tuple(
                    x + y
                    for x, y in zip(self.quats, other.quats)  # type:ignore
                )
            )

    def __sub__(self, other: ALL_DQUAT) -> ALL_DQUAT:
        if is_list(other):
            other = cast(List["DualQuaternion"], other)
            return [
                DualQuaternion(  # type:ignore
                    tuple(x - y for x, y in zip(self.quats, o.quats))
                )
                for o in other
            ]
        else:
            return DualQuaternion(
                tuple(
                    x - y
                    for x, y in zip(self.quats, other.quats)  # type:ignore
                )
            )

    def __mul__(self, other: ALL_DQUAT) -> ALL_DQUAT:
        if is_list(other):
            other = cast(List["DualQuaternion"], other)
            return [self._mult_2_dual_quaternions(self, o) for o in other]
        else:
            other = cast("DualQuaternion", other)
            return self._mult_2_dual_quaternions(self, other)

    def __truediv__(self, other: ALL_DQUAT) -> ALL_DQUAT:
        if is_list(other):
            other = cast(List["DualQuaternion"], other)
            if any(o.is_zero for o in other):
                raise ArithmeticError("cannot divide by zero")
            return [
                self._mult_2_dual_quaternions(self, o.inverse) for o in other
            ]
        else:
            other = cast("DualQuaternion", other)
            if other.is_zero:
                raise ArithmeticError("cannot divide by zero")
            return self._mult_2_dual_quaternions(self, other.inverse)

    ########################################
    #    class methods
    ########################################

    @classmethod
    def rigid_displacement(
        cls,
        axis: np.ndarray,
        angle: float,
        translation: T_VECTOR,
        rotate_first=True,
    ) -> "DualQuaternion":
        """
        Create a unit dual quaternion representing a rigid displacement:
        rotation around `axis` by `angle` and `translation`.
        If `rotate_first`, rotation happens before translation, otherwise,
        translation before rotation.
        """

        r = Quaternion.rotation_quat_from_axis_angle(axis, angle)

        if not is_vector(translation):
            raise ValueError(f"translation {translation} needs to be a vector")

        t = np.array(translation) / 2
        if rotate_first:
            q = t * r
        else:
            q = r * t

        return DualQuaternion(r, q)  # type: ignore


def is_dual_quaternion(o: Any) -> bool:
    return isinstance(o, DualQuaternion)


def is_dual_quaternion_list(lst: Any) -> bool:
    return is_list(lst) and all(is_dual_quaternion(e) for e in lst)


def is_dual_quaternion_type(o: Any) -> bool:
    return is_dual_quaternion(o) or is_dual_quaternion_list(o)
