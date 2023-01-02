import numpy as np
import pytest

from quaternions import Quaternion
from quaternions.rotation import euler_rotation, rotation_axis_angle
from quaternions.utils import is_pair, is_point, is_scalar, is_vector

########################################
#    constructor
########################################


@pytest.mark.parametrize(
    "obj, expected",
    [
        # scalar
        (-1.5, Quaternion([-1.5, 0, 0, 0])),
        (5, Quaternion([5, 0, 0, 0])),
        # vector
        (([0.4, 1, 2.5]), Quaternion([0, 0.4, 1.0, 2.5])),
        (np.array([1, -2, 3]), Quaternion([0, 1, -2, 3])),
        # point
        ([-3, 0, 2, 0], Quaternion([-3, 0, 2, 0])),
        (np.array([1.2, 1, 0, -0.6]), Quaternion([1.2, 1, 0, -0.6])),
        # pair
        ((9, [1, 2, 3]), Quaternion([9, 1, 2, 3])),
        ((0, np.array([1, 2, 3])), Quaternion([0, 1, 2, 3])),
        # quaternion
        (Quaternion(-2.5), Quaternion(-2.5)),
        (Quaternion([-1, 2, -3, 4]), Quaternion([-1, 2, -3, 4])),
    ],
)
def test_quaternion_init(obj, expected):
    result = Quaternion(obj)

    assert result == expected


@pytest.mark.parametrize(
    "obj",
    [
        "0.5",
        [0, 1, 2, 3, 4],
        list("letters"),
    ],
)
def test_quaternion_init_error(obj):
    with pytest.raises(TypeError):
        Quaternion(obj)


########################################
#    properties
########################################


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(2), [2, 0, 0, 0]),
        (Quaternion([1, 2, 3]), [0, 1, 2, 3]),
        (Quaternion(-np.array([1, 2, 3, 4])), [-1, -2, -3, -4]),
        (Quaternion((9, [4, 5, 6])), np.array([9, 4, 5, 6])),
    ],
)
def test_point(quat, expected):
    point = quat.point

    assert is_point(point)
    assert all(x == y for x, y in zip(point, expected))


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(0), 0.0),
        (Quaternion([1, 2, 3]), 0.0),
        (Quaternion([-2.5, 1, 2, 3]), -2.5),
        (Quaternion((1, [5, 6, 7])), 1.0),
    ],
)
def test_scalar_part(quat, expected):
    scalar = quat.scalar_part

    assert is_scalar(scalar)
    assert scalar == expected


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(5), [0, 0, 0]),
        (Quaternion([1, 2, 3]), np.array([1, 2, 3])),
        (Quaternion([0, -1, 2, 3]), [-1, 2, 3]),
        (Quaternion((0, [-7, -8, -9])), -np.array([7, 8, 9])),
    ],
)
def test_vector_part(quat, expected):
    vector = quat.vector_part

    assert is_vector(vector)
    assert all(x == y for x, y in zip(vector, expected))


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(0), (0, [0, 0, 0])),
        (Quaternion(np.array([4, 9, 2])), (0.0, [4.0, 9, 2])),
        (Quaternion([1, 2, 3, 4]), (1, np.array([2, 3, 4]))),
        (Quaternion((-1, [1, -2, 3])), (-1.0, [1, -2, 3])),
    ],
)
def test_pair(quat, expected):
    pair = quat.pair

    assert is_pair(pair)
    s, v = pair
    es, ev = expected
    assert s == es
    assert all(x == y for x, y in zip(v, ev))


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(0), np.zeros([2, 2])),
        (Quaternion(1), np.eye(2)),
        (Quaternion([1, 2, 3]), np.array([[1j, -2 - 3j], [2 - 3j, -1j]])),
        (
            Quaternion([5, -1, 2, 3]),
            np.array([[5 - 1j, -2 - 3j], [2 - 3j, 5 + 1j]]),
        ),
        (
            Quaternion((4, [3, 2, 1])),
            np.array([[4 + 3j, -2 - 1j], [2 - 1j, 4 - 3j]]),
        ),
    ],
)
def test_complex_matrix(quat, expected):
    matrix = quat.complex_matrix

    assert np.allclose(matrix, expected)


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(-3), True),
        (Quaternion([1, 2, 3]), False),
        (Quaternion(np.array([0, 0, 0])), True),
        (Quaternion([1, 2, 3, 4]), False),
        (Quaternion([1, 0, 0, 0]), True),
        (Quaternion((-1, [1, 2, 3])), False),
        (Quaternion((-1, [0.0, 0.0, 0.0])), True),
    ],
)
def test_is_scalar(quat, expected):
    s = quat.is_scalar

    assert s is expected


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(8), False),
        (Quaternion(0), True),
        (Quaternion([1, 2, 3]), True),
        (Quaternion([1, 2, 3, 4]), False),
        (Quaternion([0, 1, 2, 3]), True),
        (Quaternion((-1, [1, 2, 3])), False),
        (Quaternion((0.0, [1.0, 2.0, 3.0])), True),
    ],
)
def test_is_vector(quat, expected):
    v = quat.is_vector

    assert v is expected


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(1), False),
        (Quaternion(0), True),
        (Quaternion([1, 2, 3]), False),
        (Quaternion([0, 0, 0]), True),
        (Quaternion([1, 2, 3, 4]), False),
        (Quaternion([0, 0, 0, 0]), True),
        (Quaternion((-1, [1, 2, 3])), False),
        (Quaternion((0.0, [0, 0, 0])), True),
    ],
)
def test_is_zero(quat, expected):
    z = quat.is_zero

    assert z is expected


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(1), True),
        (Quaternion(-1), True),
        (Quaternion(0), False),
        (Quaternion([1, 2, 3]), False),
        (Quaternion([0, -1, 0]), True),
        (Quaternion([1, 2, 3, 4]), False),
        (Quaternion([0, 1, 0, 0]), True),
        (Quaternion((-1, [1, 2, 3])), False),
        (Quaternion((-1 / 2, [-1 / 2, 1 / 2, 1 / 2])), True),
    ],
)
def test_is_unit(quat, expected):
    result = quat.is_unit

    assert result is expected


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(0), Quaternion(0)),
        (Quaternion(-2), Quaternion(-2.0)),
        (Quaternion([0, 0, 0]), Quaternion(0)),
        (Quaternion([1, 2, 3]), Quaternion(-np.array([1, 2, 3]))),
        (Quaternion(np.array([0, 0, 0, 0])), Quaternion(0)),
        (Quaternion([1, 2, 3, 4]), Quaternion([1, -2, -3, -4])),
        (Quaternion((0.0, [0, 0, 0])), Quaternion(0)),
        (Quaternion((1, [-2, -3, -4])), Quaternion([1, 2, 3, 4])),
    ],
)
def test_conjugate(quat, expected):
    c = quat.conjugate

    assert c == expected


def test_inverse_error():
    quat = Quaternion(0.0)
    with pytest.raises(ArithmeticError):
        quat.inverse


@pytest.mark.parametrize(
    "quat, expected",
    [
        (Quaternion(2), Quaternion(0.5)),
        (Quaternion([1, 2, 3]), Quaternion(-np.array([1, 2, 3]) / 14.0)),
        (Quaternion([1, 2, 3, 4]), Quaternion(np.array([1, -2, -3, -4]) / 30)),
        (
            Quaternion((1, [-2, -3, -4])),
            Quaternion(np.array([1, 2, 3, 4]) / 30.0),
        ),
    ],
)
def test_inverse(quat, expected):
    i = quat.inverse

    assert i == expected


@pytest.mark.parametrize(
    "quat, norm, squared_norm",
    [
        (Quaternion(0), 0.0, 0.0),
        (Quaternion(-2), 2.0, 4),
        (Quaternion([0, 0, 0]), 0, 0),
        (Quaternion([1, 2, 3]), np.sqrt(14), 14.0),
        (Quaternion(np.array([0, 0, 0, 0])), 0.0, 0),
        (Quaternion([1, 2, 3, 4]), np.sqrt(30), 30),
        (Quaternion((0.0, [0, 0, 0])), 0, 0),
        (Quaternion((1, [-2, -3, -4])), np.sqrt(30), 30),
    ],
)
def test_norms(quat, norm, squared_norm):
    n = quat.norm
    sn = quat.squared_norm

    assert n == pytest.approx(norm, abs=1e-8)
    assert sn == pytest.approx(squared_norm, abs=1e-8)


##############################
# OPERATOR OVERLOADING
##############################


@pytest.mark.parametrize(
    "p, q",
    [
        (Quaternion(0), Quaternion(0)),
        (Quaternion([1, 2, 3]), Quaternion(np.array([1, 2, 3]))),
        (Quaternion([0, 1, 2, 3]), Quaternion(np.array([0, 1, 2, 3]))),
        (Quaternion((0, [1, 2, 3])), Quaternion((0, np.array([1, 2, 3])))),
    ],
)
def test_eq(p, q):
    assert p == q


@pytest.mark.parametrize(
    "p, q",
    [
        (Quaternion(2), Quaternion(-3)),
        (Quaternion([-2, 2, -2]), Quaternion(np.array([1, 2, 3]))),
        (Quaternion([-6, 3, 2, 3]), Quaternion(np.array([0, 1, 2, 3]))),
        (Quaternion((0, [-1, -2, 3])), Quaternion((0, np.array([1, 2, 3])))),
    ],
)
def test_ne(p, q):
    assert p != q


@pytest.fixture(
    scope="function",
    params=[
        # p
        # expected +p
        # expected -p
        # expected ~p
        # expected abs(p)
        (
            Quaternion(0),
            Quaternion(0),
            Quaternion(0),
            Quaternion(0),
            0,
        ),
        (
            Quaternion(-1.2),
            Quaternion(-1.2),
            Quaternion(1.2),
            Quaternion(-1.2),
            1.2,
        ),
        (
            Quaternion([1, 1, 0]),
            Quaternion([1, 1, 0]),
            Quaternion([-1, -1, 0]),
            Quaternion([-1, -1, 0]),
            np.sqrt(2),
        ),
        (
            Quaternion([-1, 1, -1, 1]),
            Quaternion([-1, 1, -1, 1]),
            Quaternion([1, -1, 1, -1]),
            Quaternion([-1, -1, 1, -1]),
            2.0,
        ),
    ],
)
def fxt_quat_unary(request):
    return request.param


@pytest.fixture(
    scope="function",
    params=[
        # p,  q
        # expected p + q
        # expected p - q
        # expected p * q
        # expected p / q
        (
            Quaternion(3),
            -1.5,
            Quaternion(1.5),
            Quaternion(4.5),
            Quaternion(-4.5),
            Quaternion(-2),
        ),
        (
            Quaternion(2),
            np.array([1, 0, 0]),
            Quaternion([2, 1, 0, 0]),
            Quaternion([2, -1, 0, 0]),
            Quaternion([2, 0, 0]),
            Quaternion((0, [-2, 0, 0])),
        ),
        (
            Quaternion(np.array([1, 0, 0])),
            Quaternion(np.array([0, 1, 0])),
            Quaternion(np.array([1, 1, 0])),
            Quaternion(np.array([1, -1, 0])),
            Quaternion(np.array([0, 0, 1])),
            Quaternion(np.array([0, 0, -1])),
        ),
        (
            Quaternion(3),
            np.array([1, 1, 1]),
            Quaternion([3, 1, 1, 1]),
            Quaternion([3, -1, -1, -1]),
            Quaternion(3 * np.array([1, 1, 1])),
            Quaternion(-np.array([1, 1, 1])),
        ),
        (
            Quaternion(np.array([1, 0, 1, 0])),
            [2, 0, 0, 1],
            Quaternion(np.array([3, 0, 1, 1])),
            Quaternion(np.array([-1, 0, 1, -1])),
            Quaternion(np.array([2, 1, 2, 1])),
            Quaternion(np.array([2, -1, 2, -1]) / 5),
        ),
        (
            Quaternion([-1, 1, 1, 0]),
            [0, 0, 1, 1],
            Quaternion(np.array([-1, 1, 2, 1])),
            Quaternion(np.array([-1, 1, 0, -1])),
            Quaternion(np.array([-1, 1, -2, 0])),
            Quaternion(np.array([1, -1, 2, 0]) / 2),
        ),
    ],
)
def fxt_bin_ops(request):
    return request.param


fxt_quat = [
    Quaternion(1.0),
    Quaternion(np.array([1, 2, 3])),
    Quaternion([1, 2, 3, 4]),
    Quaternion((-1, [-1, 0, 1])),
]

fxt_quat_struct = [
    1,
    [1, 0, 1],
    np.array([-1, 0, -2]),
    [-1, 2, -1, 0],
    np.array([-1, 0, -1, -2]),
    (0, [1, 2, 3]),
    (0, np.array([-1, 0, 0])),
]

fxt_quat_struct_list = [
    [
        [0, 1, -1],
        [-1, 1, -1],
        [2.3, 0.2, 1.5],
    ],  # list of vectors
    np.array(
        [
            [1, 2, 3],
            [-1, 0, 2],
            [-2, 1, -1],
        ]
    ),  # array of vectors
    [
        [5, 2, 1, 0],
        [0, -1, 1, -1],
        [1.1, 2.2, 1.5, -4.5],
    ],  # list of points
    np.array(
        [
            [0, 1.1, 0, 3],
            [0.2, 1.0, 0, 2],
            [-1.5, 1, 3, 1],
        ]
    ),  # array of points
    [
        (1, np.array([1, 2, 3])),
        (0, [-1, -2, 0]),
        (-1, [2, 0, -2]),
    ],  # list of pairs
    [  # list of quaternions
        Quaternion(-3.0),
        Quaternion([0, 1, 0]),
        Quaternion(np.array([2, -1, 0])),
        Quaternion([5, 2, -1, 2]),
        Quaternion(np.array([-1, 0, 2, 2])),
        Quaternion((0, [1, 2, 3])),
    ],
]


##############################
# unary ops
##############################


def test_quaternion_pos(fxt_quat_unary):
    p, expected, *_ = fxt_quat_unary
    result = +p
    assert result == expected


def test_quaternion_neg(fxt_quat_unary):
    p, _, expected, *_ = fxt_quat_unary
    result = -p
    assert result == expected


def test_quaternion_invert(fxt_quat_unary):
    p, *_, expected, _ = fxt_quat_unary
    result = ~p
    assert result == expected


def test_quaternion_abs(fxt_quat_unary):
    p, *_, expected = fxt_quat_unary
    result = abs(p)
    assert result == pytest.approx(expected)


##############################
# binary ops
##############################


def test_quaternion_add(fxt_bin_ops):
    p, q, expected, *_ = fxt_bin_ops
    result = p + q
    assert expected == result


def test_quaternion_sub(fxt_bin_ops):
    p, q, _, expected, *_ = fxt_bin_ops
    result = p - q
    assert expected == result


def test_quaternion_mul(fxt_bin_ops):
    p, q, *_, expected, _ = fxt_bin_ops
    result = p * q
    assert expected == result


def test_quaternion_truediv(fxt_bin_ops):
    p, q, *_, expected = fxt_bin_ops
    result = p / q
    assert expected == result


def test_quaternion_iadd(fxt_bin_ops):
    p, q, expected, *_ = fxt_bin_ops
    p += q
    assert p == expected


def test_quaternion_isub(fxt_bin_ops):
    p, q, _, expected, *_ = fxt_bin_ops
    p -= q
    assert p == expected


def test_quaternion_imul(fxt_bin_ops):
    p, q, *_, expected, _ = fxt_bin_ops
    p *= q
    assert p == expected


def test_quaternion_itruediv(fxt_bin_ops):
    p, q, *_, expected = fxt_bin_ops
    p /= q
    assert p == expected


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_add_struct(p, struct):
    result = p + struct
    assert all(r - Quaternion(q) == p for r, q in zip(result, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_sub_struct(p, struct):
    result = p - struct
    assert all(r + Quaternion(q) == p for r, q in zip(result, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_mul_struct(p, struct):
    result = p * struct
    assert all(r / Quaternion(q) == p for r, q in zip(result, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_truediv_struct(p, struct):
    result = p / struct
    assert all(r * Quaternion(q) == p for r, q in zip(result, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_iadd_struct(p, struct):
    before = p
    p += struct
    assert all(r - Quaternion(q) == before for r, q in zip(p, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_isub_struct(p, struct):
    before = p
    p -= struct
    assert all(r + Quaternion(q) == before for r, q in zip(p, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_imul_struct(p, struct):
    before = p
    p *= struct
    assert all(r / Quaternion(q) == before for r, q in zip(p, struct))


@pytest.mark.parametrize("p", fxt_quat)
@pytest.mark.parametrize("struct", fxt_quat_struct_list)
def test_quaternion_itruediv_struct(p, struct):
    before = p
    p /= struct
    assert all(r * Quaternion(q) == before for r, q in zip(p, struct))


@pytest.mark.parametrize("p", fxt_quat_struct)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_radd(p, q):
    result = p + q
    assert result == Quaternion(p) + q


@pytest.mark.parametrize("p", fxt_quat_struct)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rsub(p, q):
    result = p - q
    assert result == Quaternion(p) - q


@pytest.mark.parametrize("p", fxt_quat_struct)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rmul(p, q):
    result = p * q
    assert result == Quaternion(p) * q


@pytest.mark.parametrize("p", fxt_quat_struct)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rtruediv(p, q):
    result = p / q
    assert result == Quaternion(p) / q


@pytest.mark.parametrize("struct", fxt_quat_struct_list)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_radd_struct(struct, q):
    result = struct + q
    assert all(r - q == Quaternion(p) for r, p in zip(result, struct))


@pytest.mark.parametrize("struct", fxt_quat_struct_list)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rsub_struct(struct, q):
    result = struct - q
    assert all(r + q == Quaternion(p) for r, p in zip(result, struct))


@pytest.mark.parametrize("struct", fxt_quat_struct_list)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rmul_struct(struct, q):
    result = struct * q
    assert all(r / q == Quaternion(p) for r, p in zip(result, struct))


@pytest.mark.parametrize("struct", fxt_quat_struct_list)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_rtruediv_struct(struct, q):
    result = struct / q
    assert all(r * q == Quaternion(p) for r, p in zip(result, struct))


########################################
#    instance methods
########################################


@pytest.mark.parametrize("arg", fxt_quat_struct)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_group_conjugate(q, arg):
    expected = q * arg * q.inverse
    result = q.group_conjugate(arg)
    assert result == expected


@pytest.mark.parametrize("arg", fxt_quat_struct_list)
@pytest.mark.parametrize("q", fxt_quat)
def test_quaternion_group_conjugate_struct(q, arg):
    expected = q * arg * q.inverse
    result = q.group_conjugate(arg)
    assert all(r == e for r, e in zip(result, expected))


# 'AXphi' means rotation of angle phi around axis AX
UNIT_QUATERNION = {
    "x90": np.sqrt(2) / 2 * Quaternion([1, 1, 0, 0]),
    "y90": np.sqrt(2) / 2 * Quaternion([1, 0, 1, 0]),
    "z90": np.sqrt(2) / 2 * Quaternion([1, 0, 0, 1]),
}
NON_UNIT_QUATERNION = [
    Quaternion(0),
    Quaternion(-9),
    Quaternion([1, 2, 3]),
    Quaternion([1, 2, 3, 4]),
    Quaternion((-1, [3, 2, 1])),
]


EULER_SEQUENCE = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]


@pytest.mark.parametrize("q", NON_UNIT_QUATERNION)
def test_quaternion_axis_angle_fails(q):
    with pytest.raises(ArithmeticError):
        q.axis_angle()


@pytest.mark.parametrize(
    "u, exp_axis, exp_angle",
    [
        (
            Quaternion(1),
            np.array([1, 0, 0]),
            0,
        ),
        (
            1 / 2 * Quaternion([1, 1, 1, 1]),
            np.array([1, 1, 1]) / np.sqrt(3),
            2 / 3 * np.pi,
        ),
        (
            UNIT_QUATERNION["x90"],
            np.array([1, 0, 0]),
            1 / 2 * np.pi,
        ),
        (
            UNIT_QUATERNION["y90"],
            np.array([0, 1, 0]),
            1 / 2 * np.pi,
        ),
        (
            UNIT_QUATERNION["z90"],
            np.array([0, 0, 1]),
            1 / 2 * np.pi,
        ),
        (
            Quaternion((np.sqrt(3) / 2, np.array([1, 1, 0]) / 2 / np.sqrt(2))),
            np.array([1, 1, 0]) / np.sqrt(2),
            1 / 3 * np.pi,
        ),
    ],
)
def test_quaternion_axis_angle(u, exp_axis, exp_angle):
    axis, angle = u.axis_angle()

    assert np.allclose(axis, exp_axis)
    assert np.allclose(angle, exp_angle)


@pytest.mark.parametrize("q", NON_UNIT_QUATERNION)
def test_quaternion_rot_matrix_fails(q):
    with pytest.raises(ArithmeticError):
        q.rot_matrix()


@pytest.mark.parametrize(
    "quat, exp",
    [
        (Quaternion(1), np.eye(3)),
        (
            UNIT_QUATERNION["x90"],
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        ),
        (
            UNIT_QUATERNION["y90"],
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        ),
        (
            UNIT_QUATERNION["z90"],
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        ),
        (
            1 / 2 * Quaternion([1, 1, 1, 1]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ),
    ],
)
def test_quaternion_rot_matrix(quat, exp):
    result = quat.rot_matrix()

    print(result)

    assert np.allclose(result, exp)


@pytest.mark.parametrize("q", NON_UNIT_QUATERNION)
def test_quaternion_euler_angles_fails_unit(q):
    with pytest.raises(ArithmeticError):
        q.euler_angles("x", "y", "z")


@pytest.mark.parametrize(
    "axes",
    [
        "abc",
        "XYZ",
        "xxy",
        "yzy",
    ],
)
def test_quaternion_euler_angles_fails_axes(axes):
    q = Quaternion(1)

    with pytest.raises(ValueError):
        q.euler_angles(*list(axes))


@pytest.mark.parametrize("axes", EULER_SEQUENCE)
def test_quaternion_euler_angles_identity(axes):
    result = Quaternion(1).euler_angles(*list(axes))
    assert result == (0, 0, 0)


@pytest.mark.parametrize(
    "quat, axes, expected",
    [
        (UNIT_QUATERNION["x90"], "xyz", (np.pi / 2, 0, 0)),
        (UNIT_QUATERNION["y90"], "xyz", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["z90"], "xyz", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["x90"], "xzy", (np.pi / 2, 0, 0)),
        (UNIT_QUATERNION["y90"], "xzy", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["z90"], "xzy", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["y90"], "yxz", (np.pi / 2, 0, 0)),
        (UNIT_QUATERNION["x90"], "yxz", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["z90"], "yxz", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["x90"], "yzx", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["y90"], "yzx", (np.pi / 2, 0, 0)),
        (UNIT_QUATERNION["z90"], "yzx", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["x90"], "zxy", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["y90"], "zxy", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["z90"], "zxy", (np.pi / 2, 0, 0)),
        (UNIT_QUATERNION["x90"], "zyx", (0, 0, np.pi / 2)),
        (UNIT_QUATERNION["y90"], "zyx", (0, np.pi / 2, 0)),
        (UNIT_QUATERNION["z90"], "zyx", (np.pi / 2, 0, 0)),
    ],
)
def test_quaternion_euler_angles(quat, axes, expected):
    result = quat.euler_angles(*list(axes))
    assert result == expected


@pytest.mark.parametrize("quat", list(UNIT_QUATERNION.values()))
@pytest.mark.parametrize("axes", EULER_SEQUENCE)
def test_quaternion_euler_angles_euler_rotations(quat, axes):
    axis3, axis2, axis1 = list(axes)
    angle3, angle2, angle1 = quat.euler_angles(axis3, axis2, axis1)
    composition = (
        euler_rotation(axis3, angle3)
        @ euler_rotation(axis2, angle2)
        @ euler_rotation(axis1, angle1)
    )
    rot_mat = quat.rot_matrix()

    assert np.allclose(rot_mat, composition)


########################################
#    class methods
########################################


@pytest.mark.parametrize(
    "arg",
    [
        -2,
        [1, 2, 3],
        np.array([-1, 0, -1]),
        [0, 1, 2, 3],
        np.array([3, -1, 0, -1]),
        (0, [1, 2, 3]),
        (0, np.array([1, 2, 3])),
        Quaternion([1, -1, 0, 1]),
    ],
)
def test_to_quaternion(arg):
    result = Quaternion.to_quaternion(arg)
    assert result == Quaternion(arg)


@pytest.mark.parametrize(
    "arg",
    [
        [[1, 2, 3], [-1, -1, 0], [2, 4, 7]],
        [np.array([1, 2, 3]), np.array([2, 3, 4])],
        np.array([[0, 1, 0], [-1, -2, -3], [1, 1, 1]]),
        np.repeat(-1.5, 30).reshape(-1, 3),
        [[0, 1, 2, 3], [-1, -1, 0, -2]],
        [np.array([1, 2, 4, 3]), np.array([2, -3, 3, 4])],
        np.array([[3, 0, 1, 0], [2, 1, 1, 1]]),
        np.repeat(-2.5, 16).reshape(-1, 4),
        [(0, [1, 2, 3]), (0, np.array([1, 2, 3])), (-2, [-1, -2, 0])],
        [
            Quaternion(0),
            Quaternion([1, 2, 3]),
            Quaternion([1, -1, 0, 1]),
        ],
    ],
)
def test_to_quaternion_struct(arg):
    result = Quaternion.to_quaternion(arg)
    assert all(r == Quaternion(a) for r, a in zip(result, arg))


@pytest.mark.parametrize(
    "axis, angle, error",
    [
        ([1, 0], 1.0, ValueError),
        ([1, 0, 1], "1.5", ValueError),
        ([1e-17, 0, -1e-20], 1.2, ArithmeticError),
    ],
)
def test_rotation_quat_from_axis_angle_invalid(axis, angle, error):
    with pytest.raises(error):
        Quaternion.rotation_quat_from_axis_angle(axis, angle)


@pytest.mark.parametrize(
    "axis, angle, expected",
    [
        ([1, 0, 0], 0.0, Quaternion(1)),  # angle=0 -> identity rotation
        (
            [22, 0, 0],
            np.pi / 2,
            UNIT_QUATERNION["x90"],
        ),  # axis=X-axis, angle=90
        (
            [0, -3, 0],
            np.pi,  # axis=neg. Y-axis, angle=180
            Quaternion(np.array([0, 0, -1, 0])),
        ),
        (
            [0, 0, 5],
            np.pi / 3,  # axis=Z-axis, angle=60
            Quaternion(np.array([np.sqrt(3) / 2, 0, 0, 1 / 2])),
        ),
    ],
)
def test_rotation_quat_from_axis_angle(axis, angle, expected):
    result = Quaternion.rotation_quat_from_axis_angle(axis, angle)
    assert result == expected


@pytest.mark.parametrize(
    "axis, angle, vectors",
    [
        ([1, 0, 0], 1.0, list("vectors")),
        ([1, 1, 1], 1.0, np.ones(6)),
        ([1, 0, 1], 1.0, np.ones(40).reshape(-1, 4)),
    ],
)
def test_rotate_by_axis_angle_invalid(axis, angle, vectors):
    with pytest.raises(ValueError):
        Quaternion.rotate_by_axis_angle(axis, angle, vectors)


@pytest.mark.parametrize(
    "axis, angle, vector",
    [
        ([1, 0, 0], 1.0, [1, 1, 1]),
        ([1, 0, -2], 1.2, np.array([0, 1, -1])),
        (np.array([1, 2, -3]), -0.4, [-1, 1.5, 2.5]),
    ],
)
def test_rotate_by_axis_angle(axis, angle, vector):
    expected = rotation_axis_angle(np.array(axis), angle) @ np.array(vector)
    result = Quaternion.rotate_by_axis_angle(axis, angle, vector)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("n", [10, 20, 50, 100])
@pytest.mark.parametrize("angle", [1.2, np.pi / 3, 2.0, 3.3])
@pytest.mark.parametrize(
    "axis",
    [
        np.array([1, 1, 1]),
        np.array([2, 0, -1]),
        np.array([-1, 1.5, -1.3]),
    ],
)
def test_rotate_by_axis_angle_vectors(n, axis, angle):
    # create vectors pointing in axis direction
    vectors = np.tile(axis, [n, 1])
    # vectors are fix points, we expect no transform
    expected = vectors
    # rotate them
    rotated = Quaternion.rotate_by_axis_angle(axis, angle, vectors)
    # test
    assert np.allclose(rotated, expected)

    # create vectors perpendicular to axis direction
    vectors = np.tile(np.array([axis[2], 0, -axis[0]]), [n, 1])
    # calculate the expected transform
    rot_matrix = rotation_axis_angle(axis, angle)
    expected = [rot_matrix @ p for p in vectors]
    # rotate them
    rotated = Quaternion.rotate_by_axis_angle(axis, angle, vectors)
    # test
    assert np.allclose(rotated, expected)
