import numpy as np
import pandas as pd
import pytest

from quaternions.rotation import (
    euler_rotation,
    rotation_axis_angle,
    transform_cartesian_to_spherical,
    transform_spherical_to_cartesian,
)


@pytest.fixture
def cartesian_spherical_df():
    n = 8

    timestamp = pd.date_range(start="2000-01-01", freq="ms", periods=n)

    cartesian_df = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 0],
            [0, -1, 1],
        ]
    )

    spherical_df = np.array(
        [
            [1, np.pi / 2, 0],
            [1, np.pi / 2, np.pi / 2],
            [1, 0, 0],
            [np.sqrt(2), np.pi / 2, np.pi / 4],
            [np.sqrt(2), np.pi / 4, 0],
            [np.sqrt(2), np.pi / 4, np.pi / 2],
            [1, np.pi / 2, np.pi],
            [np.sqrt(2), np.pi / 4, -np.pi / 2],
        ]
    )

    return (
        pd.DataFrame(cartesian_df, index=timestamp),
        pd.DataFrame(spherical_df, index=timestamp),
    )


def test_cartesian_to_spherical(cartesian_spherical_df):
    cartesian, spherical = cartesian_spherical_df
    result = transform_cartesian_to_spherical(cartesian)
    assert np.allclose(spherical.values, result.values)


def test_spherical_to_cartesian(cartesian_spherical_df):
    cartesian, spherical = cartesian_spherical_df
    result = transform_spherical_to_cartesian(spherical)
    assert np.allclose(cartesian.values, result.values)


@pytest.mark.parametrize("axis", ["axis1", "AX", "B", "Z"])
@pytest.mark.parametrize("angle", [0, np.pi])
def test_euler_rotation_invalid_axis(axis, angle):
    with pytest.raises(ValueError):
        euler_rotation(axis, angle)


@pytest.mark.parametrize(
    "axis, angle, expected",
    [
        ("x", 0, np.eye(3),),
        ("y", 0, np.eye(3),),
        ("z", 0, np.eye(3),),
        ("x", np.pi / 2, np.array(([[1, 0, 0], [0, 0, -1], [0, 1, 0],])),),
        ("y", np.pi / 2, np.array(([[0, 0, 1], [0, 1, 0], [-1, 0, 0],])),),
        ("z", np.pi / 2, np.array(([[0, -1, 0], [1, 0, 0], [0, 0, 1],])),),
        (
            "x",
            np.pi / 4,
            np.array(
                (
                    [
                        [1, 0, 0],
                        [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                        [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                    ]
                )
            ),
        ),
        (
            "z",
            np.pi / 6,
            np.array(
                ([[np.sqrt(3) / 2, -1 / 2, 0], [1 / 2, np.sqrt(3) / 2, 0], [0, 0, 1],])
            ),
        ),
    ],
)
def test_euler_rotation(
    axis, angle, expected,
):
    result = euler_rotation(axis, angle)

    assert np.allclose(expected, result)


@pytest.fixture(
    scope="function",
    params=[
        (
            # angle=0
            np.array([2.2, -3.3, 0.2]),
            0,
            np.eye(3),
        ),
        (
            # axis=X-axis, angle=90
            np.array([3, 0, 0]),
            np.pi / 2,
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        ),
        (
            # axis=z-axis, angle=45
            np.array([0, 0, 4]),
            np.pi / 4,
            np.array(
                [
                    [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                    [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                    [0, 0, 1],
                ]
            ),
        ),
    ],
)
def axis_angle_rot_set(request):
    return request.param


def test_rotation_axis_angle(axis_angle_rot_set):
    axis, angle, expected = axis_angle_rot_set
    R = rotation_axis_angle(axis, angle)
    assert np.allclose(expected, R)
