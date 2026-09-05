import numpy as np

from human_hand.joint_angles import (
    compute_index_angles,
)


def make_straight_index() -> np.ndarray:
    points = np.zeros(
        (21, 3),
        dtype=np.float64,
    )

    points[5] = [0.0, 1.0, 0.0]
    points[6] = [0.0, 2.0, 0.0]
    points[7] = [0.0, 3.0, 0.0]
    points[8] = [0.0, 4.0, 0.0]

    return points


def test_straight_finger():
    points = make_straight_index()

    angles = compute_index_angles(
        points
    )

    assert np.isclose(
        angles.mcp_abduction,
        0.0,
        atol=1e-6,
    )

    assert np.isclose(
        angles.mcp_flexion,
        0.0,
        atol=1e-6,
    )

    assert np.isclose(
        angles.pip_flexion,
        0.0,
        atol=1e-6,
    )


def test_mcp_abduction():
    points = make_straight_index()

    angle = np.deg2rad(20.0)

    direction = np.array(
        [
            np.sin(angle),
            np.cos(angle),
            0.0,
        ]
    )

    points[6] = points[5] + direction
    points[7] = points[6] + direction
    points[8] = points[7] + direction

    angles = compute_index_angles(
        points
    )

    assert np.isclose(
        angles.mcp_abduction,
        angle,
        atol=1e-6,
    )


def test_pip_flexion():
    points = make_straight_index()

    angle = np.deg2rad(45.0)

    proximal = np.array(
        [0.0, 1.0, 0.0]
    )

    middle = np.array(
        [
            0.0,
            np.cos(angle),
            -np.sin(angle),
        ]
    )

    points[6] = points[5] + proximal
    points[7] = points[6] + middle
    points[8] = points[7] + middle

    angles = compute_index_angles(
        points
    )

    assert np.isclose(
        angles.pip_flexion,
        angle,
        atol=1e-6,
    )


def test_mcp_flexion():
    points = make_straight_index()

    angle = np.deg2rad(60.0)

    # Bend the finger at the MCP joint: the proximal
    # phalanx rotates from +y toward -z (palm direction).
    proximal = np.array(
        [
            0.0,
            np.cos(angle),
            -np.sin(angle),
        ]
    )

    points[6] = points[5] + proximal
    points[7] = points[6] + proximal
    points[8] = points[7] + proximal

    angles = compute_index_angles(
        points
    )

    assert np.isclose(
        angles.mcp_flexion,
        angle,
        atol=1e-6,
    )

    assert np.isclose(
        angles.pip_flexion,
        0.0,
        atol=1e-6,
    )