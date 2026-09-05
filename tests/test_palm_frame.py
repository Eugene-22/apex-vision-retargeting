import numpy as np
import pytest

from human_hand.palm_frame import (
    build_palm_frame,
)


def make_test_hand() -> np.ndarray:
    points = np.zeros((21, 3))

    points[0] = [0.0, 0.0, 0.0]

    points[5] = [
        1.0,
        1.0,
        0.0,
    ]

    points[9] = [
        0.0,
        1.2,
        0.0,
    ]

    points[17] = [
        -1.0,
        1.0,
        0.0,
    ]

    return points


def test_palm_frame_is_orthonormal():
    landmarks = make_test_hand()

    frame = build_palm_frame(
        landmarks
    )

    R = frame.rotation

    identity = R.T @ R

    np.testing.assert_allclose(
        identity,
        np.eye(3),
        atol=1e-6,
    )


def test_wrist_becomes_origin():
    landmarks = make_test_hand()

    frame = build_palm_frame(
        landmarks
    )

    local = frame.world_to_local(
        landmarks
    )

    np.testing.assert_allclose(
        local[0],
        np.zeros(3),
        atol=1e-6,
    )


def test_translation_invariance():
    landmarks = make_test_hand()

    frame1 = build_palm_frame(
        landmarks
    )

    local1 = frame1.world_to_local(
        landmarks
    )

    translation = np.array(
        [5.0, -3.0, 8.0]
    )

    moved = landmarks + translation

    frame2 = build_palm_frame(
        moved
    )

    local2 = frame2.world_to_local(
        moved
    )

    np.testing.assert_allclose(
        local1,
        local2,
        atol=1e-6,
    )


def rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_rotation_invariance():
    landmarks = make_test_hand()

    frame1 = build_palm_frame(
        landmarks
    )

    local1 = frame1.world_to_local(
        landmarks
    )

    rotated = landmarks @ rotation_z(
        np.deg2rad(37.0)
    ).T

    frame2 = build_palm_frame(
        rotated
    )

    local2 = frame2.world_to_local(
        rotated
    )

    np.testing.assert_allclose(
        local1,
        local2,
        atol=1e-6,
    )


def test_scale_invariance():
    landmarks = make_test_hand()

    frame1 = build_palm_frame(
        landmarks
    )

    local1 = frame1.world_to_local(
        landmarks
    )

    scaled = landmarks * 3.5

    frame2 = build_palm_frame(
        scaled
    )

    local2 = frame2.world_to_local(
        scaled
    )

    np.testing.assert_allclose(
        local1,
        local2,
        atol=1e-6,
    )


def test_degenerate_palm_geometry_raises():
    landmarks = make_test_hand()

    # index MCP and pinky MCP coincide, so the palm x-axis
    # cannot be defined.
    landmarks[17] = landmarks[5]

    with pytest.raises(ValueError):
        build_palm_frame(
            landmarks
        )

