import pytest

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange
from retargeting.index_mapping import (
    ApexIndexRange,
    ApexIndexTarget,
    linear_range_map,
    map_index_angles,
)


# ---------------------------------------------------------------------------
# linear_range_map
# ---------------------------------------------------------------------------

def test_map_source_min_to_target_min():
    result = linear_range_map(
        value=0.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )
    assert result == pytest.approx(10.0)


def test_map_source_max_to_target_max():
    result = linear_range_map(
        value=1.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )
    assert result == pytest.approx(20.0)


def test_map_source_midpoint_to_target_midpoint():
    result = linear_range_map(
        value=0.5,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )
    assert result == pytest.approx(15.0)


def test_map_below_source_min_clips_to_target_min():
    result = linear_range_map(
        value=-5.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )
    assert result == pytest.approx(10.0)


def test_map_above_source_max_clips_to_target_max():
    result = linear_range_map(
        value=99.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )
    assert result == pytest.approx(20.0)


def test_map_invalid_source_range_raises():
    with pytest.raises(ValueError):
        linear_range_map(
            value=0.5,
            source_min=1.0,
            source_max=1.0,
            target_min=0.0,
            target_max=1.0,
        )

    with pytest.raises(ValueError):
        linear_range_map(
            value=0.5,
            source_min=2.0,
            source_max=1.0,
            target_min=0.0,
            target_max=1.0,
        )


def test_map_invalid_target_range_raises():
    with pytest.raises(ValueError):
        linear_range_map(
            value=0.5,
            source_min=0.0,
            source_max=1.0,
            target_min=1.0,
            target_max=1.0,
        )


def test_map_invert_flips_direction():
    normal = linear_range_map(
        value=0.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
    )

    inverted = linear_range_map(
        value=0.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
        invert=True,
    )

    assert normal == pytest.approx(10.0)
    assert inverted == pytest.approx(20.0)


def test_map_offset_adds_constant():
    result = linear_range_map(
        value=0.0,
        source_min=0.0,
        source_max=1.0,
        target_min=10.0,
        target_max=20.0,
        offset=0.5,
    )
    assert result == pytest.approx(10.5)


# ---------------------------------------------------------------------------
# map_index_angles
# ---------------------------------------------------------------------------

def make_config() -> tuple[IndexCalibration, ApexIndexRange]:
    calibration = IndexCalibration(
        human_abduction=JointRange(-0.5, 0.5),
        human_mcp_flexion=JointRange(0.0, 1.5),
        human_pip_flexion=JointRange(0.0, 1.7),
    )

    apex_range = ApexIndexRange(
        j0_abduction=JointRange(-0.4, 0.4),
        j1_mcp_flexion=JointRange(-0.3, 1.5),
        j2_pip_flexion=JointRange(-0.1, 1.7),
    )

    return calibration, apex_range


def test_map_index_midpoint():
    calibration, apex_range = make_config()

    angles = FingerJointAngles(
        mcp_abduction=0.0,   # midpoint of [-0.5, 0.5]
        mcp_flexion=0.75,    # midpoint of [0.0, 1.5]
        pip_flexion=0.85,    # midpoint of [0.0, 1.7]
    )

    target = map_index_angles(angles, calibration, apex_range)

    assert target.j0_abduction == pytest.approx(0.0)
    assert target.j1_mcp_flexion == pytest.approx(0.6)
    assert target.j2_pip_flexion == pytest.approx(0.8)


def test_map_index_invert_abduction():
    calibration, apex_range = make_config()

    angles = FingerJointAngles(
        mcp_abduction=0.5,  # maximum human abduction
        mcp_flexion=0.0,
        pip_flexion=0.0,
    )

    target = map_index_angles(
        angles,
        calibration,
        apex_range,
        invert_abduction=True,
    )

    # maximum human -> minimum apex (inverted)
    assert target.j0_abduction == pytest.approx(-0.4)


def test_map_index_returns_apex_target_type():
    calibration, apex_range = make_config()

    angles = FingerJointAngles(
        mcp_abduction=0.0,
        mcp_flexion=0.0,
        pip_flexion=0.0,
    )

    target = map_index_angles(angles, calibration, apex_range)

    assert isinstance(target, ApexIndexTarget)
