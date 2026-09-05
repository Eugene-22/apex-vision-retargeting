import pytest

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange
from retargeting.index_mapping import map_index_angles
from robot.apex_joint_names import APEX_ACTIVE_JOINTS, INDEX_ACTIVE_JOINTS
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.index_pipeline import (
    command_index_to_apex,
    insert_index_target,
    map_index_to_full_target,
    neutral_active_target_from_urdf,
)
from robot.mock_apex_hand import MockApexHand
from robot.virtual_index import apex_index_ranges


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


@pytest.fixture
def calibration() -> IndexCalibration:
    return IndexCalibration(
        human_abduction=JointRange(-1.0, 1.0),
        human_mcp_flexion=JointRange(0.0, 2.0),
        human_pip_flexion=JointRange(0.0, 2.0),
    )


def test_neutral_active_target_is_valid(model):
    target = neutral_active_target_from_urdf(model)

    assert len(target.values) == 16
    model.validate_active_joint_values(target.as_dict())


def test_insert_index_target_changes_only_index_joints(model):
    base = neutral_active_target_from_urdf(model)
    index_range = apex_index_ranges(model)
    index = map_index_angles(
        FingerJointAngles(0.2, 0.8, 1.0),
        IndexCalibration(
            human_abduction=JointRange(-1.0, 1.0),
            human_mcp_flexion=JointRange(0.0, 2.0),
            human_pip_flexion=JointRange(0.0, 2.0),
        ),
        index_range,
    )

    full = insert_index_target(base, index)
    base_by_name = base.as_dict()
    full_by_name = full.as_dict()

    assert full_by_name["right_index_j0"] == pytest.approx(index.j0_abduction)
    assert full_by_name["right_index_j1"] == pytest.approx(index.j1_mcp_flexion)
    assert full_by_name["right_index_j2"] == pytest.approx(index.j2_pip_flexion)

    for name in APEX_ACTIVE_JOINTS:
        if name not in INDEX_ACTIVE_JOINTS:
            assert full_by_name[name] == pytest.approx(base_by_name[name])


def test_map_index_to_full_target_matches_index_mapping(model, calibration):
    human = FingerJointAngles(0.0, 1.0, 1.0)
    full = map_index_to_full_target(human, calibration, model)
    expected_index = map_index_angles(
        human,
        calibration,
        apex_index_ranges(model),
    )

    by_name = full.as_dict()
    assert by_name["right_index_j0"] == pytest.approx(expected_index.j0_abduction)
    assert by_name["right_index_j1"] == pytest.approx(expected_index.j1_mcp_flexion)
    assert by_name["right_index_j2"] == pytest.approx(expected_index.j2_pip_flexion)


def test_map_index_to_full_target_preserves_custom_base(model, calibration):
    base_values = []
    for i, name in enumerate(APEX_ACTIVE_JOINTS):
        limit = model.joint_limit(name)
        midpoint = 0.5 * (limit.lower + limit.upper)
        base_values.append(midpoint + 0.001 * i)
    base = type(neutral_active_target_from_urdf(model))(values=tuple(base_values))

    full = map_index_to_full_target(
        FingerJointAngles(0.0, 1.0, 1.0),
        calibration,
        model,
        base=base,
    )

    base_by_name = base.as_dict()
    full_by_name = full.as_dict()
    for name in APEX_ACTIVE_JOINTS:
        if name not in INDEX_ACTIVE_JOINTS:
            assert full_by_name[name] == pytest.approx(base_by_name[name])


def test_command_index_to_mock_updates_only_index_joints(model, calibration):
    hand = MockApexHand(model)
    base = neutral_active_target_from_urdf(model)
    human = FingerJointAngles(0.0, 1.0, 1.0)

    result = command_index_to_apex(
        human=human,
        calibration=calibration,
        model=model,
        hand=hand,
        base=base,
    )

    assert not result.was_clipped
    assert hand.get_joint_state().position == result.applied

    base_by_name = base.as_dict()
    applied_by_name = result.applied.as_dict()
    for name in APEX_ACTIVE_JOINTS:
        if name not in INDEX_ACTIVE_JOINTS:
            assert applied_by_name[name] == pytest.approx(base_by_name[name])


def test_command_index_to_mock_clamps_out_of_range_custom_base(model, calibration):
    values = list(neutral_active_target_from_urdf(model).values)
    limit = model.joint_limit("right_thumb_j1")
    values[APEX_ACTIVE_JOINTS.index("right_thumb_j1")] = limit.upper + 1.0
    base = type(neutral_active_target_from_urdf(model))(values=tuple(values))

    hand = MockApexHand(model)
    result = command_index_to_apex(
        human=FingerJointAngles(0.0, 1.0, 1.0),
        calibration=calibration,
        model=model,
        hand=hand,
        base=base,
    )

    assert result.was_clipped
    assert result.clipped == ("right_thumb_j1",)
    assert result.applied.as_dict()["right_thumb_j1"] == pytest.approx(limit.upper)
