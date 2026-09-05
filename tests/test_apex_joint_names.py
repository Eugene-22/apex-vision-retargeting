import pytest

from robot.apex_joint_names import (
    APEX_ACTIVE_DOF,
    APEX_ACTIVE_JOINTS,
    INDEX_ACTIVE_JOINTS,
    ApexActiveJointTarget,
    active_joint_index,
    mapping_from_sequence,
)
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


def test_canonical_order_has_16_unique_active_joints():
    assert APEX_ACTIVE_DOF == 16
    assert len(APEX_ACTIVE_JOINTS) == 16
    assert len(set(APEX_ACTIVE_JOINTS)) == 16


def test_canonical_order_matches_urdf_active_joints(model):
    assert tuple(model.active_joints) == APEX_ACTIVE_JOINTS


def test_index_active_joints_are_slice_of_canonical_order():
    start = active_joint_index("right_index_j0")
    assert APEX_ACTIVE_JOINTS[start:start + 3] == INDEX_ACTIVE_JOINTS


def test_canonical_order_excludes_mimic_joints():
    assert "right_thumb_j4" not in APEX_ACTIVE_JOINTS
    assert "right_index_j3" not in APEX_ACTIVE_JOINTS
    assert "right_middle_j3" not in APEX_ACTIVE_JOINTS
    assert "right_ring_j3" not in APEX_ACTIVE_JOINTS
    assert "right_pinky_j3" not in APEX_ACTIVE_JOINTS


def test_active_joint_target_roundtrip_mapping():
    values_by_name = {
        name: float(i) for i, name in enumerate(APEX_ACTIVE_JOINTS)
    }

    target = ApexActiveJointTarget.from_mapping(values_by_name)

    assert target.as_sequence() == tuple(float(i) for i in range(16))
    assert target.as_dict() == values_by_name


def test_active_joint_target_rejects_wrong_length():
    with pytest.raises(ValueError):
        ApexActiveJointTarget(values=(0.0,) * 15)

    with pytest.raises(ValueError):
        ApexActiveJointTarget(values=(0.0,) * 17)


def test_active_joint_target_rejects_missing_name():
    values_by_name = {
        name: 0.0 for name in APEX_ACTIVE_JOINTS
    }
    del values_by_name["right_index_j1"]

    with pytest.raises(ValueError):
        ApexActiveJointTarget.from_mapping(values_by_name)


def test_active_joint_target_rejects_extra_name():
    values_by_name = {
        name: 0.0 for name in APEX_ACTIVE_JOINTS
    }
    values_by_name["right_index_j3"] = 0.0

    with pytest.raises(ValueError):
        ApexActiveJointTarget.from_mapping(values_by_name)


def test_active_joint_index_rejects_unknown_or_mimic_joint():
    with pytest.raises(ValueError):
        active_joint_index("right_index_j3")

    with pytest.raises(ValueError):
        active_joint_index("not_a_joint")


def test_mapping_from_sequence_uses_canonical_names():
    mapped = mapping_from_sequence(range(16))

    assert list(mapped) == list(APEX_ACTIVE_JOINTS)
    assert mapped["right_thumb_j0"] == 0.0
    assert mapped["right_pinky_j2"] == 15.0
