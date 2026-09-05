import pytest

from robot.apex_joint_names import (
    APEX_ACTIVE_JOINTS,
    ApexActiveJointTarget,
)
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from safety.joint_limits import (
    clamp_active_joint_positions,
    clamp_active_joint_positions_from_urdf,
)


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


def mid_range_target(model: ApexUrdfModel) -> ApexActiveJointTarget:
    values = []
    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        values.append(0.5 * (limit.lower + limit.upper))
    return ApexActiveJointTarget(values=tuple(values))


def test_clamp_keeps_in_range_target_unchanged(model):
    target = mid_range_target(model)

    result = clamp_active_joint_positions_from_urdf(target, model)

    assert not result.was_clipped
    assert result.clipped == ()
    assert result.target == target


def test_clamp_limits_low_and_high_values(model):
    values = list(mid_range_target(model).values)
    limits = model.active_joint_limits()

    low_name = "right_index_j1"
    high_name = "right_pinky_j2"
    values[APEX_ACTIVE_JOINTS.index(low_name)] = limits[low_name].lower - 1.0
    values[APEX_ACTIVE_JOINTS.index(high_name)] = limits[high_name].upper + 1.0

    result = clamp_active_joint_positions_from_urdf(
        ApexActiveJointTarget(values=tuple(values)),
        model,
    )
    by_name = result.target.as_dict()

    assert result.was_clipped
    assert result.clipped == (low_name, high_name)
    assert by_name[low_name] == pytest.approx(limits[low_name].lower)
    assert by_name[high_name] == pytest.approx(limits[high_name].upper)


def test_clamp_preserves_canonical_order(model):
    target = mid_range_target(model)

    result = clamp_active_joint_positions_from_urdf(target, model)

    assert list(result.target.as_dict()) == list(APEX_ACTIVE_JOINTS)


def test_clamp_rejects_missing_limit(model):
    target = mid_range_target(model)
    limits = model.active_joint_limits()
    del limits["right_index_j0"]

    with pytest.raises(ValueError):
        clamp_active_joint_positions(target, limits)


def test_clamp_rejects_extra_mimic_limit(model):
    target = mid_range_target(model)
    limits = model.active_joint_limits()
    limits["right_index_j3"] = model.joint_limit("right_index_j3")

    with pytest.raises(ValueError):
        clamp_active_joint_positions(target, limits)


def test_clamped_target_validates_against_urdf(model):
    values = []
    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        values.append(limit.upper + 10.0)

    result = clamp_active_joint_positions_from_urdf(
        ApexActiveJointTarget(values=tuple(values)),
        model,
    )

    model.validate_active_joint_values(result.target.as_dict())
