import pytest

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from safety.velocity_limits import (
    limit_active_joint_velocity,
    limit_active_joint_velocity_by_joint,
)


def target(value: float) -> ApexActiveJointTarget:
    return ApexActiveJointTarget(values=(value,) * 16)


def values_with(**overrides: float) -> ApexActiveJointTarget:
    values = {name: 0.0 for name in APEX_ACTIVE_JOINTS}
    values.update(overrides)
    return ApexActiveJointTarget.from_mapping(values)


def test_velocity_limiter_keeps_small_step_unchanged():
    previous = target(0.0)
    desired = target(0.05)

    result = limit_active_joint_velocity(
        previous=previous,
        desired=desired,
        dt=0.1,
        max_velocity=1.0,
    )

    assert not result.was_limited
    assert result.limited == ()
    assert result.target == desired


def test_velocity_limiter_limits_positive_step():
    result = limit_active_joint_velocity(
        previous=target(0.0),
        desired=target(1.0),
        dt=0.1,
        max_velocity=2.0,
    )

    assert result.was_limited
    assert result.limited == APEX_ACTIVE_JOINTS
    assert result.target.values == pytest.approx((0.2,) * 16)


def test_velocity_limiter_limits_negative_step():
    result = limit_active_joint_velocity(
        previous=target(1.0),
        desired=target(0.0),
        dt=0.1,
        max_velocity=2.0,
    )

    assert result.was_limited
    assert result.target.values == pytest.approx((0.8,) * 16)


def test_velocity_limiter_preserves_canonical_order():
    previous = target(0.0)
    desired = ApexActiveJointTarget(values=tuple(float(i) for i in range(16)))

    result = limit_active_joint_velocity(
        previous=previous,
        desired=desired,
        dt=1.0,
        max_velocity=100.0,
    )

    assert list(result.target.as_dict()) == list(APEX_ACTIVE_JOINTS)


def test_velocity_limiter_rejects_invalid_dt():
    with pytest.raises(ValueError):
        limit_active_joint_velocity(target(0.0), target(1.0), 0.0, 1.0)

    with pytest.raises(ValueError):
        limit_active_joint_velocity(target(0.0), target(1.0), -0.1, 1.0)


def test_velocity_limiter_rejects_invalid_max_velocity():
    with pytest.raises(ValueError):
        limit_active_joint_velocity(target(0.0), target(1.0), 0.1, 0.0)

    with pytest.raises(ValueError):
        limit_active_joint_velocity(target(0.0), target(1.0), 0.1, float("inf"))


def test_per_joint_velocity_limiter_limits_only_fast_joints():
    previous = target(0.0)
    desired = values_with(
        right_index_j1=1.0,
        right_index_j2=0.05,
    )
    limits = {name: 10.0 for name in APEX_ACTIVE_JOINTS}
    limits["right_index_j1"] = 2.0

    result = limit_active_joint_velocity_by_joint(
        previous=previous,
        desired=desired,
        dt=0.1,
        max_velocity_by_joint=limits,
    )

    by_name = result.target.as_dict()
    assert result.limited == ("right_index_j1",)
    assert by_name["right_index_j1"] == pytest.approx(0.2)
    assert by_name["right_index_j2"] == pytest.approx(0.05)


def test_per_joint_velocity_limiter_rejects_missing_joint_limit():
    limits = {name: 1.0 for name in APEX_ACTIVE_JOINTS}
    del limits["right_index_j0"]

    with pytest.raises(ValueError):
        limit_active_joint_velocity_by_joint(
            target(0.0),
            target(1.0),
            dt=0.1,
            max_velocity_by_joint=limits,
        )


def test_per_joint_velocity_limiter_rejects_extra_mimic_limit():
    limits = {name: 1.0 for name in APEX_ACTIVE_JOINTS}
    limits["right_index_j3"] = 1.0

    with pytest.raises(ValueError):
        limit_active_joint_velocity_by_joint(
            target(0.0),
            target(1.0),
            dt=0.1,
            max_velocity_by_joint=limits,
        )
