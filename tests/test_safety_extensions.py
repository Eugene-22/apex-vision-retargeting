import pytest

from robot.apex_joint_names import ApexActiveJointTarget
from safety.acceleration_limits import limit_active_joint_acceleration
from safety.confidence_gate import gate_target


def target(value):
    return ApexActiveJointTarget(values=(value,) * 16)


def test_acceleration_limit_bounds_step():
    result = limit_active_joint_acceleration(target(0.0), target(0.0), target(1.0), 0.1, 0.5)
    assert result.was_limited
    assert result.target.values[0] == pytest.approx(0.005)


def test_acceleration_limit_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        limit_active_joint_acceleration(target(0.0), target(0.0), target(1.0), 0.0, 1.0)


def test_confidence_gate_accepts_good_confidence():
    result = gate_target(target(1.0), 0.8, target(0.0), target(0.2), 0.5)
    assert result.accepted
    assert result.target == target(1.0)


def test_confidence_gate_holds_on_lost_hand():
    result = gate_target(target(1.0), None, target(0.2), target(0.0), 0.5)
    assert not result.accepted
    assert result.reason == "lost_hand"
    assert result.target == target(0.2)


def test_confidence_gate_can_use_neutral():
    result = gate_target(target(1.0), 0.1, target(0.2), target(0.0), 0.5, "neutral")
    assert result.reason == "low_confidence"
    assert result.target == target(0.0)
