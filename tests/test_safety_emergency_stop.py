import pytest

from robot.apex_joint_names import ApexActiveJointTarget
from safety.emergency_stop import EmergencyStopLatch


def target(value: float) -> ApexActiveJointTarget:
    return ApexActiveJointTarget(values=(value,) * 16)


def test_untriggered_latch_passes_command_through():
    latch = EmergencyStopLatch()
    desired = target(1.0)

    result = latch.apply(
        desired=desired,
        neutral_target=target(0.0),
    )

    assert not latch.is_triggered
    assert not result.stopped
    assert result.reason == "not_triggered"
    assert result.target == desired


def test_trigger_latches_until_reset_reject_mode():
    latch = EmergencyStopLatch()
    desired = target(1.0)

    latch.trigger("operator_stop")

    first = latch.apply(desired, target(0.0), mode="reject")
    second = latch.apply(desired, target(0.0), mode="reject")

    assert latch.is_triggered
    assert first.stopped
    assert second.stopped
    assert first.reason == "operator_stop"
    assert second.reason == "operator_stop"
    assert first.target is None
    assert second.target is None


def test_trigger_can_output_neutral_mode():
    latch = EmergencyStopLatch()
    neutral = target(0.0)

    latch.trigger("lost_control")
    result = latch.apply(
        desired=target(1.0),
        neutral_target=neutral,
        mode="neutral",
    )

    assert result.stopped
    assert result.reason == "lost_control"
    assert result.target == neutral


def test_reset_allows_commands_again():
    latch = EmergencyStopLatch()
    desired = target(1.0)

    latch.trigger("operator_stop")
    latch.reset()
    result = latch.apply(desired, target(0.0))

    assert not latch.is_triggered
    assert latch.reason == "not_triggered"
    assert not result.stopped
    assert result.target == desired


def test_trigger_rejects_empty_reason():
    latch = EmergencyStopLatch()

    with pytest.raises(ValueError):
        latch.trigger("")


def test_apply_rejects_unknown_mode():
    latch = EmergencyStopLatch()

    with pytest.raises(ValueError):
        latch.apply(
            desired=target(1.0),
            neutral_target=target(0.0),
            mode="bad",  # type: ignore[arg-type]
        )
