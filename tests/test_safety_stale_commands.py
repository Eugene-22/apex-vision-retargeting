import pytest

from robot.apex_joint_names import ApexActiveJointTarget
from safety.stale_commands import (
    TimedApexTarget,
    apply_stale_command_policy,
)


def target(value: float) -> ApexActiveJointTarget:
    return ApexActiveJointTarget(values=(value,) * 16)


def test_fresh_command_passes_through():
    command = TimedApexTarget(target=target(1.0), timestamp=9.8)

    result = apply_stale_command_policy(
        command=command,
        current_time=10.0,
        max_age_s=0.5,
        previous_target=target(0.2),
        neutral_target=target(0.0),
    )

    assert not result.stale
    assert result.reason == "fresh_command"
    assert result.age == pytest.approx(0.2)
    assert result.target == command.target


def test_missing_command_holds_previous_by_default():
    previous = target(0.4)

    result = apply_stale_command_policy(
        command=None,
        current_time=10.0,
        max_age_s=0.5,
        previous_target=previous,
        neutral_target=target(0.0),
    )

    assert result.stale
    assert result.reason == "missing_command"
    assert result.age is None
    assert result.target == previous


def test_stale_command_holds_previous():
    previous = target(0.4)
    command = TimedApexTarget(target=target(1.0), timestamp=9.0)

    result = apply_stale_command_policy(
        command=command,
        current_time=10.0,
        max_age_s=0.5,
        previous_target=previous,
        neutral_target=target(0.0),
        fallback_mode="hold",
    )

    assert result.stale
    assert result.reason == "stale_command"
    assert result.age == pytest.approx(1.0)
    assert result.target == previous


def test_stale_command_can_fall_back_to_neutral():
    neutral = target(0.0)
    command = TimedApexTarget(target=target(1.0), timestamp=9.0)

    result = apply_stale_command_policy(
        command=command,
        current_time=10.0,
        max_age_s=0.5,
        previous_target=target(0.4),
        neutral_target=neutral,
        fallback_mode="neutral",
    )

    assert result.stale
    assert result.reason == "stale_command"
    assert result.target == neutral


def test_command_at_exact_max_age_is_fresh():
    command = TimedApexTarget(target=target(1.0), timestamp=9.5)

    result = apply_stale_command_policy(
        command=command,
        current_time=10.0,
        max_age_s=0.5,
        previous_target=target(0.4),
        neutral_target=target(0.0),
    )

    assert not result.stale
    assert result.age == pytest.approx(0.5)


def test_rejects_invalid_current_time():
    with pytest.raises(ValueError):
        apply_stale_command_policy(
            command=None,
            current_time=float("nan"),
            max_age_s=0.5,
            previous_target=target(0.4),
            neutral_target=target(0.0),
        )


def test_rejects_invalid_max_age():
    with pytest.raises(ValueError):
        apply_stale_command_policy(
            command=None,
            current_time=10.0,
            max_age_s=0.0,
            previous_target=target(0.4),
            neutral_target=target(0.0),
        )


def test_rejects_future_command_timestamp():
    with pytest.raises(ValueError):
        apply_stale_command_policy(
            command=TimedApexTarget(target=target(1.0), timestamp=10.1),
            current_time=10.0,
            max_age_s=0.5,
            previous_target=target(0.4),
            neutral_target=target(0.0),
        )


def test_rejects_invalid_fallback_mode():
    with pytest.raises(ValueError):
        apply_stale_command_policy(
            command=None,
            current_time=10.0,
            max_age_s=0.5,
            previous_target=target(0.4),
            neutral_target=target(0.0),
            fallback_mode="bad",  # type: ignore[arg-type]
        )


def test_rejects_non_finite_command_timestamp():
    with pytest.raises(ValueError):
        TimedApexTarget(target=target(1.0), timestamp=float("inf"))
