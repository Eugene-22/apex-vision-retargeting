import pytest

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.interface import ApexCommandResult, ApexInterface, ApexJointState
from safety.emergency_stop import EmergencyStopLatch
from safety.stale_commands import TimedApexTarget
from robot.mock_apex_hand import MockApexHand


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


def mid_range_target(model: ApexUrdfModel) -> ApexActiveJointTarget:
    values = []
    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        values.append(0.5 * (limit.lower + limit.upper))
    return ApexActiveJointTarget(values=tuple(values))


def test_mock_satisfies_interface_shape(model):
    hand: ApexInterface = MockApexHand(model)

    state = hand.get_joint_state()

    assert isinstance(state, ApexJointState)
    assert isinstance(state.position, ApexActiveJointTarget)
    hand.close()


def test_mock_initial_state_is_within_urdf_limits(model):
    hand = MockApexHand(model)

    model.validate_active_joint_values(
        hand.get_joint_state().position.as_dict()
    )


def test_mock_command_updates_state(model):
    hand = MockApexHand(model)
    target = mid_range_target(model)

    result = hand.command_position(target)
    state = hand.get_joint_state()

    assert isinstance(result, ApexCommandResult)
    assert not result.was_clipped
    assert result.commanded == target
    assert result.applied == target
    assert state.position == target
    assert state.timestamp == result.timestamp


def test_mock_command_clamps_out_of_range_values(model):
    values = list(mid_range_target(model).values)
    limits = model.active_joint_limits()
    name = "right_index_j1"
    values[APEX_ACTIVE_JOINTS.index(name)] = limits[name].upper + 1.0

    hand = MockApexHand(model)
    result = hand.command_position(
        ApexActiveJointTarget(values=tuple(values))
    )

    assert result.was_clipped
    assert result.clipped == (name,)
    assert result.applied.as_dict()[name] == pytest.approx(limits[name].upper)
    assert hand.get_joint_state().position == result.applied


def test_mock_initial_position_is_clamped(model):
    values = []
    limits = model.active_joint_limits()
    for name in APEX_ACTIVE_JOINTS:
        values.append(limits[name].lower - 1.0)

    hand = MockApexHand(
        model,
        initial_position=ApexActiveJointTarget(values=tuple(values)),
    )

    state = hand.get_joint_state()
    for name, value in state.position.as_dict().items():
        assert value == pytest.approx(limits[name].lower)


def test_mock_rejects_operations_after_close(model):
    hand = MockApexHand(model)
    target = mid_range_target(model)

    hand.close()

    with pytest.raises(RuntimeError):
        hand.command_position(target)

    with pytest.raises(RuntimeError):
        hand.get_joint_state()


def test_mock_optional_velocity_limit(model):
    times = iter((10.0, 11.0))
    hand = MockApexHand(
        model,
        max_velocity=0.1,
        clock=lambda: next(times),
    )

    # The allowed command step is 0.1 rad/s * 1.0 s = 0.1 rad.
    target = mid_range_target(model)
    result = hand.command_position(target)

    for previous, applied in zip(hand._neutral_target_from_limits(model).values, result.applied.values):
        assert abs(applied - previous) <= 0.1 + 1e-9


def test_mock_timed_command_uses_fresh_target(model):
    times = iter((10.0, 10.1, 10.2))
    hand = MockApexHand(model, clock=lambda: next(times))
    target = mid_range_target(model)

    result = hand.command_timed_position(
        command=TimedApexTarget(target=target, timestamp=10.0),
        max_age_s=0.5,
    )

    assert result.applied == target


def test_mock_timed_command_holds_previous_when_stale(model):
    times = iter((10.0, 11.0, 11.1))
    initial = mid_range_target(model)
    hand = MockApexHand(model, initial_position=initial, clock=lambda: next(times))
    stale_target = ApexActiveJointTarget(values=(0.0,) * 16)

    result = hand.command_timed_position(
        command=TimedApexTarget(target=stale_target, timestamp=10.0),
        max_age_s=0.5,
        fallback_mode="hold",
    )

    assert result.applied == initial


def test_mock_timed_command_uses_neutral_when_missing(model):
    times = iter((10.0, 11.0, 11.1))
    initial = mid_range_target(model)
    hand = MockApexHand(model, initial_position=initial, clock=lambda: next(times))

    result = hand.command_timed_position(
        command=None,
        max_age_s=0.5,
        fallback_mode="neutral",
    )

    assert result.applied == hand._neutral_target_from_limits(model)


def test_mock_emergency_stop_reject_mode(model):
    latch = EmergencyStopLatch()
    hand = MockApexHand(model, emergency_stop=latch)
    latch.trigger("operator_stop")

    with pytest.raises(RuntimeError):
        hand.command_position(mid_range_target(model))


def test_mock_emergency_stop_neutral_mode(model):
    latch = EmergencyStopLatch()
    initial = mid_range_target(model)
    hand = MockApexHand(
        model,
        initial_position=initial,
        emergency_stop=latch,
        emergency_stop_mode="neutral",
    )

    latch.trigger("operator_stop")
    result = hand.command_position(initial)

    assert result.applied == hand._neutral_target_from_limits(model)


def test_mock_emergency_stop_reset_allows_command(model):
    latch = EmergencyStopLatch()
    hand = MockApexHand(model, emergency_stop=latch)
    target = mid_range_target(model)

    latch.trigger("operator_stop")
    latch.reset()
    result = hand.command_position(target)

    assert result.applied == target

