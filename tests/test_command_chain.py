import pytest

from filtering.one_euro import OneEuroJointFilter
from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from safety.command_chain import SafetyCommandChain
from safety.emergency_stop import EmergencyStopLatch


@pytest.fixture
def model():
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


def target(value):
    return ApexActiveJointTarget(values=(value,) * len(APEX_ACTIVE_JOINTS))


def test_chain_filters_clamps_and_limits_velocity(model):
    chain = SafetyCommandChain(model, max_velocity=0.1, filter_=OneEuroJointFilter(min_cutoff_hz=100.0))
    result = chain.process(target(1.0), target(0.0), 0.1, 1.0, 1.0)
    assert result.filtered == target(1.0)
    assert result.velocity_limited.was_limited
    assert all(value == pytest.approx(0.01) for value in result.target.values)


def test_chain_stale_hold_uses_previous_target(model):
    previous = target(0.2)
    result = SafetyCommandChain(model, 10.0).process(target(0.8), previous, 0.1, 2.0, 1.0)
    assert result.stale
    assert result.target == previous


def test_chain_emergency_stop_rejects(model):
    latch = EmergencyStopLatch()
    latch.trigger("test")
    chain = SafetyCommandChain(model, 10.0, emergency_stop=latch)
    with pytest.raises(RuntimeError, match="Emergency stop"):
        chain.process(target(0.1), target(0.0), 0.1, 1.0, 1.0)


def test_chain_emergency_stop_neutral(model):
    latch = EmergencyStopLatch()
    latch.trigger("test")
    chain = SafetyCommandChain(model, 10.0, emergency_stop=latch, emergency_stop_mode="neutral")
    result = chain.process(target(0.1), target(0.0), 0.1, 1.0, 1.0)
    assert result.emergency_stopped
    assert result.target == chain._neutral_target()


def test_chain_low_confidence_holds_previous(model):
    chain = SafetyCommandChain(model, 10.0, confidence_threshold=0.5)
    previous = target(0.2)
    result = chain.process(target(0.8), previous, 0.1, 1.0, 1.0, confidence=0.1)
    assert not result.confidence.accepted
    assert result.confidence.reason == "low_confidence"
    assert result.target == previous


def test_chain_lost_hand_can_fallback_to_neutral(model):
    chain = SafetyCommandChain(model, 10.0, confidence_threshold=0.5, lost_hand_mode="neutral")
    result = chain.process(target(0.8), target(0.2), 0.1, 1.0, 1.0, confidence=None)
    assert result.confidence.reason == "lost_hand"
    assert result.target == chain._neutral_target()


def test_chain_stateful_process_maintains_previous_state(model):
    chain = SafetyCommandChain(model, max_velocity=1.0, max_acceleration=2.0)
    first = chain.process_stateful(target(0.0), current_time=1.0)
    second = chain.process_stateful(target(1.0), current_time=1.1)

    assert first.target == target(0.0)
    assert second.target.values[0] == pytest.approx(0.02)
    assert second.velocity_limited.was_limited


def test_chain_stateful_lost_hand_holds_last_target(model):
    chain = SafetyCommandChain(model, max_velocity=1.0, max_acceleration=2.0)
    chain.process_stateful(target(0.0), current_time=1.0)
    result = chain.process_stateful(target(1.0), current_time=1.1, confidence=None)

    assert result.confidence.reason == "lost_hand"
    assert result.target == target(0.0)
