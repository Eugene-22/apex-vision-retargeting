import pytest

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.mock_apex_hand import MockApexHand
from robot.safe_pipeline import SafeIndexPipeline
from safety.command_chain import SafetyCommandChain
from human_hand.state import HumanHandState


def state(quality=1.0):
    import numpy as np
    points = np.zeros((21, 3))
    return HumanHandState("Right", 1.0, quality, points, points, FingerJointAngles(0.0, 0.0, 0.0), quality, quality > 0.0)


def test_safe_pipeline_commands_mock_backend():
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    calibration = IndexCalibration(JointRange(-1.0, 1.0), JointRange(0.0, 2.0), JointRange(0.0, 2.0))
    hand = MockApexHand(model)
    chain = SafetyCommandChain(model, max_velocity=1.0, max_acceleration=2.0)
    result = SafeIndexPipeline(model, calibration, hand, chain).process(state())
    assert result is not None
    safety_result, command = result
    assert len(command.applied.values) == 16
    assert not safety_result.stale


def test_safe_pipeline_drops_missing_state():
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    calibration = IndexCalibration(JointRange(-1.0, 1.0), JointRange(0.0, 2.0), JointRange(0.0, 2.0))
    hand = MockApexHand(model)
    chain = SafetyCommandChain(model, max_velocity=1.0)
    assert SafeIndexPipeline(model, calibration, hand, chain).process(None) is None
