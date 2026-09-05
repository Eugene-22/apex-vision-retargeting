import pytest

from metrics.trajectory import summarize_trajectory
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.safe_pipeline import SafeIndexPipeline
from robot.mock_apex_hand import MockApexHand
from retargeting.calibration import IndexCalibration, JointRange
from human_hand.joint_angles import FingerJointAngles
from human_hand.state import HumanHandState
from safety.command_chain import SafetyCommandChain


def make_state(timestamp, quality=1.0, flexion=0.0):
    import numpy as np
    points = np.zeros((21, 3))
    angles = FingerJointAngles(0.0, flexion, flexion)
    return HumanHandState("Right", timestamp, quality, points, points, angles, quality, quality > 0.0)


def test_trajectory_summary_counts_quality_and_limits():
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    calibration = IndexCalibration(JointRange(-1.0, 1.0), JointRange(0.0, 2.0), JointRange(0.0, 2.0))
    pipeline = SafeIndexPipeline(model, calibration, MockApexHand(model), SafetyCommandChain(model, 1.0, 2.0, confidence_threshold=0.5))
    results = [pipeline.process(make_state(1.0, flexion=0.0))[0], pipeline.process(make_state(1.1, flexion=1.0, quality=0.1))[0]]
    summary = summarize_trajectory(results, 0.1)
    assert summary.samples == 2
    assert summary.low_confidence_count == 1
    assert summary.max_velocity <= 1.0 + 1e-9
    assert summary.max_acceleration <= 2.0 + 1e-9


def test_trajectory_summary_rejects_empty():
    with pytest.raises(ValueError):
        summarize_trajectory([], 0.1)
