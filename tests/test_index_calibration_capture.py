import pytest

from human_hand.joint_angles import FingerJointAngles
from retargeting.index_calibration_capture import (
    IndexCalibrationSamples,
    build_index_calibration,
    phase_summary,
    robust_percentile,
)


def angle(abd: float, mcp: float, pip: float) -> FingerJointAngles:
    return FingerJointAngles(abd, mcp, pip)


def test_robust_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        robust_percentile([], 50.0)


def test_robust_percentile_rejects_non_finite_values():
    with pytest.raises(ValueError):
        robust_percentile([0.0, float("nan")], 50.0)


def test_phase_summary_returns_medians():
    summary = phase_summary(
        (
            angle(-0.2, 0.0, 0.1),
            angle(0.0, 0.2, 0.3),
            angle(0.2, 0.4, 0.5),
        )
    )

    assert summary["mcp_abduction"] == pytest.approx(0.0)
    assert summary["mcp_flexion"] == pytest.approx(0.2)
    assert summary["pip_flexion"] == pytest.approx(0.3)


def test_build_index_calibration_from_staged_samples():
    calibration = build_index_calibration(
        IndexCalibrationSamples(
            neutral=(
                angle(0.0, -0.1, 0.0),
                angle(0.0, 0.0, 0.1),
                angle(0.0, 0.1, 0.2),
            ),
            abduction_min=(
                angle(-0.5, 0.0, 0.0),
                angle(-0.4, 0.0, 0.0),
                angle(-0.3, 0.0, 0.0),
            ),
            abduction_max=(
                angle(0.3, 0.0, 0.0),
                angle(0.4, 0.0, 0.0),
                angle(0.5, 0.0, 0.0),
            ),
            mcp_flexion_max=(
                angle(0.0, 1.1, 0.0),
                angle(0.0, 1.2, 0.0),
                angle(0.0, 1.3, 0.0),
            ),
            pip_flexion_max=(
                angle(0.0, 0.0, 1.4),
                angle(0.0, 0.0, 1.5),
                angle(0.0, 0.0, 1.6),
            ),
        )
    )

    assert calibration.human_abduction.minimum == pytest.approx(-0.48)
    assert calibration.human_abduction.maximum == pytest.approx(0.48)
    assert calibration.human_mcp_flexion.minimum == pytest.approx(-0.08)
    assert calibration.human_mcp_flexion.maximum == pytest.approx(1.28)
    assert calibration.human_pip_flexion.minimum == pytest.approx(0.02)
    assert calibration.human_pip_flexion.maximum == pytest.approx(1.58)


def test_build_index_calibration_rejects_missing_phase():
    with pytest.raises(ValueError):
        build_index_calibration(
            IndexCalibrationSamples(
                neutral=(),
                abduction_min=(angle(-0.5, 0.0, 0.0),),
                abduction_max=(angle(0.5, 0.0, 0.0),),
                mcp_flexion_max=(angle(0.0, 1.0, 0.0),),
                pip_flexion_max=(angle(0.0, 0.0, 1.0),),
            )
        )
