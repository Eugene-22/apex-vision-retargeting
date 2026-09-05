import pytest

from retargeting.calibration import (
    IndexCalibration,
    JointRange,
)


def make_calibration() -> IndexCalibration:
    return IndexCalibration(
        human_abduction=JointRange(-0.3, 0.3),
        human_mcp_flexion=JointRange(0.0, 1.5),
        human_pip_flexion=JointRange(0.0, 1.7),
    )


def test_joint_range_rejects_min_equal_max():
    with pytest.raises(ValueError):
        JointRange(minimum=1.0, maximum=1.0)


def test_joint_range_rejects_min_greater_max():
    with pytest.raises(ValueError):
        JointRange(minimum=1.0, maximum=0.0)


def test_joint_range_rejects_non_finite():
    with pytest.raises(ValueError):
        JointRange(minimum=float("nan"), maximum=1.0)

    with pytest.raises(ValueError):
        JointRange(minimum=0.0, maximum=float("inf"))


def test_joint_range_width():
    r = JointRange(-0.5, 1.5)
    assert r.width() == 2.0


def test_index_calibration_roundtrip(tmp_path):
    calibration = make_calibration()

    path = tmp_path / "index_calibration.json"

    calibration.save(path)

    loaded = IndexCalibration.load(path)

    assert loaded == calibration


def test_index_calibration_to_from_dict():
    calibration = make_calibration()

    restored = IndexCalibration.from_dict(
        calibration.to_dict()
    )

    assert restored == calibration
