import pytest

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.limit_consistency import check_active_joint_limits


@pytest.fixture(scope="module")
def model():
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


def target_for(model, offset=0.0):
    values = []
    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        values.append(min(max(offset, limit.lower), limit.upper))
    return ApexActiveJointTarget(values=tuple(values))


def test_report_contains_all_active_joints(model):
    report = check_active_joint_limits(target_for(model), model)
    assert len(report.checks) == 16
    assert report.within_limits
    assert not report.violations


def test_report_detects_violation(model):
    values = list(target_for(model).values)
    name = "right_index_j1"
    limit = model.joint_limit(name)
    values[APEX_ACTIVE_JOINTS.index(name)] = limit.upper + 0.1

    report = check_active_joint_limits(
        ApexActiveJointTarget(values=tuple(values)), model
    )

    assert not report.within_limits
    assert [item.name for item in report.violations] == [name]
    assert report.violations[0].upper_margin == pytest.approx(-0.1)


def test_report_identifies_near_limit_joint(model):
    name = "right_index_j1"
    limit = model.joint_limit(name)
    values = list(target_for(model).values)
    values[APEX_ACTIVE_JOINTS.index(name)] = limit.upper - 0.01

    report = check_active_joint_limits(
        ApexActiveJointTarget(values=tuple(values)), model
    )

    assert name in [item.name for item in report.near_limits]


def test_negative_tolerance_is_rejected(model):
    with pytest.raises(ValueError, match="tolerance"):
        check_active_joint_limits(target_for(model), model, tolerance=-1.0)
