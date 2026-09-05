"""
Index-to-mock command pipeline (M10 integration baseline).

This module composes the verified index retargeting path with the robot
interface boundary without touching real hardware:

    FingerJointAngles
        -> ApexIndexTarget
        -> ApexActiveJointTarget with neutral-filled non-index joints
        -> ApexInterface.command_position()

Conventions: all angles are radians; full commands follow
``APEX_ACTIVE_JOINTS`` order.
"""

from __future__ import annotations

from robot.apex_joint_names import (
    APEX_ACTIVE_JOINTS,
    INDEX_ACTIVE_JOINTS,
    ApexActiveJointTarget,
)
from robot.apex_urdf import ApexUrdfModel
from robot.interface import ApexCommandResult, ApexInterface
from retargeting.calibration import IndexCalibration
from retargeting.index_mapping import ApexIndexTarget, map_index_angles
from robot.virtual_index import apex_index_ranges
from human_hand.joint_angles import FingerJointAngles


def neutral_active_target_from_urdf(
    model: ApexUrdfModel,
) -> ApexActiveJointTarget:
    """
    Build a neutral full active target from URDF limits.

    Preference is 0 rad when it lies within the joint's URDF range;
    otherwise use the midpoint of that joint range. This keeps the
    result valid without inventing hand-specific neutral constants.
    """
    values: list[float] = []

    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        if limit.contains(0.0):
            values.append(0.0)
        else:
            values.append(0.5 * (limit.lower + limit.upper))

    return ApexActiveJointTarget(values=tuple(values))


def insert_index_target(
    base: ApexActiveJointTarget,
    index: ApexIndexTarget,
) -> ApexActiveJointTarget:
    """Return a full command with index j0/j1/j2 replaced."""
    values_by_name = base.as_dict()
    values_by_name[INDEX_ACTIVE_JOINTS[0]] = index.j0_abduction
    values_by_name[INDEX_ACTIVE_JOINTS[1]] = index.j1_mcp_flexion
    values_by_name[INDEX_ACTIVE_JOINTS[2]] = index.j2_pip_flexion
    return ApexActiveJointTarget.from_mapping(values_by_name)


def map_index_to_full_target(
    human: FingerJointAngles,
    calibration: IndexCalibration,
    model: ApexUrdfModel,
    base: ApexActiveJointTarget | None = None,
) -> ApexActiveJointTarget:
    """Map human index angles into a full 16-DoF Apex active target."""
    if base is None:
        base = neutral_active_target_from_urdf(model)

    index = map_index_angles(
        angles=human,
        calibration=calibration,
        apex_range=apex_index_ranges(model),
    )

    return insert_index_target(base, index)


def command_index_to_apex(
    human: FingerJointAngles,
    calibration: IndexCalibration,
    model: ApexUrdfModel,
    hand: ApexInterface,
    base: ApexActiveJointTarget | None = None,
) -> ApexCommandResult:
    """Map one human index sample and command it through an Apex backend."""
    target = map_index_to_full_target(
        human=human,
        calibration=calibration,
        model=model,
        base=base,
    )

    return hand.command_position(target)
