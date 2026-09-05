"""Minimal deterministic mapping from full human hand angles to Apex targets."""
from __future__ import annotations
from dataclasses import dataclass
from human_hand.full_state import FullHandAngles
from retargeting.calibration import JointRange
from retargeting.index_mapping import linear_range_map
from robot.apex_joint_names import ApexActiveJointTarget, APEX_ACTIVE_JOINTS

@dataclass(frozen=True)
class FingerCalibration:
    abduction: JointRange
    mcp_flexion: JointRange
    pip_flexion: JointRange

@dataclass(frozen=True)
class FullHandCalibration:
    thumb: tuple[JointRange, JointRange, JointRange, JointRange]
    fingers: dict[str, FingerCalibration]

def map_full_hand_angles(angles: FullHandAngles, calibration: FullHandCalibration, apex_ranges: dict[str, tuple[float, float]]) -> ApexActiveJointTarget:
    values = {}
    thumb_values = (angles.thumb.abduction, angles.thumb.rotation, angles.thumb.flexion, angles.thumb.ip_flexion)
    for i, value in enumerate(thumb_values):
        lo, hi = apex_ranges[f"right_thumb_j{i}"]; source = calibration.thumb[i]
        values[f"right_thumb_j{i}"] = linear_range_map(value, source.minimum, source.maximum, lo, hi)
    for finger, joint_names in (("index", ("right_index_j0","right_index_j1","right_index_j2")), ("middle", ("right_middle_j0","right_middle_j1","right_middle_j2")), ("ring", ("right_ring_j0","right_ring_j1","right_ring_j2")), ("pinky", ("right_pinky_j0","right_pinky_j1","right_pinky_j2"))):
        source = calibration.fingers[finger]; a = angles.fingers[finger]
        for name, value, rng in zip(joint_names, (a.mcp_abduction, a.mcp_flexion, a.pip_flexion), (source.abduction, source.mcp_flexion, source.pip_flexion)):
            lo, hi = apex_ranges[name]; values[name] = linear_range_map(value, rng.minimum, rng.maximum, lo, hi)
    return ApexActiveJointTarget.from_mapping(values)
