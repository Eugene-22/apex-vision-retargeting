"""
Human index finger -> Apex index finger retargeting (M2.3).

Pipeline for a single finger:

    FingerJointAngles (human, radians)
            |
            v
    linear_range_map()  per joint
            |
            v
    ApexIndexTarget (radians)

All internal values are radians. The Apex index finger maps:

    human MCP abduction -> Apex index_j0
    human MCP flexion   -> Apex index_j1
    human PIP flexion   -> Apex index_j2

The Apex index DIP joint (index_j3) mimics index_j2 1:1 and is NOT
a separate actuator target, so it is intentionally absent here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange


@dataclass(frozen=True)
class ApexIndexTarget:
    """
    Apex index-finger actuator targets (radians).

    - ``j0_abduction``  : index_j0 MCP abduction / adduction
    - ``j1_mcp_flexion``: index_j1 MCP flexion / extension
    - ``j2_pip_flexion``: index_j2 PIP flexion
    """

    j0_abduction: float
    j1_mcp_flexion: float
    j2_pip_flexion: float


@dataclass(frozen=True)
class ApexIndexRange:
    """
    Apex index-finger actuator ranges (radians).

    These are the URDF joint limits, used as the target range for the
    linear mapping. The single source of truth is the Apex URDF (see
    ``robot.apex_urdf``), not a hand-written config copy.
    """

    j0_abduction: JointRange
    j1_mcp_flexion: JointRange
    j2_pip_flexion: JointRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "j0_abduction": self.j0_abduction.to_dict(),
            "j1_mcp_flexion": self.j1_mcp_flexion.to_dict(),
            "j2_pip_flexion": self.j2_pip_flexion.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApexIndexRange":
        return cls(
            j0_abduction=JointRange.from_dict(
                data["j0_abduction"]
            ),
            j1_mcp_flexion=JointRange.from_dict(
                data["j1_mcp_flexion"]
            ),
            j2_pip_flexion=JointRange.from_dict(
                data["j2_pip_flexion"]
            ),
        )


def linear_range_map(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
    invert: bool = False,
    offset: float = 0.0,
) -> float:
    """
    Map ``value`` from ``[source_min, source_max]`` to
    ``[target_min, target_max]``.

    Baseline linear mapping:

        ratio = (value - source_min) / (source_max - source_min)
        ratio = clamp(ratio, 0, 1)
        q = target_min + ratio * (target_max - target_min)

    Features:

    - **clamping**: ratio is clipped to ``[0, 1]``.
    - **sign inversion**: ``invert=True`` flips the direction
      (``ratio -> 1 - ratio``).
    - **offset**: additive constant in the target unit (radians).
    - **invalid range rejection**: raises ``ValueError`` when the
      source or target range is degenerate (min >= max).

    All values are in radians.
    """

    if source_min >= source_max:
        raise ValueError(
            f"Invalid source range: "
            f"[{source_min}, {source_max}]"
        )

    if target_min >= target_max:
        raise ValueError(
            f"Invalid target range: "
            f"[{target_min}, {target_max}]"
        )

    ratio = (value - source_min) / (source_max - source_min)

    ratio = float(np.clip(ratio, 0.0, 1.0))

    if invert:
        ratio = 1.0 - ratio

    return target_min + ratio * (target_max - target_min) + offset


def map_index_angles(
    angles: FingerJointAngles,
    calibration: IndexCalibration,
    apex_range: ApexIndexRange,
    invert_abduction: bool = False,
    invert_mcp_flexion: bool = False,
    invert_pip_flexion: bool = False,
) -> ApexIndexTarget:
    """
    Map human index-finger angles to Apex index-finger targets.

    The per-joint ``invert_*`` flags exist because the exact human
    sign convention relative to the Apex sign convention is only
    confirmed through real experiments (M2.5). Baseline assumes no
    inversion.
    """

    j0 = linear_range_map(
        value=angles.mcp_abduction,
        source_min=calibration.human_abduction.minimum,
        source_max=calibration.human_abduction.maximum,
        target_min=apex_range.j0_abduction.minimum,
        target_max=apex_range.j0_abduction.maximum,
        invert=invert_abduction,
    )

    j1 = linear_range_map(
        value=angles.mcp_flexion,
        source_min=calibration.human_mcp_flexion.minimum,
        source_max=calibration.human_mcp_flexion.maximum,
        target_min=apex_range.j1_mcp_flexion.minimum,
        target_max=apex_range.j1_mcp_flexion.maximum,
        invert=invert_mcp_flexion,
    )

    j2 = linear_range_map(
        value=angles.pip_flexion,
        source_min=calibration.human_pip_flexion.minimum,
        source_max=calibration.human_pip_flexion.maximum,
        target_min=apex_range.j2_pip_flexion.minimum,
        target_max=apex_range.j2_pip_flexion.maximum,
        invert=invert_pip_flexion,
    )

    return ApexIndexTarget(
        j0_abduction=j0,
        j1_mcp_flexion=j1,
        j2_pip_flexion=j2,
    )
