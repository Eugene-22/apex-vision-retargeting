"""
Virtual index finger chain (M2.5).

Composes the full retargeting chain for the index finger in a single,
testable unit:

    FingerJointAngles (human, radians)
            |  map_index_angles()  with IndexCalibration
            v
    ApexIndexTarget (radians, within URDF limits)
            |  ApexUrdfModel.index_tip_position()
            v
    fingertip position (meters)

The Apex joint target ranges are derived from the URDF limits
(``apex_index_ranges``) so the URDF remains the single source of truth
for robot geometry. The human side comes from a persisted
``IndexCalibration``.

This module also holds the small validation predicates used to check
the chain (direction, joint limits, continuity, monotonicity).

Conventions: angles are radians, positions are meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange
from retargeting.index_mapping import (
    ApexIndexRange,
    ApexIndexTarget,
    map_index_angles,
)

from robot.apex_urdf import ApexUrdfModel, INDEX_ACTIVE_JOINTS


def apex_index_ranges(model: ApexUrdfModel) -> ApexIndexRange:
    """
    Apex index actuator ranges taken directly from URDF joint limits.

    This replaces the hand-written "apex" config section (M2.3) so the
    URDF is the single source of truth for robot joint limits.
    """

    j0, j1, j2 = (
        model.joints[name] for name in INDEX_ACTIVE_JOINTS
    )

    return ApexIndexRange(
        j0_abduction=JointRange(j0.lower, j0.upper),
        j1_mcp_flexion=JointRange(j1.lower, j1.upper),
        j2_pip_flexion=JointRange(j2.lower, j2.upper),
    )


@dataclass(frozen=True)
class VirtualIndexResult:
    """One step of the virtual index chain."""

    human: FingerJointAngles
    apex: ApexIndexTarget
    tip: np.ndarray = field(compare=False)  # (3,) meters


class VirtualIndexChain:
    """
    Human index angles -> Apex q -> virtual fingertip position.
    """

    def __init__(
        self,
        model: ApexUrdfModel,
        calibration: IndexCalibration,
        apex_ranges: ApexIndexRange | None = None,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.apex_ranges = (
            apex_ranges
            if apex_ranges is not None
            else apex_index_ranges(model)
        )

    def forward(self, human: FingerJointAngles) -> VirtualIndexResult:
        """Map human angles to Apex targets, then run FK to the tip."""
        apex = map_index_angles(
            human,
            self.calibration,
            self.apex_ranges,
        )

        tip = self.model.index_tip_position(
            apex.j0_abduction,
            apex.j1_mcp_flexion,
            apex.j2_pip_flexion,
        )

        return VirtualIndexResult(
            human=human,
            apex=apex,
            tip=tip,
        )


# ---------------------------------------------------------------------------
# Validation predicates
# ---------------------------------------------------------------------------

def within_limits(
    values: Iterable[float],
    lower: float,
    upper: float,
    atol: float = 1e-9,
) -> bool:
    """True when every value lies in ``[lower, upper]`` (within ``atol``)."""
    arr = np.asarray(values, dtype=float)
    return bool(
        np.all(arr >= lower - atol)
        and np.all(arr <= upper + atol)
    )


def is_monotonic(
    values: Iterable[float],
    direction: str = "increasing",
    atol: float = 1e-9,
) -> bool:
    """True when consecutive values are non-decreasing / non-increasing."""
    arr = np.asarray(values, dtype=float)
    d = np.diff(arr)

    if direction == "increasing":
        return bool(np.all(d >= -atol))
    if direction == "decreasing":
        return bool(np.all(d <= atol))

    raise ValueError(
        f"Unknown direction {direction!r}; "
        "expected 'increasing' or 'decreasing'"
    )


def is_continuous(
    values: Iterable[float],
    max_step: float,
) -> bool:
    """True when no consecutive step exceeds ``max_step`` in magnitude."""
    arr = np.asarray(values, dtype=float)
    # axis=0: difference consecutive elements (for a 2-D trajectory,
    # consecutive points — NOT the x/y/z components of one point).
    return bool(
        np.all(np.abs(np.diff(arr, axis=0)) <= max_step)
    )
