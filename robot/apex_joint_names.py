"""
Canonical Apex Hand active joint order (M6 precursor).

This module is the single source of truth for the 16 actuator command
order used by retargeting, logging, SDK commands, and future ROS output.

Conventions:

- Joint names are URDF joint names for the right Apex Hand.
- Order is thumb first, then index/middle/ring/pinky, proximal to distal.
- Mimic/passive joints are intentionally excluded:
  right_thumb_j4 and *_j3 for non-thumb fingers.
- All commanded values associated with this order are radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


THUMB_ACTIVE_JOINTS: tuple[str, ...] = (
    "right_thumb_j0",
    "right_thumb_j1",
    "right_thumb_j2",
    "right_thumb_j3",
)

INDEX_ACTIVE_JOINTS: tuple[str, ...] = (
    "right_index_j0",
    "right_index_j1",
    "right_index_j2",
)

MIDDLE_ACTIVE_JOINTS: tuple[str, ...] = (
    "right_middle_j0",
    "right_middle_j1",
    "right_middle_j2",
)

RING_ACTIVE_JOINTS: tuple[str, ...] = (
    "right_ring_j0",
    "right_ring_j1",
    "right_ring_j2",
)

PINKY_ACTIVE_JOINTS: tuple[str, ...] = (
    "right_pinky_j0",
    "right_pinky_j1",
    "right_pinky_j2",
)

APEX_ACTIVE_JOINTS: tuple[str, ...] = (
    *THUMB_ACTIVE_JOINTS,
    *INDEX_ACTIVE_JOINTS,
    *MIDDLE_ACTIVE_JOINTS,
    *RING_ACTIVE_JOINTS,
    *PINKY_ACTIVE_JOINTS,
)

APEX_ACTIVE_DOF = len(APEX_ACTIVE_JOINTS)


@dataclass(frozen=True)
class ApexActiveJointTarget:
    """
    Full Apex active joint command vector.

    ``values`` follows ``APEX_ACTIVE_JOINTS`` exactly and stores radians.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != APEX_ACTIVE_DOF:
            raise ValueError(
                f"Expected {APEX_ACTIVE_DOF} active joint values, "
                f"got {len(self.values)}"
            )

    @classmethod
    def from_mapping(
        cls,
        values_by_name: Mapping[str, float],
    ) -> "ApexActiveJointTarget":
        """Create a command vector from a mapping keyed by canonical names."""
        missing = [
            name for name in APEX_ACTIVE_JOINTS
            if name not in values_by_name
        ]
        if missing:
            raise ValueError(
                "Missing active joint values: " + ", ".join(missing)
            )

        extra = [
            name for name in values_by_name
            if name not in APEX_ACTIVE_JOINTS
        ]
        if extra:
            raise ValueError(
                "Unknown active joint values: " + ", ".join(extra)
            )

        return cls(
            values=tuple(
                float(values_by_name[name])
                for name in APEX_ACTIVE_JOINTS
            )
        )

    def as_dict(self) -> dict[str, float]:
        """Return values keyed by canonical URDF joint name."""
        return dict(zip(APEX_ACTIVE_JOINTS, self.values))

    def as_sequence(self) -> tuple[float, ...]:
        """Return values in canonical command order."""
        return self.values


def active_joint_index(joint_name: str) -> int:
    """Index of one active joint in ``APEX_ACTIVE_JOINTS``."""
    try:
        return APEX_ACTIVE_JOINTS.index(joint_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown Apex active joint: {joint_name}"
        ) from exc


def mapping_from_sequence(
    values: Sequence[float],
) -> dict[str, float]:
    """Map a 16-value command sequence to canonical joint names."""
    return ApexActiveJointTarget(
        values=tuple(float(v) for v in values)
    ).as_dict()
