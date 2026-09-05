"""
Robot interface boundary (M10 baseline).

Higher-level perception, retargeting, filtering and safety code should
depend on this protocol instead of importing the vendor Rysen SDK.

Conventions: command and state values are radians in ``APEX_ACTIVE_JOINTS``
order, timestamps use a monotonic clock in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from robot.apex_joint_names import ApexActiveJointTarget


@dataclass(frozen=True)
class ApexJointState:
    """Snapshot of active Apex joint positions."""

    position: ApexActiveJointTarget
    timestamp: float


@dataclass(frozen=True)
class ApexCommandResult:
    """Result returned after a position command is accepted by a backend."""

    commanded: ApexActiveJointTarget
    applied: ApexActiveJointTarget
    timestamp: float
    clipped: tuple[str, ...]

    @property
    def was_clipped(self) -> bool:
        return bool(self.clipped)


class ApexInterface(Protocol):
    """Common interface for mock and real Apex Hand backends."""

    def command_position(
        self,
        target: ApexActiveJointTarget,
    ) -> ApexCommandResult:
        """Command active joint positions in canonical order."""

    def get_joint_state(self) -> ApexJointState:
        """Return latest active joint state."""

    def close(self) -> None:
        """Release backend resources."""
