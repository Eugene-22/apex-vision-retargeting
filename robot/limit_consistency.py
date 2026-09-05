"""Read-only consistency checks between joint state and URDF limits."""

from __future__ import annotations

from dataclasses import dataclass

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.apex_urdf import ApexUrdfModel


@dataclass(frozen=True)
class JointLimitCheck:
    name: str
    position: float
    lower: float
    upper: float
    lower_margin: float
    upper_margin: float
    within_limits: bool


@dataclass(frozen=True)
class LimitConsistencyReport:
    checks: tuple[JointLimitCheck, ...]

    @property
    def violations(self) -> tuple[JointLimitCheck, ...]:
        return tuple(check for check in self.checks if not check.within_limits)

    @property
    def near_limits(self) -> tuple[JointLimitCheck, ...]:
        return tuple(
            check for check in self.checks
            if check.within_limits and min(check.lower_margin, check.upper_margin) <= 0.05
        )

    @property
    def within_limits(self) -> bool:
        return not self.violations


def check_active_joint_limits(
    target: ApexActiveJointTarget,
    model: ApexUrdfModel,
    tolerance: float = 1e-9,
) -> LimitConsistencyReport:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    values = target.as_dict()
    checks = []
    for name in APEX_ACTIVE_JOINTS:
        limit = model.joint_limit(name)
        position = values[name]
        checks.append(
            JointLimitCheck(
                name=name,
                position=position,
                lower=limit.lower,
                upper=limit.upper,
                lower_margin=position - limit.lower,
                upper_margin=limit.upper - position,
                within_limits=limit.lower - tolerance <= position <= limit.upper + tolerance,
            )
        )
    return LimitConsistencyReport(checks=tuple(checks))
