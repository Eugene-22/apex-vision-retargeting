"""
Mock Apex Hand backend (M10 baseline).

The mock backend implements ``ApexInterface`` without touching hardware.
It validates the software pipeline boundary by accepting only full
``ApexActiveJointTarget`` commands, applying the safety position clamp,
and storing the resulting joint state.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.apex_urdf import ApexUrdfModel
from robot.interface import ApexCommandResult, ApexJointState
from safety.emergency_stop import (
    EmergencyStopLatch,
    EmergencyStopMode,
)
from safety.joint_limits import clamp_active_joint_positions_from_urdf
from safety.stale_commands import (
    FallbackMode,
    TimedApexTarget,
    apply_stale_command_policy,
)
from safety.velocity_limits import limit_active_joint_velocity


class MockApexHand:
    """In-memory Apex backend for tests and pipeline development."""

    def __init__(
        self,
        model: ApexUrdfModel,
        initial_position: ApexActiveJointTarget | None = None,
        max_velocity: float | None = None,
        emergency_stop: EmergencyStopLatch | None = None,
        emergency_stop_mode: EmergencyStopMode = "reject",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._model = model
        self._closed = False
        self._max_velocity = max_velocity
        self._emergency_stop = emergency_stop
        self._emergency_stop_mode = emergency_stop_mode
        self._clock = clock

        if initial_position is None:
            initial_position = self._neutral_target_from_limits(model)

        clamp = clamp_active_joint_positions_from_urdf(
            initial_position,
            self._model,
        )
        self._state = ApexJointState(
            position=clamp.target,
            timestamp=self._clock(),
        )

    @staticmethod
    def _neutral_target_from_limits(
        model: ApexUrdfModel,
    ) -> ApexActiveJointTarget:
        values = []
        for name in APEX_ACTIVE_JOINTS:
            limit = model.joint_limit(name)
            if limit.contains(0.0):
                values.append(0.0)
            else:
                values.append(0.5 * (limit.lower + limit.upper))

        return ApexActiveJointTarget(values=tuple(values))

    def command_position(
        self,
        target: ApexActiveJointTarget,
    ) -> ApexCommandResult:
        """Clamp and store a full active position command."""
        if self._closed:
            raise RuntimeError("MockApexHand is closed")

        desired = target
        timestamp = self._clock()

        if self._emergency_stop is not None:
            stop = self._emergency_stop.apply(
                desired=desired,
                neutral_target=self._neutral_target_from_limits(self._model),
                mode=self._emergency_stop_mode,
            )
            if stop.target is None:
                raise RuntimeError(
                    f"Emergency stop active: {stop.reason}"
                )
            desired = stop.target

        if self._max_velocity is not None:
            dt = timestamp - self._state.timestamp
            velocity = limit_active_joint_velocity(
                previous=self._state.position,
                desired=desired,
                dt=dt,
                max_velocity=self._max_velocity,
            )
            desired = velocity.target

        clamp = clamp_active_joint_positions_from_urdf(
            desired,
            self._model,
        )
        self._state = ApexJointState(
            position=clamp.target,
            timestamp=timestamp,
        )

        return ApexCommandResult(
            commanded=target,
            applied=clamp.target,
            timestamp=timestamp,
            clipped=clamp.clipped,
        )


    def command_timed_position(
        self,
        command: TimedApexTarget | None,
        max_age_s: float,
        fallback_mode: FallbackMode = "hold",
    ) -> ApexCommandResult:
        """Apply stale-command gating before commanding this mock backend."""
        if self._closed:
            raise RuntimeError("MockApexHand is closed")

        current_time = self._clock()
        gated = apply_stale_command_policy(
            command=command,
            current_time=current_time,
            max_age_s=max_age_s,
            previous_target=self._state.position,
            neutral_target=self._neutral_target_from_limits(self._model),
            fallback_mode=fallback_mode,
        )

        return self.command_position(gated.target)

    def get_joint_state(self) -> ApexJointState:
        """Return latest mock joint state."""
        if self._closed:
            raise RuntimeError("MockApexHand is closed")

        return self._state

    def close(self) -> None:
        """Close the mock backend."""
        self._closed = True
