"""One-Euro filter for scalar and full Apex joint targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget


def _alpha(cutoff_hz: float, dt: float) -> float:
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("cutoff_hz must be finite and > 0")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroFilter:
    """One-Euro filter state for one signal, with values in signal units."""

    min_cutoff_hz: float = 1.0
    beta: float = 0.0
    derivative_cutoff_hz: float = 1.0
    _x_previous: float | None = None
    _dx_previous: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_cutoff_hz) or self.min_cutoff_hz <= 0.0:
            raise ValueError("min_cutoff_hz must be finite and > 0")
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and >= 0")
        if not math.isfinite(self.derivative_cutoff_hz) or self.derivative_cutoff_hz <= 0.0:
            raise ValueError("derivative_cutoff_hz must be finite and > 0")

    def reset(self) -> None:
        self._x_previous = None
        self._dx_previous = 0.0

    def filter(self, value: float, dt: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if self._x_previous is None:
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError("dt must be finite and > 0")
            self._x_previous = float(value)
            return float(value)

        derivative = (value - self._x_previous) / dt
        derivative_alpha = _alpha(self.derivative_cutoff_hz, dt)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._dx_previous
        )
        cutoff = self.min_cutoff_hz + self.beta * abs(filtered_derivative)
        signal_alpha = _alpha(cutoff, dt)
        filtered = signal_alpha * value + (1.0 - signal_alpha) * self._x_previous
        self._x_previous = float(filtered)
        self._dx_previous = float(filtered_derivative)
        return float(filtered)


class OneEuroJointFilter:
    """Apply one independent One-Euro filter per active Apex joint."""

    def __init__(
        self,
        min_cutoff_hz: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff_hz: float = 1.0,
    ) -> None:
        self._filters = {
            name: OneEuroFilter(min_cutoff_hz, beta, derivative_cutoff_hz)
            for name in APEX_ACTIVE_JOINTS
        }

    def reset(self) -> None:
        for item in self._filters.values():
            item.reset()

    def filter(self, target: ApexActiveJointTarget, dt: float) -> ApexActiveJointTarget:
        values = target.as_dict()
        return ApexActiveJointTarget(
            values=tuple(
                self._filters[name].filter(values[name], dt)
                for name in APEX_ACTIVE_JOINTS
            )
        )
