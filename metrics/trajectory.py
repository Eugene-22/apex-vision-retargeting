"""Synthetic trajectory metrics for safe target pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectorySummary:
    samples: int
    max_velocity: float
    max_acceleration: float
    filter_changes: int
    clamp_count: int
    low_confidence_count: int
    stale_count: int
    emergency_stop_count: int


def summarize_trajectory(results: list, dt: float) -> TrajectorySummary:
    if not results:
        raise ValueError("results must not be empty")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    max_velocity = 0.0
    max_acceleration = 0.0
    filter_changes = 0
    previous_target = None
    previous_velocity = None
    clamp_count = low_confidence_count = stale_count = emergency_stop_count = 0
    for result in results:
        clamp_count += int(result.clamped.was_clipped)
        low_confidence_count += int(not result.confidence.accepted)
        stale_count += int(result.stale)
        emergency_stop_count += int(result.emergency_stopped)
        filter_changes += int(result.filtered != result.clamped.target)
        if previous_target is not None:
            velocities = [(current - old) / dt for current, old in zip(result.target.values, previous_target.values)]
            max_velocity = max(max_velocity, max(abs(value) for value in velocities))
            if previous_velocity is not None:
                accelerations = [(current - old) / dt for current, old in zip(velocities, previous_velocity)]
                max_acceleration = max(max_acceleration, max(abs(value) for value in accelerations))
            previous_velocity = velocities
        previous_target = result.target
    return TrajectorySummary(len(results), max_velocity, max_acceleration, filter_changes, clamp_count, low_confidence_count, stale_count, emergency_stop_count)
