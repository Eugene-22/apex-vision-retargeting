"""Statistics for repeated read-only mechanical joint samples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JointSampleStats:
    joint_id: Any
    count: int
    mean: float
    stddev: float
    minimum: float
    maximum: float


def summarize_joint_samples(samples: list[Any]) -> tuple[JointSampleStats, ...]:
    if not samples:
        raise ValueError("samples must not be empty")

    by_joint: dict[Any, list[float]] = {}
    for frame in samples:
        states = getattr(frame, "joint_states", None)
        if states is None:
            states = getattr(frame, "joints", None)
        if states is None:
            states = list(frame)
        for state in states:
            by_joint.setdefault(state.joint_id, []).append(float(state.position))

    results = []
    for joint_id, values in by_joint.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        results.append(
            JointSampleStats(
                joint_id=joint_id,
                count=len(values),
                mean=mean,
                stddev=math.sqrt(variance),
                minimum=min(values),
                maximum=max(values),
            )
        )
    return tuple(results)
