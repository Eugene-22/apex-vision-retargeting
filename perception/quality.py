"""Quality checks for normalized hand perception inputs."""

from __future__ import annotations

import math
import numpy as np


def assess_landmark_quality(points: np.ndarray, confidence: float) -> tuple[float, bool]:
    if points.shape != (21, 3):
        raise ValueError("points must have shape (21, 3)")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    finite = bool(np.isfinite(points).all())
    return (confidence if finite else 0.0), (finite and confidence > 0.0)


def temporal_quality(current: np.ndarray, previous: np.ndarray | None, dt: float | None, max_speed: float = 5.0) -> tuple[float, bool]:
    if previous is None:
        return 1.0, True
    if dt is None or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0 when previous is provided")
    if not math.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("max_speed must be finite and > 0")
    speed = float(np.max(np.linalg.norm(current - previous, axis=1)) / dt)
    if speed > max_speed:
        return 0.0, False
    return max(0.0, 1.0 - speed / max_speed), True
