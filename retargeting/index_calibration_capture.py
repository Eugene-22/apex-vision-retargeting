"""
Index-finger calibration helpers (M2.3 / M13 precursor).

This module contains pure, testable logic for turning staged human index
angle samples into an ``IndexCalibration``. Camera capture lives in
``scripts/calibrate_index.py``.

Conventions: all angles are radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange


@dataclass(frozen=True)
class IndexCalibrationSamples:
    """
    Staged calibration samples for one human index finger.

    Each field stores angles in radians captured while the user holds the
    named pose. The calibration workflow is deliberately explicit so the
    source of every range endpoint is visible.
    """

    neutral: tuple[FingerJointAngles, ...]
    abduction_min: tuple[FingerJointAngles, ...]
    abduction_max: tuple[FingerJointAngles, ...]
    mcp_flexion_max: tuple[FingerJointAngles, ...]
    pip_flexion_max: tuple[FingerJointAngles, ...]


def robust_percentile(
    values: Iterable[float],
    percentile: float,
) -> float:
    """Return a finite percentile from non-empty radian samples."""
    array = np.asarray(list(values), dtype=np.float64)

    if array.size == 0:
        raise ValueError("Calibration phase has no valid samples")

    if not np.all(np.isfinite(array)):
        raise ValueError("Calibration samples must be finite")

    return float(np.percentile(array, percentile))


def _require_samples(
    name: str,
    samples: tuple[FingerJointAngles, ...],
) -> None:
    if not samples:
        raise ValueError(f"Calibration phase {name!r} has no valid samples")


def build_index_calibration(
    samples: IndexCalibrationSamples,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
) -> IndexCalibration:
    """
    Build index calibration ranges from staged samples.

    Range endpoint policy:

    - abduction min/max come from the abduction_min / abduction_max phases.
    - MCP flexion min uses the neutral/open phase; max uses mcp_flexion_max.
    - PIP flexion min uses the neutral/open phase; max uses pip_flexion_max.

    Percentiles reject short-lived perception spikes while keeping the
    result deterministic and explainable.
    """
    phases = {
        "neutral": samples.neutral,
        "abduction_min": samples.abduction_min,
        "abduction_max": samples.abduction_max,
        "mcp_flexion_max": samples.mcp_flexion_max,
        "pip_flexion_max": samples.pip_flexion_max,
    }

    for name, phase_samples in phases.items():
        _require_samples(name, phase_samples)

    abd_min = robust_percentile(
        (a.mcp_abduction for a in samples.abduction_min),
        lower_percentile,
    )
    abd_max = robust_percentile(
        (a.mcp_abduction for a in samples.abduction_max),
        upper_percentile,
    )

    mcp_min = robust_percentile(
        (a.mcp_flexion for a in samples.neutral),
        lower_percentile,
    )
    mcp_max = robust_percentile(
        (a.mcp_flexion for a in samples.mcp_flexion_max),
        upper_percentile,
    )

    pip_min = robust_percentile(
        (a.pip_flexion for a in samples.neutral),
        lower_percentile,
    )
    pip_max = robust_percentile(
        (a.pip_flexion for a in samples.pip_flexion_max),
        upper_percentile,
    )

    return IndexCalibration(
        human_abduction=JointRange(
            min(abd_min, abd_max),
            max(abd_min, abd_max),
        ),
        human_mcp_flexion=JointRange(
            min(mcp_min, mcp_max),
            max(mcp_min, mcp_max),
        ),
        human_pip_flexion=JointRange(
            min(pip_min, pip_max),
            max(pip_min, pip_max),
        ),
    )


def phase_summary(
    samples: tuple[FingerJointAngles, ...],
) -> dict[str, float]:
    """Median per-angle summary for logging / operator review."""
    _require_samples("summary", samples)

    return {
        "mcp_abduction": robust_percentile(
            (a.mcp_abduction for a in samples), 50.0
        ),
        "mcp_flexion": robust_percentile(
            (a.mcp_flexion for a in samples), 50.0
        ),
        "pip_flexion": robust_percentile(
            (a.pip_flexion for a in samples), 50.0
        ),
    }
