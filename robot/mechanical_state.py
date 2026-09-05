"""Helpers for validating SDK mechanical joint state definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MimicStateCheck:
    source_joint: Any
    mimic_joint: Any
    source_position: float
    mimic_position: float
    expected_position: float
    error: float
    consistent: bool


def sdk_joint_state_by_id(sdk_joint_states: Any) -> dict[Any, Any]:
    states = getattr(sdk_joint_states, "joint_states", None)
    if states is None:
        states = getattr(sdk_joint_states, "joints", None)
    if states is None:
        states = list(sdk_joint_states)
    return {state.joint_id: state for state in states}


def check_mimic_state(
    sdk_joint_states: Any,
    source_joint: Any,
    mimic_joint: Any,
    multiplier: float = 1.0,
    offset: float = 0.0,
    tolerance: float = 1e-6,
) -> MimicStateCheck:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    states = sdk_joint_state_by_id(sdk_joint_states)
    if source_joint not in states or mimic_joint not in states:
        raise ValueError("source and mimic joints must both be present")

    source_position = float(states[source_joint].position)
    mimic_position = float(states[mimic_joint].position)
    expected_position = multiplier * source_position + offset
    error = mimic_position - expected_position
    return MimicStateCheck(
        source_joint=source_joint,
        mimic_joint=mimic_joint,
        source_position=source_position,
        mimic_position=mimic_position,
        expected_position=expected_position,
        error=error,
        consistent=abs(error) <= tolerance,
    )
