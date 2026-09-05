"""Model-independent human hand state boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from human_hand.full_state import FullHandAngles, compute_full_hand_angles
from human_hand.joint_angles import FingerJointAngles, compute_index_angles
from human_hand.palm_frame import build_palm_frame, landmarks_to_numpy
from perception.quality import assess_landmark_quality


@dataclass(frozen=True)
class HumanHandState:
    """Normalized hand state consumed by retargeting."""

    handedness: str
    timestamp: float
    confidence: float
    world_landmarks: np.ndarray
    local_landmarks: np.ndarray
    index_angles: FingerJointAngles
    quality_score: float = 0.0
    is_valid: bool = True
    full_angles: FullHandAngles | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite seconds")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not math.isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be in [0, 1]")
        if self.world_landmarks.shape != (21, 3):
            raise ValueError("world_landmarks must have shape (21, 3)")
        if self.local_landmarks.shape != (21, 3):
            raise ValueError("local_landmarks must have shape (21, 3)")

    @classmethod
    def from_mediapipe_result(cls, result: Any, timestamp: float) -> "HumanHandState | None":
        """Convert one MediaPipe result; return None when no hand is detected."""
        world_sets = getattr(result, "hand_world_landmarks", None) or []
        if not world_sets:
            return None
        world = landmarks_to_numpy(world_sets[0])
        frame = build_palm_frame(world)
        local = frame.world_to_local(world)
        try:
            full_angles = compute_full_hand_angles(local)
        except ValueError:
            full_angles = None

        handedness = "unknown"
        handedness_sets = getattr(result, "handedness", None) or []
        if handedness_sets and handedness_sets[0]:
            category = handedness_sets[0][0]
            handedness = str(getattr(category, "category_name", getattr(category, "display_name", "unknown")))
        confidence = 0.0
        if handedness_sets and handedness_sets[0]:
            confidence = float(getattr(handedness_sets[0][0], "score", 0.0))

        quality_score, is_valid = assess_landmark_quality(world, confidence)
        index_angles = compute_index_angles(local)
        return cls(handedness, timestamp, confidence, world, local, index_angles, quality_score, is_valid, full_angles)
