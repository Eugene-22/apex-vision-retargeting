"""Minimal full-hand geometric representation in palm-local coordinates."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from human_hand.joint_angles import FingerJointAngles, compute_finger_angles, angle_between, normalize

FINGER_LANDMARKS = {
    "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20),
}

@dataclass(frozen=True)
class ThumbJointAngles:
    abduction: float
    rotation: float
    flexion: float
    ip_flexion: float

@dataclass(frozen=True)
class FullHandAngles:
    fingers: dict[str, FingerJointAngles]
    thumb: ThumbJointAngles

def compute_thumb_angles(local_points: np.ndarray) -> ThumbJointAngles:
    if local_points.shape != (21, 3):
        raise ValueError("Expected shape (21, 3)")
    cmc, mcp, ip, tip = (local_points[i] for i in (1, 2, 3, 4))
    cmc_vec = normalize(cmc - local_points[0])
    mcp_vec = normalize(mcp - cmc)
    ip_vec = normalize(ip - mcp)
    abduction = float(np.arctan2(cmc_vec[0], max(abs(cmc_vec[1]), 1e-8)))
    rotation = float(np.arctan2(mcp_vec[2], max(np.linalg.norm(mcp_vec[:2]), 1e-8)))
    flexion = float(np.arccos(np.clip(np.dot(cmc_vec, mcp_vec), -1.0, 1.0)))
    ip_flexion = angle_between(mcp_vec, ip_vec)
    return ThumbJointAngles(abduction, rotation, flexion, ip_flexion)

def compute_full_hand_angles(local_points: np.ndarray) -> FullHandAngles:
    zero = FingerJointAngles(0.0, 0.0, 0.0)
    fingers = {}
    for name, indices in FINGER_LANDMARKS.items():
        try:
            fingers[name] = compute_finger_angles(local_points, *indices)
        except ValueError:
            fingers[name] = zero
    try:
        thumb = compute_thumb_angles(local_points)
    except ValueError:
        thumb = ThumbJointAngles(0.0, 0.0, 0.0, 0.0)
    return FullHandAngles(fingers=fingers, thumb=thumb)
