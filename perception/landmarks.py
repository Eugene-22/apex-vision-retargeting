"""21-point MediaPipe convention, meters, and wrist-local coordinates.

Frame construction adapted from AnyDexRetarget (MIT); see THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
import numpy as np

FINGER_CHAINS = np.arange(1, 21).reshape(5, 4)
TIP_INDICES = np.array([4, 8, 12, 16, 20])
HAND_CONNECTIONS = tuple((a, b) for chain in FINGER_CHAINS for a, b in
                         zip([0, *chain[:-1]], chain)) + ((5, 9), (9, 13), (13, 17))


def validate_side(side: str) -> str:
    if side not in ("left", "right"):
        raise ValueError("hand_side must be 'left' or 'right'")
    return side


def validate_keypoints(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("keypoints must be a finite (21, 3) array in meters")
    return points.copy()


def hand_frame(points) -> np.ndarray:
    """Columns form a right-handed wrist basis; reject collapsed palm geometry."""
    points = validate_keypoints(points)
    backward = points[0] - points[9]
    lateral = points[5] - points[9]
    length = np.linalg.norm(backward)
    normal = np.cross(backward, lateral)
    if length < 1e-6 or np.linalg.norm(lateral) < 1e-6:
        raise ValueError("Degenerate palm: wrist/MCP points coincide")
    if np.linalg.norm(normal) / (length * np.linalg.norm(lateral)) < 1e-3:
        raise ValueError("Degenerate palm: wrist/MCP points are collinear")
    x = backward / length
    # Choose the same lateral sign as AnyDexRetarget's SVD frame.
    y = -normal / np.linalg.norm(normal)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def to_hand_frame(points, hand_side="right") -> np.ndarray:
    """Remove global pose; +z points toward fingers, lateral y depends on side.

    This is the AnyDexRetarget operator-to-MANO convention, which aligns with
    ApexHand's palm axes. Input is metric world landmarks, never image pixels.
    """
    validate_side(hand_side)
    points = validate_keypoints(points)
    transform = (np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
                 if hand_side == "right" else
                 np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]))
    return (points - points[0]) @ hand_frame(points) @ transform


@dataclass(frozen=True)
class HandObservation:
    """One tracked hand. score is handedness confidence, not landmark accuracy."""
    keypoints: np.ndarray
    hand_side: str
    timestamp: float
    score: float = 1.0
    image_landmarks: np.ndarray | None = None

    def __post_init__(self):
        validate_side(self.hand_side)
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite, nonnegative seconds")
        if not np.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be in [0, 1]")
        points = validate_keypoints(self.keypoints)
        points.setflags(write=False)
        object.__setattr__(self, "keypoints", points)
        if self.image_landmarks is not None:
            normalized = validate_keypoints(self.image_landmarks)
            normalized.setflags(write=False)
            object.__setattr__(self, "image_landmarks", normalized)

