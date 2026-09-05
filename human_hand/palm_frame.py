from dataclasses import dataclass

import numpy as np


EPS = 1e-8


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm < EPS:
        raise ValueError(
            "Cannot normalize near-zero vector."
        )

    return vector / norm


def landmarks_to_numpy(landmarks) -> np.ndarray:
    """
    Convert MediaPipe landmarks to [21, 3] numpy array.
    """

    return np.asarray(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in landmarks
        ],
        dtype=np.float64,
    )


@dataclass
class PalmFrame:
    """
    Human palm local coordinate frame.

    rotation[:, 0] = local x-axis in world coordinates
    rotation[:, 1] = local y-axis in world coordinates
    rotation[:, 2] = local z-axis in world coordinates
    """

    origin: np.ndarray
    rotation: np.ndarray
    scale: float

    def world_to_local(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Transform [N, 3] points from world coordinates
        into the palm-local coordinate frame.
        """

        centered = points - self.origin

        local = centered @ self.rotation

        return local / self.scale


def build_palm_frame(
    landmarks: np.ndarray,
) -> PalmFrame:
    """
    Build palm-local coordinate frame from 21 hand landmarks.
    """

    if landmarks.shape != (21, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3), "
            f"got {landmarks.shape}"
        )

    wrist = landmarks[0]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    pinky_mcp = landmarks[17]

    # +x: pinky MCP -> index MCP
    x_axis = normalize(
        index_mcp - pinky_mcp
    )

    # Approximate +y: wrist -> middle MCP
    y_raw = middle_mcp - wrist

    # Remove x component to guarantee x ⟂ y
    y_orthogonal = (
        y_raw
        - np.dot(y_raw, x_axis) * x_axis
    )

    y_axis = normalize(
        y_orthogonal
    )

    # Right-handed coordinate frame
    z_axis = normalize(
        np.cross(
            x_axis,
            y_axis,
        )
    )

    # Recompute y to remove numerical error
    y_axis = normalize(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    rotation = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    # Normalize different human hand sizes.
    #
    # Use average wrist -> index/pinky MCP distance
    # as characteristic palm size.
    scale = 0.5 * (
        np.linalg.norm(index_mcp - wrist)
        +
        np.linalg.norm(pinky_mcp - wrist)
    )

    if scale < EPS:
        raise ValueError(
            "Invalid palm scale."
        )

    return PalmFrame(
        origin=wrist,
        rotation=rotation,
        scale=scale,
    )