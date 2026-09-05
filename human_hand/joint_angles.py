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


def angle_between(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Return unsigned angle between two vectors in radians.
    """

    a = normalize(vector_a)
    b = normalize(vector_b)

    cosine = np.dot(a, b)

    # Avoid numerical errors such as:
    # acos(1.00000001)
    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.arccos(cosine)
    )


@dataclass
class FingerJointAngles:
    """
    Joint angles of one non-thumb finger.

    All angles are stored in radians.
    """

    mcp_abduction: float
    mcp_flexion: float
    pip_flexion: float


def compute_finger_angles(
    local_points: np.ndarray,
    mcp_index: int,
    pip_index: int,
    dip_index: int,
    tip_index: int,
) -> FingerJointAngles:
    """
    Compute MCP abduction, MCP flexion and
    PIP flexion from palm-local landmarks.

    local_points:
        shape [21, 3]

    Coordinate convention:
        +x: pinky -> index
        +y: wrist -> fingers
        +z: x cross y

    Sign convention:
        Positive MCP flexion = curling the finger
        toward the palm (toward -z).

        Positive MCP abduction = moving the finger
        away from the middle finger (toward +x for
        the index finger).

    Note: the direction of +z (palm vs dorsal) depends
    on handedness, so these signs must be re-verified
    against real Apex motion in M2.5.
    """

    if local_points.shape != (21, 3):
        raise ValueError(
            f"Expected shape (21, 3), "
            f"got {local_points.shape}"
        )

    mcp = local_points[mcp_index]
    pip = local_points[pip_index]
    dip = local_points[dip_index]
    tip = local_points[tip_index]

    # Bone directions
    proximal = normalize(
        pip - mcp
    )

    middle = normalize(
        dip - pip
    )

    # Kept for later DIP analysis
    distal = normalize(
        tip - dip
    )

    # --------------------------------
    # MCP abduction / adduction
    # --------------------------------

    # Reference direction of this finger:
    # wrist -> MCP projected onto palm plane.
    reference = np.array(
        [
            mcp[0],
            mcp[1],
            0.0,
        ],
        dtype=np.float64,
    )

    reference = normalize(reference)

    # Proximal phalanx projected onto palm plane.
    proximal_planar = np.array(
        [
            proximal[0],
            proximal[1],
            0.0,
        ],
        dtype=np.float64,
    )

    proximal_planar = normalize(
        proximal_planar
    )

    # Signed angle around palm-local z axis.
    #
    # cross(proximal_planar, reference) is positive when
    # the proximal phalanx points toward +x, i.e. abduction
    # (away from the middle finger).
    cross_z = np.cross(
        proximal_planar,
        reference,
    )[2]

    dot = np.dot(
        reference,
        proximal_planar,
    )

    mcp_abduction = np.arctan2(
        cross_z,
        dot,
    )


    # --------------------------------
    # MCP flexion / extension
    # --------------------------------

    planar_length = np.hypot(
        proximal[0],
        proximal[1],
    )

    # Flexion (curling toward the palm) moves the
    # proximal phalanx toward -z, so negate to make
    # flexion positive.
    mcp_flexion = -np.arctan2(
        proximal[2],
        planar_length,
    )

    # --------------------------------
    # PIP flexion
    #
    # Straight finger -> 0 rad
    # --------------------------------

    pip_flexion = angle_between(
        proximal,
        middle,
    )

    return FingerJointAngles(
        mcp_abduction=float(
            mcp_abduction
        ),
        mcp_flexion=float(
            mcp_flexion
        ),
        pip_flexion=float(
            pip_flexion
        ),
    )


def compute_index_angles(
    local_points: np.ndarray,
) -> FingerJointAngles:
    """
    MediaPipe index finger:

    MCP = 5
    PIP = 6
    DIP = 7
    TIP = 8
    """

    return compute_finger_angles(
        local_points=local_points,
        mcp_index=5,
        pip_index=6,
        dip_index=7,
        tip_index=8,
    )