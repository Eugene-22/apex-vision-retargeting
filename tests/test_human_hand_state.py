from types import SimpleNamespace

import numpy as np

from human_hand.state import HumanHandState


def landmarks():
    points = np.zeros((21, 3), dtype=float)
    points[0] = [0.0, 0.0, 0.0]
    points[5] = [0.2, 0.5, 0.0]
    points[9] = [0.0, 0.6, 0.0]
    points[17] = [-0.2, 0.5, 0.0]
    points[6] = [0.2, 0.8, 0.0]
    points[7] = [0.2, 1.0, 0.0]
    points[8] = [0.2, 1.2, 0.0]
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


def test_from_mediapipe_result_uses_world_landmarks():
    result = SimpleNamespace(
        hand_world_landmarks=[landmarks()],
        hand_landmarks=[],
        handedness=[[SimpleNamespace(category_name="Right", score=0.91)]],
    )
    state = HumanHandState.from_mediapipe_result(result, timestamp=12.5)

    assert state is not None
    assert state.handedness == "Right"
    assert state.confidence == 0.91
    assert state.timestamp == 12.5
    assert state.world_landmarks.shape == (21, 3)
    assert state.local_landmarks[0].tolist() == [0.0, 0.0, 0.0]
    assert state.full_angles is not None
    assert set(state.full_angles.fingers) == {"index", "middle", "ring", "pinky"}
    assert state.index_angles == state.full_angles.fingers["index"]


def test_from_mediapipe_result_returns_none_without_hand():
    result = SimpleNamespace(hand_world_landmarks=[], handedness=[])
    assert HumanHandState.from_mediapipe_result(result, timestamp=1.0) is None
