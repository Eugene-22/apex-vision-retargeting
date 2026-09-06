import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from perception import HandObservation, HandDetector, to_hand_frame, hand_frame, ObservationWriter, ReplaySource


@pytest.fixture
def hand():
    points = np.zeros((21, 3))
    for f, x in enumerate([0.055, 0.035, 0., -0.025, -0.045]):
        for j in range(4):
            points[1 + 4 * f + j] = [x, 0.05 + j * 0.023 + (0.03 if f else 0), 0.004 * j * j]
    return points


def test_wrist_frame_rigid_invariance(hand):
    transformed = hand @ Rotation.from_euler("xyz", [0.8, -0.3, 1.2]).as_matrix().T + [0.4, -2, 0.7]
    np.testing.assert_allclose(to_hand_frame(hand), to_hand_frame(transformed), atol=1e-14)
    frame = hand_frame(hand)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
    assert np.linalg.det(frame) == pytest.approx(1)
    local = to_hand_frame(hand)
    assert local[9, 2] > 0 and local[5, 1] > local[9, 1]


def test_left_right_mirror(hand):
    right = to_hand_frame(hand, "right")
    left = to_hand_frame(hand * [-1, 1, 1], "left")
    np.testing.assert_allclose(left, right * [1, -1, 1], atol=1e-14)


@pytest.mark.parametrize("bad", [np.zeros((21, 3)), np.full((21, 3), np.nan),
                                 np.zeros((20, 3)), np.tile([1, 2, 3], (21, 1))])
def test_degenerate_geometry(bad):
    with pytest.raises(ValueError):
        to_hand_frame(bad)


def test_recording_roundtrip_and_gaps(hand, tmp_path):
    path = tmp_path / "hand.jsonl"
    with ObservationWriter(path) as writer:
        writer.write(0., HandObservation(hand, "right", 0., 0.91))
        writer.write(0.1, None)
        writer.write(0.2, HandObservation(hand, "left", 0.2))
    frames = list(ReplaySource(path))
    np.testing.assert_allclose(frames[0][1].keypoints, hand)
    assert frames[0][1].score == 0.91
    assert frames[1][1] is None and frames[2][1] is None
    assert not frames[0][1].keypoints.flags.writeable


def test_bad_replay_timestamp(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"version": 1, "timestamp": float("nan"), "hand": None}) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(ReplaySource(path))


def test_detector_selects_physical_side_and_handles_missing(hand):
    detector = object.__new__(HandDetector)
    detector.hand_side, detector.min_confidence, detector.input_mirrored = "right", 0.5, False
    landmarks = [NS(x=x, y=y, z=z) for x, y, z in hand]
    result = NS(handedness=[[NS(category_name="Left", score=0.99)], [NS(category_name="Right", score=0.9)]],
                hand_world_landmarks=[landmarks, landmarks], hand_landmarks=[landmarks, landmarks])
    observation = detector._observation(result, 1.)
    assert observation.hand_side == "right" and observation.score == 0.9
    np.testing.assert_allclose(observation.keypoints, hand)
    np.testing.assert_allclose(observation.image_landmarks[:, 0], hand[:, 0])
    detector.input_mirrored = True
    mirrored = detector._observation(result, 1.)
    np.testing.assert_allclose(mirrored.keypoints, hand)
    np.testing.assert_allclose(mirrored.image_landmarks[:, 0], 1 - hand[:, 0])
    assert detector._observation(NS(handedness=[], hand_world_landmarks=[], hand_landmarks=[]), 2.) is None


def test_detector_rgb_mirror_and_monotonic_milliseconds():
    detector = object.__new__(HandDetector)
    detector.hand_side, detector.min_confidence, detector.input_mirrored = "right", 0.5, True
    detector._last_timestamp, detector._last_ms = -1., -1
    detector._mp = NS(Image=lambda **kw: kw["data"], ImageFormat=NS(SRGB=1))
    calls = []
    def detect(image, timestamp):
        calls.append((image, timestamp))
        return NS(handedness=[], hand_world_landmarks=[], hand_landmarks=[])
    detector._detector = NS(detect_for_video=detect)
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    detector.detect(bgr, 0.)
    detector.detect(bgr, 0.0001)
    np.testing.assert_array_equal(calls[0][0], bgr[:, ::-1, ::-1])
    assert [c[1] for c in calls] == [0, 1]
    with pytest.raises(ValueError):
        detector.detect(bgr, 0.0001)
