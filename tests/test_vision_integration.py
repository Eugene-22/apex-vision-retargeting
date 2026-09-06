"""Opt-in real model test using Google's official right_hands.jpg fixture.

APEX_TEST_HAND_IMAGE=/path/to/right_hands.jpg APEX_TEST_MODEL=/path/to/model.task pytest
No test downloads models or images implicitly.
"""
import os
from pathlib import Path
import numpy as np
import pytest

IMAGE = os.environ.get("APEX_TEST_HAND_IMAGE")
MODEL = os.environ.get("APEX_TEST_MODEL")


@pytest.mark.skipif(not IMAGE or not MODEL, reason="Set model and official right-hand fixture paths for vision integration")
@pytest.mark.parametrize("mirrored", [False, True])
def test_real_right_hand_and_mirror(mirrored):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from perception import HandDetector
    from retargeting import Retargeter
    frame = cv2.imread(IMAGE)
    assert frame is not None
    if mirrored:
        frame = np.ascontiguousarray(frame[:, ::-1])
    with HandDetector(MODEL, "right", input_mirrored=mirrored) as detector:
        observation = detector.detect(frame, 0.)
        assert observation is not None and observation.hand_side == "right"
        assert detector.detect(np.zeros_like(frame), 1.) is None
    with HandDetector(MODEL, "left", input_mirrored=mirrored) as detector:
        assert detector.detect(frame, 0.) is None
    r = Retargeter.from_yaml(Path(__file__).resolve().parents[1] / "config/apex_hand.yaml")
    q = r.process(observation)
    assert q is not None and q.shape == (21,) and np.isfinite(q).all()
    np.testing.assert_allclose(q[[4, 8, 12, 16, 20]], q[[3, 7, 11, 15, 19]], atol=1e-12)

