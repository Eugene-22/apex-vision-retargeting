import numpy as np
import pytest

from perception.quality import assess_landmark_quality, temporal_quality


def test_finite_landmarks_use_confidence_as_quality():
    score, valid = assess_landmark_quality(np.zeros((21, 3)), 0.8)
    assert score == pytest.approx(0.8)
    assert valid


def test_nonfinite_landmarks_are_invalid():
    points = np.zeros((21, 3))
    points[3, 0] = np.nan
    score, valid = assess_landmark_quality(points, 0.8)
    assert score == 0.0
    assert not valid


def test_temporal_quality_rejects_large_jump():
    score, valid = temporal_quality(np.ones((21, 3)), np.zeros((21, 3)), 0.1, max_speed=5.0)
    assert score == 0.0
    assert not valid


def test_temporal_quality_accepts_first_sample():
    assert temporal_quality(np.zeros((21, 3)), None, None) == (1.0, True)
