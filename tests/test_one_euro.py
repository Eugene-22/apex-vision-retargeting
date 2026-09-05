import pytest

from filtering.one_euro import OneEuroFilter, OneEuroJointFilter
from robot.apex_joint_names import ApexActiveJointTarget


def test_first_sample_passes_through():
    filter_ = OneEuroFilter()
    assert filter_.filter(0.7, 0.01) == pytest.approx(0.7)


def test_filter_reduces_step_response():
    filter_ = OneEuroFilter(min_cutoff_hz=1.0, beta=0.0)
    filter_.filter(0.0, 0.01)
    filtered = filter_.filter(1.0, 0.01)
    assert 0.0 < filtered < 1.0


def test_reset_makes_next_sample_passthrough():
    filter_ = OneEuroFilter()
    filter_.filter(0.0, 0.01)
    filter_.filter(1.0, 0.01)
    filter_.reset()
    assert filter_.filter(0.4, 0.01) == pytest.approx(0.4)


def test_filter_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="min_cutoff"):
        OneEuroFilter(min_cutoff_hz=0.0)
    with pytest.raises(ValueError, match="beta"):
        OneEuroFilter(beta=-1.0)
    filter_ = OneEuroFilter()
    with pytest.raises(ValueError, match="value"):
        filter_.filter(float("nan"), 0.01)
    with pytest.raises(ValueError, match="dt"):
        filter_.filter(0.1, 0.0)


def test_joint_filter_preserves_canonical_shape():
    filter_ = OneEuroJointFilter(min_cutoff_hz=2.0, beta=0.1)
    target = ApexActiveJointTarget(values=tuple(float(i) for i in range(16)))
    result = filter_.filter(target, 0.01)
    assert result == target
    next_result = filter_.filter(ApexActiveJointTarget(values=(1.0,) * 16), 0.01)
    assert len(next_result.values) == 16
