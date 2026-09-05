from types import SimpleNamespace

import pytest

from robot.static_sampling import summarize_joint_samples


def frame(*values):
    return SimpleNamespace(
        joint_states=[
            SimpleNamespace(joint_id=f"j{index}", position=value)
            for index, value in enumerate(values)
        ]
    )


def test_summarize_joint_samples():
    results = summarize_joint_samples([frame(1.0, 2.0), frame(3.0, 4.0)])

    assert results[0].joint_id == "j0"
    assert results[0].count == 2
    assert results[0].mean == pytest.approx(2.0)
    assert results[0].stddev == pytest.approx(1.0)
    assert results[0].minimum == pytest.approx(1.0)
    assert results[0].maximum == pytest.approx(3.0)


def test_summarize_rejects_empty_samples():
    with pytest.raises(ValueError, match="empty"):
        summarize_joint_samples([])
