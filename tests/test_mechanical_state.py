from types import SimpleNamespace

import pytest

from robot.mechanical_state import check_mimic_state, sdk_joint_state_by_id


def test_sdk_joint_state_by_id_indexes_all_states():
    states = SimpleNamespace(
        joint_states=[
            SimpleNamespace(joint_id="j2", position=0.4),
            SimpleNamespace(joint_id="j3", position=0.4),
        ]
    )
    assert set(sdk_joint_state_by_id(states)) == {"j2", "j3"}


def test_mimic_state_matches_expected_relation():
    states = SimpleNamespace(
        joint_states=[
            SimpleNamespace(joint_id="j2", position=0.4),
            SimpleNamespace(joint_id="j3", position=0.4),
        ]
    )
    result = check_mimic_state(states, "j2", "j3")
    assert result.consistent
    assert result.error == pytest.approx(0.0)


def test_mimic_state_detects_mismatch():
    states = SimpleNamespace(
        joint_states=[
            SimpleNamespace(joint_id="j2", position=0.4),
            SimpleNamespace(joint_id="j3", position=0.5),
        ]
    )
    result = check_mimic_state(states, "j2", "j3")
    assert not result.consistent
    assert result.error == pytest.approx(0.1)


def test_mimic_state_rejects_missing_joint():
    with pytest.raises(ValueError, match="both"):
        check_mimic_state(
            SimpleNamespace(joint_states=[SimpleNamespace(joint_id="j2", position=0.4)]),
            "j2",
            "j3",
        )
