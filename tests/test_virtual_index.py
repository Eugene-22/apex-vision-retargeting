import numpy as np
import pytest

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration, JointRange
from retargeting.index_mapping import map_index_angles
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.virtual_index import (
    VirtualIndexChain,
    apex_index_ranges,
    is_continuous,
    is_monotonic,
    within_limits,
)


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


@pytest.fixture(scope="module")
def calibration() -> IndexCalibration:
    # Placeholder human ROM (radians). Flexion ranges include their
    # extension side so that human neutral (0) maps to Apex neutral
    # (0) — see CLAUDE.md 8.3.1. Replace with a real per-user
    # calibration in M13.
    return IndexCalibration(
        human_abduction=JointRange(-0.349066, 0.349066),
        human_mcp_flexion=JointRange(-0.349066, 1.570796),
        human_pip_flexion=JointRange(-0.087266, 1.745329),
    )


@pytest.fixture(scope="module")
def chain(model, calibration) -> VirtualIndexChain:
    return VirtualIndexChain(model, calibration)


# ---------------------------------------------------------------------------
# Apex ranges derived from URDF (single source of truth)
# ---------------------------------------------------------------------------

def test_apex_index_ranges_match_urdf(model):
    ranges = apex_index_ranges(model)

    j0 = model.joints["right_index_j0"]
    j1 = model.joints["right_index_j1"]
    j2 = model.joints["right_index_j2"]

    assert (ranges.j0_abduction.minimum, ranges.j0_abduction.maximum) == (
        j0.lower,
        j0.upper,
    )
    assert (ranges.j1_mcp_flexion.minimum, ranges.j1_mcp_flexion.maximum) == (
        j1.lower,
        j1.upper,
    )
    assert (ranges.j2_pip_flexion.minimum, ranges.j2_pip_flexion.maximum) == (
        j2.lower,
        j2.upper,
    )


# ---------------------------------------------------------------------------
# Chain composition
# ---------------------------------------------------------------------------

def test_forward_matches_mapping_then_fk(chain):
    human = FingerJointAngles(0.1, 0.5, 0.9)

    result = chain.forward(human)

    expected_apex = map_index_angles(
        human, chain.calibration, chain.apex_ranges
    )
    assert result.apex == expected_apex

    expected_tip = chain.model.index_tip_position(
        expected_apex.j0_abduction,
        expected_apex.j1_mcp_flexion,
        expected_apex.j2_pip_flexion,
    )
    np.testing.assert_allclose(result.tip, expected_tip)


# ---------------------------------------------------------------------------
# Direction + monotonicity through the full chain
# ---------------------------------------------------------------------------

def test_abduction_sweep_direction(chain):
    cal = chain.calibration

    j0s, ys = [], []
    for a in np.linspace(
        cal.human_abduction.minimum,
        cal.human_abduction.maximum,
        21,
    ):
        result = chain.forward(FingerJointAngles(a, 0.0, 0.0))
        j0s.append(result.apex.j0_abduction)
        ys.append(result.tip[1])

    assert is_monotonic(j0s, "increasing")
    assert is_monotonic(ys, "increasing")


def test_mcp_flexion_direction(chain):
    cal = chain.calibration

    j1s, zs = [], []
    for f in np.linspace(
        0.0, cal.human_mcp_flexion.maximum, 21
    ):
        result = chain.forward(FingerJointAngles(0.0, f, 0.0))
        j1s.append(result.apex.j1_mcp_flexion)
        zs.append(result.tip[2])

    assert is_monotonic(j1s, "increasing")
    assert is_monotonic(zs, "decreasing")


def test_pip_flexion_direction(chain):
    cal = chain.calibration

    j2s, zs = [], []
    for f in np.linspace(
        0.0, cal.human_pip_flexion.maximum, 21
    ):
        result = chain.forward(FingerJointAngles(0.0, 0.0, f))
        j2s.append(result.apex.j2_pip_flexion)
        zs.append(result.tip[2])

    assert is_monotonic(j2s, "increasing")
    assert is_monotonic(zs, "decreasing")


def test_combined_curl_joints_monotonic_and_continuous(chain):
    cal = chain.calibration

    j1s, j2s = [], []
    tips = []
    mcp = np.linspace(
        0.0, cal.human_mcp_flexion.maximum, 21
    )
    pip = np.linspace(
        0.0, cal.human_pip_flexion.maximum, 21
    )
    for m, p in zip(mcp, pip):
        result = chain.forward(FingerJointAngles(0.0, m, p))
        j1s.append(result.apex.j1_mcp_flexion)
        j2s.append(result.apex.j2_pip_flexion)
        tips.append(result.tip)

    assert is_monotonic(j1s, "increasing")
    assert is_monotonic(j2s, "increasing")

    # As both joints flex together the fingertip eventually wraps
    # around the palm (a fist), so tip-z is NOT monotonic over the
    # full closure — but the trajectory must remain continuous
    # (no jumps > 20 mm between consecutive samples).
    steps = np.linalg.norm(np.diff(np.asarray(tips), axis=0), axis=1)
    assert np.all(steps <= 0.02)


# ---------------------------------------------------------------------------
# Joint limits hold across the whole human ROM
# ---------------------------------------------------------------------------

def test_targets_within_urdf_limits_over_rom(chain):
    cal = chain.calibration
    ranges = chain.apex_ranges

    for a in np.linspace(
        cal.human_abduction.minimum, cal.human_abduction.maximum, 5
    ):
        for m in np.linspace(
            cal.human_mcp_flexion.minimum,
            cal.human_mcp_flexion.maximum,
            5,
        ):
            for p in np.linspace(
                cal.human_pip_flexion.minimum,
                cal.human_pip_flexion.maximum,
                5,
            ):
                apex = chain.forward(
                    FingerJointAngles(a, m, p)
                ).apex

                assert within_limits(
                    [apex.j0_abduction],
                    ranges.j0_abduction.minimum,
                    ranges.j0_abduction.maximum,
                )
                assert within_limits(
                    [apex.j1_mcp_flexion],
                    ranges.j1_mcp_flexion.minimum,
                    ranges.j1_mcp_flexion.maximum,
                )
                assert within_limits(
                    [apex.j2_pip_flexion],
                    ranges.j2_pip_flexion.minimum,
                    ranges.j2_pip_flexion.maximum,
                )


# ---------------------------------------------------------------------------
# Validation predicates
# ---------------------------------------------------------------------------

def test_within_limits():
    assert within_limits([0.0, 0.5, 1.0], 0.0, 1.0)
    assert not within_limits([0.0, 1.5], 0.0, 1.0)
    assert not within_limits([-0.1, 0.5], 0.0, 1.0)


def test_is_monotonic():
    assert is_monotonic([0.0, 1.0, 2.0], "increasing")
    assert is_monotonic([2.0, 1.0, 0.0], "decreasing")
    assert not is_monotonic([0.0, 2.0, 1.0], "increasing")
    assert not is_monotonic([0.0, 1.0, 0.5], "decreasing")


def test_is_monotonic_unknown_direction():
    with pytest.raises(ValueError):
        is_monotonic([0.0, 1.0], "sideways")


def test_is_continuous():
    assert is_continuous([0.0, 1.0, 2.0], 1.0)
    assert not is_continuous([0.0, 2.0], 1.0)
