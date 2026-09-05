import numpy as np
import pytest

from robot.apex_urdf import (
    DEFAULT_RIGHT_URDF,
    INDEX_TIP_LINK,
    ApexUrdfModel,
)


@pytest.fixture(scope="module")
def model() -> ApexUrdfModel:
    return ApexUrdfModel(DEFAULT_RIGHT_URDF)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_urdf_parsed(model):
    assert len(model.joints) > 0
    assert model.root_link == "right_palm_link"


def test_joint_space_counts(model):
    assert len(model.active_joints) == 16
    assert len(model.revolute_joints) == 21
    assert len(model.mimic_joints) == 5


def test_mimic_joint_names(model):
    assert set(model.mimic_joints) == {
        "right_thumb_j4",
        "right_index_j3",
        "right_middle_j3",
        "right_ring_j3",
        "right_pinky_j3",
    }


def test_index_j3_mimics_j2(model):
    mimic = model.joints["right_index_j3"].mimic

    assert mimic is not None
    assert mimic.target == "right_index_j2"
    assert mimic.multiplier == 1.0
    assert mimic.offset == 0.0


def test_index_joint_limits(model):
    j0 = model.joints["right_index_j0"]
    j1 = model.joints["right_index_j1"]
    j2 = model.joints["right_index_j2"]

    assert (j0.lower, j0.upper) == pytest.approx(
        (-0.4363, 0.4363)
    )
    assert (j1.lower, j1.upper) == pytest.approx(
        (-0.349, 1.5707)
    )
    assert (j2.lower, j2.upper) == pytest.approx(
        (-0.0872, 1.7453)
    )


def test_index_chain_hierarchy(model):
    chain = model.chain_to(INDEX_TIP_LINK)

    # Root-side joints first, ending at the tip.
    assert chain[0] == "right_palm_base_fix"
    assert chain[-1] == "right_index_tip_fix"

    assert "right_index_j0" in chain
    assert "right_index_j3" in chain

    # The index finger hangs off the palm base.
    assert model.joints["right_index_j0"].parent == "right_palm_base"
    assert model.joints["right_index_j0"].child == "right_index_link0"


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def test_fk_root_is_identity(model):
    t = model.forward_kinematics("right_palm_link", {})
    np.testing.assert_allclose(t, np.eye(4), atol=1e-9)


def test_fk_neutral_tip_position(model):
    p = model.index_tip_position(0.0, 0.0, 0.0)

    # Independently hand-computed from the URDF origin/axis data.
    expected = np.array([0.00765028, 0.034496, 0.232472])

    np.testing.assert_allclose(p, expected, atol=1e-4)


def test_fk_j0_abduction_direction(model):
    # j0 = MCP abduction/adduction: the tip sweeps laterally (+y),
    # while the x coordinate stays fixed.
    j0_values = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])

    tips = np.array(
        [model.index_tip_position(q, 0.0, 0.0) for q in j0_values]
    )

    ys = tips[:, 1]
    xs = tips[:, 0]

    # Monotonic lateral sweep.
    assert np.all(np.diff(ys) > 0)
    # x is unchanged by abduction (rotation about the x axis).
    np.testing.assert_allclose(xs, xs[0], atol=1e-6)


def test_fk_j1_flexion_direction(model):
    # j1 = MCP flexion/extension: positive flexion curls the tip, so its
    # z coordinate decreases as flexion increases; y stays fixed.
    j1_values = np.array([0.0, 0.3, 0.6, 0.9, 1.2])

    tips = np.array(
        [model.index_tip_position(0.0, q, 0.0) for q in j1_values]
    )

    zs = tips[:, 2]
    ys = tips[:, 1]

    assert np.all(np.diff(zs) < 0)
    np.testing.assert_allclose(ys, ys[0], atol=1e-6)


def test_fk_j2_flexion_direction(model):
    # j2 = PIP flexion/extension: positive flexion also curls the tip
    # (z decreases).
    j2_values = np.array([0.0, 0.35, 0.7, 1.05, 1.4])

    tips = np.array(
        [model.index_tip_position(0.0, 0.0, q) for q in j2_values]
    )

    zs = tips[:, 2]
    ys = tips[:, 1]

    assert np.all(np.diff(zs) < 0)
    np.testing.assert_allclose(ys, ys[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Mimic (DIP coupling)
# ---------------------------------------------------------------------------

def test_mimic_matches_explicit_mechanical_joint(model):
    # Active space (j0, j1, j2) with automatic mimic resolution must
    # equal the full mechanical space with j3 = j2 set explicitly.
    j2 = 0.7

    p_active = model.index_tip_position(0.0, 0.0, j2)

    q_mechanical = {
        "right_index_j0": 0.0,
        "right_index_j1": 0.0,
        "right_index_j2": j2,
        "right_index_j3": j2,  # DIP mimics PIP 1:1
    }

    p_mechanical = model.link_position(
        INDEX_TIP_LINK, q_mechanical
    )

    np.testing.assert_allclose(
        p_active, p_mechanical, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_fk_missing_active_joint_raises(model):
    with pytest.raises(ValueError):
        model.forward_kinematics(
            INDEX_TIP_LINK,
            {"right_index_j0": 0.0},  # j1, j2 missing
        )


def test_fk_unknown_link_raises(model):
    with pytest.raises(ValueError):
        model.chain_to("does_not_exist")


# ---------------------------------------------------------------------------
# Active joint limits
# ---------------------------------------------------------------------------

def test_active_joint_limits_cover_active_space(model):
    limits = model.active_joint_limits()

    assert set(limits) == set(model.active_joints)
    assert len(limits) == 16

    for name, limit in limits.items():
        joint = model.joints[name]
        assert limit.lower == joint.lower
        assert limit.upper == joint.upper
        assert limit.lower < limit.upper


def test_joint_limit_api_matches_index_limits(model):
    j0 = model.joint_limit("right_index_j0")
    j1 = model.joint_limit("right_index_j1")
    j2 = model.joint_limit("right_index_j2")

    assert (j0.lower, j0.upper) == pytest.approx((-0.4363, 0.4363))
    assert (j1.lower, j1.upper) == pytest.approx((-0.349, 1.5707))
    assert (j2.lower, j2.upper) == pytest.approx((-0.0872, 1.7453))


def test_validate_active_joint_values_accepts_mid_range_values(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    model.validate_active_joint_values(q)


def test_validate_active_joint_values_rejects_out_of_limit(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    limit = model.joint_limit("right_index_j1")
    q["right_index_j1"] = limit.upper + 0.01

    with pytest.raises(ValueError):
        model.validate_active_joint_values(q)


def test_validate_active_joint_values_rejects_mimic_joint(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    q["right_index_j3"] = 0.0

    with pytest.raises(ValueError):
        model.validate_active_joint_values(q)


def test_validate_active_joint_values_rejects_fixed_joint(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    q["right_palm_base_fix"] = 0.0

    with pytest.raises(ValueError):
        model.validate_active_joint_values(q)


def test_validate_active_joint_values_rejects_unknown_joint(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    q["not_a_joint"] = 0.0

    with pytest.raises(ValueError):
        model.validate_active_joint_values(q)


def test_validate_active_joint_values_rejects_missing_joint(model):
    q = {}
    for name, limit in model.active_joint_limits().items():
        q[name] = 0.5 * (limit.lower + limit.upper)

    del q["right_index_j0"]

    with pytest.raises(ValueError):
        model.validate_active_joint_values(q)

