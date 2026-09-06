from pathlib import Path
import numpy as np
import pytest
from retargeting import ApexHandModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["right", "left"])
def model(request):
    return ApexHandModel(ROOT / f"vendor/rysen-sdk/urdf/apex_hand_{request.param}.urdf", request.param)


def test_sdk_order_and_distal_geometry(model):
    assert model.joint_names[:6] == ("f0_joint0", "f0_joint1", "f0_joint2", "f0_joint3", "f0_joint4", "f1_joint0")
    assert model.joint_names[-1] == "f4_joint3"
    q = model.neutral_keypoints
    np.testing.assert_allclose(np.linalg.norm(q[[4, 8, 12, 16, 20]] - q[[3, 7, 11, 15, 19]], axis=1),
                               [0.032, 0.028, 0.028, 0.028, 0.028], atol=1e-10)
    # Independent known URDF base placement + abduction-link length at neutral.
    sign = 1 if model.hand_side == "right" else -1
    np.testing.assert_allclose(q[5], [0.0069979, sign * 0.034496, 0.1315], atol=1e-6)


def test_analytical_jacobian_against_central_difference(model):
    q = model.lower + (model.upper - model.lower) * np.random.default_rng(42).uniform(0.2, 0.8, 21)
    _, jac = model.keypoints(q, with_jacobian=True)
    for i in range(21):
        step = np.zeros(21)
        step[i] = 1e-6
        numerical = (model.keypoints(q + step) - model.keypoints(q - step)) / 2e-6
        np.testing.assert_allclose(jac[:, :, i], numerical, atol=2e-9, rtol=1e-5)
    np.testing.assert_array_equal(jac[8, :, 9:], 0)


def test_reject_bad_qpos(model):
    with pytest.raises(ValueError):
        model.keypoints(np.full(21, np.nan))


def test_mimic_jacobian_and_independent_limits(model):
    assert model.num_independent_joints == 16
    x = model.independent_lower + 0.6 * (model.independent_upper - model.independent_lower)
    full = model.expand(x)
    assert np.all(full >= model.lower) and np.all(full <= model.upper)
    np.testing.assert_allclose(full[[4, 8, 12, 16, 20]], full[[3, 7, 11, 15, 19]])
    _, full_jac = model.keypoints(full, with_jacobian=True)
    jac = full_jac @ model.expansion
    for i in range(16):
        step = np.zeros(16)
        step[i] = 1e-6
        numerical = (model.keypoints(model.expand(x + step)) - model.keypoints(model.expand(x - step))) / 2e-6
        np.testing.assert_allclose(jac[:, :, i], numerical, atol=2e-9)
