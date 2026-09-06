from pathlib import Path
from types import SimpleNamespace as NS
import json
import subprocess
import sys
import numpy as np
import pytest
from perception import HandObservation, ObservationWriter
from retargeting import Retargeter
from retargeting.optimizer import HandOptimizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["right", "left"])
def retargeter(request):
    return Retargeter.from_yaml(ROOT / "config/apex_hand.yaml", request.param)


@pytest.mark.parametrize("mode", ["vector", "adaptive"])
def test_loss_gradient(retargeter, mode):
    model = retargeter.model
    optimizer = HandOptimizer(model, mode=mode)
    q = model.lower + 0.4 * (model.upper - model.lower)
    targets = model.keypoints(q + 0.05)
    alphas = np.array([0.7, 0.7, 0.3, 0., 0.1]) if mode == "adaptive" else np.zeros(5)
    previous = q - 0.03
    fun = lambda x: optimizer.loss_and_grad(x, targets, targets, alphas, previous)
    _, gradient = fun(q)
    numerical = np.empty(21)
    for i in range(21):
        step = np.zeros(21)
        step[i] = 1e-6
        numerical[i] = (fun(q + step)[0] - fun(q - step)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=2e-6)


def test_recovers_reachable_pose_and_respects_limits(retargeter):
    retargeter.optimizer.mode = "vector"
    retargeter.optimizer.max_iterations = 150
    model = retargeter.model
    truth = model.expand(model.independent_lower + 0.4 * (model.independent_upper - model.independent_lower))
    targets = model.keypoints(truth)
    q, info = retargeter.retarget_verbose(targets, apply_filter=False, input_frame="robot")
    errors = np.linalg.norm(model.keypoints(q)[[4, 8, 12, 16, 20]] - targets[[4, 8, 12, 16, 20]], axis=1)
    assert info["accepted"]
    assert info["cost"] < info["initial_cost"] * 0.001
    assert errors.max() < 0.001
    assert np.all(q >= model.lower) and np.all(q <= model.upper)
    np.testing.assert_allclose(q[[4, 8, 12, 16, 20]], q[[3, 7, 11, 15, 19]], atol=1e-12)


def test_pinch_contact_improves_over_vector(retargeter):
    model = retargeter.model
    targets = model.keypoints(model.lower + 0.35 * (model.upper - model.lower))
    # Both solvers see identical posture targets; adaptive additionally closes index/thumb.
    result = []
    for mode in ("vector", "adaptive"):
        optimizer = HandOptimizer(model, mode=mode, max_iterations=100)
        q, info = optimizer.solve(targets, targets, np.array([0.7, 0.7, 0, 0, 0]) if mode == "adaptive" else np.zeros(5))
        assert info["accepted"]
        points = model.keypoints(q)
        result.append(np.linalg.norm(points[4] - points[8]))
    assert result[1] < result[0] * 0.5


def test_rate_limit_reset_and_tracking_gaps(retargeter):
    model = retargeter.model
    retargeter.optimizer.mode = "vector"
    points = model.neutral_keypoints
    q1 = retargeter.retarget(points, input_frame="robot", timestamp=0.)
    bent = model.keypoints(model.lower + 0.5 * (model.upper - model.lower))
    q2 = retargeter.retarget(bent, input_frame="robot", timestamp=0.02)
    assert np.max(np.abs(q2 - q1)) <= 3 * 0.02 + 1e-12
    assert retargeter.process(None, 0.1) is None
    assert retargeter.optimizer.last_qpos is not None
    assert retargeter.process(None, 0.8) is None
    assert retargeter.optimizer.last_qpos is None
    assert retargeter._last_output is None
    with pytest.raises(ValueError, match="increasing"):
        retargeter.process(None, 0.7)


def test_invalid_input_does_not_poison_state(retargeter):
    good = retargeter.model.neutral_keypoints
    retargeter.retarget(good, input_frame="robot", timestamp=0.)
    previous = retargeter.optimizer.last_qpos.copy()
    with pytest.raises(ValueError):
        retargeter.retarget(np.full((21, 3), np.nan), timestamp=0.1)
    np.testing.assert_array_equal(retargeter.optimizer.last_qpos, previous)
    collapsed = HandObservation(np.zeros((21, 3)), retargeter.hand_side, 0.2)
    assert retargeter.process(collapsed) is None


def test_failed_solver_keeps_previous_output(retargeter, monkeypatch):
    points = retargeter.model.neutral_keypoints
    previous = retargeter.retarget(points, input_frame="robot")
    monkeypatch.setattr("retargeting.optimizer.minimize", lambda *a, **kw: NS(
        x=np.full(21, np.nan), success=False, status=4, message="failure", nit=1))
    q, info = retargeter.retarget_verbose(points, input_frame="robot")
    assert not info["accepted"]
    np.testing.assert_array_equal(q, previous)


def test_config_resolves_relative_to_yaml(retargeter, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    other = Retargeter.from_yaml(ROOT / "config/apex_hand.yaml", retargeter.hand_side)
    assert other.model.urdf_path.is_file()


def test_replay_cli_end_to_end(tmp_path):
    retargeter = Retargeter.from_yaml(ROOT / "config/apex_hand.yaml")
    path, output = tmp_path / "input.jsonl", tmp_path / "joints.jsonl"
    with ObservationWriter(path) as writer:
        writer.write(0., HandObservation(retargeter.model.neutral_keypoints, "right", 0.))
        writer.write(0.1, None)
    result = subprocess.run([sys.executable, "-m", "retargeting", "--replay", str(path),
                             "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2 and len(rows[0]["qpos"]) == 21 and rows[0]["valid"]
    assert rows[1]["qpos"] is None and not rows[1]["valid"]


def test_replay_cli_prevents_input_overwrite(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text("unchanged")
    result = subprocess.run([sys.executable, "-m", "retargeting", "--replay", str(path),
                             "--record", str(path)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert path.read_text() == "unchanged"
