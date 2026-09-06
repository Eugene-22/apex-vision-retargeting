"""Vector matching and pinch-aware IK, inspired by AnyDexRetarget.

Analytical gradients are passed to SciPy SLSQP. Distances are evaluated in cm
for conditioning, whereas all public keypoints and thresholds are in meters.
"""
import time
import numpy as np
from scipy.optimize import minimize
from perception.landmarks import FINGER_CHAINS, TIP_INDICES, validate_keypoints


class HandOptimizer:
    def __init__(self, model, mode="adaptive", huber_delta=0.02,
                 regularization=0.02, contact_weight=2.0, direction_weight=0.5,
                 pinch_close=0.02, pinch_open=0.05, pinch_alpha=0.7,
                 max_iterations=60, ftol=1e-6):
        if mode not in ("adaptive", "vector"):
            raise ValueError("optimizer mode must be 'adaptive' or 'vector'")
        values = [huber_delta, regularization, contact_weight, direction_weight,
                  pinch_close, pinch_open, pinch_alpha, ftol]
        if not np.isfinite(values).all():
            raise ValueError("Optimizer parameters must be finite")
        if (huber_delta <= 0 or min(regularization, contact_weight, direction_weight) < 0
                or not 0 <= pinch_close < pinch_open or not 0 <= pinch_alpha <= 1
                or isinstance(max_iterations, bool) or int(max_iterations) != max_iterations
                or max_iterations < 1 or ftol <= 0):
            raise ValueError("Invalid optimizer parameters")
        self.model, self.mode = model, mode
        self.huber_delta = huber_delta * 100
        self.regularization = regularization
        self.contact_weight, self.direction_weight = contact_weight, direction_weight
        self.pinch_close, self.pinch_open, self.pinch_alpha = pinch_close, pinch_open, pinch_alpha
        self.max_iterations, self.ftol = int(max_iterations), ftol
        self.last_qpos = None

    def pinch_alphas(self, points):
        distances = np.linalg.norm(points[TIP_INDICES[1:]] - points[4], axis=1)
        non_thumb = self.pinch_alpha * np.clip(
            (self.pinch_open - distances) / (self.pinch_open - self.pinch_close), 0, 1)
        return np.r_[non_thumb.max(), non_thumb] if self.mode == "adaptive" else np.zeros(5)

    def loss_and_grad(self, qpos, targets, pinch_targets, alphas, previous=None):
        points, jac = self.model.keypoints(qpos, with_jacobian=True)
        points, jac = points * 100, jac * 100
        cost, grad = 0., np.zeros(21)

        def add_huber(residual, derivative, weights, delta):
            distance = np.linalg.norm(residual, axis=1)
            loss = np.where(distance <= delta, 0.5 * distance**2,
                            delta * (distance - 0.5 * delta))
            factor = np.minimum(1., delta / np.maximum(distance, 1e-12))
            return float(weights @ loss), np.einsum(
                "n,ni,nij->j", weights * factor, residual, derivative)

        indices = FINGER_CHAINS[:, 1:].reshape(-1)
        weights = np.repeat(1 - alphas, 3) / 15
        c, g = add_huber(points[indices] - targets[indices] * 100,
                          jac[indices], weights, self.huber_delta)
        cost += c
        grad += g
        if self.mode == "adaptive":
            c, g = add_huber(points[TIP_INDICES] - pinch_targets[TIP_INDICES] * 100,
                              jac[TIP_INDICES], alphas / 5, self.huber_delta)
            cost += c
            grad += g
            # Distal directions retain the fingertip orientation during contact.
            vectors = points[TIP_INDICES] - points[TIP_INDICES - 1]
            lengths = np.linalg.norm(vectors, axis=1)
            unit = vectors / np.maximum(lengths[:, None], 1e-12)
            target_vectors = targets[TIP_INDICES] - targets[TIP_INDICES - 1]
            target_unit = target_vectors / np.maximum(np.linalg.norm(target_vectors, axis=1)[:, None], 1e-12)
            normalization = (np.eye(3)[None] - unit[:, :, None] * unit[:, None, :]) / lengths[:, None, None]
            unit_jac = normalization @ (jac[TIP_INDICES] - jac[TIP_INDICES - 1])
            c, g = add_huber(unit - target_unit, unit_jac, alphas * self.direction_weight / 5, 0.5)
            cost += c
            grad += g
            partner = 1 + int(np.argmax(alphas[1:]))
            contact = points[4] - points[TIP_INDICES[partner]]
            weight = self.contact_weight * alphas[partner]
            cost += weight * float(contact @ contact)
            grad += 2 * weight * contact @ (jac[4] - jac[TIP_INDICES[partner]])
        if previous is not None:
            delta = qpos - previous
            cost += self.regularization * float(delta @ delta)
            grad += 2 * self.regularization * delta
        return cost, grad

    def solve(self, targets, pinch_targets, alphas):
        targets = validate_keypoints(targets)
        pinch_targets = validate_keypoints(pinch_targets)
        previous = None if self.last_qpos is None else self.last_qpos.copy()
        initial = self.model.neutral.copy() if previous is None else previous.copy()
        start = time.perf_counter()
        full_objective = lambda q: self.loss_and_grad(q, targets, pinch_targets, alphas, previous)
        def objective(independent):
            cost, grad = full_objective(self.model.expand(independent))
            return cost, grad @ self.model.expansion
        initial_cost = full_objective(initial)[0]
        result = minimize(objective, initial[self.model.independent_indices], method="SLSQP", jac=True,
                          bounds=list(zip(self.model.independent_lower, self.model.independent_upper)),
                          options={"maxiter": self.max_iterations, "ftol": self.ftol})
        candidate = np.asarray(result.x)
        finite = candidate.shape == (self.model.num_independent_joints,) and np.isfinite(candidate).all()
        if finite:
            candidate = self.model.expand(np.clip(candidate, self.model.independent_lower, self.model.independent_upper))
            cost = full_objective(candidate)[0]
        else:
            cost = float("inf")
        # Iteration limits may still yield useful progress, but never promote a
        # failed solve, nonfinite result, or a cost increase to a valid command.
        accepted = bool(finite and np.isfinite(cost) and cost <= initial_cost + 1e-9
                        and (result.success or result.status == 9))
        qpos = candidate.copy() if accepted else initial
        if accepted:
            self.last_qpos = qpos.copy()
        info = {"success": bool(result.success), "accepted": accepted,
                "status": int(result.status), "message": str(result.message),
                "iterations": int(result.nit), "cost": float(cost if accepted else initial_cost),
                "initial_cost": float(initial_cost), "solve_ms": (time.perf_counter() - start) * 1000,
                "pinch_alphas": alphas.copy()}
        return qpos, info

    def reset(self):
        self.last_qpos = None
