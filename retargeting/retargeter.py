"""High-level world-landmark to ApexHand joint-angle pipeline."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from perception.landmarks import (FINGER_CHAINS, HandObservation, to_hand_frame,
                                  validate_keypoints, validate_side)
from .kinematics import ApexHandModel
from .optimizer import HandOptimizer


class Retargeter:
    def __init__(self, config, hand_side="right", base_dir=None):
        self.hand_side = validate_side(hand_side)
        self.config = deepcopy(config)
        base_dir = Path(base_dir or Path(__file__).resolve().parents[1])
        robot = self.config.get("robot", {})
        path = robot.get("urdf", "vendor/rysen-sdk/urdf/apex_hand_{hand_side}.urdf")
        path = Path(path.format(hand_side=hand_side))
        self.model = ApexHandModel(path if path.is_absolute() else base_dir / path,
                                   hand_side, robot.get("tip_offsets"))
        self.optimizer = HandOptimizer(self.model, **self.config.get("optimizer", {}))
        options = self.config.get("retarget", {})
        allowed = {"smoothing_alpha", "tracking_timeout", "max_joint_speed", "nominal_fps",
                   "scale_to_robot", "match_bone_lengths", "segment_scaling", "rotation_degrees"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown retarget options: {sorted(unknown)}")
        self.smoothing_alpha = float(options.get("smoothing_alpha", 0.35))
        self.tracking_timeout = float(options.get("tracking_timeout", 0.5))
        self.nominal_fps = float(options.get("nominal_fps", 30))
        speed = float(options.get("max_joint_speed", 3.0))
        if (not np.isfinite([self.smoothing_alpha, self.tracking_timeout, self.nominal_fps, speed]).all()
                or not 0 < self.smoothing_alpha <= 1 or min(self.tracking_timeout, self.nominal_fps, speed) <= 0):
            raise ValueError("Invalid filtering/timing options")
        self.speed_limits = np.minimum(self.model.velocity_limits, speed)
        self.scale_to_robot = options.get("scale_to_robot", True)
        self.match_bone_lengths = options.get("match_bone_lengths", True)
        if not isinstance(self.scale_to_robot, bool) or not isinstance(self.match_bone_lengths, bool):
            raise ValueError("Scaling switches must be booleans")
        self.segment_scaling = np.asarray(options.get("segment_scaling", np.ones((5, 3))), dtype=float)
        if (self.segment_scaling.shape != (5, 3) or not np.isfinite(self.segment_scaling).all()
                or np.any(self.segment_scaling <= 0)):
            raise ValueError("segment_scaling must be a positive (5, 3) array")
        rotation = options.get("rotation_degrees", [0, 0, 0])
        if isinstance(rotation, dict):
            rotation = rotation.get(hand_side, [0, 0, 0])
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3,) or not np.isfinite(rotation).all():
            raise ValueError("rotation_degrees must contain three finite XYZ angles")
        self.rotation = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
        neutral = self.model.neutral_keypoints
        self._bone_lengths = np.linalg.norm(np.diff(neutral[FINGER_CHAINS], axis=1), axis=2)
        self.reset()

    @classmethod
    def from_yaml(cls, yaml_path, hand_side="right"):
        path = Path(yaml_path).resolve()
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return cls(config, hand_side, base_dir=path.parent)

    @property
    def num_joints(self):
        return self.model.num_joints

    @property
    def joint_names(self):
        return self.model.joint_names

    def prepare_targets(self, raw_keypoints, input_frame="world"):
        points = validate_keypoints(raw_keypoints)
        if input_frame == "robot":
            # Explicit path for already calibrated URDF-frame targets and FK tests.
            return points, points.copy(), self.optimizer.pinch_alphas(points)
        if input_frame != "world":
            raise ValueError("input_frame must be 'world' or 'robot'")
        # Pinch detection uses human distances before robot-size scaling.
        alphas = self.optimizer.pinch_alphas(points)
        local = to_hand_frame(points, self.hand_side) @ self.rotation.T
        if self.scale_to_robot:
            local *= np.linalg.norm(self.model.neutral_keypoints[9]) / np.linalg.norm(local[9])
        uniform = local.copy()
        targets = local.copy()
        for finger, chain in enumerate(FINGER_CHAINS):
            vectors = np.diff(local[chain], axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            if np.any(lengths < 1e-6):
                raise ValueError("Degenerate finger: adjacent landmarks coincide")
            if self.match_bone_lengths:
                vectors *= (self._bone_lengths[finger] / lengths)[:, None]
                # Long-finger MCP bases are fixed by the robot palm geometry.
                # Thumb CMC remains mobile, so keep its observed target position.
                if finger > 0:
                    targets[chain[0]] = self.model.neutral_keypoints[chain[0]]
            vectors *= self.segment_scaling[finger, :, None]
            for segment in range(3):
                targets[chain[segment + 1]] = targets[chain[segment]] + vectors[segment]
        pinch_targets = targets.copy()
        partner = 1 + int(np.argmax(alphas[1:]))
        if alphas[partner] > 0:
            beta = alphas[partner] / max(self.optimizer.pinch_alpha, 1e-12)
            for tip in (4, int(FINGER_CHAINS[partner, -1])):
                pinch_targets[tip] = (1 - beta) * targets[tip] + beta * uniform[tip]
        return targets, pinch_targets, alphas

    def retarget(self, raw_keypoints, apply_filter=True, input_frame="world", timestamp=None):
        qpos, _ = self.retarget_verbose(raw_keypoints, apply_filter, input_frame, timestamp)
        return qpos

    def retarget_verbose(self, raw_keypoints, apply_filter=True, input_frame="world", timestamp=None):
        targets, pinch_targets, alphas = self.prepare_targets(raw_keypoints, input_frame)
        if timestamp is not None:
            self._validate_timestamp(timestamp)
            if self._last_valid_time is not None and timestamp - self._last_valid_time > self.tracking_timeout:
                self._reset_motion()
        dt = (1 / self.nominal_fps if timestamp is None or self._last_valid_time is None
              else timestamp - self._last_valid_time)
        qpos, info = self.optimizer.solve(targets, pinch_targets, alphas)
        unfiltered = qpos.copy()
        if info["accepted"]:
            if apply_filter and self._last_output is not None:
                # Equivalent configured alpha at nominal FPS; robust to variable frame intervals.
                alpha = 1 - (1 - self.smoothing_alpha) ** (dt * self.nominal_fps)
                qpos = self._last_output + alpha * (qpos - self._last_output)
                qpos = self._last_output + np.clip(qpos - self._last_output,
                           -self.speed_limits * dt, self.speed_limits * dt)
            qpos = np.clip(qpos, self.model.lower, self.model.upper)
            self._last_output = qpos.copy()
            self.optimizer.last_qpos = qpos.copy()
            self._last_valid_time = timestamp
        elif self._last_output is not None:
            qpos = self._last_output.copy()
        if timestamp is not None:
            self._last_timestamp = timestamp
        info.update({"qpos_unfiltered": unfiltered, "qpos": qpos.copy(),
                     "targets": targets, "pinch_targets": pinch_targets})
        self.last_info = info
        return qpos, info

    def process(self, observation: HandObservation | None, timestamp=None):
        """Streaming API: missing, wrong-side or unusable frames yield None.

        Supply timestamps even for missing frames. No command is emitted while
        tracking is lost. Long gaps clear the warm start and filter together.
        """
        if observation is not None:
            if timestamp is not None and timestamp != observation.timestamp:
                raise ValueError("Timestamp does not match observation")
            timestamp = observation.timestamp
        if timestamp is None:
            raise ValueError("process requires a timestamp for missing frames")
        self._validate_timestamp(timestamp)
        if observation is not None and observation.hand_side == self.hand_side:
            try:
                qpos, info = self.retarget_verbose(observation.keypoints, timestamp=timestamp)
                return qpos if info["accepted"] else None
            except ValueError as exc:
                self.last_info = {"accepted": False, "message": str(exc)}
        else:
            self.last_info = {"accepted": False, "message": "Hand not detected"}
        self._last_timestamp = timestamp
        if self._last_valid_time is not None and timestamp - self._last_valid_time > self.tracking_timeout:
            self._reset_motion()
        return None

    def _validate_timestamp(self, timestamp):
        if (not np.isfinite(timestamp) or timestamp < 0
                or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
            raise ValueError("Timestamps must be finite, nonnegative and strictly increasing")

    def _reset_motion(self):
        self.optimizer.reset()
        self._last_output = None
        self._last_valid_time = None

    def reset(self):
        self._reset_motion()
        self._last_timestamp = None
        self.last_info = {}

