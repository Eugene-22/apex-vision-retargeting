"""Adapt the bundled Rysen Python SDK, not the older upstream SDK spelling."""
import importlib.util
import sys
import threading
import time
import numpy as np


class RobotError(RuntimeError):
    pass


# SDK IDs are verified by enum name before any connection is attempted.
SDK_JOINT_ENUM_NAMES = (
    "THUMB_CMC_ABD", "THUMB_CMC_ROT", "THUMB_CMC_FLEX", "THUMB_MCP_FLEX", "THUMB_IP_FLEX",
    "INDEX_MCP_ABD", "INDEX_MCP_FLEX", "INDEX_PIP_FLEX", "INDEX_DIP_FLEX",
    "MIDDLE_MCP_ABD", "MIDDLE_MCP_FLEX", "MIDDLE_PIP_FLEX", "MIDDLE_DIP_FLEX",
    "RING_MCP_ABD", "RING_MCP_FLEX", "RING_PIP_FLEX", "RING_DIP_FLEX",
    "LITTLE_MCP_ABD", "LITTLE_MCP_FLEX", "LITTLE_PIP_FLEX", "LITTLE_DIP_FLEX")


def enum_value(value):
    return int(getattr(value, "value", value))


def load_sdk(path):
    if not path.is_file():
        raise RobotError(f"SDK wrapper not found: {path}")
    spec = importlib.util.spec_from_file_location("apex_bundled_rysen_sdk", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RobotError("Cannot load bundled Rysen SDK. Use Python 3.10 and the vendor's "
                         "Linux runtime dependencies (or the rysen_sdk Docker image). "
                         f"Original error: {exc}") from exc
    return module


class SDKBackend:
    """Connection and feedback only until enable() is explicitly called."""
    def __init__(self, config, model, sdk_module=None, clock=time.monotonic):
        self.config, self.model = config, model
        self._module, self._sdk = sdk_module, None
        self._clock = clock
        self._feedback = threading.Condition()
        self._positions, self._received_at = None, None
        self._feedback_error = None
        self._feedback_sequence = 0
        # Keep the exact bound-method object alive through native teardown.
        # SDK 1.5.2 drops its callback reference while the GIL is released;
        # destroying the last bound-method reference there can segfault.
        self._feedback_callback = None
        self._registered = False
        self._enable_attempted = False
        self._connected = False

    def _check(self, operation, result):
        # ERROR_CODE_OK is zero, so bool(result) would invert success/failure.
        if result != self._module.ErrorCode.ERROR_CODE_OK:
            raise RobotError(f"{operation} failed: {result}")

    def connect(self):
        if self._sdk is not None:
            raise RobotError("Backend is already connected; create a new controller to reconnect")
        if not self.config.ip:
            raise RobotError("A right-hand device IP is required for hardware operation")
        self._module = self._module or load_sdk(self.config.sdk_path)
        module = self._module
        self._joint_ids = []
        for expected, name in enumerate(SDK_JOINT_ENUM_NAMES):
            joint = getattr(module.JointId, "JOINT_ID_" + name)
            if enum_value(joint) != expected:
                raise RobotError(f"SDK joint order mismatch: {name}")
            self._joint_ids.append(joint)
        self._sdk = module.Rysen()
        try:
            self._check("load right URDF", self._sdk.set_robot_model_from_urdf(str(self.model.urdf_path)))
            self._check("connect", self._sdk.connect(self.config.ip, module.ConnectionType.CONNECTION_TYPE_ETHERNET))
            self._connected = True
            if self._sdk.get_hand_dir() != module.HandDir.RIGHT:
                raise RobotError("Connected device is not a RIGHT hand; refusing to enable it")
            self.check_health()
            speeds, accels, torques = [], [], []
            for i, joint in enumerate(self._joint_ids):
                speed, accel = module.MaxJointSpeed(), module.MaxJointAccel()
                speed.joint_id = accel.joint_id = joint
                speed.speed = min(self.config.max_speed, float(self.model.velocity_limits[i]))
                accel.accel = self.config.max_accel
                speeds.append(speed)
                accels.append(accel)
            for i in range(5):
                torque = module.MaxFingerTorque()
                torque.finger_id, torque.torque = module.FingerId(i), self.config.finger_torque_pct
                torques.append(torque)
            self._check("configure speed", self._sdk.set_max_joint_speed(speeds))
            self._check("configure acceleration", self._sdk.set_max_joint_accel(accels))
            self._check("configure torque", self._sdk.set_max_finger_torque(torques))
            # Mark attempted registration so failures also trigger unregister.
            self._feedback_callback = self._on_feedback
            self._registered = True
            self._check("register feedback", self._sdk.register_joint_states_callback(
                self._feedback_callback, freq_hz=self.config.feedback_rate))
            deadline = self._clock() + self.config.startup_timeout
            with self._feedback:
                while self._positions is None and self._feedback_error is None:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise RobotError("Timed out waiting for complete measured joint positions")
                    self._feedback.wait(min(remaining, 0.05))
            self.read_positions()
        except BaseException as exc:
            # Initialization failures also clean up outside controller.__exit__.
            print(f"Stopping SDK initialization: {exc}", file=sys.stderr, flush=True)
            try:
                self.close()
            except Exception as cleanup_error:
                raise RobotError(f"{exc}; cleanup also failed: {cleanup_error}") from exc
            raise

    def _on_feedback(self, states):
        # Never throw through a C++ callback. Invalid feedback latches a fault.
        try:
            values = {}
            for joint in states.joint_states:
                index = enum_value(joint.joint_id)
                if index in values or not 0 <= index < 21:
                    raise ValueError("Duplicate or unknown joint ID")
                values[index] = float(joint.position)
            if set(values) != set(range(21)):
                raise ValueError("Feedback must contain all 21 joint IDs")
            positions = np.array([values[i] for i in range(21)])
            if not np.isfinite(positions).all():
                raise ValueError("Nonfinite joint feedback")
            with self._feedback:
                self._positions = positions
                self._received_at = self._clock()
                self._feedback_sequence += 1
                self._feedback.notify_all()
        except Exception as exc:
            with self._feedback:
                self._feedback_error = str(exc)
                self._feedback.notify_all()

    def read_positions(self):
        with self._feedback:
            if self._feedback_error:
                raise RobotError(f"Invalid feedback: {self._feedback_error}")
            if self._positions is None or self._clock() - self._received_at > self.config.feedback_timeout:
                raise RobotError("Joint feedback is missing or stale")
            return self._positions.copy()

    def check_health(self):
        if self._sdk is None or not self._connected or not self._sdk.is_connected():
            raise RobotError("Robot connection was lost")
        self._check("hardware health", self._sdk.get_hardware_error_code())

    def enable(self):
        self.check_health()
        self.read_positions()
        self._enable_attempted = True
        self._check("enable fingers", self._sdk.set_all_fingers_enabled())

    def send(self, qpos):
        qpos = self.model.validate_qpos(qpos)
        if (np.any(qpos < self.model.lower - 1e-9) or np.any(qpos > self.model.upper + 1e-9)
                or not np.allclose(qpos, self.model.expand(qpos[self.model.independent_indices]), atol=1e-6, rtol=0)):
            raise RobotError("Command violates right-hand limits or mimic coupling")
        if not self._enable_attempted:
            raise RobotError("Enable the right hand before sending commands")
        params = []
        for joint, position in zip(self._joint_ids, qpos):
            item = self._module.MoveJPositionFollowParam()
            item.id, item.position = joint, float(position)
            params.append(item)
        self._check("position follow", self._sdk.move_j_position_follow(params))

    def disable(self):
        if self._sdk is not None and self._enable_attempted:
            self._check("disable fingers", self._sdk.set_all_fingers_disabled())
            self._enable_attempted = False

    def return_to_neutral(self, *, timeout=2.0):
        """Vendor example's blocking MoveJoint to zero, then verify feedback.

        timeout bounds feedback verification after MoveJoint returns, not the
        native blocking move itself. Never enable an unstarted/faulted session.
        """
        self.check_health()
        self.read_positions()
        if not self._enable_attempted:
            raise RobotError("Cannot return to neutral before fingers are enabled")
        target = np.zeros(len(self._joint_ids))
        if (np.any(target < self.model.lower) or np.any(target > self.model.upper)
                or not np.allclose(target, self.model.expand(target[self.model.independent_indices]))):
            raise RobotError("Zero pose is incompatible with the right-hand URDF")
        commands = []
        for i, joint in enumerate(self._joint_ids):
            item = self._module.JointControlParam()
            item.joint_id, item.position = joint, 0.0
            item.velocity = min(0.5, self.config.max_speed, float(self.model.velocity_limits[i]))
            item.acceleration = min(2.0, self.config.max_accel)
            commands.append(item)
        with self._feedback:
            sequence = self._feedback_sequence
        self._check("return to neutral", self._sdk.move_joint(commands))
        deadline = time.monotonic() + timeout
        while True:
            self.check_health()
            with self._feedback:
                positions = self.read_positions()
                if (self._feedback_sequence > sequence
                        and np.all(np.abs(positions - target) <= np.radians(2.0))):
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RobotError("Return to neutral was not confirmed by fresh feedback within 2 degrees")
                self._feedback.wait(min(remaining, 0.02))

    def close(self):
        if self._sdk is None:
            return
        errors = []
        try:
            try:
                self.disable()
            except Exception as exc:
                errors.append(str(exc))
            if self._registered:
                try:
                    self._check("unregister feedback", self._sdk.unregister_joint_states_callback())
                except Exception as exc:
                    errors.append(str(exc))
            try:
                self._check("disconnect", self._sdk.disconnect())
            except Exception as exc:
                errors.append(str(exc))
        finally:
            self._sdk = None
            # Release under the GIL, after unregister/disconnect/destruction.
            self._feedback_callback = None
            self._connected = self._registered = self._enable_attempted = False
        if errors:
            raise RobotError("; ".join(errors))


class DryRunBackend:
    """In-memory command sink. No native SDK import, sockets, or physical motion."""
    def __init__(self, model):
        self.positions = model.neutral.copy()
        self.connected = self.enabled = False
        self.command_count = 0

    def connect(self):
        self.connected = True

    def check_health(self):
        if not self.connected:
            raise RobotError("Dry-run backend is disconnected")

    def read_positions(self):
        self.check_health()
        return self.positions.copy()

    def enable(self):
        self.check_health()
        self.enabled = True

    def send(self, qpos):
        if not self.enabled:
            raise RobotError("Dry-run backend is not enabled")
        self.positions = np.array(qpos, copy=True)
        self.command_count += 1

    def disable(self):
        self.enabled = False

    def return_to_neutral(self):
        self.check_health()
        if not self.enabled:
            raise RobotError("Dry-run backend is not enabled")
        self.positions = np.zeros_like(self.positions)

    def close(self):
        self.disable()
        self.connected = False
