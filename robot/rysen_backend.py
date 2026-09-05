"""
Rysen SDK backend for real-time Apex Hand control.

This module is the only project robot backend layer that imports the vendor
``rysen_apexhand_sdk`` package. Higher layers should use ``ApexInterface`` and
canonical ``APEX_ACTIVE_JOINTS`` values in radians.

Commands are sent as position targets in the canonical active-joint order.
"""

from __future__ import annotations

import sys
import time
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, APEX_OFFICIAL_LIMITS, ApexActiveJointTarget
from robot.interface import ApexCommandResult, ApexJointState

DEFAULT_RYSEN_IP = "192.168.0.103"

SDK_JOINT_ID_ATTRS: dict[str, str] = {
    "right_thumb_j0": "JOINT_ID_THUMB_J0",
    "right_thumb_j1": "JOINT_ID_THUMB_J1",
    "right_thumb_j2": "JOINT_ID_THUMB_J2",
    "right_thumb_j3": "JOINT_ID_THUMB_J3",
    "right_index_j0": "JOINT_ID_INDEX_J0",
    "right_index_j1": "JOINT_ID_INDEX_J1",
    "right_index_j2": "JOINT_ID_INDEX_J2",
    "right_middle_j0": "JOINT_ID_MIDDLE_J0",
    "right_middle_j1": "JOINT_ID_MIDDLE_J1",
    "right_middle_j2": "JOINT_ID_MIDDLE_J2",
    "right_ring_j0": "JOINT_ID_RING_J0",
    "right_ring_j1": "JOINT_ID_RING_J1",
    "right_ring_j2": "JOINT_ID_RING_J2",
    "right_pinky_j0": "JOINT_ID_PINKY_J0",
    "right_pinky_j1": "JOINT_ID_PINKY_J1",
    "right_pinky_j2": "JOINT_ID_PINKY_J2",
    "right_thumb_j4": "JOINT_ID_THUMB_J4",
    "right_index_j3": "JOINT_ID_INDEX_J3",
    "right_middle_j3": "JOINT_ID_MIDDLE_J3",
    "right_ring_j3": "JOINT_ID_RING_J3",
    "right_pinky_j3": "JOINT_ID_PINKY_J3",
}


class RysenBackendError(RuntimeError):
    """Raised when the Rysen SDK backend cannot complete an operation."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vendor_python_path(root: Path | None = None) -> Path:
    base = root if root is not None else project_root()
    return base.parent / "rysen-sdk" / "python"


def ensure_vendor_python_path(root: Path | None = None) -> Path:
    path = vendor_python_path(root)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def import_rysen_sdk(root: Path | None = None) -> tuple[Any, Any, Any, Any]:
    ensure_vendor_python_path(root)
    from rysen_apexhand_sdk import ConnectionType, ErrorCode, JointId, Rysen

    return Rysen, ConnectionType, ErrorCode, JointId


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def sdk_joint_id_map(joint_id_enum: Any) -> dict[Any, str]:
    mapping: dict[Any, str] = {}
    for joint_name in (*APEX_ACTIVE_JOINTS, "right_thumb_j4", "right_index_j3", "right_middle_j3", "right_ring_j3", "right_pinky_j3"):
        attr = SDK_JOINT_ID_ATTRS[joint_name]
        try:
            sdk_id = getattr(joint_id_enum, attr)
        except AttributeError as exc:
            raise RysenBackendError(
                f"SDK JointId is missing expected attribute {attr}"
            ) from exc
        mapping[sdk_id] = joint_name
    return mapping


def extract_sdk_joint_positions(
    sdk_joint_states: Any,
    sdk_to_project_joint: dict[Any, str],
) -> ApexActiveJointTarget:
    raw_states = getattr(sdk_joint_states, "joint_states", None)
    if raw_states is None:
        raw_states = getattr(sdk_joint_states, "joints", None)
    if raw_states is None:
        try:
            raw_states = list(sdk_joint_states)
        except TypeError as exc:
            raise RysenBackendError("SDK joint states are not iterable") from exc

    values_by_name: dict[str, float] = {}
    for state in raw_states:
        joint_id = getattr(state, "joint_id", None)
        if joint_id not in sdk_to_project_joint:
            continue
        position = getattr(state, "position", None)
        if position is None:
            raise RysenBackendError(
                f"SDK joint state for {enum_name(joint_id)} has no position"
            )
        values_by_name[sdk_to_project_joint[joint_id]] = float(position)

    missing = [name for name in APEX_ACTIVE_JOINTS if name not in values_by_name]
    if missing:
        raise RysenBackendError(
            "Missing active SDK joint states: " + ", ".join(missing)
        )

    return ApexActiveJointTarget.from_mapping(values_by_name)


class RealApexHand:
    """Real-time Rysen SDK Apex backend."""

    def __init__(
        self,
        ip: str = DEFAULT_RYSEN_IP,
        log_dir: str | Path | None = None,
        sdk: Any | None = None,
        connection_type: Any | None = None,
        error_code: Any | None = None,
        joint_id_enum: Any | None = None,
        auto_connect: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ip = ip
        self._clock = clock
        self._closed = False
        self._connected = False

        if sdk is None:
            Rysen, ConnectionType, ErrorCode, JointId = import_rysen_sdk()
            sdk = Rysen()
            if connection_type is None:
                connection_type = ConnectionType.CONNECTION_TYPE_ETHERNET
            if error_code is None:
                error_code = ErrorCode
            if joint_id_enum is None:
                joint_id_enum = JointId
        else:
            if connection_type is None or error_code is None or joint_id_enum is None:
                raise ValueError(
                    "Injected sdk requires connection_type, error_code, and joint_id_enum"
                )

        self._sdk = sdk
        self._connection_type = connection_type
        self._error_code = error_code
        self._sdk_to_project_joint = sdk_joint_id_map(joint_id_enum)
        self._project_to_sdk_joint = {name: sdk_id for sdk_id, name in self._sdk_to_project_joint.items()}

        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._sdk.set_log_path(str(log_dir))
            self._sdk.enable_logging(True)

        if auto_connect:
            self.connect()

    @property
    def is_connected(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self._sdk.is_connected())
        except AttributeError:
            return self._connected

    def connect(self) -> None:
        if self._closed:
            raise RysenBackendError("RealApexHand is closed")
        if self.is_connected:
            self._connected = True
            return

        ret = self._sdk.connect(self._ip, self._connection_type)
        if ret != self._error_code.ERROR_CODE_OK:
            raise RysenBackendError(f"SDK connect failed: {enum_name(ret)}")
        self._connected = True

    def configure_motion(self, *, max_speed: float = 0.5, max_accel: float = 1.0, torque_nmm: float = 200.0) -> None:
        """Configure conservative SDK-wide motion bounds when supported."""
        try:
            from rysen_apexhand_sdk import JointId, MaxJointAccel, MaxJointSpeed
        except ImportError:
            return
        speeds, accels = [], []
        for joint_id in self._sdk_to_project_joint:
            speed, accel = MaxJointSpeed(), MaxJointAccel()
            speed.joint_id = accel.joint_id = joint_id
            speed.speed = float(max_speed)
            accel.accel = float(max_accel)
            speeds.append(speed); accels.append(accel)
        self._sdk.set_max_joint_speed(speeds)
        self._sdk.set_max_joint_accel(accels)

    def enable(self) -> None:
        try:
            from rysen_apexhand_sdk import ErrorCode
            ret = self._sdk.set_all_fingers_enabled()
            if ret != ErrorCode.ERROR_CODE_OK:
                raise RysenBackendError(f"enable fingers failed: {enum_name(ret)}")
        except AttributeError:
            from rysen_apexhand_sdk import ErrorCode, FingerId
            ids = [getattr(FingerId, n) for n in ("FINGER_ID_THUMB", "FINGER_ID_INDEX", "FINGER_ID_MIDDLE", "FINGER_ID_RING", "FINGER_ID_PINKY") if hasattr(FingerId, n)]
            ret = self._sdk.set_finger_enabled(ids)
            if ret != ErrorCode.ERROR_CODE_OK:
                raise RysenBackendError(f"enable fingers failed: {enum_name(ret)}")

    def disable(self) -> None:
        if self._closed:
            return
        try:
            self._sdk.set_all_fingers_disabled()
        except AttributeError:
            try:
                from rysen_apexhand_sdk import FingerId
                ids = [getattr(FingerId, n) for n in ("FINGER_ID_THUMB", "FINGER_ID_INDEX", "FINGER_ID_MIDDLE", "FINGER_ID_RING", "FINGER_ID_PINKY") if hasattr(FingerId, n)]
                self._sdk.set_finger_disabled(ids)
            except Exception:
                pass

    def command_position(self, target: ApexActiveJointTarget) -> ApexCommandResult:
        """Send one official position-follow tick with hardware-safe limits."""
        if self._closed or not self.is_connected:
            raise RysenBackendError("RealApexHand is not connected")
        try:
            from rysen_apexhand_sdk import ErrorCode, create_move_j_position_follow_param
        except ImportError as exc:
            raise RysenBackendError("official position-follow API is unavailable") from exc
        values = target.as_dict()
        applied_values = {}
        clipped = []
        for name in APEX_ACTIVE_JOINTS:
            lo, hi = APEX_OFFICIAL_LIMITS[name]
            value = float(values[name])
            if not math.isfinite(value):
                raise RysenBackendError(f"non-finite target for {name}")
            applied = min(hi - 1e-3, max(lo + 1e-3, value))
            applied_values[name] = applied
            if applied != value:
                clipped.append(name)
        coupled = {
            "right_thumb_j4": "right_thumb_j3",
            "right_index_j3": "right_index_j2",
            "right_middle_j3": "right_middle_j2",
            "right_ring_j3": "right_ring_j2",
            "right_pinky_j3": "right_pinky_j2",
        }
        for destination, source in coupled.items():
            applied_values[destination] = applied_values[source]
        params = [
            create_move_j_position_follow_param(
                self._project_to_sdk_joint[name], applied_values[name], 200.0
            )
            for name in (*APEX_ACTIVE_JOINTS, *coupled)
        ]
        ret = self._sdk.move_j_position_follow(params)
        if ret == ErrorCode.ERROR_CODE_OUT_OF_RANGE:
            raise RysenBackendError("move_j_position_follow rejected out-of-range target")
        if ret != ErrorCode.ERROR_CODE_OK:
            raise RysenBackendError(f"move_j_position_follow failed: {enum_name(ret)}")
        applied = ApexActiveJointTarget.from_mapping({name: applied_values[name] for name in APEX_ACTIVE_JOINTS})
        return ApexCommandResult(target, applied, self._clock(), tuple(clipped))

    def get_joint_state(self) -> ApexJointState:
        if self._closed:
            raise RysenBackendError("RealApexHand is closed")
        if not self.is_connected:
            raise RysenBackendError("RealApexHand is not connected")

        position = extract_sdk_joint_positions(
            self._sdk.get_joint_states(),
            self._sdk_to_project_joint,
        )
        return ApexJointState(position=position, timestamp=self._clock())

    def close(self) -> None:
        if self._closed:
            return
        if self._connected or self.is_connected:
            ret = self._sdk.disconnect()
            if ret != self._error_code.ERROR_CODE_OK:
                raise RysenBackendError(f"SDK disconnect failed: {enum_name(ret)}")
        self._connected = False
        self._closed = True


MVP_ALLOWED_JOINTS = ("right_index_j0", "right_index_j1")
MVP_MAX_TRAVEL_RAD = 0.15
MVP_MIN_VELOCITY_RAD_S = 0.1
MVP_MAX_VELOCITY_RAD_S = 0.2
MVP_MAX_ACCELERATION_RAD_S2 = 0.10

class RealApexHandMvp:
    """Explicitly armed, single-joint hardware bring-up controller.

    This is intentionally separate from the general backend: it permits only
    one index joint per call and requires ``confirm_movement=True``. Values are
    radians, rad/s, and rad/s^2.
    """
    def __init__(self, backend: RealApexHand, *, confirm_movement: bool = False,
                 joint_limits: dict[str, tuple[float, float]] | None = None,
                 finger_id_enum: Any | None = None,
                 joint_control_param: Any | None = None):
        self.backend = backend
        self.confirm_movement = confirm_movement
        self.joint_limits = joint_limits or {}
        self.finger_id_enum = finger_id_enum
        self.joint_control_param = joint_control_param

    def move_one_joint(self, joint_name: str, target: float, *,
                       velocity: float = 0.1, acceleration: float = 0.05) -> ApexCommandResult:
        if not self.confirm_movement:
            raise RysenBackendError("MVP movement requires confirm_movement=True")
        if joint_name not in MVP_ALLOWED_JOINTS:
            raise RysenBackendError(f"MVP joint not allowed: {joint_name}")
        if abs(velocity) < MVP_MIN_VELOCITY_RAD_S or abs(velocity) > MVP_MAX_VELOCITY_RAD_S or abs(acceleration) > MVP_MAX_ACCELERATION_RAD_S2:
            raise RysenBackendError("MVP velocity/acceleration safety bound exceeded")
        state = self.backend.get_joint_state()
        current = state.position.as_dict()[joint_name]
        if abs(target - current) > MVP_MAX_TRAVEL_RAD:
            raise RysenBackendError("MVP target travel exceeds safety bound")
        if joint_name in self.joint_limits:
            lo, hi = self.joint_limits[joint_name]
            if not lo <= target <= hi:
                raise RysenBackendError("MVP target outside supplied joint limits")
        sdk = self.backend._sdk
        jid = next(k for k, v in self.backend._sdk_to_project_joint.items() if v == joint_name)
        finger = getattr(type(sdk), "__mro__", ())
        try:
            if self.finger_id_enum is None or self.joint_control_param is None:
                from rysen_apexhand_sdk import FingerId, JointControlParam
                self.finger_id_enum = FingerId
                self.joint_control_param = JointControlParam
            finger_id = self.finger_id_enum.FINGER_ID_INDEX
            cmd = self.joint_control_param()
            cmd.joint_id, cmd.position, cmd.velocity, cmd.acceleration = jid, target, abs(velocity), abs(acceleration)
            ret = sdk.set_finger_enabled([finger_id])
            if ret != self.backend._error_code.ERROR_CODE_OK:
                raise RysenBackendError(f"enable index failed: {enum_name(ret)}")
            ret = sdk.move_joint([cmd])
            if ret != self.backend._error_code.ERROR_CODE_OK:
                raise RysenBackendError(f"move_joint failed: {enum_name(ret)}")
            values = state.position.as_dict()
            values[joint_name] = float(target)
            applied = ApexActiveJointTarget.from_mapping(values)
            return ApexCommandResult(commanded=applied, applied=applied, timestamp=self.backend._clock(), clipped=())
        finally:
            sdk.set_finger_disabled([finger_id])
