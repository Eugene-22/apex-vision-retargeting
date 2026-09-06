from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import sys
import time
import textwrap
import numpy as np
import pytest
from robot import RobotConfig, RightHandController, RobotError, SDKBackend, DryRunBackend
from robot.sdk import SDK_JOINT_ENUM_NAMES, load_sdk
from retargeting import ApexHandModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    return ApexHandModel(ROOT / "vendor/rysen-sdk/urdf/apex_hand_right.urdf")


class Clock:
    def __init__(self):
        self.now = 10.
    def __call__(self):
        return self.now


class FakeSDK:
    def __init__(self, model):
        self.calls = []
        self.results = {}
        self.connected = False
        self.direction = 1
        self.positions = model.expand(np.full(16, 0.15))
        self.feedback_values = list(enumerate(self.positions))
        self.callback = None
        self.auto_feedback = True

    def result(self, name, *args):
        self.calls.append((name, args))
        return self.results.get(name, 0)
    def connect(self, *args):
        result = self.result("connect", *args)
        self.connected = result == 0
        return result
    def disconnect(self):
        self.connected = False
        return self.result("disconnect")
    def is_connected(self):
        return self.connected
    def get_hand_dir(self):
        return self.direction
    def set_robot_model_from_urdf(self, path):
        return self.result("urdf", path)
    def get_hardware_error_code(self):
        return self.result("health")
    def set_max_joint_speed(self, value):
        return self.result("speed", value)
    def set_max_joint_accel(self, value):
        return self.result("accel", value)
    def set_max_finger_torque(self, value):
        return self.result("torque", value)
    def register_joint_states_callback(self, callback, freq_hz):
        self.callback = callback
        if self.auto_feedback:
            self.emit()
        return self.result("register", freq_hz)
    def emit(self):
        # Deliberately reverse incoming SDK state order; adapter must sort by ID.
        self.callback(NS(joint_states=[NS(joint_id=i, position=p) for i, p in reversed(self.feedback_values)]))
    def unregister_joint_states_callback(self):
        return self.result("unregister")
    def set_all_fingers_enabled(self):
        return self.result("enable")
    def set_all_fingers_disabled(self):
        return self.result("disable")
    def move_j_position_follow(self, params):
        return self.result("send", params)
    def move_joint(self, params):
        result = self.result("move_joint", params)
        if result == 0:
            self.positions = np.array([p.position for p in params])
            self.feedback_values = list(enumerate(self.positions))
            if self.auto_feedback:
                self.emit()
        return result


def sdk_module(fake):
    joint = IntEnum("JointId", {"JOINT_ID_" + name: i for i, name in enumerate(SDK_JOINT_ENUM_NAMES)})
    class Follow:
        # Disallow inventing torque_nmm or joint_id for the local SDK struct.
        __slots__ = ("id", "position")
    return NS(Rysen=lambda: fake, JointId=joint, FingerId=lambda i: i,
              ErrorCode=NS(ERROR_CODE_OK=0), HandDir=NS(RIGHT=1, LEFT=0),
              ConnectionType=NS(CONNECTION_TYPE_ETHERNET=1),
              MaxJointSpeed=NS, MaxJointAccel=NS, MaxFingerTorque=NS,
              MoveJPositionFollowParam=Follow, JointControlParam=NS)


@pytest.fixture
def backend(model):
    fake, clock = FakeSDK(model), Clock()
    b = SDKBackend(RobotConfig(ip="192.0.2.1"), model, sdk_module(fake), clock=clock)
    return b, fake, clock


def names(fake):
    return [name for name, _ in fake.calls]


def test_sdk_right_connection_feedback_and_local_structs(backend, model):
    b, fake, clock = backend
    b.connect()
    assert "enable" not in names(fake) and "send" not in names(fake)
    np.testing.assert_allclose(b.read_positions(), fake.positions)
    assert str(model.urdf_path) in fake.calls[0][1]
    b.enable()
    b.send(model.neutral)
    params = next(args[0] for name, args in fake.calls if name == "send")
    assert [int(p.id) for p in params] == list(range(21))
    assert [p.position for p in params] == model.neutral.tolist()
    b.close()
    assert names(fake)[-3:] == ["disable", "unregister", "disconnect"]


def test_actual_vendor_python_wrapper_with_native_stub(model, monkeypatch):
    """Run the real bundled Python methods over a fake native extension."""
    fake = FakeSDK(model)
    native = sdk_module(fake)
    native.ConnectionType = IntEnum("ConnectionType", {"CONNECTION_TYPE_ETHERNET": 1})
    native.FingerId = IntEnum("FingerId", {f"FINGER_{i}": i for i in range(5)})
    native.ErrorCode = IntEnum("ErrorCode", {"ERROR_CODE_OK": 0, "ERROR_CODE_INVALID_ARGUMENT": 6})
    native.HandDir = IntEnum("HandDir", {"LEFT": 0, "RIGHT": 1})
    native.HandSensorImage = NS
    native.JointControlParam = NS
    monkeypatch.setitem(sys.modules, "_rysen_sdk", native)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    module = load_sdk(RobotConfig().sdk_path)
    backend = SDKBackend(RobotConfig(ip="192.0.2.1"), model, sdk_module=module)
    backend.connect()
    backend.enable()
    backend.send(model.neutral)
    backend.return_to_neutral()
    backend.close()
    assert "send" in names(fake) and names(fake)[-1] == "disconnect"


def test_left_hardware_refused_before_enable(backend):
    b, fake, _ = backend
    fake.direction = 0
    with pytest.raises(RobotError, match="RIGHT"):
        b.connect()
    assert "enable" not in names(fake) and "send" not in names(fake)
    assert names(fake)[-1] == "disconnect"


def test_zero_pose_uses_vendor_move_joint_and_conservative_limits(backend):
    b, fake, _ = backend
    b.config = replace(b.config, max_speed=0.2, max_accel=0.8)
    b.connect()
    b.enable()
    b.return_to_neutral()
    params = next(args[0] for name, args in fake.calls if name == "move_joint")
    assert [int(p.joint_id) for p in params] == list(range(21))
    assert all(p.position == 0 and p.velocity == 0.2 and p.acceleration == 0.8 for p in params)
    assert fake.positions.tolist() == [0.] * 21
    b.close()


@pytest.mark.parametrize("cause", ["disabled", "hardware_error", "stale_feedback"])
def test_return_to_neutral_refuses_unhealthy_or_disabled_session(backend, cause):
    b, fake, clock = backend
    b.connect()
    if cause != "disabled":
        b.enable()
    if cause == "hardware_error":
        fake.results["health"] = 10
    elif cause == "stale_feedback":
        clock.now += 1
    with pytest.raises(RobotError):
        b.return_to_neutral()
    assert "move_joint" not in names(fake)
    b.close()


@pytest.mark.parametrize("feedback", ["unchanged_zero", "outside_tolerance"])
def test_return_to_neutral_requires_new_feedback_near_zero(backend, feedback):
    b, fake, _ = backend
    if feedback == "unchanged_zero":
        fake.feedback_values = list(enumerate(np.zeros(21)))
    b.connect()
    b.enable()
    def move(params):
        if feedback == "outside_tolerance":
            fake.emit()  # Fresh, but still at the old bent pose.
        return fake.result("move_joint", params)
    fake.move_joint = move
    with pytest.raises(RobotError, match="not confirmed"):
        b.return_to_neutral(timeout=0.03)
    b.close()


def test_return_to_neutral_reports_move_error(backend):
    b, fake, _ = backend
    b.connect()
    b.enable()
    fake.results["move_joint"] = 7
    with pytest.raises(RobotError, match="return to neutral failed"):
        b.return_to_neutral()
    b.close()


@pytest.mark.parametrize("operation", ["urdf", "connect", "health", "speed", "accel", "torque", "register"])
def test_initialization_failure_disconnects_without_motion(backend, operation):
    b, fake, _ = backend
    fake.results[operation] = 7
    with pytest.raises(RobotError):
        b.connect()
    assert "enable" not in names(fake) and "send" not in names(fake)
    assert "disconnect" in names(fake)


def test_error_code_zero_is_success_and_send_failure_raises(backend, model):
    b, fake, _ = backend
    b.connect()
    b.enable()
    fake.results["send"] = 3
    with pytest.raises(RobotError, match="position follow"):
        b.send(model.neutral)
    b.close()


def test_feedback_staleness_and_invalid_packet(backend):
    b, fake, clock = backend
    b.connect()
    clock.now += 0.6
    with pytest.raises(RobotError, match="stale"):
        b.read_positions()
    fake.feedback_values[2] = (2, float("nan"))
    fake.emit()
    with pytest.raises(RobotError, match="Nonfinite"):
        b.read_positions()
    b.close()


def test_missing_joint_packet_refused(backend):
    b, fake, _ = backend
    fake.feedback_values.pop()
    with pytest.raises(RobotError, match="21 joint"):
        b.connect()
    assert "enable" not in names(fake)


def test_missing_startup_feedback_does_not_use_zero_pose(model):
    fake = FakeSDK(model)
    fake.auto_feedback = False
    b = SDKBackend(RobotConfig(ip="192.0.2.1", startup_timeout=0.02), model, sdk_module(fake))
    with pytest.raises(RobotError, match="Timed out"):
        b.connect()
    assert "enable" not in names(fake)


def test_cleanup_disconnects_even_if_disable_fails(backend):
    b, fake, _ = backend
    b.connect()
    b.enable()
    fake.results["disable"] = 1
    with pytest.raises(RobotError, match="disable"):
        b.close()
    assert names(fake)[-1] == "disconnect"


def test_initialization_error_is_reported_before_cleanup(backend, monkeypatch, capsys):
    b, fake, _ = backend
    fake.results["register"] = 7
    original_close = b.close
    def close():
        assert "Stopping SDK initialization: register feedback failed: 7" in capsys.readouterr().err
        original_close()
    monkeypatch.setattr(b, "close", close)
    with pytest.raises(RobotError, match="register feedback failed"):
        b.connect()
    assert names(fake)[-2:] == ["unregister", "disconnect"]


@pytest.mark.skipif(sys.implementation.name != "cpython" or sys.platform != "linux",
                    reason="Models the Linux CPython vendor callback release")
@pytest.mark.parametrize("registration_fails", [False, True])
def test_native_callback_release_without_gil(registration_fails):
    # Match the SDK 1.5.2 teardown: a native holder drops its Python callable
    # reference without the GIL. Run in a child so a regression cannot kill pytest.
    script = textwrap.dedent('''
        import ctypes
        import resource
        import sys
        import weakref
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        sys.path.insert(0, "tests")
        from test_robot import FakeSDK, sdk_module, ROOT
        from robot import SDKBackend, RobotConfig, RobotError
        from retargeting import ApexHandModel

        incref = ctypes.pythonapi.Py_IncRef
        incref.argtypes = [ctypes.py_object]
        incref.restype = None
        # CDLL releases the GIL, whereas pythonapi/PyDLL keeps it.
        decref = ctypes.CDLL(None).Py_DecRef
        decref.argtypes = [ctypes.c_void_p]
        decref.restype = None

        class NativeHolder(FakeSDK):
            def register_joint_states_callback(self, callback, freq_hz):
                result = super().register_joint_states_callback(callback, freq_hz)
                self.callback_ref = weakref.ref(callback)
                self.address = id(callback)
                incref(callback)
                self.callback = None  # Only the native holder owns this reference.
                return result

            def unregister_joint_states_callback(self):
                decref(self.address)
                self.address = None
                assert self.callback_ref() is not None
                return super().unregister_joint_states_callback()

            def disconnect(self):
                assert self.callback_ref() is not None
                return super().disconnect()

        model = ApexHandModel(ROOT / "vendor/rysen-sdk/urdf/apex_hand_right.urdf")
        fake = NativeHolder(model)
        if sys.argv[1] == "True":
            fake.results["register"] = 7
        backend = SDKBackend(RobotConfig(ip="192.0.2.1"), model, sdk_module(fake))
        try:
            backend.connect()
        except RobotError:
            assert sys.argv[1] == "True"
        else:
            backend.close()
        assert fake.callback_ref() is None
        backend.close()  # Idempotent teardown must not release twice.
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(registration_fails)],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def controller(model, monkeypatch):
    clock = Clock()
    b = DryRunBackend(model)
    b.positions = model.expand(np.full(16, 0.15))
    c = RightHandController(backend=b, clock=clock)
    # Numerical tick tests advance an explicit clock; watchdog is tested with a real thread separately.
    monkeypatch.setattr(c, "_run", lambda: None)
    c.connect()
    yield c, b, clock
    c.close()


def test_measured_startup_speed_limit_and_mimic(controller):
    c, b, clock = controller
    measured = b.positions.copy()
    target = c.model.expand(np.full(16, 0.6))
    c.submit(target, observed_at=clock())
    np.testing.assert_allclose(c.last_command, measured)
    clock.now += 0.02
    c._tick()
    command = c.last_command
    assert np.max(np.abs(command - measured)) <= 0.02 + 1e-12
    np.testing.assert_array_equal(command[[4, 8, 12, 16, 20]], command[[3, 7, 11, 15, 19]])
    assert b.command_count == 1


def test_startup_limit_error_reports_measured_joint_without_enabling(controller):
    c, b, clock = controller
    b.positions[11] = b.positions[12] = np.radians(-10.)
    with pytest.raises(RobotError, match=r"SDK id 11.*-10.00 deg.*-5.00"):
        c.submit(c.model.neutral, observed_at=clock())
    assert not b.enabled and b.command_count == 0


def test_original_error_is_reported_before_cleanup(controller, monkeypatch, capsys):
    c, _, _ = controller
    error = RobotError("original startup error")
    def close(*, return_to_neutral):
        assert not return_to_neutral
        assert "original startup error" in capsys.readouterr().err
        raise RobotError("cleanup error")
    with monkeypatch.context() as patch:
        patch.setattr(c, "close", close)
        with pytest.raises(RobotError, match="original startup error; shutdown failed: cleanup error"):
            c.__exit__(RobotError, error, None)


def test_normal_exit_returns_to_zero_before_disable(controller, monkeypatch):
    c, b, clock = controller
    c.submit(c.model.neutral, observed_at=clock())
    events = []
    original_return, original_close = b.return_to_neutral, b.close
    def return_to_neutral():
        assert c._stop.is_set() and not c._thread.is_alive()
        assert b.enabled and b.connected
        events.append("return")
        original_return()
    def close():
        assert b.positions.tolist() == [0.] * 21
        events.append("close")
        original_close()
    monkeypatch.setattr(b, "return_to_neutral", return_to_neutral)
    monkeypatch.setattr(b, "close", close)
    c.__exit__(None, None, None)
    c.close()
    assert events == ["return", "close"]
    assert not b.enabled and not b.connected


@pytest.mark.parametrize("reason", ["not_started", "fault", "exception", "interrupt", "config_off"])
def test_exit_skips_return_for_faults_interrupts_and_unstarted_sessions(controller, monkeypatch, reason):
    c, b, clock = controller
    if reason != "not_started":
        c.submit(c.model.neutral, observed_at=clock())
    if reason == "fault":
        c._trip(RobotError("tracking timeout"))
    if reason == "config_off":
        c.config = replace(c.config, return_to_neutral_on_exit=False)
    monkeypatch.setattr(b, "return_to_neutral", lambda: pytest.fail("Unexpected return motion"))
    error = KeyboardInterrupt() if reason == "interrupt" else RuntimeError("failure") if reason == "exception" else None
    c.__exit__(type(error) if error else None, error, None)
    assert not b.enabled and not b.connected


@pytest.mark.parametrize("error", [RobotError("return failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_return_still_disables_and_disconnects(controller, monkeypatch, error):
    c, b, clock = controller
    c.submit(c.model.neutral, observed_at=clock())
    def fail():
        raise error
    monkeypatch.setattr(b, "return_to_neutral", fail)
    with pytest.raises(type(error)):
        c.close()
    assert not b.enabled and not b.connected


def test_upstream_16_actuated_input_and_clamping(controller):
    c, _, clock = controller
    c.submit_actuated(np.full(16, 9.), observed_at=clock())
    assert np.all(c._target <= c.model.upper + 1e-12)
    np.testing.assert_array_equal(c._target[[4, 8, 12, 16, 20]], c._target[[3, 7, 11, 15, 19]])


def test_reject_left_nan_wrong_shape_and_invalid_coupling(controller):
    c, b, clock = controller
    with pytest.raises(ValueError, match="Left-hand"):
        c.submit(c.model.neutral, hand_side="left")
    for q in [np.zeros(20), np.full(21, np.nan)]:
        with pytest.raises(ValueError):
            c.submit(q)
    q = c.model.neutral.copy()
    q[8] += 0.1
    with pytest.raises(ValueError, match="mimic"):
        c.submit(q)
    assert not b.enabled and b.command_count == 0


def test_out_of_order_or_stale_targets_rejected(controller):
    c, _, clock = controller
    with pytest.raises(ValueError, match="stale"):
        c.submit(c.model.neutral, observed_at=clock() - 1.)
    c.submit(c.model.neutral, observed_at=clock())
    with pytest.raises(ValueError, match="increasing"):
        c.submit(c.model.neutral, observed_at=clock())


def test_watchdog_disables_without_more_perception_frames(model):
    config = RobotConfig(command_rate=100, target_timeout=0.06)
    backend = DryRunBackend(model)
    with RightHandController(config, backend=backend) as c:
        c.submit(model.neutral)
        assert c._stop.wait(1.0), "Independent watchdog did not fire"
        c._thread.join(timeout=1.)
        assert not backend.enabled
        with pytest.raises(RobotError, match="timed out"):
            c.check_health()
        with pytest.raises(RobotError):
            c.submit(model.neutral)


def test_worker_send_failure_latches_fault_and_disables(model):
    backend = DryRunBackend(model)
    def fail(qpos):
        raise RobotError("simulated command error")
    backend.send = fail
    with RightHandController(backend=backend) as c:
        c.submit(model.neutral)
        assert c._stop.wait(1.)
        c._thread.join(timeout=1.)
        assert not backend.enabled
        with pytest.raises(RobotError, match="simulated command"):
            c.check_health()


def test_worker_stale_feedback_disables(model):
    backend = DryRunBackend(model)
    with RightHandController(backend=backend) as c:
        c.submit(model.neutral)
        def stale():
            raise RobotError("stale feedback")
        backend.read_positions = stale
        assert c._stop.wait(1.)
        c._thread.join(timeout=1.)
        assert not backend.enabled
        with pytest.raises(RobotError, match="stale feedback"):
            c.check_health()


def test_slow_sdk_read_cannot_send_an_expired_target(controller):
    c, b, clock = controller
    c.submit(c.model.neutral, observed_at=clock())
    def slow_read():
        clock.now += 0.6
        return b.positions.copy()
    b.read_positions = slow_read
    with pytest.raises(RobotError, match="timed out"):
        c._tick()
    assert b.command_count == 0


def test_no_sdk_import_in_dry_run(monkeypatch):
    monkeypatch.setattr("robot.sdk.load_sdk", lambda path: pytest.fail("dry run loaded SDK"))
    with RightHandController(dry_run=True) as c:
        c.submit(c.model.neutral)


@pytest.mark.parametrize("option", [dict(hand_side="left"), dict(max_speed=float("nan")),
                                    dict(finger_torque_pct=101), dict(target_timeout=0.001),
                                    dict(return_to_neutral_on_exit="false")])
def test_bad_robot_config(option):
    with pytest.raises(ValueError):
        RobotConfig(**option)


def test_robot_config_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = RobotConfig.from_yaml(ROOT / "config/robot_right.yaml")
    assert config.urdf_path.is_file() and config.sdk_path.is_file()


def test_cli_dry_run_replay(tmp_path, model):
    from perception import HandObservation, ObservationWriter
    path = tmp_path / "input.jsonl"
    with ObservationWriter(path) as writer:
        for i in range(3):
            writer.write(i * 0.1, HandObservation(model.neutral_keypoints, "right", i * 0.1))
    result = subprocess.run([sys.executable, "-m", "robot", "--replay", str(path)], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    import json
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 3 and all(r["hand_side"] == "right" and r["dry_run"] for r in rows)
    assert rows[-1]["command_count"] > 0


def test_cli_execute_needs_explicit_ip():
    result = subprocess.run([sys.executable, "-m", "robot", "--execute"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 2 and "requires --ip" in result.stderr
