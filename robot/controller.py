"""Latest-target right-hand control, measured startup, and independent watchdog."""
import threading
import time
import sys
import numpy as np
from retargeting import ApexHandModel
from .config import RobotConfig
from .sdk import SDKBackend, DryRunBackend, RobotError


class RightHandController:
    def __init__(self, config=None, *, dry_run=False, backend=None, clock=time.monotonic):
        self.config = config or RobotConfig()
        self.model = ApexHandModel(self.config.urdf_path, "right")
        self.backend = backend if backend is not None else (
            DryRunBackend(self.model) if dry_run else SDKBackend(self.config, self.model))
        self._clock = clock
        self._lock, self._io_lock = threading.RLock(), threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._connected = self._started = self._closed = False
        self._fault = None
        self._target = self._target_time = self._last_tick = self._last_command = None
        self.command_count = 0
        # Independent speed bounds must also satisfy each driven mimic joint.
        self._speeds = np.full(self.model.num_independent_joints, np.inf)
        for i, row in enumerate(self.model.expansion):
            for j in np.flatnonzero(row):
                self._speeds[j] = min(self._speeds[j], self.config.max_speed / abs(row[j]),
                                      self.model.velocity_limits[i] / abs(row[j]))

    @property
    def started(self):
        with self._lock:
            return self._started

    @property
    def last_command(self):
        with self._lock:
            return None if self._last_command is None else self._last_command.copy()

    def connect(self):
        if self._closed or self._connected:
            raise RobotError("Create a new controller for a new connection")
        self.backend.connect()
        self._connected = True
        return self

    def prepare_target(self, qpos, *, hand_side="right"):
        if hand_side != "right":
            raise ValueError("Left-hand targets cannot be sent to the right-hand device")
        q = self.model.validate_qpos(qpos)
        expanded = self.model.expand(q[self.model.independent_indices])
        if not np.allclose(q, expanded, atol=1e-6, rtol=0):
            raise ValueError("21-joint target violates URDF mimic coupling; use submit_actuated for 16 joints")
        # Clamp independent limits once and then expand, preserving coupling.
        return self.model.expand(np.clip(q[self.model.independent_indices],
                                self.model.independent_lower, self.model.independent_upper))

    def submit_actuated(self, qpos, *, hand_side="right", observed_at=None):
        """16-angle reference order: thumb 4, index/middle/ring/pinky 3 each."""
        self.submit(self.model.expand(qpos), hand_side=hand_side, observed_at=observed_at)

    def submit(self, qpos, *, hand_side="right", observed_at=None):
        """Replace the latest target; first valid target enables measured startup.

        observed_at is a local time.monotonic() capture time, NOT video/JSONL
        timestamps. Passing the capture time includes perception/IK latency.
        """
        target = self.prepare_target(qpos, hand_side=hand_side)
        now = self._clock()
        observed_at = now if observed_at is None else observed_at
        if (not np.isfinite(observed_at) or observed_at > now
                or now - observed_at > self.config.target_timeout):
            raise ValueError("Target capture time is invalid, in the future, or stale")
        self.check_health()
        with self._lock:
            if self._target_time is not None and observed_at <= self._target_time:
                raise ValueError("Capture times must be strictly increasing")
            self._target, self._target_time = target, observed_at
            needs_start = not self._started
        if needs_start:
            try:
                with self._io_lock:
                    self.backend.check_health()
                    measured = self.model.validate_qpos(self.backend.read_positions())
                    current = measured[self.model.independent_indices]
                    outside = ((current < self.model.independent_lower - 0.001)
                               | (current > self.model.independent_upper + 0.001))
                    if np.any(outside):
                        details = []
                        for index in np.flatnonzero(outside):
                            joint = self.model.independent_indices[index]
                            details.append(
                                f"{self.model.joint_names[joint]} (SDK id {joint}): "
                                f"{np.degrees(current[index]):.2f} deg, allowed "
                                f"[{np.degrees(self.model.independent_lower[index]):.2f}, "
                                f"{np.degrees(self.model.independent_upper[index]):.2f}] deg")
                        raise RobotError("Measured startup position lies outside right-hand URDF limits; "
                                         "check device zero/calibration: " + "; ".join(details))
                    baseline = self.model.expand(np.clip(current, self.model.independent_lower,
                                                         self.model.independent_upper))
                    self.backend.enable()
                    with self._lock:
                        self._last_command = baseline
                        self._last_tick = self._clock()
                        self._started = True
                self._thread = threading.Thread(target=self._run, name="apex-right-control", daemon=True)
                self._thread.start()
            except BaseException as exc:
                self._trip(exc)
                raise

    def check_health(self):
        with self._lock:
            if self._fault is not None:
                raise RobotError(f"Right-hand control stopped: {self._fault}")
            if not self._connected or self._closed:
                raise RobotError("Controller is not connected")

    def _tick(self):
        with self._io_lock:
            if self._stop.is_set():
                return
            self.backend.check_health()
            self.backend.read_positions()  # requires fresh callback feedback on hardware
            # Check age after SDK reads as well: those calls can take time.
            now = self._clock()
            with self._lock:
                if self._target is None or now - self._target_time > self.config.target_timeout:
                    raise RobotError("Tracking target timed out; fingers disabled")
                target, previous = self._target.copy(), self._last_command.copy()
                # Never accumulate a large step budget across a delayed control tick.
                dt = min(max(now - self._last_tick, 0.), 2 / self.config.command_rate)
            ids = self.model.independent_indices
            step = np.clip(target[ids] - previous[ids], -self._speeds * dt, self._speeds * dt)
            command = self.model.expand(previous[ids] + step)
            self.backend.send(command)
            with self._lock:
                self._last_command, self._last_tick = command, now
                self.command_count += 1

    def _run(self):
        period = 1 / self.config.command_rate
        while not self._stop.wait(period):
            try:
                self._tick()
            except Exception as exc:
                self._trip(exc)
                return

    def _trip(self, error):
        with self._lock:
            self._fault = str(error)
            self._stop.set()
        try:
            with self._io_lock:
                self.backend.disable()
        except Exception as exc:
            with self._lock:
                self._fault += f"; disabling fingers also failed: {exc}"

    def close(self, *, return_to_neutral=True):
        if self._closed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RobotError("SDK call did not return; software shutdown could not be confirmed")
        try:
            with self._io_lock:
                error = None
                try:
                    if (return_to_neutral and self.config.return_to_neutral_on_exit
                            and self._connected and self._started and self._fault is None):
                        print("Returning RIGHT hand to zero pose before disconnect...", file=sys.stderr, flush=True)
                        self.backend.return_to_neutral()
                        print("Zero pose confirmed; disabling fingers.", file=sys.stderr, flush=True)
                except BaseException as exc:
                    error = exc
                # Disable/disconnect even if the move fails or is interrupted.
                try:
                    self.backend.close()
                except Exception as cleanup_error:
                    if error is None:
                        raise
                    raise RobotError(f"{error}; shutdown failed: {cleanup_error}") from error
                if error is not None:
                    raise error
        finally:
            self._connected, self._closed = False, True

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            # A native teardown crash must not hide the triggering error.
            print(f"Stopping right-hand controller: {exc}", file=sys.stderr, flush=True)
        try:
            self.close(return_to_neutral=exc is None)
        except Exception as cleanup_error:
            if exc is None:
                raise
            raise RobotError(f"{exc}; shutdown failed: {cleanup_error}") from exc
