"""
Capture human index-finger calibration from a live camera.

The workflow captures staged poses and writes a suggested JSON
``IndexCalibration``. It does not command robot hardware.

Run from the dev container:

    python3 scripts/calibrate_index.py

Use ``--output configs/calibration/index.json`` only when you are ready to
replace the active placeholder calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from human_hand.joint_angles import FingerJointAngles
from human_hand.state import HumanHandState
from perception.quality import temporal_quality
from perception.mediapipe_hand import MediaPipeHandDetector
from retargeting.index_calibration_capture import (
    IndexCalibrationSamples,
    build_index_calibration,
    phase_summary,
)


MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "index_calibration_suggested.json"
CAMERA_DEVICE = "/dev/video0"
WIDTH = 1280
HEIGHT = 720
CAMERA_FPS = 30


@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    instruction: str


PHASES = (
    Phase(
        "neutral",
        "neutral/open",
        "Hold the hand open and relaxed.",
    ),
    Phase(
        "abduction_min",
        "minimum abduction",
        "Swing the index finger to the negative/inner side and hold.",
    ),
    Phase(
        "abduction_max",
        "maximum abduction",
        "Swing the index finger to the positive/outer side and hold.",
    ),
    Phase(
        "mcp_flexion_max",
        "maximum MCP flexion",
        "Bend the index finger mainly at MCP and hold.",
    ),
    Phase(
        "pip_flexion_max",
        "maximum PIP flexion",
        "Bend the index finger mainly at PIP and hold.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture staged human index calibration."
    )
    parser.add_argument("--camera", default=CAMERA_DEVICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase-seconds", type=float, default=5.0)
    parser.add_argument("--prep-seconds", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def open_camera(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    return cap


def detect_index_angles(
    detector: MediaPipeHandDetector,
    frame: np.ndarray,
    timestamp_ms: int,
    previous_world: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[FingerJointAngles | None, np.ndarray | None]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(rgb, timestamp_ms)

    state = HumanHandState.from_mediapipe_result(
        result,
        timestamp=timestamp_ms / 1000.0,
    )
    if state is None:
        return None, previous_world
    _, valid = temporal_quality(
        state.world_landmarks,
        previous_world,
        dt if previous_world is not None else None,
    )
    if not valid:
        return None, state.world_landmarks
    return state.index_angles, state.world_landmarks


def deg(rad: float) -> float:
    return float(np.degrees(rad))


def capture_phase(
    cap: cv2.VideoCapture,
    detector: MediaPipeHandDetector,
    phase: Phase,
    phase_seconds: float,
    prep_seconds: float,
    start_time: float,
    show: bool,
) -> tuple[FingerJointAngles, ...]:
    print()
    print(f"=== {phase.label} ===")
    print(phase.instruction)
    print(f"Prepare for {prep_seconds:.1f}s...")
    time.sleep(prep_seconds)
    print(f"Recording for {phase_seconds:.1f}s")

    samples: list[FingerJointAngles] = []
    previous_world = None
    previous_sample_time = None
    frames = 0
    end_time = time.monotonic() + phase_seconds

    while time.monotonic() < end_time:
        ok, frame = cap.read()
        if not ok:
            continue

        frames += 1
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        sample_time = time.monotonic()
        try:
            angles, previous_world = detect_index_angles(
                detector, frame, timestamp_ms, previous_world,
                sample_time - previous_sample_time if previous_sample_time is not None else None,
            )
        except ValueError:
            angles = None
        previous_sample_time = sample_time

        if angles is not None:
            samples.append(angles)

        if show:
            cv2.imshow("index calibration", frame)
            if cv2.waitKey(1) == 27:
                break

    rate = 100.0 * len(samples) / frames if frames else 0.0
    print(
        f"Captured {len(samples)}/{frames} valid samples "
        f"({rate:.1f}% detection)"
    )

    if samples:
        summary = phase_summary(tuple(samples))
        print(
            "median deg | "
            f"abd={deg(summary['mcp_abduction']):6.1f} "
            f"mcp={deg(summary['mcp_flexion']):6.1f} "
            f"pip={deg(summary['pip_flexion']):6.1f}"
        )

    return tuple(samples)


def main() -> int:
    args = parse_args()
    cap = open_camera(args.camera)
    detector = MediaPipeHandDetector(MODEL_PATH, num_hands=1)
    start_time = time.monotonic()

    phase_data: dict[str, tuple[FingerJointAngles, ...]] = {}

    try:
        print(f"Camera: {args.camera}")
        print(f"Output: {args.output}")

        for phase in PHASES:
            samples = capture_phase(
                cap=cap,
                detector=detector,
                phase=phase,
                phase_seconds=args.phase_seconds,
                prep_seconds=args.prep_seconds,
                start_time=start_time,
                show=args.show,
            )
            phase_data[phase.key] = samples

        for phase in PHASES:
            count = len(phase_data[phase.key])
            if count < args.min_samples:
                print(
                    f"ERROR: phase {phase.label!r} only has {count} "
                    f"valid samples; need at least {args.min_samples}."
                )
                return 1

        calibration = build_index_calibration(
            IndexCalibrationSamples(
                neutral=phase_data["neutral"],
                abduction_min=phase_data["abduction_min"],
                abduction_max=phase_data["abduction_max"],
                mcp_flexion_max=phase_data["mcp_flexion_max"],
                pip_flexion_max=phase_data["pip_flexion_max"],
            )
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        calibration.save(args.output)

        print()
        print("=== Suggested Index Calibration ===")
        print(
            f"abduction : {deg(calibration.human_abduction.minimum):6.1f} "
            f"to {deg(calibration.human_abduction.maximum):6.1f} deg"
        )
        print(
            f"MCP flex  : {deg(calibration.human_mcp_flexion.minimum):6.1f} "
            f"to {deg(calibration.human_mcp_flexion.maximum):6.1f} deg"
        )
        print(
            f"PIP flex  : {deg(calibration.human_pip_flexion.minimum):6.1f} "
            f"to {deg(calibration.human_pip_flexion.maximum):6.1f} deg"
        )
        print(f"Saved: {args.output}")
        return 0

    finally:
        cap.release()
        detector.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
