"""Right-hand camera teleoperation or hardware-free command validation."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from perception import HandDetector, ObservationWriter, ReplaySource, VideoSource
from perception.landmarks import HAND_CONNECTIONS
from retargeting import Retargeter
from .config import RobotConfig, ROOT
from .controller import RightHandController


def main(argv=None):
    parser = argparse.ArgumentParser(description="Right ApexHand controller (dry-run unless --execute)")
    parser.add_argument("--execute", action="store_true", help="Enable and move the physical RIGHT hand")
    parser.add_argument("--ip", help="Actual right-hand device address")
    parser.add_argument("--robot-config", type=Path, default=ROOT / "config/robot_right.yaml")
    parser.add_argument("--retarget-config", type=Path, default=ROOT / "config/apex_hand.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models/hand_landmarker.task")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--replay", type=Path, help="Recorded landmarks; available in dry-run mode")
    parser.add_argument("--input-mirrored", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if args.execute and args.replay:
        parser.error("Hardware execution requires live camera input; use dry-run for recorded targets")
    if args.replay and args.show:
        parser.error("Replay has no image preview")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    inputs = [args.robot_config, args.retarget_config, args.model, args.replay]
    if args.record and args.record.resolve() in {p.resolve() for p in inputs if p is not None}:
        parser.error("Recording path must not overwrite an input/configuration file")
    try:
        config = RobotConfig.from_yaml(args.robot_config)
        if args.ip:
            config = replace(config, ip=args.ip)
        if args.execute and not config.ip:
            parser.error("--execute requires --ip or ip in robot_right.yaml")
        retargeter = Retargeter.from_yaml(args.retarget_config, "right")
        controller = RightHandController(config, dry_run=not args.execute)
        if retargeter.model.urdf_path != controller.model.urdf_path:
            raise ValueError("Retargeter and controller must use the same right-hand URDF")
        print("Mode: PHYSICAL RIGHT HAND" if args.execute else "Mode: DRY RUN (no SDK/hardware)", file=sys.stderr)
        with ExitStack() as stack:
            recorder = stack.enter_context(ObservationWriter(args.record)) if args.record else None
            if args.replay:
                stream = ((timestamp, observation, None) for timestamp, observation in ReplaySource(args.replay, "right"))
            else:
                detector = stack.enter_context(HandDetector(args.model, "right", input_mirrored=args.input_mirrored))
                capture = stack.enter_context(VideoSource(args.camera))
                # Detection happens below, after taking the wall-clock capture timestamp.
                stream = ((timestamp, None, frame) for timestamp, frame in capture)
            if args.show:
                import cv2
                stack.callback(cv2.destroyAllWindows)
            # Enter controller last so it disables/closes before releasing the camera.
            stack.enter_context(controller)
            replay_start = time.monotonic()
            replay_zero = None
            for count, (timestamp, observation, frame) in enumerate(stream, 1):
                if args.replay:
                    if replay_zero is None:
                        replay_zero = timestamp
                    delay = replay_start + timestamp - replay_zero - time.monotonic()
                    while delay > 0:
                        time.sleep(min(delay, 0.02))
                        controller.check_health()
                        delay = replay_start + timestamp - replay_zero - time.monotonic()
                captured_at = time.monotonic()
                controller.check_health()
                if frame is not None:
                    observation = detector.detect(frame, timestamp)
                if recorder:
                    recorder.write(timestamp, observation)
                qpos = retargeter.process(observation, timestamp)
                if qpos is not None:
                    controller.submit(qpos, hand_side="right", observed_at=captured_at)
                controller.check_health()
                command = controller.last_command
                print(json.dumps({"timestamp": timestamp, "hand_side": "right", "dry_run": not args.execute,
                      "tracking": qpos is not None, "target": None if qpos is None else qpos.tolist(),
                      "last_command": None if command is None else command.tolist(),
                      "command_count": controller.command_count}), flush=True)
                if args.show:
                    if observation is not None:
                        h, w = frame.shape[:2]
                        points = (observation.image_landmarks[:, :2] * [w, h]).astype(int)
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 200, 0), 2)
                    cv2.imshow("RIGHT ApexHand" if args.execute else "RIGHT ApexHand dry-run", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.max_frames and count >= args.max_frames:
                    break
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
