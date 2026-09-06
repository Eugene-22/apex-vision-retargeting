"""Camera/video/replay -> joint angles; never connects to robot hardware."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from perception import HandDetector, VideoSource, ReplaySource, ObservationWriter
from perception.landmarks import HAND_CONNECTIONS
from .retargeter import Retargeter


def main(argv=None):
    parser = argparse.ArgumentParser(description="ApexHand perception and retargeting (joint output only)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=None)
    source.add_argument("--video", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/apex_hand.yaml")
    parser.add_argument("--model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--optimizer", choices=["adaptive", "vector"])
    parser.add_argument("--input-mirrored", action="store_true")
    parser.add_argument("--show", action="store_true", help="Show input image and landmark overlay")
    parser.add_argument("--record", type=Path, help="Record input landmarks as JSONL")
    parser.add_argument("--output", type=Path, help="Write joint JSONL (default stdout)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.replay and args.show:
        parser.error("--show requires camera/video input")
    inputs = [p.resolve() for p in (args.replay, args.video, args.config, args.model) if p]
    outputs = [p.resolve() for p in (args.record, args.output) if p]
    if len(set(outputs)) != len(outputs) or set(inputs) & set(outputs):
        parser.error("Input, record and output files must have distinct paths")
    try:
        retargeter = Retargeter.from_yaml(args.config, args.hand)
        if args.optimizer:
            retargeter.optimizer.mode = args.optimizer
        with ExitStack() as stack:
            output = stack.enter_context(args.output.open("w", encoding="utf-8")) if args.output else sys.stdout
            recorder = stack.enter_context(ObservationWriter(args.record)) if args.record else None
            detector = None
            if args.replay:
                stream = ((timestamp, observation, None) for timestamp, observation in ReplaySource(args.replay, args.hand))
            else:
                detector = stack.enter_context(HandDetector(args.model, args.hand,
                                                input_mirrored=args.input_mirrored))
                capture = stack.enter_context(VideoSource(str(args.video) if args.video else
                                                          (args.camera if args.camera is not None else 0)))
                stream = ((timestamp, detector.detect(frame, timestamp), frame) for timestamp, frame in capture)
            if args.show:
                import cv2
                stack.callback(cv2.destroyAllWindows)
            for count, (timestamp, observation, frame) in enumerate(stream, 1):
                if recorder:
                    recorder.write(timestamp, observation)
                qpos = retargeter.process(observation, timestamp)
                info = retargeter.last_info
                item = {"timestamp": timestamp, "hand_side": args.hand,
                        "tracked": observation is not None, "valid": qpos is not None,
                        "joint_names": list(retargeter.joint_names), "joint_ids": list(range(21)),
                        "qpos": qpos.tolist() if qpos is not None else None, "units": "radians",
                        "solver": {k: v for k, v in info.items() if k not in
                                   ("qpos", "qpos_unfiltered", "targets", "pinch_targets", "pinch_alphas")}}
                print(json.dumps(item, allow_nan=False), file=output, flush=True)
                if args.show:
                    if observation is not None and observation.image_landmarks is not None:
                        h, w = frame.shape[:2]
                        points = (observation.image_landmarks[:, :2] * [w, h]).astype(int)
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 220, 0), 2)
                    cv2.putText(frame, "TRACKING" if qpos is not None else "NO VALID HAND",
                                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    cv2.imshow("ApexHand perception", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.max_frames is not None and count >= args.max_frames:
                    break
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
