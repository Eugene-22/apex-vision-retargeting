#!/usr/bin/env python3
"""Read-only multi-frame static sample of the Apex mechanical state."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand, import_rysen_sdk
from robot.static_sampling import summarize_joint_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Apex static joint sampling")
    parser.add_argument("--ip", default=DEFAULT_RYSEN_IP)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--period", type=float, default=0.01)
    parser.add_argument("--log-dir", default=str(ROOT / "outputs" / "real_apex_logs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0 or args.period < 0.0:
        print("ERROR: samples must be positive and period non-negative")
        return 1

    hand = RealApexHand(ip=args.ip, log_dir=args.log_dir)
    try:
        frames = []
        for index in range(args.samples):
            frames.append(hand._sdk.get_joint_states())
            if index + 1 < args.samples:
                time.sleep(args.period)

        _, _, _, joint_id = import_rysen_sdk()
        names = {value: name for name, value in vars(joint_id).items() if name.startswith("JOINT_ID_")}
        print("READ-ONLY Apex static sampling: no enable and no motion")
        print(f"samples: {len(frames)} period_s: {args.period:.6f}")
        for result in summarize_joint_samples(frames):
            name = names.get(result.joint_id, str(result.joint_id))
            print(
                f"{name}: mean={result.mean:.9f} rad stddev={result.stddev:.9f} rad "
                f"min={result.minimum:.9f} max={result.maximum:.9f} count={result.count}"
            )
        return 0
    finally:
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())
