#!/usr/bin/env python3
"""Inspect read-only SDK calibration capabilities and live mechanical state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.mechanical_state import check_mimic_state
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand, import_rysen_sdk
from robot.sdk_capabilities import inspect_sdk_capabilities


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Apex SDK calibration capability inspection")
    parser.add_argument("--ip", default=DEFAULT_RYSEN_IP)
    parser.add_argument("--log-dir", default=str(ROOT / "outputs" / "real_apex_logs"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    Rysen, _, _, joint_id = import_rysen_sdk()
    sdk = Rysen()
    report = inspect_sdk_capabilities(sdk)
    print("SDK calibration inspection: read-only; no calibration, enable, or motion")
    print(f"calibration_methods: {report.calibration_methods or 'none'}")
    print(f"position_offset_methods: {report.position_offset_methods or 'none'}")
    print(f"read_only_methods: {report.read_only_methods}")

    hand = RealApexHand(ip=args.ip, log_dir=args.log_dir)
    try:
        states = hand._sdk.get_joint_states()
        by_name = {name: getattr(joint_id, attr) for name, attr in {
            "index_j2": "JOINT_ID_INDEX_J2", "index_j3": "JOINT_ID_INDEX_J3",
            "middle_j2": "JOINT_ID_MIDDLE_J2", "middle_j3": "JOINT_ID_MIDDLE_J3",
            "ring_j2": "JOINT_ID_RING_J2", "ring_j3": "JOINT_ID_RING_J3",
            "pinky_j2": "JOINT_ID_PINKY_J2", "pinky_j3": "JOINT_ID_PINKY_J3",
            "thumb_j3": "JOINT_ID_THUMB_J3", "thumb_j4": "JOINT_ID_THUMB_J4",
        }.items()}
        print("mimic_checks:")
        failures = 0
        for source, mimic in (("index_j2", "index_j3"), ("middle_j2", "middle_j3"),
                              ("ring_j2", "ring_j3"), ("pinky_j2", "pinky_j3"),
                              ("thumb_j3", "thumb_j4")):
            result = check_mimic_state(states, by_name[source], by_name[mimic])
            print(f"  {source}->{mimic}: error={result.error:.9f} rad consistent={result.consistent}")
            failures += int(not result.consistent)
        return 0 if failures == 0 else 2
    finally:
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())
