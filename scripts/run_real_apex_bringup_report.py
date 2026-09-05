#!/usr/bin/env python3
"""Generate a read-only Apex hardware bring-up report."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.latency import benchmark_operation
from robot.apex_urdf import DEFAULT_RIGHT_URDF, ApexUrdfModel
from robot.bringup_report import BringupReport
from robot.limit_consistency import check_active_joint_limits
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand, import_rysen_sdk
from robot.static_sampling import summarize_joint_samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Apex hardware bring-up report")
    parser.add_argument("--ip", default=DEFAULT_RYSEN_IP)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--period", type=float, default=0.01)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "hardware_bringup_report.json"))
    parser.add_argument("--log-dir", default=str(ROOT / "outputs" / "real_apex_logs"))
    return parser.parse_args(argv)


def object_value(obj, name: str | None = None, default: str = "unknown") -> str:
    value = getattr(obj, name, default) if name is not None else obj
    return str(getattr(value, "name", value))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0 or args.period < 0.0:
        print("ERROR: samples must be positive and period non-negative")
        return 1

    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    _, _, _, joint_id = import_rysen_sdk()
    hand = RealApexHand(ip=args.ip, log_dir=args.log_dir)
    try:
        sdk = hand._sdk
        version = sdk.get_version_info()
        frames = []
        for index in range(args.samples):
            frames.append(sdk.get_joint_states())
            if index + 1 < args.samples:
                time.sleep(args.period)

        state = hand.get_joint_state()
        limits = check_active_joint_limits(state.position, model)
        latency = benchmark_operation(hand.get_joint_state, samples=min(args.samples, 20))
        stats = summarize_joint_samples(frames)
        names = {value: name for name, value in vars(joint_id).items() if name.startswith("JOINT_ID_")}

        blockers = []
        if limits.violations:
            blockers.append("active joint positions violate official URDF limits")
        if latency.p95_s > 0.005 or latency.failures:
            blockers.append("joint-state read latency exceeds 5 ms or has failures")
        error_code = object_value(sdk.get_hardware_error_code())
        if error_code != "ERROR_CODE_OK":
            blockers.append(f"hardware error code is {error_code}")

        report = BringupReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ip=args.ip,
            sdk_version=object_value(version, "sdk_version"),
            firmware_version=object_value(version, "hand_firmware_version"),
            hardware_uid=str(sdk.get_hardware_uid()),
            hand_dir=object_value(sdk.get_hand_dir()),
            hardware_error_code=error_code,
            latency={
                "count": latency.count,
                "failures": latency.failures,
                "mean_s": latency.mean_s,
                "median_s": latency.median_s,
                "p95_s": latency.p95_s,
                "max_s": latency.max_s,
                "min_s": latency.min_s,
            },
            joint_statistics=[
                {
                    "joint_id": names.get(item.joint_id, str(item.joint_id)),
                    "count": item.count,
                    "mean_rad": item.mean,
                    "stddev_rad": item.stddev,
                    "min_rad": item.minimum,
                    "max_rad": item.maximum,
                }
                for item in stats
            ],
            limit_violations=[
                {
                    "name": item.name,
                    "position_rad": item.position,
                    "lower_rad": item.lower,
                    "upper_rad": item.upper,
                    "lower_margin_rad": item.lower_margin,
                    "upper_margin_rad": item.upper_margin,
                }
                for item in limits.violations
            ],
            status="BLOCKED" if blockers else "PASS",
            blockers=blockers,
        )
        report.write_json(args.output)
        print(f"report: {args.output}")
        print(f"status: {report.status}")
        print(f"blockers: {len(report.blockers)}")
        for blocker in report.blockers:
            print(f"- {blocker}")
        return 0 if report.status == "PASS" else 2
    finally:
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())
