#!/usr/bin/env python3
"""Compare read-only Apex joint state against official URDF limits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.apex_urdf import DEFAULT_RIGHT_URDF, ApexUrdfModel
from robot.limit_consistency import check_active_joint_limits
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Apex joint/URDF limit check")
    parser.add_argument("--ip", default=DEFAULT_RYSEN_IP)
    parser.add_argument("--urdf", default=str(DEFAULT_RIGHT_URDF))
    parser.add_argument("--log-dir", default=str(ROOT / "outputs" / "real_apex_logs"))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tolerance < 0.0:
        print("ERROR: --tolerance must be non-negative")
        return 1

    model = ApexUrdfModel(args.urdf)
    hand = RealApexHand(ip=args.ip, log_dir=args.log_dir)
    try:
        state = hand.get_joint_state()
        report = check_active_joint_limits(state.position, model, args.tolerance)
        print("READ-ONLY Apex joint limit check: no enable and no motion")
        print(f"urdf: {model.path}")
        print(f"active_joints: {len(report.checks)}")
        for check in report.checks:
            status = "OK" if check.within_limits else "VIOLATION"
            print(
                f"{status} {check.name}: position={check.position:.6f} rad "
                f"range=[{check.lower:.6f}, {check.upper:.6f}] rad "
                f"margins=({check.lower_margin:.6f}, {check.upper_margin:.6f})"
            )
        print(f"violations: {len(report.violations)}")
        print(f"near_limits_<=0.05rad: {len(report.near_limits)}")
        return 0 if report.within_limits else 2
    finally:
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())
