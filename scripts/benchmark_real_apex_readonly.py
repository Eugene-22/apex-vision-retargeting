#!/usr/bin/env python3
"""Read-only latency benchmark for Apex Hand SDK joint-state reads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.latency import benchmark_operation, format_latency_summary
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Apex SDK joint-state latency benchmark")
    parser.add_argument("--ip", default=DEFAULT_RYSEN_IP, help="Apex Hand IP address")
    parser.add_argument("--samples", type=int, default=100, help="Number of read samples")
    parser.add_argument("--period", type=float, default=0.01, help="Delay between reads in seconds")
    parser.add_argument(
        "--log-dir",
        default=str(project_root() / "outputs" / "real_apex_logs"),
        help="SDK log directory inside this project",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples <= 0:
        print("ERROR: --samples must be positive")
        return 1
    if args.period < 0.0:
        print("ERROR: --period must be non-negative")
        return 1

    print("READ-ONLY Apex SDK latency benchmark: joint-state reads only; no enable and no motion")
    hand = RealApexHand(ip=args.ip, log_dir=args.log_dir)
    try:
        first_state = hand.get_joint_state()
        print(f"connected: {hand.is_connected}")
        print(f"active_count: {len(first_state.position.values)}")
        print(f"samples: {args.samples}")
        print(f"period_s: {args.period:.6f}")

        summary = benchmark_operation(
            operation=hand.get_joint_state,
            samples=args.samples,
            sleep_s=args.period,
        )
        print("read_joint_state_latency:", format_latency_summary(summary))
        return 0 if summary.failures == 0 else 2
    finally:
        hand.close()
        print("closed")


if __name__ == "__main__":
    raise SystemExit(main())
