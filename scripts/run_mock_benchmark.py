#!/usr/bin/env python3
"""Run a deterministic mock safety-pipeline benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_hand.joint_angles import FingerJointAngles
from human_hand.state import HumanHandState
from metrics.benchmark_report import BenchmarkReport
from metrics.latency import benchmark_operation
from metrics.trajectory import summarize_trajectory
from retargeting.calibration import IndexCalibration, JointRange
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.mock_apex_hand import MockApexHand
from robot.safe_pipeline import SafeIndexPipeline
from safety.command_chain import SafetyCommandChain


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic mock benchmark")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "mock_benchmark_report.json"))
    args = parser.parse_args(argv)
    if args.samples <= 1:
        print("ERROR: samples must be greater than one")
        return 1
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    calibration = IndexCalibration(JointRange(-1.0, 1.0), JointRange(0.0, 2.0), JointRange(0.0, 2.0))
    hand = MockApexHand(model)
    pipeline = SafeIndexPipeline(model, calibration, hand, SafetyCommandChain(model, 1.0, 2.0, confidence_threshold=0.5))
    results = []
    for index in range(args.samples):
        quality = 0.1 if index % 15 == 0 and index else 1.0
        state = HumanHandState("Right", 1.0 + index * 0.01, quality, __import__('numpy').zeros((21, 3)), __import__('numpy').zeros((21, 3)), FingerJointAngles(0.0, min(index * 0.01, 1.0), min(index * 0.01, 1.0)), quality, True)
        results.append(pipeline.process(state)[0])
    trajectory = summarize_trajectory(results, 0.01)
    latency = benchmark_operation(lambda: pipeline.process(None), samples=10)
    report = BenchmarkReport.from_summaries("mock_safe_index_pipeline", latency, trajectory, {"samples": args.samples, "lost_hand_events": 0})
    report.write_json(args.output)
    print(f"report: {args.output}")
    print(f"trajectory_samples: {trajectory.samples}")
    print(f"max_velocity: {trajectory.max_velocity:.6f} rad/s")
    print(f"max_acceleration: {trajectory.max_acceleration:.6f} rad/s^2")
    print(f"low_confidence: {trajectory.low_confidence_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
