"""
Run a synthetic Human Index -> Mock Apex pipeline sample.

No camera and no hardware are used. This script is a quick integration
check for the M2.3 retargeting, M6 full target order, M9 safety clamp,
and M10 mock robot boundary.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.index_pipeline import command_index_to_apex
from robot.mock_apex_hand import MockApexHand


CONFIG_PATH = PROJECT_ROOT / "configs" / "calibration" / "index.json"


def deg(rad: float) -> float:
    return float(np.degrees(rad))


def main() -> int:
    calibration = IndexCalibration.load(CONFIG_PATH)
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    hand = MockApexHand(model)

    human = FingerJointAngles(
        mcp_abduction=0.0,
        mcp_flexion=0.6,
        pip_flexion=0.8,
    )

    result = command_index_to_apex(
        human=human,
        calibration=calibration,
        model=model,
        hand=hand,
    )

    applied = result.applied.as_dict()

    print("=== Synthetic Index -> Mock Apex ===")
    print(
        "Human deg | "
        f"abd={deg(human.mcp_abduction):6.1f} "
        f"mcp={deg(human.mcp_flexion):6.1f} "
        f"pip={deg(human.pip_flexion):6.1f}"
    )
    print(
        "Apex deg  | "
        f"j0={deg(applied['right_index_j0']):6.1f} "
        f"j1={deg(applied['right_index_j1']):6.1f} "
        f"j2={deg(applied['right_index_j2']):6.1f}"
    )
    print(f"clipped   | {result.clipped if result.clipped else 'none'}")
    print(f"timestamp | {result.timestamp:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
