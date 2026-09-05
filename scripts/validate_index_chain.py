"""
Synthetic end-to-end validation of the index retargeting chain (M2.5).

Runs the full chain on synthetic human index angles:

    human angles -> calibrated mapping -> Apex q -> FK -> fingertip

and checks the properties required by CLAUDE.md section 10:

    direction, magnitude, joint limits, continuity, monotonicity

This is a virtual (simulation) validation: no camera and no hardware.
Run with:

    python3 scripts/validate_index_chain.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from human_hand.joint_angles import FingerJointAngles
from retargeting.calibration import IndexCalibration
from robot.apex_urdf import ApexUrdfModel, DEFAULT_RIGHT_URDF
from robot.virtual_index import (
    VirtualIndexChain,
    is_monotonic,
    within_limits,
)


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "calibration"
    / "index.json"
)

N_SWEEP = 25


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def sweep(
    chain: VirtualIndexChain,
    angles: list[FingerJointAngles],
) -> tuple[list[float], list[float], list[float], np.ndarray]:
    """Run the chain over a sequence; return per-joint Apex values + tips."""
    j0s, j1s, j2s = [], [], []
    tips = []

    for human in angles:
        result = chain.forward(human)
        j0s.append(result.apex.j0_abduction)
        j1s.append(result.apex.j1_mcp_flexion)
        j2s.append(result.apex.j2_pip_flexion)
        tips.append(result.tip)

    return j0s, j1s, j2s, np.asarray(tips)


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"Missing config: {CONFIG_PATH}")
        return 1

    calibration = IndexCalibration.load(CONFIG_PATH)
    model = ApexUrdfModel(DEFAULT_RIGHT_URDF)
    chain = VirtualIndexChain(model, calibration)

    apex_ranges = chain.apex_ranges

    checks: list[Check] = []

    # ---------------------------------------------------------------
    # Sweep definitions (all within the calibrated human ROM)
    # ---------------------------------------------------------------

    abd_angles = [
        FingerJointAngles(a, 0.0, 0.0)
        for a in np.linspace(
            calibration.human_abduction.minimum,
            calibration.human_abduction.maximum,
            N_SWEEP,
        )
    ]

    # Flexion direction sweeps go from neutral (0) to max: "flexing"
    # the finger, not hyper-extending it. Starting at the extension
    # minimum would first straighten the finger (tip z rising), which
    # is correct motion but not the monotonic curl we are checking.
    mcp_angles = [
        FingerJointAngles(0.0, f, 0.0)
        for f in np.linspace(
            0.0,
            calibration.human_mcp_flexion.maximum,
            N_SWEEP,
        )
    ]

    pip_angles = [
        FingerJointAngles(0.0, 0.0, f)
        for f in np.linspace(
            0.0,
            calibration.human_pip_flexion.maximum,
            N_SWEEP,
        )
    ]

    mcp_ramp = np.linspace(
        0.0, calibration.human_mcp_flexion.maximum, N_SWEEP
    )
    pip_ramp = np.linspace(
        0.0, calibration.human_pip_flexion.maximum, N_SWEEP
    )
    curl_angles = [
        FingerJointAngles(0.0, m, p)
        for m, p in zip(mcp_ramp, pip_ramp)
    ]

    # Full-ROM sweeps (extension minimum -> flexion maximum) used only
    # for the joint-limit check, so the extension side is also covered.
    # (j0 abduction full ROM is already covered by the abd sweep above.)
    rom_mcp = [
        FingerJointAngles(0.0, f, 0.0)
        for f in np.linspace(
            calibration.human_mcp_flexion.minimum,
            calibration.human_mcp_flexion.maximum,
            N_SWEEP,
        )
    ]
    rom_pip = [
        FingerJointAngles(0.0, 0.0, f)
        for f in np.linspace(
            calibration.human_pip_flexion.minimum,
            calibration.human_pip_flexion.maximum,
            N_SWEEP,
        )
    ]

    # ---------------------------------------------------------------
    # Run sweeps
    # ---------------------------------------------------------------

    j0_abd, j1_abd, j2_abd, tip_abd = sweep(chain, abd_angles)
    j0_mcp, j1_mcp, j2_mcp, tip_mcp = sweep(chain, mcp_angles)
    j0_pip, j1_pip, j2_pip, tip_pip = sweep(chain, pip_angles)
    j0_curl, j1_curl, j2_curl, tip_curl = sweep(chain, curl_angles)
    _, j1_rom_mcp, _, _ = sweep(chain, rom_mcp)
    _, _, j2_rom_pip, _ = sweep(chain, rom_pip)

    # ---------------------------------------------------------------
    # Joint limits (all sweeps, all joints)
    # ---------------------------------------------------------------

    all_j0 = j0_abd + j0_mcp + j0_pip + j0_curl
    all_j1 = j1_abd + j1_mcp + j1_pip + j1_curl + j1_rom_mcp
    all_j2 = j2_abd + j2_mcp + j2_pip + j2_curl + j2_rom_pip

    checks.append(
        Check(
            "j0 within URDF limits",
            within_limits(
                all_j0,
                apex_ranges.j0_abduction.minimum,
                apex_ranges.j0_abduction.maximum,
            ),
        )
    )
    checks.append(
        Check(
            "j1 within URDF limits",
            within_limits(
                all_j1,
                apex_ranges.j1_mcp_flexion.minimum,
                apex_ranges.j1_mcp_flexion.maximum,
            ),
        )
    )
    checks.append(
        Check(
            "j2 within URDF limits",
            within_limits(
                all_j2,
                apex_ranges.j2_pip_flexion.minimum,
                apex_ranges.j2_pip_flexion.maximum,
            ),
        )
    )

    # ---------------------------------------------------------------
    # Direction + monotonicity
    # ---------------------------------------------------------------

    checks.append(
        Check(
            "abduction: j0 monotonic increasing",
            is_monotonic(j0_abd, "increasing"),
        )
    )
    checks.append(
        Check(
            "abduction: tip sweeps laterally (+y)",
            is_monotonic(tip_abd[:, 1], "increasing"),
        )
    )

    checks.append(
        Check(
            "MCP flexion: j1 monotonic increasing",
            is_monotonic(j1_mcp, "increasing"),
        )
    )
    checks.append(
        Check(
            "MCP flexion: tip curls (z decreasing)",
            is_monotonic(tip_mcp[:, 2], "decreasing"),
        )
    )

    checks.append(
        Check(
            "PIP flexion: j2 monotonic increasing",
            is_monotonic(j2_pip, "increasing"),
        )
    )
    checks.append(
        Check(
            "PIP flexion: tip curls (z decreasing)",
            is_monotonic(tip_pip[:, 2], "decreasing"),
        )
    )

    # ---------------------------------------------------------------
    # Continuity (combined curl)
    # ---------------------------------------------------------------

    step = np.linalg.norm(np.diff(tip_curl, axis=0), axis=1)
    max_step = float(np.max(step))
    # The combined curl spans ~90 deg over 25 steps; the smooth arc
    # step is ~1-2 cm. This only rejects discontinuities (> 20 mm).
    checks.append(
        Check(
            "curl: tip trajectory continuous",
            bool(np.all(step <= 0.02)),
            detail=f"max step = {max_step * 1000:.2f} mm",
        )
    )

    # ---------------------------------------------------------------
    # Magnitude
    # ---------------------------------------------------------------

    abd_span = float(np.ptp(tip_abd[:, 1]))
    mcp_drop = float(tip_mcp[0, 2] - tip_mcp[-1, 2])
    pip_drop = float(tip_pip[0, 2] - tip_pip[-1, 2])

    checks.append(
        Check(
            "abduction: lateral span meaningful (>1 mm)",
            abd_span > 0.001,
            detail=f"span = {abd_span * 1000:.1f} mm",
        )
    )
    checks.append(
        Check(
            "MCP flexion: tip drop meaningful (>5 mm)",
            mcp_drop > 0.005,
            detail=f"drop = {mcp_drop * 1000:.1f} mm",
        )
    )
    checks.append(
        Check(
            "PIP flexion: tip drop meaningful (>5 mm)",
            pip_drop > 0.005,
            detail=f"drop = {pip_drop * 1000:.1f} mm",
        )
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("=== M2.5 Virtual Index Validation ===")
    print(f"URDF:   {model.path}")
    print(f"Config: {CONFIG_PATH}")
    print()

    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        suffix = f"  ({check.detail})" if check.detail else ""
        print(f"[{mark}] {check.name}{suffix}")

    print()
    n_fail = sum(1 for c in checks if not c.passed)
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
