#!/usr/bin/env python3
"""One-shot, guarded Apex Hand MVP bring-up command."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand, RealApexHandMvp, RysenBackendError

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Guarded one-shot Apex index-joint MVP")
    p.add_argument("--ip", default=DEFAULT_RYSEN_IP)
    p.add_argument("--joint", choices=("right_index_j0","right_index_j1"), default="right_index_j0")
    p.add_argument("--delta", type=float, default=0.03, help="relative target displacement in radians")
    p.add_argument("--velocity", type=float, default=0.1)
    p.add_argument("--acceleration", type=float, default=0.05)
    p.add_argument("--confirm-movement", action="store_true", help="required to send one motion command")
    return p.parse_args(argv)

def main(argv=None):
    a=parse_args(argv)
    if abs(a.delta)>0.15 or abs(a.velocity)<0.1 or abs(a.velocity)>0.2 or abs(a.acceleration)>0.10:
        print("ERROR: MVP safety bound exceeded", file=sys.stderr); return 2
    hand=None
    try:
        hand=RealApexHand(ip=a.ip)
        state=hand.get_joint_state(); current=state.position.as_dict()[a.joint]
        print(f"connected={hand.is_connected} joint={a.joint} current_rad={current:.6f}")
        if not a.confirm_movement:
            print("READ-ONLY: no movement sent (add --confirm-movement for one command)")
            return 0
        target=current+a.delta
        result=RealApexHandMvp(hand, confirm_movement=True).move_one_joint(a.joint,target,velocity=a.velocity,acceleration=a.acceleration)
        print(f"command_sent target_rad={result.applied.as_dict()[a.joint]:.6f}")
        return 0
    except (RysenBackendError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 3
    finally:
        if hand is not None: hand.close()
if __name__=="__main__": raise SystemExit(main())
