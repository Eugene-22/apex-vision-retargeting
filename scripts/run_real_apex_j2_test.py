#!/usr/bin/env python3
"""Guarded one-shot fixed-target verification for right_index_j2."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from robot.rysen_backend import DEFAULT_RYSEN_IP, RealApexHand, RysenBackendError

def main(argv=None):
 p=argparse.ArgumentParser(description="Fixed target right_index_j2 verification")
 p.add_argument("--ip",default=DEFAULT_RYSEN_IP)
 p.add_argument("--confirm-movement",action="store_true")
 p.add_argument("--confirm-j2-limit-warning",action="store_true")
 a=p.parse_args(argv)
 if not (a.confirm_movement and a.confirm_j2_limit_warning):
  print("READ-ONLY: require --confirm-movement and --confirm-j2-limit-warning"); return 0
 hand=None
 try:
  hand=RealApexHand(ip=a.ip); before=hand.get_joint_state(); b=before.position.as_dict()["right_index_j2"]
  print(f"connected=True current_j2_rad={b:.6f} target_j2_rad=0.000000")
  if abs(b)>0.15: raise RysenBackendError("current j2 is outside fixed-test travel guard")
  from rysen_apexhand_sdk import FingerId, JointId, JointControlParam
  cmd=JointControlParam(); cmd.joint_id=JointId.JOINT_ID_INDEX_J2; cmd.position=0.0; cmd.velocity=0.1; cmd.acceleration=0.05
  ok=hand._error_code.ERROR_CODE_OK
  ret=hand._sdk.set_finger_enabled([FingerId.FINGER_ID_INDEX])
  if ret!=ok: raise RysenBackendError(f"enable failed: {ret}")
  try:
   ret=hand._sdk.move_joint([cmd])
   if ret!=ok: raise RysenBackendError(f"move_joint failed: {ret}")
   after=hand.get_joint_state(); after_j2=after.position.as_dict()["right_index_j2"]; print(f"command_sent after_j2_rad={after_j2:.6f}")
  finally: hand._sdk.set_finger_disabled([FingerId.FINGER_ID_INDEX])
  return 0
 except (RysenBackendError,OSError,ImportError) as e:
  print(f"ERROR: {e}",file=sys.stderr); return 3
 finally:
  if hand is not None: hand.close()
if __name__=="__main__": raise SystemExit(main())
