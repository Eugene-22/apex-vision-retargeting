#!/usr/bin/env python3
"""Safe camera -> perception -> retargeting -> Apex realtime controller."""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from filtering.one_euro import EMAJointFilter
from human_hand.state import HumanHandState
from perception.mediapipe_hand import MediaPipeHandDetector
from retargeting.calibration import JointRange
from retargeting.full_mapping import (
    FingerCalibration,
    FullHandCalibration,
    map_full_hand_angles,
)
from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.rysen_backend import RealApexHand, RysenBackendError

def pair(v): return JointRange(float(v[0]),float(v[1]))

def load_config(path): return json.loads(Path(path).read_text())
def build_calibration(data):
 h=data['human_calibration']; return FullHandCalibration(tuple(pair(x) for x in h['thumb']),{k:FingerCalibration(pair(v['abduction']),pair(v['mcp_flexion']),pair(v['pip_flexion'])) for k,v in h['fingers'].items()})
def clamp(target,limits):
 values={}; clipped=[]
 for name,value in target.as_dict().items():
  if not math.isfinite(value): raise ValueError(f'non-finite target: {name}')
  lo,hi=limits[name]; applied=max(lo,min(hi,float(value))); values[name]=applied
  if applied != value: clipped.append(name)
 return ApexActiveJointTarget.from_mapping(values),clipped
@dataclass
class RealtimePipeline:
 detector: MediaPipeHandDetector
 hand: RealApexHand
 filt: EMAJointFilter
 calibration: FullHandCalibration
 limits: dict[str,tuple[float,float]]
 min_confidence: float
 max_step_rad_s: float
 previous: ApexActiveJointTarget|None=None
 previous_time: float|None=None
 failures: int=0
 def process(self,rgb,stamp_ms,now):
  result=self.detector.detect(rgb,stamp_ms)
  state=HumanHandState.from_mediapipe_result(result,now)
  return self.process_result(result,state,now)
 def process_result(self,result,state,now):
  if state is None or not state.is_valid or state.full_angles is None or state.quality_score < self.min_confidence:
   return 'lost',None
  raw=map_full_hand_angles(state.full_angles,self.calibration,self.limits)
  dt=max(now-(self.previous_time or now-1/30),1e-3)
  target,_=clamp(self.filt.filter(raw,dt),self.limits)
  limited_values = {}
  limited = False
  if self.previous is not None:
   for name, value, previous in zip(APEX_ACTIVE_JOINTS, target.values, self.previous.values):
    step = self.max_step_rad_s * dt
    bounded = previous + max(-step, min(step, value - previous))
    limited_values[name] = bounded
    limited = limited or bounded != value
   target = ApexActiveJointTarget.from_mapping(limited_values)
  try:
   self.hand.command_position(target)
  except Exception as exc:
   self.failures+=1
   if self.failures>=3: raise RysenBackendError(f'3 consecutive command failures: {exc}') from exc
   return f'send_error={type(exc).__name__}',None
  self.failures=0; self.previous=target; self.previous_time=now
  return ('tracked limited' if limited else 'tracked sent'),target


def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--config',default=str(ROOT/'config/realtime_robot.json')); p.add_argument('--model',default=str(ROOT/'models/hand_landmarker.task')); p.add_argument('--camera'); p.add_argument('--ip'); p.add_argument('--show',action='store_true'); p.add_argument('--output-video',default=None); p.add_argument('--landmarks-jsonl',default=None); a=p.parse_args(argv)
 cfg=load_config(a.config); camera=a.camera or cfg['camera']['device']; ip=a.ip or cfg['robot']['ip']; cap=cv2.VideoCapture(camera,cv2.CAP_V4L2)
 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 if not cap.isOpened(): raise RuntimeError(f'Could not open camera: {camera}')
 cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
 for prop,key in ((cv2.CAP_PROP_FRAME_WIDTH,'width'),(cv2.CAP_PROP_FRAME_HEIGHT,'height'),(cv2.CAP_PROP_FPS,'fps')): cap.set(prop,cfg['camera'][key])
 detector=hand=None; writer=landmark_file=None; start=time.monotonic(); count=last_log=0
 try:
  if a.show and not os.environ.get('DISPLAY'):
   print('warning: --show requested but DISPLAY is unavailable; video will still be saved if configured',flush=True)
  if a.output_video:
   Path(a.output_video).parent.mkdir(parents=True,exist_ok=True)
  if a.landmarks_jsonl:
   Path(a.landmarks_jsonl).parent.mkdir(parents=True,exist_ok=True); landmark_file=open(a.landmarks_jsonl,'a',encoding='utf-8')
  detector=MediaPipeHandDetector(a.model,num_hands=1,min_detection_confidence=0.35,min_presence_confidence=0.35,min_tracking_confidence=0.35); hand=RealApexHand(ip=ip,log_dir=cfg['robot'].get('log_dir')); hand.configure_motion(max_speed=cfg['robot'].get('max_speed',0.5), max_accel=cfg['robot'].get('max_accel',1.0), torque_nmm=cfg['robot'].get('torque_nmm',200.0)); hand.enable(); limits={k:tuple(v) for k,v in cfg['joint_limits'].items()}; track=cfg['tracking']
  pipeline=RealtimePipeline(detector,hand,EMAJointFilter(float(cfg['filter'].get('alpha', 0.65))),build_calibration(cfg),limits,float(track['min_confidence']),float(track.get('max_step_rad_s',2.4)))
  print(f'camera={camera} robot_ip={ip} connected={hand.is_connected}',flush=True)
  while True:
   ok,frame=cap.read()
   if not ok: continue
   tick_start=time.perf_counter(); now=time.monotonic(); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); result=detector.detect(rgb,int((now-start)*1000)); state=HumanHandState.from_mediapipe_result(result,now); status,target=pipeline.process_result(result,state,now); count+=1; tick_ms=(time.perf_counter()-tick_start)*1000
   annotated=frame.copy(); points=[]
   sets=getattr(result,'hand_landmarks',[]) or []
   if sets:
    points=[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in sets[0]]
    for point in sets[0]: cv2.circle(annotated,(int(point.x*annotated.shape[1]),int(point.y*annotated.shape[0])),4,(0,255,0),-1)
    for a_idx,b_idx in ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)): cv2.line(annotated,(int(sets[0][a_idx].x*annotated.shape[1]),int(sets[0][a_idx].y*annotated.shape[0])),(int(sets[0][b_idx].x*annotated.shape[1]),int(sets[0][b_idx].y*annotated.shape[0])),(255,180,0),2)
   if a.output_video:
    if writer is None: writer=cv2.VideoWriter(a.output_video,cv2.VideoWriter_fourcc(*'mp4v'),cfg['camera']['fps'],(annotated.shape[1],annotated.shape[0]))
    writer.write(annotated)
   if a.show and os.environ.get('DISPLAY'):
    cv2.imshow('apex realtime hand landmarks',annotated)
    if cv2.waitKey(1)&0xff==27: break
   if landmark_file and points: landmark_file.write(json.dumps({'timestamp':now,'status':status,'landmarks':points})+'\n'); landmark_file.flush()
   if now-last_log>=track['print_interval_seconds']:
    print(f'fps={count/max(now-start,1e-6):.1f} hand={status} target={target.values if target else None} robot_send={hand.is_connected} tick_ms={tick_ms:.1f} landmarks={len(points)}',flush=True)
    if points: print('landmarks=' + json.dumps(points,separators=(',',':')),flush=True)
    last_log=now
 except KeyboardInterrupt: pass
 finally:
  if detector: detector.close()
  if writer: writer.release()
  if landmark_file: landmark_file.close()
  cv2.destroyAllWindows()
  cap.release()
  if hand: hand.close()
if __name__=='__main__': main()
