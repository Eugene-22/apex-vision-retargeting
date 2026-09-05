import numpy as np
from human_hand.full_state import compute_full_hand_angles

def test_full_hand_straight_geometry():
    points=np.zeros((21,3))
    for m,p,d,t in ((5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
        points[m]=[0,1,0]; points[p]=[0,2,0]; points[d]=[0,3,0]; points[t]=[0,4,0]
    points[1]=[1,0,0]; points[2]=[2,0,0]; points[3]=[3,0,0]; points[4]=[4,0,0]
    state=compute_full_hand_angles(points)
    assert set(state.fingers)=={"index","middle","ring","pinky"}
    assert all(abs(a.pip_flexion)<1e-8 for a in state.fingers.values())
    assert state.thumb.ip_flexion == 0.0
