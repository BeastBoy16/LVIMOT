import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_sliding_window import CarlaSlidingWindowEstimator


def test_sliding_window_marginalization():
    estimator = CarlaSlidingWindowEstimator(window_size=4)
    for f in range(6):
        t = f * 0.1
        pose = np.array([0.0, 0.0, 0.0, f * 1.0, 0.0, 0.0])
        vel = np.array([10.0, 0.0, 0.0])
        imu_preint = {'dt': 0.1, 'delta_p': np.array([1.0, 0.0, 0.0]), 'delta_v': np.zeros(3), 'delta_R': np.eye(3)} if f > 0 else None
        estimator.add_keyframe_data(f, t, pose, vel, imu_preintegration=imu_preint, prev_frame_id=f - 1 if f > 0 else None)
    assert len(estimator.factor_graph.keyframe_ids) <= 4
    assert estimator.current_prior is not None
