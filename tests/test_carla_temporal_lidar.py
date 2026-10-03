import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_temporal_lidar import CarlaTemporalLiDARMatcher


def test_temporal_lidar_planar_matching():
    matcher = CarlaTemporalLiDARMatcher(max_center_distance=1.0, min_normal_cos=0.85)
    feats_0 = [
        {'center': np.array([5.0, 2.0, 0.0]), 'normal': np.array([0.0, 0.0, 1.0]), 'planarity': 0.05},
        {'center': np.array([10.0, -3.0, 1.0]), 'normal': np.array([0.0, 1.0, 0.0]), 'planarity': 0.08}
    ]
    lms_0 = matcher.update(feats_0, frame_id=0, timestamp=0.0)
    assert len(lms_0) == 2

    feats_1 = [
        {'center': np.array([5.05, 2.02, 0.01]), 'normal': np.array([0.01, 0.0, 0.99]), 'planarity': 0.06},
        {'center': np.array([9.98, -3.01, 1.02]), 'normal': np.array([0.0, 0.99, 0.01]), 'planarity': 0.07}
    ]
    lms_1 = matcher.update(feats_1, frame_id=1, timestamp=0.1)
    assert len(lms_1) == 2
