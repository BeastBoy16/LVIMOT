import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_4d_mapping import Carla4DEnvironmentMapper


def test_4d_mapping():
    mapper = Carla4DEnvironmentMapper(static_voxel_size=0.5)
    ego_pose = {'location': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}}
    lidar_pts = np.array([[10.0, 5.0, 0.0], [10.0, -5.0, 0.0], [5.0, 0.0, 0.0]])
    active_tracks = [{'track_id': 1, 'position': [5.0, 0.0, 0.0], 'velocity': [2.0, 0.0, 0.0], 'class': 'vehicle', 'motion_state': 'DYNAMIC'}]
    mapper.add_frame_data(0, 0.0, lidar_pts, [], active_tracks, ego_pose)
    static_pts = mapper.get_static_map_points()
    assert len(static_pts) > 0
