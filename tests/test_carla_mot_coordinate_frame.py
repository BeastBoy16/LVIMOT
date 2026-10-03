import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from carla_pipeline_core import CarlaLVIMOTCore
from carla_evaluation import CarlaEvaluator


def _bare_core():
    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.limits = SimpleNamespace(
        object_max_range=55.0,
        object_min_height=-3.0,
        object_max_height=4.0,
    )
    return core


def test_gt_and_estimated_tracks_compare_in_current_ego_frame():
    core = _bare_core()
    # CARLA ego at an arbitrary world position and yaw. An object is 10 m
    # forward and 2 m right in the ego frame.
    gt_pose = {
        "location": {"x": -50.0, "y": 70.0, "z": 0.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 90.0},
    }
    R = core._carla_world_rotation(gt_pose["rotation"])
    ego_t = np.array([-50.0, 70.0, 0.0])
    p_ego = np.array([10.0, 2.0, 0.0])
    p_world = R @ p_ego + ego_t
    gt_objects = [{
        "actor_id": 7,
        "location": {"x": p_world[0], "y": p_world[1], "z": p_world[2]},
        "velocity": {"x": 0.0, "y": 3.0, "z": 0.0},
    }]

    gt_eval = core._gt_objects_current_ego(gt_objects, gt_pose)
    assert np.allclose(gt_eval[0]["location"], p_ego, atol=1e-8)

    # Estimator local world is independent of CARLA world. Its current pose is
    # arbitrary; the track represents the same 10 m forward / 2 m right target.
    est_pose = np.array([0.0, 0.0, np.deg2rad(15.0), 4.0, -3.0, 0.0])
    est_pose_dict = core.pose_dict_from_vector(est_pose)
    R_est = core._carla_world_rotation(est_pose_dict["rotation"])
    p_track_world = R_est @ p_ego + est_pose[3:]
    tracks = [{"track_id": 11, "position": p_track_world.tolist(), "velocity": [0.0, 0.0, 0.0]}]
    track_eval = core._tracks_current_ego(tracks, est_pose)
    assert np.allclose(track_eval[0]["position"], p_ego, atol=1e-8)

    res = CarlaEvaluator(distance_threshold=1.0).evaluate_tracking_sequence([track_eval], [gt_eval])
    assert res["total_matches"] == 1
    assert res["mota"] == 1.0
    assert res["mean_nearest_bev_distance"] < 1e-8
    assert res["evaluation_frame"] == "current_lidar_sensor_frame"


def test_gt_observability_is_measured_from_current_ego():
    core = _bare_core()
    gt_pose = {
        "location": {"x": 100.0, "y": -200.0, "z": 0.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": -45.0},
    }
    R = core._carla_world_rotation(gt_pose["rotation"])
    t = np.array([100.0, -200.0, 0.0])
    near = R @ np.array([20.0, 0.0, 0.0]) + t
    far = R @ np.array([80.0, 0.0, 0.0]) + t
    objs = [
        {"actor_id": 1, "location": near.tolist()},
        {"actor_id": 2, "location": far.tolist()},
    ]
    ego_objs = core._gt_objects_current_ego(objs, gt_pose)
    kept = core._filter_gt_observable_current_ego(ego_objs)
    assert [x["actor_id"] for x in kept] == [1]


def test_track_frame_transform_preserves_camera_bbox_and_metadata():
    core = _bare_core()
    est_pose = np.array([0.0, 0.0, np.deg2rad(20.0), 3.0, -2.0, 0.0])
    pose_dict = core.pose_dict_from_vector(est_pose)
    R = core._carla_world_rotation(pose_dict["rotation"])
    p_ego = np.array([12.0, -1.5, 0.5])
    p_world = R @ p_ego + est_pose[3:]

    tracks = [{
        "track_id": 5,
        "class": "vehicle",
        "position": p_world.tolist(),
        "velocity": [1.0, 0.0, 0.0],
        "bbox": [420.0, 180.0, 500.0, 240.0],
        "score": 0.91,
        "camera_confidence": 0.87,
        "camera_evidence": True,
        "hits": 7,
    }]

    transformed = core._tracks_current_ego(tracks, est_pose)
    assert len(transformed) == 1
    out = transformed[0]
    assert np.allclose(out["position"], p_ego, atol=1e-8)
    assert out["bbox"] == [420.0, 180.0, 500.0, 240.0]
    assert out["score"] == 0.91
    assert out["camera_confidence"] == 0.87
    assert out["camera_evidence"] is True
    assert out["hits"] == 7

    # This is the exact gate used by the camera+LiDAR evaluation path.  A
    # transformed confirmed track must not disappear merely because its
    # coordinate frame changed.
    scored = [
        tr for tr in transformed
        if tr.get("bbox") is not None and len(tr.get("bbox", [])) == 4
    ]
    assert len(scored) == 1
