import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from carla_camera_lidar_detector import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig
from carla_pipeline_core import CarlaLVIMOTCore
from carla_multimodal_features import CarlaMultimodalFeatureBuilder


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[70.0, 65.0, 130.0, 135.0]], dtype=np.float32)
        self.cls = np.array([2], dtype=np.float32)  # COCO car
        self.conf = np.array([0.9], dtype=np.float32)


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def predict(self, **kwargs):
        return [FakeResult()]


class FakeCalibration:
    width = 200
    height = 200
    lidar_transform = {
        "location": {"x": 1.0, "y": 0.0, "z": 2.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }

    def lidar_to_opencv(self, points):
        p = np.asarray(points, dtype=np.float64)
        return np.column_stack([p[:, 1], -p[:, 2], p[:, 0]])

    def project_lidar(self, points):
        c = self.lidar_to_opencv(points)
        valid = c[:, 2] > 0.1
        uv = np.full((len(c), 2), np.nan, dtype=np.float64)
        uv[valid, 0] = 100.0 + 100.0 * c[valid, 0] / c[valid, 2]
        uv[valid, 1] = 100.0 + 100.0 * c[valid, 1] / c[valid, 2]
        return uv, valid


def test_yolo_camera_lidar_detection_is_sensor_only_and_3d():
    cal = FakeCalibration()
    detector = CarlaCameraLiDARObjectDetector(
        cal,
        CameraLiDARDetectionConfig(min_lidar_points=4, max_range=55.0),
        model=FakeModel(),
    )
    rng = np.random.default_rng(4)
    obj = np.column_stack([
        rng.normal(10.0, 0.35, 80),
        rng.normal(0.0, 0.6, 80),
        rng.normal(0.0, 0.5, 80),
        np.ones(80),
    ])
    background = np.column_stack([
        rng.normal(30.0, 1.0, 30),
        rng.normal(0.0, 2.0, 30),
        rng.normal(0.0, 1.0, 30),
        np.ones(30),
    ])
    proposals = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8), np.vstack([obj, background]))
    assert len(proposals) == 1
    p = proposals[0]
    assert p["class"] == "vehicle"
    assert p["camera_evidence"] is True
    assert abs(float(p["location"][0]) - 10.0) < 2.0
    assert p["num_points"] >= 4


def test_nested_lidar_extrinsic_is_applied_everywhere():
    cal = FakeCalibration()
    builder = CarlaMultimodalFeatureBuilder(cal)
    p = builder.lidar_to_ego(np.array([[10.0, 2.0, 0.0]]))[0]
    assert np.allclose(p, [11.0, 2.0, 2.0])

    core = CarlaLVIMOTCore.__new__(CarlaLVIMOTCore)
    core.calibration = cal
    core.detector_mode = "camera_lidar"
    core.limits = SimpleNamespace(object_max_range=55.0)
    gt_pose = {
        "location": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }
    gt = [{
        "actor_id": 7,
        "type": "vehicle.test",
        "location": {"x": 11.0, "y": 2.0, "z": 2.0},
        "velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
    }]
    converted = core._gt_objects_current_lidar(gt, gt_pose)
    assert np.allclose(converted[0]["location"], [10.0, 2.0, 0.0])


def test_camera_lidar_detector_accepts_targets_beyond_legacy_55m_when_sensor_range_allows_it():
    cal = FakeCalibration()
    detector = CarlaCameraLiDARObjectDetector(
        cal,
        CameraLiDARDetectionConfig(min_lidar_points=4, max_range=100.0),
        model=FakeModel(),
    )
    rng = np.random.default_rng(17)
    obj = np.column_stack([
        rng.normal(80.0, 0.35, 80),
        rng.normal(0.0, 0.7, 80),
        rng.normal(0.0, 0.5, 80),
        np.ones(80),
    ])
    proposals = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8), obj)
    assert len(proposals) == 1
    assert 70.0 < float(proposals[0]["location"][0]) < 90.0
    assert detector.last_diagnostics["max_range_m"] == 100.0
