import numpy as np

from carla_sensor_objects import CarlaLiDARObjectDetector


def test_lidar_object_detector_rejects_common_static_clutter():
    rng = np.random.default_rng(3)
    ground = np.c_[
        rng.uniform(0, 35, 7000),
        rng.uniform(-12, 12, 7000),
        rng.normal(0, 0.02, 7000),
    ]
    car = np.c_[
        rng.uniform(10, 14.2, 700),
        rng.uniform(1.05, 2.95, 700),
        rng.uniform(0.2, 1.6, 700),
    ]
    motorcycle = np.c_[
        rng.uniform(21, 23.2, 220),
        rng.uniform(-4.4, -3.6, 220),
        rng.uniform(0.2, 1.7, 220),
    ]
    pole = np.c_[
        rng.uniform(17.9, 18.1, 200),
        rng.uniform(7.9, 8.1, 200),
        rng.uniform(0.2, 3.3, 200),
    ]
    wall = np.c_[
        rng.uniform(5, 15, 1000),
        rng.uniform(-9.1, -8.9, 1000),
        rng.uniform(0.3, 2.5, 1000),
    ]

    detector = CarlaLiDARObjectDetector(
        voxel_size=0.45,
        min_points=18,
        max_proposals=12,
    )
    proposals = detector.detect(np.vstack([ground, car, motorcycle, pole, wall]))

    assert len(proposals) == 2
    lengths = sorted(float(p["dimensions"]["length"]) for p in proposals)
    assert 1.5 < lengths[0] < 3.0
    assert 3.5 < lengths[1] < 5.0
    assert all(float(p["score"]) >= 0.42 for p in proposals)


def test_camera_evidence_rejects_weak_lidar_only_fragment():
    detector = CarlaLiDARObjectDetector(min_vehicle_score=0.52, max_proposals=8)
    weak = {
        'location': np.array([12.0, 5.0, 1.0]),
        'dimensions': {'length': 3.0, 'width': 1.3, 'height': 1.2},
        'score': 0.60,
        'num_points': 60,
        'min_corner': np.array([10.5, 4.3, 0.2]),
        'max_corner': np.array([13.5, 5.7, 1.8]),
    }
    strong = dict(weak)
    strong['location'] = np.array([18.0, -4.0, 1.0])
    strong['score'] = 0.82
    out = detector.add_camera_evidence([weak, strong], [], (375, 1242, 3))
    assert len(out) == 1
    assert float(out[0]['score']) >= 0.68
