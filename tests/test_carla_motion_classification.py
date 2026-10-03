import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_motion_classification import CarlaMotionClassifier, MotionState


def test_static_track_classification():
    classifier = CarlaMotionClassifier(static_speed_threshold=0.5, dynamic_speed_threshold=1.2)
    history = [
        {'timestamp': 0.0, 'position': np.array([10.0, 5.0, 0.0])},
        {'timestamp': 0.1, 'position': np.array([10.01, 5.0, 0.0])},
        {'timestamp': 0.2, 'position': np.array([10.0, 5.02, 0.0])},
        {'timestamp': 0.3, 'position': np.array([10.01, 5.01, 0.0])},
    ]
    res = classifier.classify_track(history, 'vehicle')
    assert res['state'] == MotionState.STATIC


def test_dynamic_track_classification():
    classifier = CarlaMotionClassifier(static_speed_threshold=0.5, dynamic_speed_threshold=1.2)
    history = [
        {'timestamp': 0.0, 'position': np.array([0.0, 0.0, 0.0])},
        {'timestamp': 0.1, 'position': np.array([1.0, 0.0, 0.0])},
        {'timestamp': 0.2, 'position': np.array([2.0, 0.0, 0.0])},
        {'timestamp': 0.3, 'position': np.array([3.0, 0.0, 0.0])},
    ]
    res = classifier.classify_track(history, 'vehicle')
    assert res['state'] == MotionState.DYNAMIC


def test_stopped_dynamic_vehicle():
    classifier = CarlaMotionClassifier(static_speed_threshold=0.5, dynamic_speed_threshold=1.2)
    history = [
        {'timestamp': 0.0, 'position': np.array([0.0, 0.0, 0.0])},
        {'timestamp': 0.1, 'position': np.array([2.0, 0.0, 0.0])},
        {'timestamp': 0.2, 'position': np.array([4.0, 0.0, 0.0])},
        {'timestamp': 0.3, 'position': np.array([4.01, 0.0, 0.0])},
        {'timestamp': 0.4, 'position': np.array([4.01, 0.0, 0.0])},
    ]
    res = classifier.classify_track(history, 'vehicle')
    assert res['state'] == MotionState.STOPPED_DYNAMIC
