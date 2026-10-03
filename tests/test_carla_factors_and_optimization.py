import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from carla_factors import IMUPreintegrationFactor
from carla_factor_graph import CarlaFactorGraph


def test_imu_factor_residual():
    dt = 0.1
    g = np.array([0.0, 0.0, -9.80665])
    v_i = np.array([10.0, 0.0, 0.0])
    acc = np.array([1.0, 0.0, 0.0])
    v_j = v_i + (acc + g) * dt
    p_i = np.array([0.0, 0.0, 0.0])
    p_j = p_i + v_i * dt + 0.5 * (acc + g) * (dt ** 2)
    factor = IMUPreintegrationFactor(0.5 * acc * (dt ** 2), acc * dt, np.eye(3), dt, gravity=g)
    residual = factor.compute_residual(np.eye(3), p_i, v_i, np.eye(3), p_j, v_j)
    assert np.allclose(residual, 0.0, atol=1e-5)


def test_factor_graph_optimization():
    fg = CarlaFactorGraph()
    fg.add_keyframe(0, timestamp=0.0)
    fg.add_keyframe(1, timestamp=0.1)
    dt = 0.1
    g = np.array([0.0, 0.0, -9.80665])
    acc = np.array([0.5, 0.0, 0.0])
    fg.add_imu_factor(0, 1, 0.5 * acc * (dt ** 2), acc * dt, np.eye(3), dt)
    init_poses = {0: np.zeros(6), 1: np.array([0.0, 0.0, 0.0, 1.2, 0.1, -0.05])}
    init_vels = {0: np.array([10.0, 0.0, 0.0]), 1: np.array([10.2, 0.1, 0.0])}
    res = fg.optimize(init_poses, init_vels)
    assert res['success'] or res['cost'] < 1.0
