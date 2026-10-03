import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class CarlaLiDARDeskewer:
    """Approximate CARLA LiDAR deskewer with bounded vectorized computation."""

    def __init__(self, scan_duration=0.1):
        self.scan_duration = float(scan_duration)

    def create_point_times(self, num_points):
        if num_points <= 0:
            return np.empty(0, dtype=np.float64)
        return np.linspace(0.0, self.scan_duration, int(num_points), dtype=np.float64)

    def interpolate_pose(
        self,
        alpha,
        translation_start,
        translation_end,
        rotation_start,
        rotation_end,
    ):
        translation = (1.0 - alpha) * np.asarray(translation_start) + alpha * np.asarray(translation_end)
        rotations = Rotation.from_matrix(np.stack([rotation_start, rotation_end], axis=0))
        rotation = Slerp([0.0, 1.0], rotations)([float(alpha)])[0]
        return translation, rotation

    def deskew(
        self,
        points,
        translation_start,
        translation_end,
        rotation_start,
        rotation_end,
    ):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("Expected LiDAR points with shape (N, 4)")
        n = len(points)
        if n == 0:
            return points.copy()

        t0 = np.asarray(translation_start, dtype=np.float64).reshape(3)
        t1 = np.asarray(translation_end, dtype=np.float64).reshape(3)
        R0 = np.asarray(rotation_start, dtype=np.float64).reshape(3, 3)
        R1 = np.asarray(rotation_end, dtype=np.float64).reshape(3, 3)

        alphas = np.linspace(0.0, 1.0, n, dtype=np.float64)
        rotations = Rotation.from_matrix(np.stack([R0, R1], axis=0))
        R_points = Slerp([0.0, 1.0], rotations)(alphas).as_matrix()
        translations = (1.0 - alphas[:, None]) * t0 + alphas[:, None] * t1

        xyz = np.asarray(points[:, :3], dtype=np.float64)
        world = np.einsum("nij,nj->ni", R_points, xyz, optimize=True) + translations
        # p_ref = R0.T @ (p_world - t0). Row-vector equivalent is @ R0.
        ref = (world - t0) @ R0

        result = np.empty((n, 4), dtype=np.float64)
        result[:, :3] = ref
        result[:, 3] = points[:, 3]
        return result
