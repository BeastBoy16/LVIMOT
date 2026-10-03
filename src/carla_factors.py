import numpy as np
from typing import Dict, Optional, Tuple, Union


class PosePriorFactor:
    def __init__(self, pose: np.ndarray, weight_rotation: float = 50.0, weight_position: float = 50.0):
        self.pose = np.asarray(pose, dtype=np.float64).reshape(6)
        self.weight_rotation = float(weight_rotation)
        self.weight_position = float(weight_position)

    def compute_residual(self, current_pose: np.ndarray) -> np.ndarray:
        current = np.asarray(current_pose, dtype=np.float64).reshape(6)
        residual = current - self.pose
        # Wrap Euler-angle residuals into [-pi, pi].
        residual[:3] = (residual[:3] + np.pi) % (2.0 * np.pi) - np.pi
        residual[:3] *= self.weight_rotation
        residual[3:] *= self.weight_position
        return residual


class VelocityPriorFactor:
    """Unary velocity prior used by sensor-only stationary mode."""

    def __init__(self, velocity: np.ndarray, weight: float = 20.0):
        self.velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
        self.weight = float(weight)

    def compute_residual(self, current_velocity: np.ndarray) -> np.ndarray:
        current = np.asarray(current_velocity, dtype=np.float64).reshape(3)
        return self.weight * (current - self.velocity)


class IMUPreintegrationFactor:
    def __init__(
        self,
        delta_p: np.ndarray,
        delta_v: np.ndarray,
        delta_R: np.ndarray,
        dt: float,
        gravity: Optional[np.ndarray] = None,
        weight_p: float = 10.0,
        weight_v: float = 5.0,
        weight_R: float = 20.0
    ):
        self.delta_p = np.asarray(delta_p, dtype=np.float64)
        self.delta_v = np.asarray(delta_v, dtype=np.float64)
        self.delta_R = np.asarray(delta_R, dtype=np.float64)
        self.dt = float(dt)
        self.gravity = (
            gravity
            if gravity is not None
            else np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        )
        self.weight_p = weight_p
        self.weight_v = weight_v
        self.weight_R = weight_R

    @staticmethod
    def log_so3(R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if abs(theta) < 1e-6:
            return 0.5 * np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ],
                dtype=np.float64
            )

        sin_theta = np.sin(theta)

        if abs(sin_theta) < 1e-8:
            return np.zeros(3, dtype=np.float64)

        factor = theta / (2.0 * sin_theta)

        return factor * np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ],
            dtype=np.float64
        )

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        v_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
        v_j: np.ndarray
    ) -> np.ndarray:
        dt = self.dt
        g = self.gravity

        r_p = (
            R_i.T
            @ (p_j - p_i - v_i * dt - 0.5 * g * (dt ** 2))
            - self.delta_p
        )

        r_v = (
            R_i.T
            @ (v_j - v_i - g * dt)
            - self.delta_v
        )

        R_err = self.delta_R.T @ (R_i.T @ R_j)
        r_R = self.log_so3(R_err)

        return np.hstack(
            [
                self.weight_p * r_p,
                self.weight_v * r_v,
                self.weight_R * r_R
            ]
        )


class LiDARRelativePoseFactor:
    """Binary SE(3) relative-pose constraint from scan registration.

    ``delta_R`` and ``delta_t`` map a point expressed in the current ego frame
    into the previous ego frame:

        p_prev = delta_R @ p_curr + delta_t

    For ego poses represented as ego->world transforms, the predicted transform
    is ``R_i.T @ R_j`` and ``R_i.T @ (p_j - p_i)``.
    """

    def __init__(
        self,
        delta_R: np.ndarray,
        delta_t: np.ndarray,
        weight_translation: float = 8.0,
        weight_rotation: float = 12.0,
    ):
        self.delta_R = np.asarray(delta_R, dtype=np.float64).reshape(3, 3)
        self.delta_t = np.asarray(delta_t, dtype=np.float64).reshape(3)
        self.weight_translation = float(weight_translation)
        self.weight_rotation = float(weight_rotation)

    @staticmethod
    def _log_so3(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(R))
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        vee = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=np.float64,
        )
        if theta < 1e-7:
            return 0.5 * vee
        sin_theta = float(np.sin(theta))
        if abs(sin_theta) < 1e-8:
            return np.zeros(3, dtype=np.float64)
        return (theta / (2.0 * sin_theta)) * vee

    def compute_residual(
        self,
        R_i: np.ndarray,
        p_i: np.ndarray,
        R_j: np.ndarray,
        p_j: np.ndarray,
    ) -> np.ndarray:
        R_i = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
        R_j = np.asarray(R_j, dtype=np.float64).reshape(3, 3)
        p_i = np.asarray(p_i, dtype=np.float64).reshape(3)
        p_j = np.asarray(p_j, dtype=np.float64).reshape(3)

        R_pred = R_i.T @ R_j
        t_pred = R_i.T @ (p_j - p_i)
        R_err = self.delta_R.T @ R_pred
        r_rot = self._log_so3(R_err)
        r_trans = t_pred - self.delta_t
        return np.hstack(
            [
                self.weight_translation * r_trans,
                self.weight_rotation * r_rot,
            ]
        )


class LiDARPlanarFactor:
    def __init__(
        self,
        point_lidar: np.ndarray,
        plane_normal_w: np.ndarray,
        plane_center_w: np.ndarray,
        weight: float = 5.0
    ):
        self.point_lidar = np.asarray(point_lidar, dtype=np.float64)
        self.plane_normal_w = np.asarray(
            plane_normal_w,
            dtype=np.float64
        )

        self.plane_normal_w /= max(
            np.linalg.norm(self.plane_normal_w),
            1e-6
        )

        self.plane_center_w = np.asarray(
            plane_center_w,
            dtype=np.float64
        )

        self.weight = weight

    def compute_residual(
        self,
        R_ego: np.ndarray,
        t_ego: np.ndarray
    ) -> float:
        p_w = R_ego @ self.point_lidar + t_ego

        dist = float(
            np.dot(
                self.plane_normal_w,
                p_w - self.plane_center_w
            )
        )

        return self.weight * dist


class VisualReprojectionFactor:
    def __init__(
        self,
        measurement_uv: np.ndarray,
        K: np.ndarray,
        landmark_3d_w: np.ndarray,
        weight: float = 1.0,
        camera_translation_ego: Optional[np.ndarray] = None,
        camera_rotation_ego: Optional[np.ndarray] = None,
        use_carla_camera_axes: bool = False,
    ):
        self.z_uv = np.asarray(measurement_uv, dtype=np.float64)
        self.K = np.asarray(K, dtype=np.float64)
        self.landmark_3d_w = np.asarray(landmark_3d_w, dtype=np.float64)
        self.weight = float(weight)
        self.camera_translation_ego = (
            None if camera_translation_ego is None
            else np.asarray(camera_translation_ego, dtype=np.float64).reshape(3)
        )
        self.camera_rotation_ego = (
            None if camera_rotation_ego is None
            else np.asarray(camera_rotation_ego, dtype=np.float64).reshape(3, 3)
        )
        self.use_carla_camera_axes = bool(use_carla_camera_axes)

    def compute_residual(
        self,
        R_cam_w: np.ndarray,
        t_cam_w: np.ndarray
    ) -> np.ndarray:
        # Backward-compatible path: callers may provide world->camera directly.
        if self.camera_translation_ego is None or self.camera_rotation_ego is None:
            p_c = R_cam_w @ self.landmark_3d_w + t_cam_w
        else:
            # In the factor graph R_cam_w/t_cam_w are passed as world->ego.
            p_ego = R_cam_w @ self.landmark_3d_w + t_cam_w
            p_carla_cam = self.camera_rotation_ego.T @ (p_ego - self.camera_translation_ego)
            if self.use_carla_camera_axes:
                # CARLA camera: X forward, Y right, Z up.
                # OpenCV: X right, Y down, Z forward.
                p_c = np.array([p_carla_cam[1], -p_carla_cam[2], p_carla_cam[0]], dtype=np.float64)
            else:
                p_c = p_carla_cam

        if p_c[2] <= 0.1 or not np.isfinite(p_c).all():
            return np.array([100.0, 100.0], dtype=np.float64) * self.weight
        uv_h = self.K @ p_c
        predicted = uv_h[:2] / uv_h[2]
        return self.weight * (predicted - self.z_uv[:2])


class DynamicObjectFactor:
    def __init__(
        self,
        obs_local: np.ndarray,
        dt_from_origin: float,
        weight_pos: float = 5.0
    ):
        self.obs_local = np.asarray(
            obs_local,
            dtype=np.float64
        )

        self.dt = float(dt_from_origin)
        self.weight_pos = weight_pos

    def compute_residual(
        self,
        R_ego: np.ndarray,
        t_ego: np.ndarray,
        obj_p0: np.ndarray,
        obj_vel: np.ndarray
    ) -> np.ndarray:
        p_obj_pred = (
            obj_p0
            + obj_vel * self.dt
        )

        p_obj_meas = (
            R_ego @ self.obs_local
            + t_ego
        )

        return self.weight_pos * (
            p_obj_pred - p_obj_meas
        )


class MarginalizationFactor:
    """
    Square-root representation of a marginalized quadratic prior.

    The stored quadratic form is:

        0.5 * dx.T @ H_prior @ dx + b_prior.T @ dx

    The least-squares residual is constructed as:

        r = sqrt_info @ dx + residual_offset

    such that:

        sqrt_info.T @ sqrt_info = H_prior
        sqrt_info.T @ residual_offset = b_prior
    """

    def __init__(
        self,
        linearization_point: np.ndarray,
        H_prior: np.ndarray,
        b_prior: np.ndarray
    ):
        self.x_lin = np.asarray(
            linearization_point,
            dtype=np.float64
        ).copy()

        H = np.asarray(
            H_prior,
            dtype=np.float64
        )

        b = np.asarray(
            b_prior,
            dtype=np.float64
        ).reshape(-1)

        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(
                "H_prior must be a square matrix."
            )

        if H.shape[0] != len(self.x_lin):
            raise ValueError(
                "H_prior dimension must match linearization_point."
            )

        if len(b) != len(self.x_lin):
            raise ValueError(
                "b_prior dimension must match linearization_point."
            )

        H = 0.5 * (H + H.T)

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        max_eigenvalue = max(
            float(np.max(np.abs(eigenvalues))),
            1.0
        )

        tolerance = 1e-10 * max_eigenvalue

        keep = eigenvalues > tolerance

        if np.any(keep):
            values = np.maximum(
                eigenvalues[keep],
                0.0
            )

            vectors = eigenvectors[:, keep]

            self.sqrt_info = (
                np.sqrt(values)[:, None]
                * vectors.T
            )

            self.residual_offset = np.linalg.lstsq(
                self.sqrt_info.T,
                b,
                rcond=None
            )[0]
        else:
            self.sqrt_info = np.zeros(
                (0, len(self.x_lin)),
                dtype=np.float64
            )

            self.residual_offset = np.zeros(
                0,
                dtype=np.float64
            )

        self.H_prior = (
            self.sqrt_info.T
            @ self.sqrt_info
        )

        self.b_prior = (
            self.sqrt_info.T
            @ self.residual_offset
        )

    def compute_residual(
        self,
        current_x: np.ndarray
    ) -> np.ndarray:
        current_x = np.asarray(
            current_x,
            dtype=np.float64
        )

        if len(current_x) != len(self.x_lin):
            raise ValueError(
                "Current state dimension does not match "
                "the marginalization prior."
            )

        dx = current_x - self.x_lin

        if self.sqrt_info.shape[0] == 0:
            return np.zeros(
                0,
                dtype=np.float64
            )

        return (
            self.sqrt_info @ dx
            + self.residual_offset
        )