import numpy as np


class IMUPreintegrator:
    """Lightweight two-sample IMU preintegration.

    Acceleration is treated as body-frame specific force. Gravity is therefore
    not subtracted here; it is handled by the IMU factor in the world frame.
    Rotation uses the SO(3) exponential map rather than I + [w dt]x.
    """

    def __init__(self, min_dt: float = 1e-4, max_dt: float = 1.0):
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v, dtype=np.float64)
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    @classmethod
    def exp_so3(cls, rotation_vector: np.ndarray) -> np.ndarray:
        w = np.asarray(rotation_vector, dtype=np.float64)
        theta = float(np.linalg.norm(w))
        K = cls._skew(w)
        if theta < 1e-10:
            return np.eye(3, dtype=np.float64) + K + 0.5 * (K @ K)
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta * theta)
        return np.eye(3, dtype=np.float64) + A * K + B * (K @ K)

    def _dt(self, motion_k, motion_k1) -> float:
        t0 = motion_k.get("timestamp")
        t1 = motion_k1.get("timestamp")
        if t0 is not None and t1 is not None:
            dt = float(t1) - float(t0)
        else:
            dt = float(motion_k.get("dt", motion_k1.get("dt", 0.1)))
        if not np.isfinite(dt) or dt <= 0:
            dt = float(motion_k.get("dt", 0.1))
        return float(np.clip(dt, self.min_dt, self.max_dt))

    def integrate(self, motion_k, motion_k1):
        dt = self._dt(motion_k, motion_k1)
        a_k = np.asarray(motion_k["acceleration"], dtype=np.float64)
        a_k1 = np.asarray(motion_k1["acceleration"], dtype=np.float64)
        w_k = np.asarray(motion_k["gyro"], dtype=np.float64)
        w_k1 = np.asarray(motion_k1["gyro"], dtype=np.float64)

        acceleration = 0.5 * (a_k + a_k1)
        angular_velocity = 0.5 * (w_k + w_k1)
        delta_theta = angular_velocity * dt
        delta_R = self.exp_so3(delta_theta)
        delta_v = acceleration * dt
        delta_p = 0.5 * acceleration * (dt ** 2)

        return {
            "dt": dt,
            "delta_p": delta_p,
            "delta_v": delta_v,
            "delta_theta": delta_theta,
            "delta_R": delta_R,
        }
