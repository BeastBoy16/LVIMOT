import numpy as np


class CarlaLiDARCameraProjector:

    def __init__(self, calibration):
        """
        LiDAR -> camera image projection for CARLA.

        Parameters
        ----------
        calibration : CarlaCalibration
            CARLA calibration object.
        """

        self.calibration = calibration
        self.K = calibration.get_camera_matrix()

        if self.K is None:
            raise ValueError("Camera intrinsic matrix is missing.")

    def project(self, lidar_points, image_shape):
        """
        Project CARLA LiDAR points onto the camera image.

        Parameters
        ----------
        lidar_points : numpy.ndarray
            LiDAR points with shape (N, 3) or (N, 4).

        image_shape : tuple
            Image shape: (height, width, channels).

        Returns
        -------
        projected_points : numpy.ndarray
            Image coordinates with shape (M, 2).

        depths : numpy.ndarray
            Camera depth for each projected point.

        valid_lidar_points : numpy.ndarray
            Original LiDAR XYZ points corresponding to
            projected points.

        valid_indices : numpy.ndarray
            Indices of the original LiDAR points.
        """

        points = np.asarray(
            lidar_points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "LiDAR points must have shape (N, >=3)."
            )

        xyz = points[:, :3]

        # --------------------------------------------------
        # Remove invalid points
        # --------------------------------------------------

        finite = np.isfinite(xyz).all(axis=1)

        if not np.any(finite):
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64)
            )

        original_indices = np.nonzero(finite)[0]
        valid_xyz = xyz[finite]

        # --------------------------------------------------
        # LiDAR -> OpenCV camera coordinates
        # --------------------------------------------------

        camera_points = self.calibration.lidar_to_opencv(
            valid_xyz
        )

        # --------------------------------------------------
        # Keep points in front of camera
        # --------------------------------------------------

        positive_depth = camera_points[:, 2] > 0

        if not np.any(positive_depth):
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int64)
            )

        camera_points = camera_points[positive_depth]
        valid_xyz = valid_xyz[positive_depth]
        original_indices = original_indices[positive_depth]

        # --------------------------------------------------
        # Perspective projection
        # --------------------------------------------------

        homogeneous = (
            self.K @ camera_points.T
        ).T

        depths = camera_points[:, 2]

        u = (
            homogeneous[:, 0]
            / depths
        )

        v = (
            homogeneous[:, 1]
            / depths
        )

        # --------------------------------------------------
        # Keep points inside image
        # --------------------------------------------------

        height = image_shape[0]
        width = image_shape[1]

        inside = (
            (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )

        projected_points = np.column_stack(
            (
                u[inside],
                v[inside]
            )
        )

        depths = depths[inside]
        valid_lidar_points = valid_xyz[inside]
        valid_indices = original_indices[inside]

        return (
            projected_points,
            depths,
            valid_lidar_points,
            valid_indices
        )
