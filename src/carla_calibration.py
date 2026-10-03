import os
import json
import numpy as np


class CarlaCalibration:

    def __init__(self, calibration_file):

        self.calibration_file = os.path.expanduser(
            calibration_file
        )

        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(
                f"CARLA calibration file not found: "
                f"{self.calibration_file}"
            )

        self._load_calibration()

    # ========================================================
    # Load calibration
    # ========================================================

    def _load_calibration(self):

        with open(self.calibration_file, "r") as f:
            data = json.load(f)

        self.data = data

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        camera = data["camera"]

        self.width = int(camera["width"])
        self.height = int(camera["height"])
        self.fov = float(camera["fov"])

        self.K = np.array(
            camera["intrinsic"],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # LiDAR
        # ----------------------------------------------------

        lidar = data["lidar"]

        self.lidar_channels = int(
            lidar["channels"]
        )

        self.lidar_range = float(
            lidar["range"]
        )

        # ----------------------------------------------------
        # Sensor transforms
        # ----------------------------------------------------

        self.camera_transform = camera[
            "transform"
        ]

        self.lidar_transform = lidar[
            "transform"
        ]

        self.imu_transform = data[
            "imu"
        ]["transform"]

        # ----------------------------------------------------
        # LiDAR → camera sensor transform
        # ----------------------------------------------------

        self.T_lidar_to_camera = np.array(
            data["lidar_to_camera_sensor"],
            dtype=np.float64
        )

    # ========================================================
    # Camera intrinsics
    # ========================================================

    def get_camera_matrix(self):

        return self.K.copy()

    # ========================================================
    # LiDAR → camera sensor coordinates
    # ========================================================

    def lidar_to_camera(self, points):

        points = np.asarray(
            points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "LiDAR points must have shape (N, >=3)"
            )

        xyz = points[:, :3]

        ones = np.ones(
            (xyz.shape[0], 1),
            dtype=np.float64
        )

        homogeneous = np.hstack(
            (xyz, ones)
        )

        transformed = (
            self.T_lidar_to_camera
            @ homogeneous.T
        ).T

        return transformed[:, :3]

    # ========================================================
    # CARLA camera coordinates → OpenCV coordinates
    # ========================================================

    def carla_camera_to_opencv(self, points):

        points = np.asarray(
            points,
            dtype=np.float64
        )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Points must have shape (N, 3)"
            )

        x_carla = points[:, 0]
        y_carla = points[:, 1]
        z_carla = points[:, 2]

        # CARLA camera:
        #
        # X = forward
        # Y = right
        # Z = up
        #
        # OpenCV:
        #
        # X = right
        # Y = down
        # Z = forward

        x_cv = y_carla
        y_cv = -z_carla
        z_cv = x_carla

        return np.column_stack(
            (x_cv, y_cv, z_cv)
        )

    # ========================================================
    # LiDAR → OpenCV camera coordinates
    # ========================================================

    def lidar_to_opencv(self, points):

        camera_points = self.lidar_to_camera(
            points
        )

        return self.carla_camera_to_opencv(
            camera_points
        )

    # ========================================================
    # Project 3D points → image pixels
    # ========================================================

    def project_lidar(self, points):

        camera_points = self.lidar_to_opencv(
            points
        )

        z = camera_points[:, 2]

        valid = z > 0

        projected = np.full(
            (len(points), 2),
            np.nan,
            dtype=np.float64
        )

        if np.any(valid):

            pts = camera_points[valid]

            homogeneous = (
                self.K @ pts.T
            ).T

            projected[valid, 0] = (
                homogeneous[:, 0]
                / homogeneous[:, 2]
            )

            projected[valid, 1] = (
                homogeneous[:, 1]
                / homogeneous[:, 2]
            )

        return projected, valid

    # ========================================================
    # Print information
    # ========================================================

    def print_info(self):

        print("CARLA Calibration")
        print("-----------------")

        print(
            "Image size:",
            self.width,
            "x",
            self.height
        )

        print(
            "FOV:",
            self.fov
        )

        print(
            "Camera matrix:"
        )

        print(self.K)

        print(
            "LiDAR channels:",
            self.lidar_channels
        )

        print(
            "LiDAR range:",
            self.lidar_range
        )

        print(
            "LiDAR → Camera:"
        )

        print(
            self.T_lidar_to_camera
        )
