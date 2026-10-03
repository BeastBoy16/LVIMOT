import numpy as np


class CarlaLiDARTransform:

    def __init__(self, calibration):
        self.calibration = calibration

    def rotation_matrix(self, rotation):
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        return np.array([
            [
                cp * cy,
                cy * sp * sr - sy * cr,
                -cy * sp * cr - sy * sr
            ],
            [
                cp * sy,
                sy * sp * sr + cy * cr,
                -sy * sp * cr + cy * sr
            ],
            [
                sp,
                -cp * sr,
                cp * cr
            ]
        ], dtype=np.float64)

    def lidar_to_ego(self, points):
        """
        LiDAR sensor coordinates -> ego vehicle coordinates.

        CARLA calibration contains the LiDAR sensor transform
        relative to the ego vehicle.
        """

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "points must have shape (N, >=3)"
            )

        lidar_transform = self.calibration.lidar_transform

        translation = np.array([
            float(lidar_transform["location"]["x"]),
            float(lidar_transform["location"]["y"]),
            float(lidar_transform["location"]["z"])
        ])

        rotation = lidar_transform.get(
            "rotation",
            {}
        )

        R = self.rotation_matrix(rotation)

        xyz = points[:, :3]

        # LiDAR local -> ego coordinates
        ego = (
            R @ xyz.T
        ).T + translation

        return ego

    def ego_to_world(self, points, ego_pose):
        """
        Ego vehicle coordinates -> world coordinates.
        """

        points = np.asarray(
            points,
            dtype=np.float64
        )

        location = np.array([
            float(ego_pose["location"]["x"]),
            float(ego_pose["location"]["y"]),
            float(ego_pose["location"]["z"])
        ])

        R = self.rotation_matrix(
            ego_pose["rotation"]
        )

        world = (
            R @ points.T
        ).T + location

        return world

    def lidar_to_world(self, points, ego_pose):
        """
        LiDAR sensor coordinates -> world coordinates.
        """

        ego_points = self.lidar_to_ego(points)

        return self.ego_to_world(
            ego_points,
            ego_pose
        )
