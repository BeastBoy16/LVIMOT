import numpy as np


class CarlaBoundingBoxProjector:

    def __init__(self, calibration):
        self.calibration = calibration
        self.K = calibration.get_camera_matrix()

        if self.K is None:
            raise ValueError(
                "Camera intrinsic matrix is missing."
            )

    def _rotation_matrix(self, rotation):
        """
        CARLA rotation convention.

        rotation contains degrees:
            roll
            pitch
            yaw
        """

        roll = np.deg2rad(
            float(rotation.get("roll", 0.0))
        )
        pitch = np.deg2rad(
            float(rotation.get("pitch", 0.0))
        )
        yaw = np.deg2rad(
            float(rotation.get("yaw", 0.0))
        )

        cr = np.cos(roll)
        sr = np.sin(roll)

        cp = np.cos(pitch)
        sp = np.sin(pitch)

        cy = np.cos(yaw)
        sy = np.sin(yaw)

        R = np.array([
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

        return R

    def world_to_ego(self, points, ego_pose):
        """
        Transform world coordinates into the ego-vehicle frame.
        """

        location = ego_pose["location"]

        ego_translation = np.array([
            float(location["x"]),
            float(location["y"]),
            float(location["z"])
        ], dtype=np.float64)

        R_world_ego = self._rotation_matrix(
            ego_pose["rotation"]
        )

        points = np.asarray(
            points,
            dtype=np.float64
        )

        # Inverse rigid transform:
        #
        # p_ego = R^T (p_world - t)

        return (
            R_world_ego.T
            @ (points - ego_translation).T
        ).T

    def ego_to_camera(self, points):
        """
        Transform ego coordinates into CARLA camera
        sensor coordinates.

        Camera transform is relative to the ego vehicle.
        """

        camera_transform = self.calibration.camera_transform

        location = camera_transform["location"]

        camera_translation = np.array([
            float(location["x"]),
            float(location["y"]),
            float(location["z"])
        ], dtype=np.float64)

        camera_rotation = camera_transform.get(
            "rotation",
            {}
        )

        R_ego_camera = self._rotation_matrix(
            camera_rotation
        )

        points = np.asarray(
            points,
            dtype=np.float64
        )

        return (
            R_ego_camera.T
            @ (points - camera_translation).T
        ).T

    def carla_camera_to_opencv(self, points):
        """
        CARLA camera coordinates:

            X = forward
            Y = right
            Z = up

        OpenCV camera coordinates:

            X = right
            Y = down
            Z = forward
        """

        points = np.asarray(
            points,
            dtype=np.float64
        )

        return np.column_stack([
            points[:, 1],
            -points[:, 2],
            points[:, 0]
        ])

    def project_world_corners(
        self,
        world_corners,
        ego_pose,
        image_shape
    ):
        """
        Project world-space CARLA bounding-box corners
        into image pixels.

        Returns:
            projected
            valid
            bbox
        """

        corners = np.asarray(
            world_corners,
            dtype=np.float64
        )

        if corners.ndim != 2 or corners.shape[1] != 3:
            raise ValueError(
                "world_corners must have shape (N, 3)."
            )

        # --------------------------------------------------
        # WORLD -> EGO
        # --------------------------------------------------

        ego_points = self.world_to_ego(
            corners,
            ego_pose
        )

        # --------------------------------------------------
        # EGO -> CAMERA
        # --------------------------------------------------

        camera_points = self.ego_to_camera(
            ego_points
        )

        # --------------------------------------------------
        # CARLA CAMERA -> OPENCV
        # --------------------------------------------------

        cv_points = self.carla_camera_to_opencv(
            camera_points
        )

        depth = cv_points[:, 2]

        valid = (
            np.isfinite(cv_points).all(axis=1)
            & (depth > 0)
        )

        projected = np.full(
            (len(corners), 2),
            np.nan,
            dtype=np.float64
        )

        if not np.any(valid):
            return projected, valid, None

        points = cv_points[valid]

        homogeneous = (
            self.K @ points.T
        ).T

        uv = np.column_stack([
            homogeneous[:, 0] / homogeneous[:, 2],
            homogeneous[:, 1] / homogeneous[:, 2]
        ])

        projected[valid] = uv

        # --------------------------------------------------
        # Enclosing image bounding box
        # --------------------------------------------------

        height = image_shape[0]
        width = image_shape[1]

        x1 = np.min(uv[:, 0])
        y1 = np.min(uv[:, 1])
        x2 = np.max(uv[:, 0])
        y2 = np.max(uv[:, 1])

        # Reject boxes that don't intersect image.
        if (
            x2 < 0
            or y2 < 0
            or x1 >= width
            or y1 >= height
        ):
            return projected, valid, None

        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))

        if x2 <= x1 or y2 <= y1:
            return projected, valid, None

        bbox = [
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ]

        return projected, valid, bbox
