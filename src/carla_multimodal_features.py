import numpy as np


class CarlaMultimodalFeatureBuilder:
    """
    Build multimodal features for CARLA objects.

    LiDAR planar features are expected in LiDAR sensor coordinates.
    They are transformed:

        LiDAR -> Ego -> World -> Object-local

    before 3D bounding-box filtering.

    Camera FAST keypoints and BRIEF descriptors are filtered using
    the projected 2D bounding box.
    """

    def __init__(self, calibration, max_lidar_distance=None):
        self.calibration = calibration
        self.max_lidar_distance = max_lidar_distance

    # ---------------------------------------------------------
    # CARLA rotation
    # ---------------------------------------------------------

    def _rotation_matrix(self, rotation):
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

    # ---------------------------------------------------------
    # LiDAR -> Ego
    # ---------------------------------------------------------

    def lidar_to_ego(self, points):
        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        transform = self.calibration.lidar_transform
        location = transform.get("location", transform)

        translation = np.array([
            float(location.get("x", 0.0)),
            float(location.get("y", 0.0)),
            float(location.get("z", 0.0))
        ], dtype=np.float64)

        rotation = transform.get("rotation", {})
        R = self._rotation_matrix(rotation)

        return (R @ points.T).T + translation

    # ---------------------------------------------------------
    # LiDAR normal -> Ego
    # ---------------------------------------------------------

    def lidar_normals_to_ego(self, normals):
        normals = np.asarray(normals, dtype=np.float64)

        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError("normals must have shape (N, 3)")

        transform = self.calibration.lidar_transform
        rotation = transform.get("rotation", {})
        R = self._rotation_matrix(rotation)

        return (R @ normals.T).T

    # ---------------------------------------------------------
    # Ego -> World
    # ---------------------------------------------------------

    def ego_to_world(self, points, ego_pose):
        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        location = ego_pose["location"]

        if isinstance(location, dict):
            translation = np.array([
                float(location["x"]),
                float(location["y"]),
                float(location["z"])
            ], dtype=np.float64)
        else:
            translation = np.array([
                float(location[0]),
                float(location[1]),
                float(location[2])
            ], dtype=np.float64)

        R = self._rotation_matrix(
            ego_pose["rotation"]
        )

        return (R @ points.T).T + translation

    # ---------------------------------------------------------
    # Ego normal -> World
    # ---------------------------------------------------------

    def ego_normals_to_world(self, normals, ego_pose):
        normals = np.asarray(normals, dtype=np.float64)

        R = self._rotation_matrix(
            ego_pose["rotation"]
        )

        return (R @ normals.T).T

    # ---------------------------------------------------------
    # LiDAR -> World
    # ---------------------------------------------------------

    def lidar_to_world(self, points, ego_pose):
        ego_points = self.lidar_to_ego(points)

        return self.ego_to_world(
            ego_points,
            ego_pose
        )

    # ---------------------------------------------------------
    # LiDAR normals -> World
    # ---------------------------------------------------------

    def lidar_normals_to_world(self, normals, ego_pose):
        ego_normals = self.lidar_normals_to_ego(normals)

        return self.ego_normals_to_world(
            ego_normals,
            ego_pose
        )

    # ---------------------------------------------------------
    # World -> Object local
    # ---------------------------------------------------------

    def world_to_object(self, points, obj):
        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")

        location = obj["location"]

        if isinstance(location, dict):
            center = np.array([
                float(location["x"]),
                float(location["y"]),
                float(location["z"])
            ], dtype=np.float64)
        else:
            center = np.array([
                float(location[0]),
                float(location[1]),
                float(location[2])
            ], dtype=np.float64)

        R = self._rotation_matrix(
            obj["rotation"]
        )

        return (
            R.T @ (points - center).T
        ).T

    # ---------------------------------------------------------
    # 3D box test
    # ---------------------------------------------------------

    def points_inside_3d_box(self, points_world, obj):
        points_world = np.asarray(
            points_world,
            dtype=np.float64
        )

        local = self.world_to_object(
            points_world,
            obj
        )

        dimensions = obj["dimensions"]

        half_dimensions = np.array([
            float(dimensions[0]) / 2.0,
            float(dimensions[1]) / 2.0,
            float(dimensions[2]) / 2.0
        ])

        return np.all(
            np.abs(local) <= half_dimensions,
            axis=1
        )

    # ---------------------------------------------------------
    # LiDAR planar feature extraction
    # ---------------------------------------------------------

    def extract_lidar_features(
        self,
        planar_features,
        obj,
        ego_pose
    ):
        """
        planar_features is the actual output of
        CarlaLiDARPlanarFeatureExtractor.extract():

            [
                {
                    "point": (3,),
                    "normal": (3,),
                    "eigenvalues": ...,
                    "surface_variation": ...,
                    "planarity": ...
                },
                ...
            ]

        Points and normals initially exist in LiDAR coordinates.
        """

        if planar_features is None:
            return {
                "points": np.empty((0, 3)),
                "normals": np.empty((0, 3)),
                "planarity": np.empty((0,)),
                "surface_variation": np.empty((0,)),
                "count": 0
            }

        if len(planar_features) == 0:
            return {
                "points": np.empty((0, 3)),
                "normals": np.empty((0, 3)),
                "planarity": np.empty((0,)),
                "surface_variation": np.empty((0,)),
                "count": 0
            }

        lidar_points = np.asarray([
            feature["point"]
            for feature in planar_features
        ], dtype=np.float64)

        lidar_normals = np.asarray([
            feature["normal"]
            for feature in planar_features
        ], dtype=np.float64)

        planarity = np.asarray([
            feature["planarity"]
            for feature in planar_features
        ], dtype=np.float64)

        surface_variation = np.asarray([
            feature["surface_variation"]
            for feature in planar_features
        ], dtype=np.float64)

        # -----------------------------------------------------
        # LiDAR -> World
        # -----------------------------------------------------

        world_points = self.lidar_to_world(
            lidar_points,
            ego_pose
        )

        world_normals = self.lidar_normals_to_world(
            lidar_normals,
            ego_pose
        )

        # -----------------------------------------------------
        # World -> Object box
        # -----------------------------------------------------

        inside = self.points_inside_3d_box(
            world_points,
            obj
        )

        selected_points = world_points[inside]
        selected_normals = world_normals[inside]
        selected_planarity = planarity[inside]
        selected_surface_variation = surface_variation[inside]

        # -----------------------------------------------------
        # Optional distance filtering
        # -----------------------------------------------------

        if (
            self.max_lidar_distance is not None
            and len(selected_points) > 0
        ):
            center = np.asarray(
                obj["location"],
                dtype=np.float64
            )

            distances = np.linalg.norm(
                selected_points - center,
                axis=1
            )

            keep = (
                distances <=
                float(self.max_lidar_distance)
            )

            selected_points = selected_points[keep]
            selected_normals = selected_normals[keep]
            selected_planarity = selected_planarity[keep]
            selected_surface_variation = (
                selected_surface_variation[keep]
            )

        return {
            "points": selected_points,
            "normals": selected_normals,
            "planarity": selected_planarity,
            "surface_variation": selected_surface_variation,
            "count": int(len(selected_points))
        }

    # ---------------------------------------------------------
    # Camera features
    # ---------------------------------------------------------

    def extract_camera_features(self, keypoints, descriptors, bbox):
        """
        Extract camera keypoints and corresponding BRIEF descriptors
        that lie inside the given 2D bounding box.

        keypoints:
            List of cv2.KeyPoint objects.

        descriptors:
            NumPy array with one descriptor per keypoint.

        bbox:
            [x1, y1, x2, y2].
        """

        if keypoints is None or descriptors is None:
            return []

        if len(keypoints) != len(descriptors):
            raise ValueError(
                "Number of keypoints must match number of descriptors"
            )

        bbox = np.asarray(bbox, dtype=np.float64)

        if bbox.shape != (4,):
            raise ValueError("bbox must have shape (4,)")

        x1, y1, x2, y2 = bbox

        camera_features = []

        for keypoint, descriptor in zip(keypoints, descriptors):

            # OpenCV KeyPoint -> numeric (x, y)
            x, y = keypoint.pt

            if x1 <= x <= x2 and y1 <= y <= y2:

                camera_features.append({
                    "keypoint": np.array(
                        [x, y],
                        dtype=np.float64
                    ),
                    "descriptor": np.asarray(
                        descriptor,
                        dtype=np.uint8
                    )
                })

        return camera_features

    def build_object(
        self,
        obj,
        association,
        projected_bbox,
        lidar_features,
        keypoints,
        descriptors
    ):
        camera_features = self.extract_camera_features(
            keypoints,
            descriptors,
            projected_bbox
        )

        return {
            "actor_id": int(obj["actor_id"]),
            "type": obj["type"],

            "location": np.asarray(
                obj["location"],
                dtype=np.float64
            ),

            "dimensions": np.asarray(
                obj["dimensions"],
                dtype=np.float64
            ),

            "rotation": {
                "yaw": float(
                    obj["rotation"]["yaw"]
                ),
                "pitch": float(
                    obj["rotation"]["pitch"]
                ),
                "roll": float(
                    obj["rotation"]["roll"]
                )
            },

            "bounding_box": obj["bounding_box"],

            "projected_bbox": np.asarray(
                projected_bbox,
                dtype=np.float64
            ),

            "association": {
                "lidar_id": int(
                    association["lidar_id"]
                ),
                "camera_id": int(
                    association["camera_id"]
                ),
                "iou": float(
                    association["iou"]
                )
            },

            "lidar_features": lidar_features,
            "camera_features": camera_features
        }

    # ---------------------------------------------------------
    # Build all objects
    # ---------------------------------------------------------

    def build(
        self,
        objects,
        associations,
        projected_boxes,
        planar_features,
        ego_pose,
        keypoints,
        descriptors
    ):
        """
        Build multimodal features for all associated objects.

        Parameters
        ----------
        objects:
            CARLA objects from CarlaDetector.

        associations:
            Output from CarlaObjectAssociator.

        projected_boxes:
            actor_id -> projected 2D bounding box.

        planar_features:
            Actual output from CarlaLiDARPlanarFeatureExtractor.

        ego_pose:
            CARLA ego pose for the current frame.

        keypoints:
            FAST keypoints.

        descriptors:
            BRIEF descriptors.
        """

        object_map = {
            int(obj["actor_id"]): obj
            for obj in objects
        }

        multimodal_objects = []

        for association in associations:

            actor_id = int(
                association["lidar_id"]
            )

            if actor_id not in object_map:
                continue

            if actor_id not in projected_boxes:
                continue

            obj = object_map[actor_id]

            projected_bbox = projected_boxes[
                actor_id
            ]["bbox"]

            lidar_features = (
                self.extract_lidar_features(
                    planar_features,
                    obj,
                    ego_pose
                )
            )

            multimodal_object = self.build_object(
                obj=obj,
                association=association,
                projected_bbox=projected_bbox,
                lidar_features=lidar_features,
                keypoints=keypoints,
                descriptors=descriptors
            )

            multimodal_objects.append(
                multimodal_object
            )

        return multimodal_objects
