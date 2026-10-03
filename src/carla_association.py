import numpy as np


class CarlaObjectAssociator:

    def __init__(self, iou_threshold=0.1):
        self.iou_threshold = float(iou_threshold)

    def calculate_iou(self, box_a, box_b):
        """
        Calculate IoU between two 2D bounding boxes.

        Box format:
            [x1, y1, x2, y2]
        """

        box_a = np.asarray(box_a, dtype=np.float64)
        box_b = np.asarray(box_b, dtype=np.float64)

        if box_a.shape != (4,) or box_b.shape != (4,):
            raise ValueError(
                "Bounding boxes must have shape (4,)"
            )

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        # Intersection
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0.0, ix2 - ix1)
        intersection_height = max(0.0, iy2 - iy1)

        intersection_area = (
            intersection_width * intersection_height
        )

        # Areas
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def associate(
        self,
        lidar_objects,
        camera_objects,
        projected_boxes
    ):
        """
        Associate 3D/LiDAR objects with camera objects.

        Parameters
        ----------
        lidar_objects : list
            CARLA 3D objects.

        camera_objects : list
            Camera-side 2D objects.

        projected_boxes : dict
            Dictionary:

                actor_id -> {
                    "bbox": [x1, y1, x2, y2],
                    "class": object_type
                }

        Returns
        -------
        associations : list
            One-to-one matched object pairs.
        """

        associations = []

        used_camera_ids = set()

        for lidar_object in lidar_objects:

            actor_id = int(lidar_object["actor_id"])
            object_class = lidar_object["type"]

            # The 3D object must have a valid projected box.
            if actor_id not in projected_boxes:
                continue

            projected_box = projected_boxes[actor_id]["bbox"]

            best_iou = 0.0
            best_camera_object = None

            for camera_object in camera_objects:

                camera_id = int(camera_object["actor_id"])

                # One-to-one matching
                if camera_id in used_camera_ids:
                    continue

                # Object class must match
                if camera_object["type"] != object_class:
                    continue

                camera_box = camera_object["bbox"]

                iou = self.calculate_iou(
                    projected_box,
                    camera_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_camera_object = camera_object

            if (
                best_camera_object is not None
                and best_iou >= self.iou_threshold
            ):

                associations.append(
                    {
                        "lidar_id": actor_id,
                        "camera_id": int(
                            best_camera_object["actor_id"]
                        ),
                        "class": object_class,
                        "iou": float(best_iou),
                    }
                )

                used_camera_ids.add(
                    int(best_camera_object["actor_id"])
                )

        return associations
