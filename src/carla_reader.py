import os
import json
import numpy as np
import cv2


class CarlaReader:

    def __init__(self, carla_root, sequence, camera="image_02"):

        self.carla_root = os.path.expanduser(carla_root)
        self.sequence = f"{int(sequence):04d}"
        self.camera = camera

        self.sequence_dir = os.path.join(
            self.carla_root,
            "sequences",
            self.sequence
        )

        self.image_dir = os.path.join(
            self.sequence_dir,
            self.camera
        )

        self.lidar_dir = os.path.join(
            self.sequence_dir,
            "velodyne"
        )

        self.imu_dir = os.path.join(
            self.sequence_dir,
            "imu"
        )

        self.pose_dir = os.path.join(
            self.sequence_dir,
            "poses"
        )

        self.label_dir = os.path.join(
            self.sequence_dir,
            "labels"
        )

        self.calibration_path = os.path.join(
            self.carla_root,
            "calibration",
            "carla_calibration.json"
        )

        self.timestamp_path = os.path.join(
            self.sequence_dir,
            "timestamps.txt"
        )

        if not os.path.isdir(self.sequence_dir):
            raise FileNotFoundError(
                f"CARLA sequence not found: {self.sequence_dir}"
            )

    # ========================================================
    # Utility
    # ========================================================

    def _frame_name(self, frame):
        return f"{int(frame):06d}"

    def _load_json(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        with open(path, "r") as f:
            return json.load(f)

    # ========================================================
    # LiDAR
    # ========================================================

    def load_lidar(self, frame):

        filename = self._frame_name(frame) + ".bin"

        path = os.path.join(
            self.lidar_dir,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LiDAR file not found: {path}"
            )

        points = np.fromfile(
            path,
            dtype=np.float32
        )

        if points.size % 4 != 0:
            raise ValueError(
                f"Invalid LiDAR file: {path}"
            )

        points = points.reshape(-1, 4)

        return points

    # ========================================================
    # Camera
    # ========================================================

    def load_image(self, frame):

        filename = self._frame_name(frame) + ".png"

        path = os.path.join(
            self.image_dir,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )

        image = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        # CARLA images were saved as RGBA.
        # OpenCV reads them as BGRA.
        # Convert to BGR because the existing
        # LVIMOT OpenCV pipeline uses BGR images.

        if image.ndim == 3 and image.shape[2] == 4:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2BGR
            )

        elif image.ndim == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        return image

    # ========================================================
    # IMU
    # ========================================================

    def load_imu(self, frame):

        filename = self._frame_name(frame) + ".json"

        path = os.path.join(
            self.imu_dir,
            filename
        )

        return self._load_json(path)

    # ========================================================
    # Pose
    # ========================================================

    def load_pose(self, frame):

        filename = self._frame_name(frame) + ".json"

        path = os.path.join(
            self.pose_dir,
            filename
        )

        return self._load_json(path)

    # ========================================================
    # Ground-truth labels
    # ========================================================

    def load_labels(self, frame):

        filename = self._frame_name(frame) + ".json"

        path = os.path.join(
            self.label_dir,
            filename
        )

        return self._load_json(path)

    # ========================================================
    # Calibration
    # ========================================================

    def load_calibration(self):

        return self._load_json(
            self.calibration_path
        )

    # ========================================================
    # Timestamp
    # ========================================================

    def load_timestamp(self, frame):

        if not os.path.exists(self.timestamp_path):
            raise FileNotFoundError(
                f"Timestamp file not found: {self.timestamp_path}"
            )

        with open(self.timestamp_path, "r") as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        frame = int(frame)

        if frame < 0 or frame >= len(lines):

            raise IndexError(
                f"Frame {frame} out of range. "
                f"Available frames: 0-{len(lines) - 1}"
            )

        # CARLA timestamps.txt format:
        #
        # Current dataset: one timestamp per line
        #
        # Example:
        # 2355.909665
        # 2356.009665
        # 2356.109665
        #
        # Older datasets may contain:
        # dataset_frame  carla_frame  timestamp
        #
        # Support both formats.

        parts = lines[frame].split()

        if len(parts) == 1:
            return float(parts[0])

        if len(parts) >= 3:
            return float(parts[2])

        raise ValueError(
            f"Invalid timestamp line at frame {frame}: "
            f"{lines[frame]}"
        )

    # ========================================================
    # Complete frame
    # ========================================================

    def get_frame(self, frame):

        return {
            "image": self.load_image(frame),
            "lidar": self.load_lidar(frame),
            "imu": self.load_imu(frame),
            "pose": self.load_pose(frame),
            "labels": self.load_labels(frame),
            "timestamp": self.load_timestamp(frame)
        }

    # ========================================================
    # Sequence information
    # ========================================================

    def num_frames(self):

        if not os.path.exists(self.timestamp_path):
            raise FileNotFoundError(
                f"Timestamp file not found: {self.timestamp_path}"
            )

        with open(self.timestamp_path, "r") as f:

            return sum(
                1
                for line in f
                if line.strip()
            )
