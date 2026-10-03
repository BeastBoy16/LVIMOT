from __future__ import annotations
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import psutil
except ImportError:
    psutil = None

from carla_runtime_config import DEFAULT_LIMITS, PipelineLimits
from carla_calibration import CarlaCalibration
from carla_lidar_features import CarlaLiDARPlanarFeatureExtractor
from carla_multimodal_features import CarlaMultimodalFeatureBuilder
from carla_lidar_deskew import CarlaLiDARDeskewer
from carla_imu import CarlaIMUPreprocessor
from imu_preintegration import IMUPreintegrator
from fast_features import FASTFeatureDetector
from brief_features import BRIEFDescriptorExtractor
from carla_temporal_visual import CarlaTemporalVisualTracker
from carla_temporal_lidar import CarlaTemporalLiDARMatcher
from carla_motion_classification import CarlaMotionClassifier
from carla_tracking import CarlaMultiObjectTracker
from carla_sliding_window import CarlaSlidingWindowEstimator
from carla_4d_mapping import Carla4DEnvironmentMapper
from carla_evaluation import CarlaEvaluator
from carla_lidar_camera_projection import CarlaLiDARCameraProjector
from carla_sensor_objects import CarlaLiDARObjectDetector, build_sensor_multimodal_objects
from carla_camera_lidar_detector import CarlaCameraLiDARObjectDetector, CameraLiDARDetectionConfig
from carla_box_projection import CarlaBoundingBoxProjector


class CarlaLVIMOTCore:
    """Shared sensor-only LVIMOT core for offline and live CARLA.

    Camera/LiDAR/IMU enter the estimator. CARLA pose and actor labels may be
    supplied only through ``ground_truth_*`` arguments and are used exclusively
    for evaluation histories; they never influence prediction, factors, tracks,
    or the map.
    """

    GRAVITY = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

    def __init__(
        self,
        calibration: CarlaCalibration,
        limits: PipelineLimits = DEFAULT_LIMITS,
        brief_bytes: int = 32,
        voxel_size: float = 0.20,
        detector_mode: str = "lidar",
        yolo_model: str = "yolov8n.pt",
        yolo_confidence: float = 0.12,
        yolo_image_size: int = 1280,
        yolo_device: str = "auto",
    ):
        self.calibration = calibration
        self.limits = limits
        calibration_range = float(getattr(calibration, "lidar_range", 0.0) or 0.0)
        configured_range = float(getattr(limits, "object_max_range", 0.0) or 0.0)
        if calibration_range > 0.0 and configured_range > 0.0:
            self.object_sensor_range = min(calibration_range, configured_range)
        elif calibration_range > 0.0:
            self.object_sensor_range = calibration_range
        elif configured_range > 0.0:
            self.object_sensor_range = configured_range
        else:
            self.object_sensor_range = 55.0
        self.detector_mode = str(detector_mode).strip().lower()
        if self.detector_mode not in {"lidar", "camera_lidar"}:
            raise ValueError("detector_mode must be 'lidar' or 'camera_lidar'")
        self.lidar_features = CarlaLiDARPlanarFeatureExtractor(
            voxel_size=voxel_size,
            max_candidate_points=limits.max_lidar_planar_candidates,
            max_query_points=limits.max_lidar_planar_queries,
        )
        self.multimodal_builder = CarlaMultimodalFeatureBuilder(calibration)
        self.lidar_projector = CarlaLiDARCameraProjector(calibration)
        self.gt_box_projector = CarlaBoundingBoxProjector(calibration)
        self.lidar_deskewer = CarlaLiDARDeskewer(scan_duration=0.1)
        self.imu_preprocessor = CarlaIMUPreprocessor(dt=0.1)
        self.imu_preintegrator = IMUPreintegrator()
        self.fast = FASTFeatureDetector(threshold=20, nonmax_suppression=True)
        self.brief = BRIEFDescriptorExtractor(bytes=brief_bytes)
        self.visual_tracker = CarlaTemporalVisualTracker(
            max_hamming_distance=40,
            ratio_threshold=0.85,
            max_features=limits.max_visual_features,
            max_tracks=limits.max_visual_tracks,
            history_size=limits.visual_history_size,
        )
        self.lidar_tracker = CarlaTemporalLiDARMatcher(
            max_center_distance=1.0,
            max_features=limits.max_planar_features_for_temporal,
            max_landmarks=limits.max_planar_landmarks,
            history_size=limits.lidar_landmark_history_size,
        )
        self.motion_classifier = CarlaMotionClassifier()
        self.object_tracker = CarlaMultiObjectTracker(
            distance_threshold=limits.tracker_distance_threshold,
            max_missed=limits.tracker_max_missed,
            min_hits=min(limits.tracker_min_hits, limits.tracker_fast_confirm_hits),
            use_ego_compensation=True,
            motion_classifier=self.motion_classifier,
            history_size=limits.tracker_history_size,
            new_track_min_score=(
                limits.tracker_new_track_min_score
                if self.detector_mode == "camera_lidar"
                else limits.tracker_new_track_min_score
            ),
            publish_min_score=(
                limits.tracker_publish_min_score
                if self.detector_mode == "camera_lidar"
                else limits.tracker_new_track_min_score
            ),
            publish_max_missed=(
                limits.tracker_publish_max_missed if self.detector_mode == "camera_lidar" else 0
            ),
            gap_publish_min_hits=limits.tracker_gap_publish_min_hits,
            min_camera_support=(
                limits.tracker_min_camera_support if self.detector_mode == "camera_lidar" else 0
            ),
            fast_confirm_hits=limits.tracker_fast_confirm_hits,
            fast_confirm_min_score=limits.tracker_fast_confirm_min_score,
            fast_confirm_min_support=limits.tracker_fast_confirm_min_support,
            weak_confirm_min_score=limits.tracker_weak_confirm_min_score,
            weak_confirm_min_camera_confidence=limits.tracker_weak_confirm_min_camera_confidence,
            weak_confirm_min_support=limits.tracker_weak_confirm_min_support,
            weak_confirm_hits=limits.tracker_weak_confirm_hits,
            static_suppress_min_hits=limits.tracker_static_suppress_min_hits,
            static_suppress_min_confidence=limits.tracker_static_suppress_min_confidence,
            fallback_min_hits=limits.tracker_fallback_min_hits,
            fallback_min_support=limits.tracker_fallback_min_support,
        )
        self.sliding_window = CarlaSlidingWindowEstimator(
            window_size=limits.sliding_window_size,
            max_lidar_factors_per_frame=limits.max_lidar_factors_per_frame,
            max_visual_factors_per_frame=limits.max_visual_factors_per_frame,
            max_object_factors_per_frame=limits.max_object_factors_per_frame,
            max_objects=limits.max_graph_objects,
            optimizer_max_nfev=limits.optimizer_max_nfev,
        )
        self.mapper_4d = Carla4DEnvironmentMapper(
            static_voxel_size=voxel_size,
            max_static_voxels=limits.max_map_static_voxels,
            max_planar_landmarks=limits.max_map_planar_landmarks,
            max_dynamic_history=limits.max_dynamic_tube_history,
        )
        self.evaluator = CarlaEvaluator()
        self.sensor_object_detector = CarlaLiDARObjectDetector(
            voxel_size=limits.object_voxel_size,
            min_points=limits.object_min_points,
            max_range=self.object_sensor_range,
            min_height=limits.object_min_height,
            max_height=limits.object_max_height,
            max_proposals=limits.max_object_proposals,
            max_points=limits.object_max_points,
            min_vehicle_score=limits.object_min_vehicle_score,
        )
        self.camera_lidar_detector = None
        if self.detector_mode == "camera_lidar":
            self.camera_lidar_detector = CarlaCameraLiDARObjectDetector(
                calibration,
                CameraLiDARDetectionConfig(
                    model_path=str(yolo_model),
                    confidence=float(yolo_confidence),
                    image_size=int(yolo_image_size),
                    max_detections=limits.max_object_proposals,
                    max_range=self.object_sensor_range,
                    device=str(yolo_device),
                    half=None,
                ),
            )
        # Camera inference is independent of planar-feature extraction.  A
        # single persistent worker overlaps CUDA YOLO + LiDAR projection with
        # CPU planar geometry without ever running two YOLO calls concurrently.
        self._detector_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="lvimot-detector")
            if self.detector_mode == "camera_lidar"
            else None
        )

        self._last_pose: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None
        self._last_imu_motion: Optional[Dict] = None
        self._previous_planar_ego: Optional[List[Dict]] = None
        self._previous_pose_for_lidar: Optional[np.ndarray] = None
        self._visual_landmarks_world: Dict[int, np.ndarray] = {}

        self.tracking_history: List[List[Dict]] = []
        self.ground_truth_history: List[List[Dict]] = []
        self.estimated_pose_history: List[np.ndarray] = []
        self.ground_truth_pose_history: List[np.ndarray] = []

        # Evaluation-only anchors.  Localization keeps its existing pose
        # convention; object tracking must use CARLA's left-handed world
        # convention so GT actors land in the same ego-local frame as LiDAR.
        self._gt_anchor_R_localization: Optional[np.ndarray] = None
        self._gt_anchor_R_tracking: Optional[np.ndarray] = None
        self._gt_anchor_t: Optional[np.ndarray] = None

        self.camera_translation_ego, self.camera_rotation_ego = self._camera_extrinsic()
        # Sensor-integrity state. Exact hashes are cheap enough for CARLA frames
        # and let the offline/live core reject capture freezes without GT.
        self._last_image_hash: Optional[bytes] = None
        self._last_lidar_hash: Optional[bytes] = None
        self._stationary_streak = 0
        self._stationary_release_streak = 0
        self._stationary_active = False
        self._stationary_anchor_pose: Optional[np.ndarray] = None
        self._last_stationary_lidar_metric: Optional[float] = None
        self._previous_lidar_range_signature: Optional[np.ndarray] = None
        # V24 zero-velocity guard: independent image/LiDAR translation evidence.
        # These buffers contain sensor data only; no GT or evaluator state enters.
        self._previous_motion_gray: Optional[np.ndarray] = None
        self._previous_motion_lidar_xyz: Optional[np.ndarray] = None
        # V25 full metric LiDAR relative-pose diagnostic/factor state.
        self._last_lidar_se3: Dict = {"accepted": False, "reason": "uninitialized"}
        # V25.1: retain the full current->previous rigid transform already
        # estimated by the V24 motion ICP.  V25 previously discarded its
        # direction and later re-estimated SE(3) from plane residuals alone,
        # which is degenerate on road-dominated scenes.
        self._last_lidar_motion_transform: Optional[Dict] = None


    @staticmethod
    def euler_to_rot(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
        return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    @staticmethod
    def rot_to_euler(R: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_euler("xyz")

    @staticmethod
    def pose_vector_from_dict(pose: Dict) -> np.ndarray:
        r = pose.get("rotation", {})
        p = pose.get("location", {})
        return np.array(
            [
                np.deg2rad(float(r.get("roll", 0.0))),
                np.deg2rad(float(r.get("pitch", 0.0))),
                np.deg2rad(float(r.get("yaw", 0.0))),
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
            ],
            dtype=np.float64,
        )

    @classmethod
    def pose_dict_from_vector(cls, pose: np.ndarray) -> Dict:
        pose = np.asarray(pose, dtype=np.float64)
        return {
            "location": {"x": float(pose[3]), "y": float(pose[4]), "z": float(pose[5])},
            "rotation": {
                "roll": float(np.rad2deg(pose[0])),
                "pitch": float(np.rad2deg(pose[1])),
                "yaw": float(np.rad2deg(pose[2])),
            },
        }

    @staticmethod
    def _rotation_from_carla_dict(rotation: Dict) -> np.ndarray:
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
        return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    @staticmethod
    def _carla_world_rotation(rotation: Dict) -> np.ndarray:
        """CARLA left-handed actor rotation matrix (ego/local -> world).

        CARLA uses X-forward, Y-right, Z-up in a left-handed world.  SciPy's
        Rotation class is right-handed, so it must not be used to convert
        CARLA actor/world coordinates for MOT evaluation.  This matrix matches
        the coordinate convention used by the tracker and multimodal transforms.
        """
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ], dtype=np.float64)

    def _camera_extrinsic(self) -> Tuple[np.ndarray, np.ndarray]:
        transform = self.calibration.camera_transform
        loc = transform.get("location", transform)
        t = np.array(
            [float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), float(loc.get("z", 0.0))],
            dtype=np.float64,
        )
        R = self._rotation_from_carla_dict(transform.get("rotation", {}))
        return t, R

    def _memory_mb(self) -> float:
        if psutil is None:
            return 0.0
        try:
            return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
        except Exception:
            return 0.0

    def _predict_state(self, imu_preintegrated: Optional[Dict], timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        if self._last_pose is None:
            return np.zeros(6, dtype=np.float64), np.zeros(3, dtype=np.float64)
        prev_pose = self._last_pose
        prev_vel = self._last_velocity if self._last_velocity is not None else np.zeros(3)
        dt = max(float(timestamp) - float(self._last_timestamp), 1e-4)
        pose = prev_pose.copy()
        velocity = prev_vel.copy()
        if imu_preintegrated is None:
            pose[3:] = prev_pose[3:] + prev_vel * dt
            return pose, velocity

        dt = float(imu_preintegrated.get("dt", dt))
        R_prev = self.euler_to_rot(prev_pose[:3])
        dR = np.asarray(imu_preintegrated.get("delta_R", np.eye(3)), dtype=np.float64)
        dp = np.asarray(imu_preintegrated.get("delta_p", np.zeros(3)), dtype=np.float64)
        dv = np.asarray(imu_preintegrated.get("delta_v", np.zeros(3)), dtype=np.float64)
        R_new = R_prev @ dR
        pose[:3] = self.rot_to_euler(R_new)
        pose[3:] = prev_pose[3:] + prev_vel * dt + 0.5 * self.GRAVITY * dt * dt + R_prev @ dp
        velocity = prev_vel + self.GRAVITY * dt + R_prev @ dv
        return pose, velocity

    def _feature_to_ego(self, feat: Dict) -> Dict:
        center_lidar = np.asarray(feat.get("center", feat.get("point", np.zeros(3))), dtype=np.float64)
        point_lidar = np.asarray(feat.get("point", center_lidar), dtype=np.float64)
        normal_lidar = np.asarray(feat.get("normal", [0.0, 0.0, 1.0]), dtype=np.float64)
        center = self.multimodal_builder.lidar_to_ego(center_lidar.reshape(1, 3))[0]
        point = self.multimodal_builder.lidar_to_ego(point_lidar.reshape(1, 3))[0]
        normal = self.multimodal_builder.lidar_normals_to_ego(normal_lidar.reshape(1, 3))[0]
        normal /= max(np.linalg.norm(normal), 1e-9)
        return {
            "center": center,
            "point": point,
            "normal": normal,
            "planarity": float(feat.get("planarity", 0.0)),
        }

    def _bounded_planar_ego(self, planar_features: List[Dict]) -> List[Dict]:
        if not planar_features:
            return []
        limit = self.limits.max_planar_features_for_temporal
        if len(planar_features) <= limit:
            selected = planar_features
        else:
            scores = np.asarray([float(f.get("planarity", 0.0)) for f in planar_features])
            ids = np.argpartition(scores, -limit)[-limit:]
            selected = [planar_features[int(i)] for i in ids]
        return [self._feature_to_ego(f) for f in selected]

    def _lidar_factor_features(self, current_planar_ego: List[Dict], current_pose: np.ndarray) -> List[Dict]:
        if self._previous_planar_ego is None or self._previous_pose_for_lidar is None:
            return []
        R_prev = self.euler_to_rot(self._previous_pose_for_lidar[:3])
        t_prev = self._previous_pose_for_lidar[3:]
        R_curr = self.euler_to_rot(current_pose[:3])
        t_curr = current_pose[3:]
        # Previous ego coordinates -> current ego coordinates.
        R_rel = R_curr.T @ R_prev
        t_rel = R_curr.T @ (t_prev - t_curr)
        matches = self.lidar_tracker.match_planar_features(
            self._previous_planar_ego,
            current_planar_ego,
            R_pred=R_rel,
            t_pred=t_rel,
        )
        factors = []
        for prev_idx, curr_idx in matches[: self.limits.max_lidar_factors_per_frame]:
            prev = self._previous_planar_ego[prev_idx]
            curr = current_planar_ego[curr_idx]
            plane_center_w = R_prev @ np.asarray(prev["center"]) + t_prev
            plane_normal_w = R_prev @ np.asarray(prev["normal"])
            plane_normal_w /= max(np.linalg.norm(plane_normal_w), 1e-9)
            factors.append(
                {
                    "point": np.asarray(curr["point"], dtype=np.float64),
                    "center": plane_center_w,
                    "normal": plane_normal_w,
                    "planarity": float(curr.get("planarity", 0.0)),
                }
            )
        return factors

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )

    def _lidar_se3_relative_measurement(
        self,
        current_planar_ego: List[Dict],
        current_pose_prediction: np.ndarray,
        expected_translation_m: Optional[float] = None,
        expected_rotation_rad: Optional[float] = None,
    ) -> Dict:
        """Robust scan-to-scan point-to-plane SE(3) measurement.

        The measurement maps coordinates in the current ego frame into the
        previous ego frame.  It uses only LiDAR planar geometry already present
        in the LVIMOT front end; CARLA pose/labels never enter this calculation.
        """
        rejected = {
            "accepted": False,
            "delta_R": np.eye(3, dtype=np.float64),
            "delta_t": np.zeros(3, dtype=np.float64),
            "matches": 0,
            "inliers": 0,
            "rmse_m": None,
            "translation_m": None,
            "rotation_rad": None,
            "point_rmse_m": None,
            "source": "none",
            "reason": "unavailable",
            "weight_translation": float(self.limits.lidar_se3_translation_weight),
            "weight_rotation": float(self.limits.lidar_se3_rotation_weight),
        }
        if not bool(getattr(self.limits, "lidar_se3_enabled", True)):
            rejected["reason"] = "disabled"
            return rejected
        if self._previous_planar_ego is None or self._previous_pose_for_lidar is None:
            rejected["reason"] = "no_previous_scan"
            return rejected
        if not self._previous_planar_ego or not current_planar_ego:
            rejected["reason"] = "no_planar_features"
            return rejected

        prev_pose = np.asarray(self._previous_pose_for_lidar, dtype=np.float64).reshape(6)
        curr_pose = np.asarray(current_pose_prediction, dtype=np.float64).reshape(6)
        R_prev = self.euler_to_rot(prev_pose[:3])
        R_curr = self.euler_to_rot(curr_pose[:3])
        p_prev = prev_pose[3:]
        p_curr = curr_pose[3:]

        # Predicted current->previous transform from the IMU/previous graph state.
        R_cp = R_prev.T @ R_curr
        t_cp = R_prev.T @ (p_curr - p_prev)

        # V25.1: the V24 motion ICP already solved a robust full current->previous
        # transform on a non-ground-biased cloud.  Prefer that transform as the
        # SE(3) seed.  Point-to-plane geometry is used to validate/refine only
        # when safe; it is no longer allowed to invent road-tangent motion.
        motion_seed = getattr(self, "_last_lidar_motion_transform", None)
        use_motion_seed = bool(motion_seed and motion_seed.get("accepted", False))
        if use_motion_seed:
            R_cp = np.asarray(motion_seed["delta_R"], dtype=np.float64).reshape(3, 3)
            t_cp = np.asarray(motion_seed["delta_t"], dtype=np.float64).reshape(3)

        # Matcher expects previous->current.
        R_pc = R_cp.T
        t_pc = -R_pc @ t_cp
        matches = self.lidar_tracker.match_planar_features(
            self._previous_planar_ego,
            current_planar_ego,
            R_pred=R_pc,
            t_pred=t_pc,
        )
        min_matches = max(12, int(self.limits.lidar_se3_min_matches))
        if len(matches) < min_matches:
            rejected["matches"] = int(len(matches))
            rejected["reason"] = "too_few_matches"
            return rejected

        prev_centers = np.asarray(
            [self._previous_planar_ego[i]["center"] for i, _ in matches], dtype=np.float64
        )
        prev_normals = np.asarray(
            [self._previous_planar_ego[i]["normal"] for i, _ in matches], dtype=np.float64
        )
        curr_centers = np.asarray(
            [current_planar_ego[j]["center"] for _, j in matches], dtype=np.float64
        )
        curr_normals = np.asarray(
            [current_planar_ego[j]["normal"] for _, j in matches], dtype=np.float64
        )
        prev_normals /= np.maximum(np.linalg.norm(prev_normals, axis=1, keepdims=True), 1e-9)
        curr_normals /= np.maximum(np.linalg.norm(curr_normals, axis=1, keepdims=True), 1e-9)

        R = R_cp.copy()
        t = t_cp.copy()
        seed_R = R.copy()
        seed_t = t.copy()
        trim_fraction = float(np.clip(self.limits.lidar_se3_trim_fraction, 0.45, 1.0))
        huber = max(1e-3, float(self.limits.lidar_se3_huber_m))
        # The raw-cloud ICP is already a full-rank 3-D rigid fit.  When it is
        # available, keep it as the metric transform and use planar geometry as
        # a validation signal only.  This removes V25's road-plane degeneracy.
        max_iterations = 0 if use_motion_seed else max(1, int(self.limits.lidar_se3_max_iterations))
        final_ids = np.arange(len(matches), dtype=np.int64)

        for _ in range(max_iterations):
            transformed = (R @ curr_centers.T).T + t
            transformed_normals = (R @ curr_normals.T).T
            transformed_normals /= np.maximum(
                np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-9
            )
            normal_cos = np.abs(np.einsum("ij,ij->i", transformed_normals, prev_normals))
            residual = np.einsum("ij,ij->i", prev_normals, transformed - prev_centers)
            valid = np.isfinite(residual) & np.isfinite(normal_cos) & (normal_cos >= 0.80)
            ids = np.flatnonzero(valid)
            if len(ids) < min_matches:
                rejected["matches"] = int(len(matches))
                rejected["inliers"] = int(len(ids))
                rejected["reason"] = "normal_inconsistency"
                return rejected

            abs_r = np.abs(residual[ids])
            if trim_fraction < 1.0 and len(ids) > min_matches:
                order = np.argsort(abs_r)
                keep = max(min_matches, int(round(len(ids) * trim_fraction)))
                ids = ids[order[:keep]]
                abs_r = np.abs(residual[ids])
            final_ids = ids

            weights = np.ones(len(ids), dtype=np.float64)
            large = abs_r > huber
            weights[large] = huber / np.maximum(abs_r[large], 1e-12)
            sqrt_w = np.sqrt(weights)

            A = np.empty((len(ids), 6), dtype=np.float64)
            b = -residual[ids]
            for row, idx in enumerate(ids):
                p = transformed[idx]
                n = prev_normals[idx]
                A[row, :3] = n @ (-self._skew(p))
                A[row, 3:] = n
            Aw = A * sqrt_w[:, None]
            bw = b * sqrt_w
            H = Aw.T @ Aw
            g = Aw.T @ bw
            # Small damping keeps weakly observable directions close to the
            # IMU-predicted initialization instead of inventing motion.
            H += 1e-5 * np.eye(6, dtype=np.float64)
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(Aw, bw, rcond=None)[0]
            if not np.isfinite(delta).all():
                rejected["reason"] = "nonfinite_update"
                return rejected

            d_rot = delta[:3]
            d_trans = delta[3:]
            rot_norm = float(np.linalg.norm(d_rot))
            trans_norm = float(np.linalg.norm(d_trans))
            if rot_norm > 0.08:
                d_rot *= 0.08 / rot_norm
            if trans_norm > 0.50:
                d_trans *= 0.50 / trans_norm
            R_step = Rotation.from_rotvec(d_rot).as_matrix()
            R = R_step @ R
            t = R_step @ t + d_trans
            if np.linalg.norm(d_rot) < 2e-5 and np.linalg.norm(d_trans) < 2e-4:
                break

        transformed = (R @ curr_centers.T).T + t
        transformed_normals = (R @ curr_normals.T).T
        transformed_normals /= np.maximum(
            np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-9
        )
        normal_cos = np.abs(np.einsum("ij,ij->i", transformed_normals, prev_normals))
        residual = np.einsum("ij,ij->i", prev_normals, transformed - prev_centers)
        valid_final = np.isfinite(residual) & np.isfinite(normal_cos) & (normal_cos >= 0.80)
        valid_ids = np.flatnonzero(valid_final)
        if use_motion_seed:
            # Trim planar validation exactly as in the legacy solver, but never
            # alter the already full-rank rigid transform.
            if len(valid_ids) >= min_matches and trim_fraction < 1.0:
                order = np.argsort(np.abs(residual[valid_ids]))
                keep = max(min_matches, int(round(len(valid_ids) * trim_fraction)))
                valid_ids = valid_ids[order[:keep]]
            final_ids = valid_ids
        if len(final_ids) == 0:
            rejected["reason"] = "no_final_inliers"
            return rejected
        rmse = float(np.sqrt(np.mean(np.square(residual[final_ids]))))
        translation = float(np.linalg.norm(t))
        rotation = float(Rotation.from_matrix(R).magnitude())
        point_rmse = None
        if use_motion_seed:
            point_rmse = motion_seed.get("point_rmse_m")
        point_quality = bool(point_rmse is None or (np.isfinite(float(point_rmse)) and float(point_rmse) <= 0.45))
        translation_consistent = True
        if expected_translation_m is not None and np.isfinite(float(expected_translation_m)):
            expected_t = max(0.0, float(expected_translation_m))
            translation_consistent = abs(translation - expected_t) <= max(0.15, 0.50 * expected_t)
        rotation_consistent = True
        if expected_rotation_rad is not None and np.isfinite(float(expected_rotation_rad)):
            expected_r = max(0.0, float(expected_rotation_rad))
            rotation_consistent = abs(rotation - expected_r) <= max(0.04, 1.5 * expected_r)
        accepted = bool(
            len(final_ids) >= min_matches
            and np.isfinite(rmse)
            and rmse <= float(self.limits.lidar_se3_accept_rmse_m)
            and translation <= float(self.limits.lidar_se3_max_translation_m)
            and rotation <= float(self.limits.lidar_se3_max_rotation_rad)
            and point_quality
            and translation_consistent
            and rotation_consistent
        )
        quality = float(np.clip(0.10 / max(rmse, 0.03), 0.50, 1.50))
        # V25.3: the accepted motion-seed path is a dense 3-D rigid ICP with
        # hundreds of correspondences, while the legacy IMU/planar factors are
        # intentionally lightweight for V13 runtime.  Give the binary LiDAR
        # factor a covariance-like translation weight so optimization preserves
        # the measured metric displacement instead of averaging it several
        # centimetres toward the weaker motion model every frame.
        if use_motion_seed and accepted:
            point_sigma = 0.05 if point_rmse is None else float(np.clip(float(point_rmse), 0.02, 0.08))
            translation_weight = float(np.clip(1.0 / point_sigma, 16.0, 40.0))
            rotation_weight = max(12.0, float(self.limits.lidar_se3_rotation_weight) * min(quality, 1.25))
        else:
            translation_weight = float(self.limits.lidar_se3_translation_weight) * quality
            rotation_weight = float(self.limits.lidar_se3_rotation_weight) * quality
        return {
            "accepted": accepted,
            "delta_R": R,
            "delta_t": t,
            "matches": int(len(matches)),
            "inliers": int(len(final_ids)),
            "rmse_m": rmse,
            "translation_m": translation,
            "rotation_rad": rotation,
            "point_rmse_m": point_rmse,
            "source": ("v24_rigid_icp_validated_by_planes" if use_motion_seed else "planar_se3_fallback"),
            "reason": (
                "accepted"
                if accepted
                else (
                    "point_quality"
                    if not point_quality
                    else (
                        "translation_consistency"
                        if not translation_consistent
                        else ("rotation_consistency" if not rotation_consistent else "quality_gate")
                    )
                )
            ),
            "weight_translation": translation_weight,
            "weight_rotation": rotation_weight,
        }

    def _select_fast_keypoints(self, keypoints):
        if len(keypoints) <= self.limits.max_visual_features:
            return list(keypoints)
        return sorted(keypoints, key=lambda kp: float(getattr(kp, "response", 0.0)), reverse=True)[
            : self.limits.max_visual_features
        ]

    def _visual_observations(
        self,
        visual_tracks: List[Dict],
        deskewed_lidar: np.ndarray,
        estimated_pose: np.ndarray,
        sensor_boxes: List[Dict],
    ) -> List[Dict]:
        projected, depths, lidar_xyz, _ = self.lidar_projector.project(deskewed_lidar, (self.calibration.height, self.calibration.width, 3))
        tree = cKDTree(projected) if len(projected) else None
        R_w_e = self.euler_to_rot(estimated_pose[:3])
        t_w_e = estimated_pose[3:]
        active_ids = {int(t["track_id"]) for t in visual_tracks}
        self._visual_landmarks_world = {
            tid: lm for tid, lm in self._visual_landmarks_world.items() if tid in active_ids
        }
        observations = []
        for track in visual_tracks:
            if len(observations) >= self.limits.max_visual_factors_per_frame:
                break
            if int(track.get("length", 0)) < 2 or track.get("object_id") is not None:
                continue
            tid = int(track["track_id"])
            uv = np.asarray(track["point"], dtype=np.float64)
            if tid not in self._visual_landmarks_world:
                if tree is None:
                    continue
                dist, idx = tree.query(uv, k=1, distance_upper_bound=4.0)
                if not np.isfinite(dist) or idx >= len(lidar_xyz):
                    continue
                p_lidar = np.asarray(lidar_xyz[int(idx)], dtype=np.float64)
                p_ego = self.multimodal_builder.lidar_to_ego(p_lidar.reshape(1, 3))[0]
                # Skip extremely near/far or non-finite depth anchors.
                if not np.isfinite(p_ego).all() or np.linalg.norm(p_ego) < 1.0 or np.linalg.norm(p_ego) > 80.0:
                    continue
                self._visual_landmarks_world[tid] = R_w_e @ p_ego + t_w_e
                continue
            observations.append(
                {
                    "measurement_uv": uv,
                    "K": self.calibration.K,
                    "landmark_3d_w": self._visual_landmarks_world[tid],
                    "weight": 0.5,
                    "camera_translation_ego": self.camera_translation_ego,
                    "camera_rotation_ego": self.camera_rotation_ego,
                    "use_carla_camera_axes": True,
                }
            )
        return observations

    def _object_factor_observations(self, active_tracks: List[Dict], pose: np.ndarray) -> List[Dict]:
        R = self.euler_to_rot(pose[:3])
        t = pose[3:]
        # Dynamic-object variables are expensive. Only stable, currently seen
        # tracks enter the graph; MOT itself still retains all confirmed tracks.
        candidates = [
            tr for tr in active_tracks
            if int(tr.get("hits", 0)) >= 3
            and int(tr.get("missed", 0)) == 0
            and float(tr.get("score", 0.0)) >= 0.25
        ]
        candidates.sort(key=lambda tr: (-int(tr.get("hits", 0)), -float(tr.get("score", 0.0))))
        result = []
        for track in candidates[: self.limits.max_object_factors_per_frame]:
            p_w = np.asarray(track["position"], dtype=np.float64)
            p_local = R.T @ (p_w - t)
            result.append(
                {
                    "track_id": int(track["track_id"]),
                    "location": p_local,
                    "velocity": np.asarray(track.get("velocity", np.zeros(3)), dtype=np.float64),
                }
            )
        return result

    @staticmethod
    def _vec3(value) -> np.ndarray:
        if isinstance(value, dict):
            return np.array([
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            ], dtype=np.float64)
        return np.asarray(value, dtype=np.float64).reshape(3)

    @classmethod
    def _corner_array(cls, corners) -> np.ndarray:
        """Normalize CARLA bounding-box corner encodings to ``(N, 3)``.

        The dataset capture scripts are not perfectly uniform across all
        sequences.  Some label files store ``world_corners`` as a normal
        list-of-lists, while others serialize vectors as dictionaries (for
        example a list/dict of ``{x, y, z}`` objects or an ``{x: [...],
        y: [...], z: [...]}`` structure).  Evaluation must accept all of
        those encodings without changing the detector or estimator.
        """
        if corners is None:
            return np.empty((0, 3), dtype=np.float64)

        def point_from(value):
            if isinstance(value, dict) and any(k in value for k in ("x", "y", "z")):
                x, y, z = value.get("x"), value.get("y"), value.get("z")
                # A single vector encoded as {x: scalar, y: scalar, z: scalar}.
                if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
                    return np.array([float(x), float(y), float(z)], dtype=np.float64)
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return None
            if arr.size == 3:
                return arr.astype(np.float64, copy=False)
            return None

        if isinstance(corners, dict):
            # Column-style encoding: {"x": [...], "y": [...], "z": [...]}.
            if all(k in corners for k in ("x", "y", "z")):
                x, y, z = corners["x"], corners["y"], corners["z"]
                if not (np.isscalar(x) and np.isscalar(y) and np.isscalar(z)):
                    try:
                        xa = np.asarray(x, dtype=np.float64).reshape(-1)
                        ya = np.asarray(y, dtype=np.float64).reshape(-1)
                        za = np.asarray(z, dtype=np.float64).reshape(-1)
                        n = min(len(xa), len(ya), len(za))
                        if n > 0:
                            return np.column_stack([xa[:n], ya[:n], za[:n]])
                    except (TypeError, ValueError):
                        pass
                pt = point_from(corners)
                if pt is not None:
                    return pt.reshape(1, 3)

            # Named/numeric-key dictionary of corner vectors.  Preserve
            # insertion order, which is the JSON order used by Python.
            values = list(corners.values())
            pts = [point_from(v) for v in values]
            pts = [v for v in pts if v is not None]
            if pts:
                return np.vstack(pts)

        if isinstance(corners, (list, tuple)):
            pts = [point_from(v) for v in corners]
            pts = [v for v in pts if v is not None]
            if pts:
                return np.vstack(pts)

        try:
            arr = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError):
            return np.empty((0, 3), dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        if arr.size and arr.size % 3 == 0:
            return arr.reshape(-1, 3)
        return np.empty((0, 3), dtype=np.float64)

    def _gt_objects_current_ego(self, gt_objects: List[Dict], gt_pose: Dict) -> List[Dict]:
        """Evaluation-only CARLA world actors -> current GT ego frame.

        Tracking estimates are born from LiDAR measurements in the current ego
        frame and are then ego-motion compensated into the estimator world.
        Evaluating both hypotheses and GT back in their respective *current ego*
        frames avoids any arbitrary frame-0/world handedness offset while still
        using GT only after estimation is complete.
        """
        ego_t = self._vec3(gt_pose.get("location", {}))
        R_w_e = self._carla_world_rotation(gt_pose.get("rotation", {}))
        R_e_w = R_w_e.T
        output = []
        for obj in gt_objects:
            item = dict(obj)
            p_w = self._vec3(obj.get("location", [0.0, 0.0, 0.0]))
            item["location"] = (R_e_w @ (p_w - ego_t)).tolist()
            vel = obj.get("velocity")
            if vel is not None:
                item["velocity"] = (R_e_w @ self._vec3(vel)).tolist()
            output.append(item)
        return output

    def _tracks_current_ego(self, tracks: List[Dict], estimated_pose: np.ndarray) -> List[Dict]:
        """Estimator-world tracks -> current estimated ego frame for evaluation.

        Uses the exact CARLA-style rotation convention employed by
        CarlaMultiObjectTracker._transform_to_world(), making this transform its
        inverse.  No GT quantity enters this conversion.
        """
        pose_dict = self.pose_dict_from_vector(estimated_pose)
        ego_t = np.asarray(estimated_pose[3:], dtype=np.float64)
        R_w_e = self._carla_world_rotation(pose_dict.get("rotation", {}))
        R_e_w = R_w_e.T
        output = []
        for tr in tracks:
            # Preserve all non-spatial observation metadata (especially the
            # RGB bounding box).  Camera+LiDAR MOT evaluation uses that box
            # as its primary association cue.  Earlier builds rebuilt a tiny
            # track dictionary here and silently discarded ``bbox``; the
            # evaluation path then filtered every otherwise-valid track, which
            # produced the contradictory result "Tracks 3" but
            # "Outputs 0".
            item = dict(tr)
            item["track_id"] = int(tr["track_id"])
            item["class"] = tr.get("class", "object")
            item["position"] = (
                R_e_w @ (np.asarray(tr["position"], dtype=np.float64) - ego_t)
            ).tolist()
            item["velocity"] = (
                R_e_w
                @ np.asarray(tr.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            ).tolist()
            if tr.get("bbox") is not None:
                item["bbox"] = [float(v) for v in tr["bbox"]]
            output.append(item)
        return output

    def _current_object_range(self) -> float:
        value = float(getattr(self, "object_sensor_range", 0.0) or 0.0)
        if value > 0.0:
            return value
        limits = getattr(self, "limits", None)
        configured = float(getattr(limits, "object_max_range", 0.0) or 0.0)
        if configured > 0.0:
            return configured
        calibration = getattr(self, "calibration", None)
        sensor = float(getattr(calibration, "lidar_range", 0.0) or 0.0)
        return sensor if sensor > 0.0 else 55.0

    def _lidar_extrinsic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ego<-LiDAR translation and rotation from calibration."""
        transform = self.calibration.lidar_transform
        loc = transform.get("location", transform)
        t_e_l = np.array([
            float(loc.get("x", 0.0)),
            float(loc.get("y", 0.0)),
            float(loc.get("z", 0.0)),
        ], dtype=np.float64)
        R_e_l = self._carla_world_rotation(transform.get("rotation", {}))
        return t_e_l, R_e_l

    def _ego_to_lidar_point(self, point_ego: np.ndarray) -> np.ndarray:
        t_e_l, R_e_l = self._lidar_extrinsic()
        return R_e_l.T @ (np.asarray(point_ego, dtype=np.float64) - t_e_l)

    def _ego_to_lidar_vector(self, vector_ego: np.ndarray) -> np.ndarray:
        _, R_e_l = self._lidar_extrinsic()
        return R_e_l.T @ np.asarray(vector_ego, dtype=np.float64)

    def _gt_objects_current_lidar(self, gt_objects: List[Dict], gt_pose: Dict) -> List[Dict]:
        """Evaluation-only CARLA world actor states -> raw LiDAR sensor frame.

        The GT comparison point is the geometric centre of the CARLA 3-D
        bounding box when world-space corners are available.  CARLA actor
        locations are commonly near the actor origin/road contact point, while
        the sensor detector estimates the visible 3-D object body.  Using the
        box centre makes the evaluation geometry comparable without feeding any
        GT quantity back into estimation or tracking.
        """
        ego_t = self._vec3(gt_pose.get("location", {}))
        R_w_e = self._carla_world_rotation(gt_pose.get("rotation", {}))
        R_e_w = R_w_e.T
        output = []
        for obj in gt_objects:
            type_name = str(obj.get("type", ""))
            if type_name and not type_name.startswith("vehicle."):
                continue
            item = dict(obj)

            p_world = self._vec3(obj.get("location", [0.0, 0.0, 0.0]))
            bbox = obj.get("bounding_box", {}) or {}
            corners = bbox.get("world_corners")
            corners_arr = self._corner_array(corners)
            if len(corners_arr) >= 4:
                finite = np.isfinite(corners_arr).all(axis=1)
                if np.any(finite):
                    p_world = np.mean(corners_arr[finite], axis=0)

            p_ego = R_e_w @ (p_world - ego_t)
            item["location"] = self._ego_to_lidar_point(p_ego).tolist()
            vel = obj.get("velocity")
            if vel is not None:
                v_ego = R_e_w @ self._vec3(vel)
                item["velocity"] = self._ego_to_lidar_vector(v_ego).tolist()
            output.append(item)
        return output

    def _project_gt_camera_boxes(
        self,
        gt_objects: List[Dict],
        gt_pose: Dict,
        image_shape: Tuple[int, ...],
    ) -> Dict[int, List[float]]:
        """Project GT 3-D box corners into the RGB image, evaluation only.

        This intentionally reuses the original CARLA box-projection path that
        was already validated on the recorded dataset.  Actor-centre projection
        is not a reliable camera-visibility test because the actor origin can be
        at road level or outside the crop while part of its 3-D box is visible.
        """
        boxes: Dict[int, List[float]] = {}
        for obj in gt_objects:
            type_name = str(obj.get("type", ""))
            if type_name and not type_name.startswith("vehicle."):
                continue
            actor_id = int(obj.get("actor_id", obj.get("id", -1)))
            bbox = obj.get("bounding_box", {}) or {}
            corners = self._corner_array(bbox.get("world_corners"))
            if len(corners) < 4:
                continue
            try:
                _, _, projected_box = self.gt_box_projector.project_world_corners(
                    corners, gt_pose, image_shape
                )
            except Exception:
                projected_box = None
            if projected_box is not None:
                boxes[actor_id] = [float(v) for v in projected_box]
        return boxes

    def _tracks_current_lidar(self, tracks: List[Dict], estimated_pose: np.ndarray) -> List[Dict]:
        """Estimator-world tracks -> current raw LiDAR sensor frame for scoring."""
        ego_tracks = self._tracks_current_ego(tracks, estimated_pose)
        output = []
        for tr in ego_tracks:
            item = dict(tr)
            item["position"] = self._ego_to_lidar_point(
                np.asarray(tr["position"], dtype=np.float64)
            ).tolist()
            item["velocity"] = self._ego_to_lidar_vector(
                np.asarray(tr.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            ).tolist()
            output.append(item)
        return output

    def _gt_lidar_support_by_bbox(
        self,
        projected_gt_boxes: Dict[int, List[float]],
        lidar_points: Optional[np.ndarray],
    ) -> Dict[int, int]:
        """Count actual LiDAR returns inside each evaluation-only GT image box.

        Camera+LiDAR detection is observable only if the RGB box is visible and
        the LiDAR actually contributes points inside that box.  This is a more
        faithful observability test than thresholding the GT actor-centre range,
        especially for long vehicles and for datasets whose LiDAR range is not
        the old hard-coded 55 m value.
        """
        if lidar_points is None or not projected_gt_boxes:
            return {int(k): 0 for k in projected_gt_boxes}
        pts = np.asarray(lidar_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return {int(k): 0 for k in projected_gt_boxes}

        xyz = pts[:, :3]
        finite_xyz = np.isfinite(xyz).all(axis=1)
        camera_xyz = self.calibration.lidar_to_opencv(xyz)
        depth = camera_xyz[:, 2]
        uv, valid_projection = self.calibration.project_lidar(xyz)
        valid = finite_xyz & valid_projection & np.isfinite(uv).all(axis=1)
        valid &= (depth > 0.1) & (depth <= float(self._current_object_range()))

        support: Dict[int, int] = {}
        for actor_id, bbox in projected_gt_boxes.items():
            if bbox is None or len(bbox) != 4:
                support[int(actor_id)] = 0
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            inside = (
                valid
                & (uv[:, 0] >= x1)
                & (uv[:, 0] <= x2)
                & (uv[:, 1] >= y1)
                & (uv[:, 1] <= y2)
            )
            support[int(actor_id)] = int(np.count_nonzero(inside))
        return support

    def _filter_gt_observable_current_lidar(
        self,
        gt_objects_lidar: List[Dict],
        projected_gt_boxes: Optional[Dict[int, List[float]]] = None,
        lidar_points: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Evaluation-only observability gate in the detector frame.

        In camera+LiDAR mode the eligibility rule mirrors the actual sensor
        detector: a GT 3-D box must intersect the RGB image *and* receive real
        LiDAR returns inside that projected box.  CARLA labels are used only to
        define the evaluation box; the sensor returns decide observability.
        
        For LiDAR-only mode, the geometric centre is range-gated using the
        effective LiDAR range from calibration.
        """
        projected_gt_boxes = projected_gt_boxes or {}
        lidar_support = self._gt_lidar_support_by_bbox(projected_gt_boxes, lidar_points)
        min_support = 1
        if self.camera_lidar_detector is not None:
            min_support = int(max(1, self.camera_lidar_detector.config.min_lidar_points))

        output = []
        for obj in gt_objects_lidar:
            p = np.asarray(obj.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
            if not np.isfinite(p).all():
                continue
            item = dict(obj)
            actor_id = int(obj.get("actor_id", obj.get("id", -1)))

            if self.detector_mode == "camera_lidar":
                bbox = projected_gt_boxes.get(actor_id)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if self.camera_lidar_detector is not None:
                    cfg = self.camera_lidar_detector.config
                    min_w = float(getattr(cfg, "min_eval_bbox_width_px", 0.0) or 0.0)
                    min_h = float(getattr(cfg, "min_eval_bbox_height_px", 0.0) or 0.0)
                    min_area = float(getattr(cfg, "min_eval_bbox_area_px", 0.0) or 0.0)
                    if bw < min_w or bh < min_h or (bw * bh) < min_area:
                        continue
                support_count = int(lidar_support.get(actor_id, 0))
                if support_count < min_support:
                    continue
                item["bbox"] = list(bbox)
                item["bbox_area_px"] = float(bw * bh)
                item["lidar_support"] = support_count
                item["evaluation_range_m"] = float(np.linalg.norm(p[:2]))
                output.append(item)
                continue

            r_xy = float(np.linalg.norm(p[:2]))
            if r_xy < 1.0 or r_xy > self._current_object_range():
                continue
            item["evaluation_range_m"] = r_xy
            output.append(item)
        return output

    def _filter_gt_observable_current_ego(self, gt_objects_ego: List[Dict]) -> List[Dict]:
        """Evaluation-only gate expressed directly in the LiDAR/ego frame."""
        output = []
        for obj in gt_objects_ego:
            p = np.asarray(obj.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
            if not np.isfinite(p).all():
                continue
            r_xy = float(np.linalg.norm(p[:2]))
            if r_xy < 1.5 or r_xy > self._current_object_range():
                continue
            # Sensor proposals represent above-road object surfaces; allow the
            # actor origin to sit below the proposal centre without dropping it.
            if p[2] < self.limits.object_min_height - 3.0 or p[2] > self.limits.object_max_height + 3.0:
                continue
            output.append(obj)
        return output

    def _gt_pose_local(self, gt_pose: Dict) -> np.ndarray:
        """Ground-truth ego pose in the estimator's local frame (evaluation only)."""
        pose = self.pose_vector_from_dict(gt_pose)
        R_localization = self.euler_to_rot(pose[:3])
        R_tracking = self._carla_world_rotation(gt_pose.get("rotation", {}))
        t = pose[3:]
        if self._gt_anchor_t is None:
            self._gt_anchor_t = t.copy()
            # Preserve the already-validated localization evaluation convention.
            self._gt_anchor_R_localization = R_localization.copy()
            # MOT uses the actual CARLA left-handed actor/world convention.
            self._gt_anchor_R_tracking = R_tracking.copy()
        R_local = self._gt_anchor_R_localization.T @ R_localization
        t_local = self._gt_anchor_R_localization.T @ (t - self._gt_anchor_t)
        return np.hstack([self.rot_to_euler(R_local), t_local])

    def _gt_tracking_ego_position(self, gt_pose: Dict) -> np.ndarray:
        if self._gt_anchor_R_tracking is None or self._gt_anchor_t is None:
            return np.zeros(3, dtype=np.float64)
        p = gt_pose.get("location", {})
        t = np.array([
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("z", 0.0)),
        ], dtype=np.float64)
        return self._gt_anchor_R_tracking.T @ (t - self._gt_anchor_t)

    def _gt_objects_local(self, gt_objects: List[Dict]) -> List[Dict]:
        """CARLA world actor states -> frame-0 ego coordinates, evaluation only."""
        if self._gt_anchor_R_tracking is None or self._gt_anchor_t is None:
            return []
        output = []
        R0_T = self._gt_anchor_R_tracking.T
        for obj in gt_objects:
            item = dict(obj)
            loc = obj.get("location", [0, 0, 0])
            p = (
                np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
                if isinstance(loc, dict)
                else np.asarray(loc, dtype=np.float64)
            )
            item["location"] = (R0_T @ (p - self._gt_anchor_t)).tolist()
            vel = obj.get("velocity")
            if vel is not None:
                v = (
                    np.array([vel["x"], vel["y"], vel["z"]], dtype=np.float64)
                    if isinstance(vel, dict)
                    else np.asarray(vel, dtype=np.float64)
                )
                item["velocity"] = (R0_T @ v).tolist()
            output.append(item)
        return output

    def _filter_gt_observable(self, gt_objects_local: List[Dict], ego_tracking_position: np.ndarray) -> List[Dict]:
        """Evaluation-only LiDAR observability gate, identical in range to detector."""
        ego_p = np.asarray(ego_tracking_position, dtype=np.float64)
        output = []
        for obj in gt_objects_local:
            p = np.asarray(obj.get("location", [0, 0, 0]), dtype=np.float64)
            rel = p - ego_p
            if not np.isfinite(rel).all():
                continue
            r_xy = float(np.linalg.norm(rel[:2]))
            # Match the detector's near and far LiDAR limits.
            if r_xy < 1.5 or r_xy > self._current_object_range():
                continue
            if rel[2] < self.limits.object_min_height - 2.0 or rel[2] > self.limits.object_max_height + 3.0:
                continue
            output.append(obj)
        return output

    def _array_digest(self, array: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(array)
        view = memoryview(arr).cast("B")
        return hashlib.blake2b(
            view,
            digest_size=max(4, int(self.limits.stale_hash_digest_bytes)),
        ).digest()

    def _lidar_range_signature(self, raw_lidar: np.ndarray) -> np.ndarray:
        """Build an estimator-independent angular range image from raw LiDAR.

        The signature is computed before deskewing so estimator drift cannot
        prevent stationary detection. Moving traffic changes only a minority of
        angular cells; ego motion changes the static background broadly.
        """
        pts = np.asarray(raw_lidar, dtype=np.float64)
        az_bins = max(36, int(self.limits.stationary_lidar_azimuth_bins))
        el_bins = max(8, int(self.limits.stationary_lidar_elevation_bins))
        signature = np.full(az_bins * el_bins, np.nan, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return signature
        xyz = pts[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if len(xyz) == 0:
            return signature
        xy = np.linalg.norm(xyz[:, :2], axis=1)
        r = np.linalg.norm(xyz, axis=1)
        valid = (r >= 2.0) & (r <= min(float(self._current_object_range()), 100.0)) & (xy > 1e-3)
        xyz = xyz[valid]
        r = r[valid]
        xy = xy[valid]
        if len(r) == 0:
            return signature
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        el = np.arctan2(xyz[:, 2], xy)
        # CARLA LiDAR vertical FoV varies by capture configuration. Clamp to a
        # generous road-driving envelope and map every valid point deterministically.
        el = np.clip(el, np.deg2rad(-35.0), np.deg2rad(25.0))
        ai = np.floor((az + np.pi) / (2.0 * np.pi) * az_bins).astype(np.int32)
        ei = np.floor((el - np.deg2rad(-35.0)) / np.deg2rad(60.0) * el_bins).astype(np.int32)
        ai = np.clip(ai, 0, az_bins - 1)
        ei = np.clip(ei, 0, el_bins - 1)
        idx = ei * az_bins + ai
        temp = np.full_like(signature, np.inf)
        np.minimum.at(temp, idx, r)
        temp[~np.isfinite(temp)] = np.nan
        return temp

    def _lidar_stationary_metric(self, raw_lidar: np.ndarray) -> tuple[Optional[float], float]:
        """Return median angular range change and overlap fraction.

        This is robust to a limited number of moving actors and, unlike the old
        planar-feature NN metric, does not depend on which planar samples happen
        to be selected on each frame.
        """
        current = self._lidar_range_signature(raw_lidar)
        previous = self._previous_lidar_range_signature
        self._previous_lidar_range_signature = current.copy()
        if previous is None or previous.shape != current.shape:
            return None, 0.0
        valid = np.isfinite(previous) & np.isfinite(current)
        denom = max(1, int(np.count_nonzero(np.isfinite(previous) | np.isfinite(current))))
        overlap = float(np.count_nonzero(valid) / denom)
        if np.count_nonzero(valid) < 64:
            return None, overlap
        delta = np.abs(current[valid] - previous[valid])
        # The median rejects moving cars/occlusions. A light upper trim further
        # protects against large-depth swaps at object boundaries.
        delta = delta[np.isfinite(delta)]
        if len(delta) == 0:
            return None, overlap
        if len(delta) >= 20:
            cutoff = np.percentile(delta, 90.0)
            delta = delta[delta <= cutoff]
        return float(np.median(delta)), overlap


    def _visual_background_motion_metric(self, image: np.ndarray) -> tuple[Optional[float], float, int]:
        """Return robust frame-to-frame background flow in pixels.

        Constant-velocity ego translation can have near-zero gyro and linear
        acceleration.  Median optical-flow *magnitude* over many corners is a
        cheap sensor-only veto for that case.  A moving car occupies only a
        minority of the image, so the median remains near zero for a parked ego.
        """
        if cv2 is None:
            return None, 0.0, 0
        arr = np.asarray(image)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            return None, 0.0, 0
        gray = np.ascontiguousarray(gray.astype(np.uint8, copy=False))
        previous = self._previous_motion_gray
        self._previous_motion_gray = gray.copy()
        if previous is None or previous.shape != gray.shape:
            return None, 0.0, 0

        points0 = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=320,
            qualityLevel=0.01,
            minDistance=8.0,
            blockSize=5,
            useHarrisDetector=False,
        )
        if points0 is None or len(points0) < 8:
            return None, 0.0, 0
        points1, status, _err = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        if points1 is None or status is None:
            return None, 0.0, 0
        mask = status.reshape(-1).astype(bool)
        p0 = points0.reshape(-1, 2)[mask]
        p1 = points1.reshape(-1, 2)[mask]
        finite = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
        p0 = p0[finite]
        p1 = p1[finite]
        if len(p0) < 8:
            return None, 0.0, int(len(p0))

        flow = np.linalg.norm(p1 - p0, axis=1)
        flow = flow[np.isfinite(flow)]
        if len(flow) == 0:
            return None, 0.0, 0
        # Trim only the largest 20% so independently moving actors do not
        # dominate a stationary camera while broad ego parallax is retained.
        if len(flow) >= 20:
            cutoff = float(np.percentile(flow, 80.0))
            trimmed = flow[flow <= cutoff]
            if len(trimmed) >= 8:
                flow_for_median = trimmed
            else:
                flow_for_median = flow
        else:
            flow_for_median = flow
        median_flow = float(np.median(flow_for_median))
        moving_fraction = float(
            np.mean(flow > float(self.limits.stationary_visual_track_motion_px))
        )
        return median_flow, moving_fraction, int(len(flow))

    def _motion_lidar_cloud(self, raw_lidar: np.ndarray) -> np.ndarray:
        """Return a bounded non-ground-biased cloud for motion-only ICP."""
        pts = np.asarray(raw_lidar, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        xyz = pts[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if len(xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        xyz = xyz[(radius >= 3.0) & (radius <= min(85.0, float(self._current_object_range())))]
        if len(xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)

        # Reject most road returns.  Continuous ground is translation-ambiguous
        # for point-to-point ICP; trees, poles, facades and curbs provide useful
        # static structure.  The percentile form avoids assuming a fixed sensor Z.
        z_cut = float(np.percentile(xyz[:, 2], 35.0)) + 0.25
        nonground = xyz[xyz[:, 2] > z_cut]
        if len(nonground) >= 300:
            xyz = nonground

        # Deterministic XY voxel representatives bound CPU cost.
        voxel = 0.35
        keys = np.floor(xyz[:, :2] / voxel).astype(np.int32)
        _, first = np.unique(keys, axis=0, return_index=True)
        xyz = xyz[np.sort(first)]
        max_points = max(300, int(self.limits.stationary_lidar_icp_max_points))
        if len(xyz) > max_points:
            ids = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
            xyz = xyz[ids]
        return np.asarray(xyz, dtype=np.float64)

    @staticmethod
    def _rigid_transform_svd(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        src = np.asarray(source, dtype=np.float64)
        dst = np.asarray(target, dtype=np.float64)
        src_c = np.mean(src, axis=0)
        dst_c = np.mean(dst, axis=0)
        H = (src - src_c).T @ (dst - dst_c)
        U, _S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0.0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = dst_c - R @ src_c
        return R, t

    def _lidar_translation_motion_metric(self, raw_lidar: np.ndarray) -> tuple[Optional[float], Optional[float], int]:
        """Estimate current->previous rigid LiDAR motion without estimator/GT input.

        V24 used only the translation/rotation magnitudes from this robust ICP.
        V25.1 also retains the full transform so the moving pose factor uses the
        same well-behaved registration instead of reconstructing tangent motion
        from a road-dominated point-to-plane system.
        """
        current = self._motion_lidar_cloud(raw_lidar)
        previous = self._previous_motion_lidar_xyz
        self._previous_motion_lidar_xyz = current.copy()
        self._last_lidar_motion_transform = None
        if previous is None or len(previous) < 80 or len(current) < 80:
            return None, None, 0

        src = current.copy()
        R_total = np.eye(3, dtype=np.float64)
        t_total = np.zeros(3, dtype=np.float64)
        tree = cKDTree(previous)
        used = 0
        max_corr = float(self.limits.stationary_lidar_icp_max_correspondence_m)
        trim_fraction = float(np.clip(self.limits.stationary_lidar_icp_trim_fraction, 0.30, 1.0))
        iterations = max(1, int(self.limits.stationary_lidar_icp_iterations))

        for _ in range(iterations):
            dist, idx = tree.query(src, k=1, workers=1)
            valid = np.isfinite(dist) & (dist <= max_corr)
            if np.count_nonzero(valid) < 60:
                break
            ids = np.flatnonzero(valid)
            if len(ids) >= 80 and trim_fraction < 1.0:
                order = np.argsort(dist[ids])
                keep = max(60, int(round(len(ids) * trim_fraction)))
                ids = ids[order[:keep]]
            src_sel = src[ids]
            dst_sel = previous[idx[ids]]
            R_step, t_step = self._rigid_transform_svd(src_sel, dst_sel)
            src = (R_step @ src.T).T + t_step
            R_total = R_step @ R_total
            t_total = R_step @ t_total + t_step
            used = int(len(ids))
            if np.linalg.norm(t_step) < 1e-4 and Rotation.from_matrix(R_step).magnitude() < 1e-4:
                break

        if used < 60:
            return None, None, used

        # Re-score after the final transform.  This is a true 3-D residual, so
        # translation tangent to a road plane cannot hide behind near-zero
        # point-to-plane error.
        final_dist, _ = tree.query(src, k=1, workers=1)
        final_valid = np.isfinite(final_dist) & (final_dist <= max_corr)
        if np.count_nonzero(final_valid) >= 60:
            d = np.sort(final_dist[final_valid])
            keep = max(60, int(round(len(d) * trim_fraction)))
            point_rmse = float(np.sqrt(np.mean(np.square(d[:keep]))))
        else:
            point_rmse = None

        translation = float(np.linalg.norm(t_total[:2]))
        rotation = float(Rotation.from_matrix(R_total).magnitude())
        self._last_lidar_motion_transform = {
            "accepted": True,
            "delta_R": R_total.copy(),
            "delta_t": t_total.copy(),
            "translation_m": float(np.linalg.norm(t_total)),
            "rotation_rad": rotation,
            "point_rmse_m": point_rmse,
            "correspondences": used,
        }
        return translation, rotation, used

    def _sensor_integrity_state(
        self,
        image_hash: bytes,
        lidar_hash: bytes,
        imu_motion: Dict,
        lidar_stationary_metric: Optional[float],
        visual_motion_px: Optional[float] = None,
        visual_motion_fraction: float = 0.0,
        visual_motion_tracks: int = 0,
        lidar_translation_m: Optional[float] = None,
        lidar_rotation_rad: Optional[float] = None,
        lidar_motion_correspondences: int = 0,
    ) -> Dict:
        image_duplicate = self._last_image_hash is not None and image_hash == self._last_image_hash
        lidar_duplicate = self._last_lidar_hash is not None and lidar_hash == self._last_lidar_hash
        gyro = np.asarray(imu_motion.get("gyro", np.zeros(3)), dtype=np.float64)
        accel = np.asarray(imu_motion.get("acceleration", np.zeros(3)), dtype=np.float64)
        gyro_norm = float(np.linalg.norm(gyro))
        accel_norm = float(np.linalg.norm(accel))
        accel_gravity_error = abs(accel_norm - float(np.linalg.norm(self.GRAVITY)))
        accel_zero_error = accel_norm
        gyro_quiet = gyro_norm <= float(self.limits.stationary_gyro_threshold_rad_s)
        accel_quiet = (
            accel_gravity_error <= float(self.limits.stationary_accel_tolerance_m_s2)
            or accel_zero_error <= float(self.limits.stationary_accel_zero_tolerance_m_s2)
        )
        imu_quiet = bool(gyro_quiet and accel_quiet)
        lidar_overlap = float(getattr(self, "_current_lidar_stationary_overlap", 0.0))
        lidar_stationary = (
            lidar_stationary_metric is not None
            and lidar_stationary_metric <= float(self.limits.stationary_lidar_range_delta_m)
            and lidar_overlap >= float(self.limits.stationary_lidar_min_overlap)
        )
        strong_lidar_motion = (
            lidar_stationary_metric is not None
            and lidar_stationary_metric > 2.0 * float(self.limits.stationary_lidar_range_delta_m)
        )

        # A duplicate RGB frame is only called stale when another sensor says
        # the world/sensor state changed. Exact repeated frames while genuinely
        # stationary remain valid observations, but do not need to create motion.
        stale_rgb = bool(image_duplicate and not lidar_duplicate and (strong_lidar_motion or not imu_quiet))
        stale_lidar = bool(lidar_duplicate and not image_duplicate and (not imu_quiet or not lidar_stationary))

        visual_motion_veto = bool(
            visual_motion_px is not None
            and int(visual_motion_tracks) >= int(self.limits.stationary_visual_min_tracks)
            and (
                float(visual_motion_px) > float(self.limits.stationary_visual_flow_threshold_px)
                or float(visual_motion_fraction) > float(self.limits.stationary_visual_motion_fraction_threshold)
            )
        )
        lidar_translation_veto = bool(
            (
                lidar_translation_m is not None
                and float(lidar_translation_m) > float(self.limits.stationary_lidar_translation_threshold_m)
            )
            or (
                lidar_rotation_rad is not None
                and float(lidar_rotation_rad) > float(self.limits.stationary_lidar_rotation_threshold_rad)
            )
        )
        translation_motion_veto = bool(visual_motion_veto or lidar_translation_veto)

        stationary_candidate = bool(
            imu_quiet
            and lidar_stationary
            and not stale_lidar
            and not translation_motion_veto
        )
        if stationary_candidate:
            self._stationary_streak += 1
            self._stationary_release_streak = 0
        else:
            self._stationary_streak = 0
            if self._stationary_active:
                self._stationary_release_streak += 1
        if not self._stationary_active:
            self._stationary_active = self._stationary_streak >= max(1, int(self.limits.stationary_confirm_frames))
        elif self._stationary_release_streak >= max(1, int(self.limits.stationary_release_frames)):
            self._stationary_active = False
            self._stationary_release_streak = 0
        self._last_stationary_lidar_metric = lidar_stationary_metric

        return {
            "rgb_duplicate": bool(image_duplicate),
            "lidar_duplicate": bool(lidar_duplicate),
            "stale_rgb": bool(stale_rgb),
            "stale_lidar": bool(stale_lidar),
            "imu_quiet": bool(imu_quiet),
            "gyro_norm_rad_s": gyro_norm,
            "accel_norm_m_s2": accel_norm,
            "accel_gravity_error_m_s2": accel_gravity_error,
            "accel_zero_error_m_s2": accel_zero_error,
            "gyro_quiet": bool(gyro_quiet),
            "accel_quiet": bool(accel_quiet),
            "lidar_stationary_metric_m": lidar_stationary_metric,
            "lidar_stationary_overlap": lidar_overlap,
            "lidar_stationary": bool(lidar_stationary),
            "visual_motion_px": visual_motion_px,
            "visual_motion_fraction": float(visual_motion_fraction),
            "visual_motion_tracks": int(visual_motion_tracks),
            "visual_motion_veto": bool(visual_motion_veto),
            "lidar_translation_m": lidar_translation_m,
            "lidar_rotation_rad": lidar_rotation_rad,
            "lidar_motion_correspondences": int(lidar_motion_correspondences),
            "lidar_translation_veto": bool(lidar_translation_veto),
            "translation_motion_veto": bool(translation_motion_veto),
            "stationary_candidate": stationary_candidate,
            "stationary_streak": int(self._stationary_streak),
            "stationary_active": bool(self._stationary_active),
        }

    def _tracker_lidar_fallback_objects(
        self,
        deskewed_lidar: np.ndarray,
        image_shape,
        camera_object_count: int,
    ) -> List[Dict]:
        """Current-frame LiDAR support for already-established camera tracks.

        This fallback is deliberately asymmetric: it may update an existing
        identity when YOLO misses it, but every returned observation carries
        ``allow_new_track=False`` so LiDAR clutter can never create a new ID in
        camera+LiDAR mode. CARLA labels/actor IDs are not used.
        """
        if self.detector_mode != "camera_lidar":
            return []
        established = self.object_tracker.established_track_count()
        if established <= int(camera_object_count):
            return []
        proposals = self.sensor_object_detector.detect(deskewed_lidar)
        if not proposals:
            return []
        proposals = self.sensor_object_detector.attach_camera_boxes(
            proposals, self.calibration, image_shape
        )
        out: List[Dict] = []
        cfg = self.camera_lidar_detector.config if self.camera_lidar_detector is not None else None
        for proposal in proposals:
            bbox = proposal.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            support = int(proposal.get("num_points", 0) or 0)
            if support < int(self.limits.tracker_fallback_min_support):
                continue
            score = float(proposal.get("score", proposal.get("vehicle_likeness", 0.0)))
            if score < float(self.limits.object_min_vehicle_score):
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if cfg is not None:
                if bw < float(cfg.min_eval_bbox_width_px) or bh < float(cfg.min_eval_bbox_height_px):
                    continue
                if bw * bh < float(cfg.min_eval_bbox_area_px):
                    continue
            item = dict(proposal)
            item["type"] = "vehicle"
            item["class"] = "vehicle"
            item["score"] = score
            item["camera_confidence"] = 0.0
            item["camera_evidence"] = False
            item["camera_support"] = support
            item["lidar_fallback"] = True
            item["allow_new_track"] = False
            out.append(item)
        out.sort(
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -int(item.get("camera_support", 0)),
            )
        )
        return out[: max(1, int(self.limits.tracker_lidar_fallback_max_proposals))]

    def process(
        self,
        frame_id: int,
        timestamp: float,
        image: np.ndarray,
        lidar: np.ndarray,
        imu_data: Dict,
        ground_truth_ego_pose: Optional[Dict] = None,
        ground_truth_objects: Optional[List[Dict]] = None,
        include_lidar_points: bool = False,
    ) -> Dict:
        stage = {}
        t0 = time.perf_counter()
        timestamp = float(timestamp)

        imu_motion = self.imu_preprocessor.process(imu_data)
        imu_preintegrated = None
        if self._last_imu_motion is not None:
            imu_preintegrated = self.imu_preintegrator.integrate(self._last_imu_motion, imu_motion)
        stage["imu_ms"] = (time.perf_counter() - t0) * 1000.0

        image_hash = self._array_digest(image)
        lidar_hash = self._array_digest(np.asarray(lidar))

        t = time.perf_counter()
        pose_init, velocity_init = self._predict_state(imu_preintegrated, timestamp)
        deskewed_lidar = np.asarray(lidar)
        if self._last_pose is not None:
            try:
                deskewed_lidar = self.lidar_deskewer.deskew(
                    np.asarray(lidar),
                    self._last_pose[3:],
                    pose_init[3:],
                    self.euler_to_rot(self._last_pose[:3]),
                    self.euler_to_rot(pose_init[:3]),
                )
            except Exception:
                deskewed_lidar = np.asarray(lidar)
        stage["deskew_ms"] = (time.perf_counter() - t) * 1000.0

        # Start camera+LiDAR object inference before CPU geometry so CUDA/CPU
        # work overlaps.  The future is consumed before tracking, preserving
        # exactly the same per-frame dependency order and measurements.
        detector_future = None
        detector_submit_t = time.perf_counter()
        if self.detector_mode == "camera_lidar" and self._detector_executor is not None:
            detector_future = self._detector_executor.submit(
                self.camera_lidar_detector.detect, image, deskewed_lidar
            )

        t = time.perf_counter()
        planar_features = self.lidar_features.extract(deskewed_lidar)
        planar_ego = self._bounded_planar_ego(planar_features)
        stage["lidar_features_ms"] = (time.perf_counter() - t) * 1000.0

        lidar_stationary_metric, lidar_stationary_overlap = self._lidar_stationary_metric(np.asarray(lidar))
        self._current_lidar_stationary_overlap = float(lidar_stationary_overlap)
        visual_motion_px, visual_motion_fraction, visual_motion_tracks = (
            self._visual_background_motion_metric(image)
        )
        lidar_translation_m, lidar_rotation_rad, lidar_motion_correspondences = (
            self._lidar_translation_motion_metric(np.asarray(lidar))
        )
        integrity = self._sensor_integrity_state(
            image_hash,
            lidar_hash,
            imu_motion,
            lidar_stationary_metric,
            visual_motion_px=visual_motion_px,
            visual_motion_fraction=visual_motion_fraction,
            visual_motion_tracks=visual_motion_tracks,
            lidar_translation_m=lidar_translation_m,
            lidar_rotation_rad=lidar_rotation_rad,
            lidar_motion_correspondences=lidar_motion_correspondences,
        )
        lidar_se3 = self._lidar_se3_relative_measurement(
            planar_ego,
            pose_init,
            expected_translation_m=lidar_translation_m,
            expected_rotation_rad=lidar_rotation_rad,
        )
        self._last_lidar_se3 = lidar_se3
        if (
            lidar_se3.get("accepted", False)
            and self._last_pose is not None
            and not integrity["stationary_active"]
            and not integrity["stale_lidar"]
        ):
            # Use LiDAR odometry only as the nonlinear solver initialization;
            # the same transform is also added as a binary graph factor below.
            # The factor graph remains the final fused estimator.
            prev_pose = np.asarray(self._last_pose, dtype=np.float64)
            R_prev = self.euler_to_rot(prev_pose[:3])
            R_seed = R_prev @ np.asarray(lidar_se3["delta_R"], dtype=np.float64)
            p_seed = prev_pose[3:] + R_prev @ np.asarray(lidar_se3["delta_t"], dtype=np.float64)
            pose_init = pose_init.copy()
            pose_init[:3] = self.rot_to_euler(R_seed)
            pose_init[3:] = p_seed
            if self._last_timestamp is not None:
                dt_seed = max(1e-3, float(timestamp - self._last_timestamp))
                lidar_velocity = (p_seed - prev_pose[3:]) / dt_seed
                velocity_init = 0.8 * lidar_velocity + 0.2 * np.asarray(velocity_init, dtype=np.float64)

        if integrity["stationary_active"]:
            # Freeze the reference at the first confidently stationary pose.
            # This is a sensor-only zero-motion update; GT never enters it.
            if self._stationary_anchor_pose is None:
                self._stationary_anchor_pose = (
                    self._last_pose.copy() if self._last_pose is not None else pose_init.copy()
                )
            pose_init = self._stationary_anchor_pose.copy()
            velocity_init = np.zeros(3, dtype=np.float64)
        elif not integrity["stationary_active"]:
            self._stationary_anchor_pose = None

        t = time.perf_counter()
        if integrity["stale_rgb"]:
            fast_keypoints_all = []
            fast_keypoints = []
            brief_keypoints = []
            descriptors = np.empty((0, self.brief.bytes), dtype=np.uint8)
            brief_points = np.empty((0, 2), dtype=np.float64)
        else:
            fast_keypoints_all = self.fast.detect(image)
            fast_keypoints = self._select_fast_keypoints(fast_keypoints_all)
            brief_keypoints, descriptors = self.brief.compute(image, fast_keypoints)
            if descriptors is None:
                descriptors = np.empty((0, self.brief.bytes), dtype=np.uint8)
            brief_points = (
                np.asarray([[kp.pt[0], kp.pt[1]] for kp in brief_keypoints], dtype=np.float64)
                if brief_keypoints
                else np.empty((0, 2), dtype=np.float64)
            )
        stage["camera_features_ms"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        detector_result = None
        if detector_future is not None:
            detector_result = detector_future.result()
            stage["detector_wall_ms"] = (time.perf_counter() - detector_submit_t) * 1000.0
            stage["detector_wait_ms"] = (time.perf_counter() - t) * 1000.0
        elif self.detector_mode == "camera_lidar" and not integrity["stale_rgb"] and not integrity["stale_lidar"]:
            detector_result = self.camera_lidar_detector.detect(image, deskewed_lidar)
            stage["detector_wall_ms"] = (time.perf_counter() - t) * 1000.0
            stage["detector_wait_ms"] = stage["detector_wall_ms"]
        else:
            stage["detector_wall_ms"] = 0.0
            stage["detector_wait_ms"] = 0.0

        if integrity["stale_lidar"]:
            # 3-D object measurements are invalid without a fresh LiDAR cloud.
            sensor_objects = []
        elif integrity["stale_rgb"]:
            # Do not reuse stale camera boxes. Fall back to current LiDAR-only
            # proposals so MOT can continue without camera-driven stale objects.
            sensor_objects = self.sensor_object_detector.detect(deskewed_lidar)
        elif detector_result is not None:
            sensor_objects = detector_result
        else:
            sensor_objects = self.sensor_object_detector.detect(deskewed_lidar)
            sensor_objects = self.sensor_object_detector.attach_camera_boxes(sensor_objects, self.calibration, image.shape)
            sensor_objects = self.sensor_object_detector.add_camera_evidence(sensor_objects, brief_keypoints, image.shape)
        camera_boxes = [
            {"id": i, "bbox": obj["bbox"]}
            for i, obj in enumerate(sensor_objects)
            if obj.get("bbox") is not None
        ]
        multimodal_objects = build_sensor_multimodal_objects(
            sensor_objects,
            planar_features,
            brief_keypoints,
            descriptors,
        )
        stage["objects_ms"] = (time.perf_counter() - t) * 1000.0

        temporal_t0 = time.perf_counter()
        t = time.perf_counter()
        visual_tracks = self.visual_tracker.update(
            brief_points,
            descriptors,
            frame_id=frame_id,
            timestamp=timestamp,
            object_boxes_2d=([] if integrity["stale_rgb"] else camera_boxes),
        )
        if integrity["stale_rgb"]:
            visual_tracks = []
        stage["temporal_visual_ms"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        R_pred = self.euler_to_rot(pose_init[:3])
        temporal_lidar = self.lidar_tracker.update(
            ([] if integrity["stale_lidar"] else planar_ego),
            frame_id=frame_id,
            timestamp=timestamp,
            R_world_ego=R_pred,
            p_world_ego=pose_init[3:],
        )
        stage["temporal_lidar_ms"] = (time.perf_counter() - t) * 1000.0
        stage["temporal_ms"] = (time.perf_counter() - temporal_t0) * 1000.0

        t = time.perf_counter()
        estimated_pose_dict = self.pose_dict_from_vector(pose_init)
        # Object proposals are born in LiDAR sensor coordinates.  The tracker
        # expects ego-local measurements before applying ego->world motion
        # compensation.  Missing this extrinsic translation/rotation biases
        # every track and weakens association.
        tracking_objects = []
        for obj in sensor_objects:
            item = dict(obj)
            p_lidar = np.asarray(obj.get("location", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(1, 3)
            item["location"] = self.multimodal_builder.lidar_to_ego(p_lidar)[0]
            tracking_objects.append(item)

        # V21: when an established camera track temporarily loses its YOLO box,
        # use current LiDAR geometry as a support-only measurement. These
        # observations cannot create new IDs and are penalized in association,
        # so direct camera+LiDAR observations always win when both exist.
        fallback_objects = []
        fallback_t0 = time.perf_counter()
        if not integrity["stale_lidar"]:
            fallback_objects = self._tracker_lidar_fallback_objects(
                deskewed_lidar, image.shape, len(sensor_objects)
            )
        for obj in fallback_objects:
            item = dict(obj)
            p_lidar = np.asarray(obj.get("location", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(1, 3)
            item["location"] = self.multimodal_builder.lidar_to_ego(p_lidar)[0]
            tracking_objects.append(item)
        stage["tracking_fallback_ms"] = (time.perf_counter() - fallback_t0) * 1000.0
        stage["tracking_fallback_candidates"] = int(len(fallback_objects))

        active_tracks = self.object_tracker.update(
            tracking_objects,
            timestamp=timestamp,
            ego_pose=estimated_pose_dict,
            suppress_static_publication=bool(integrity.get("stationary_active", False)),
        )
        stage["tracking_ms"] = (time.perf_counter() - t) * 1000.0

        optimization_t0 = time.perf_counter()
        t = time.perf_counter()
        lidar_factors = (
            [] if integrity["stale_lidar"] else self._lidar_factor_features(planar_ego, pose_init)
        )
        # Visual landmark anchoring does not need every raw LiDAR return.
        # Use a deterministic spatially distributed subset while preserving
        # the full deskewed cloud for object detection and per-frame JSON output.
        visual_lidar = deskewed_lidar[:: max(1, int(self.limits.map_point_stride))]
        visual_observations = (
            []
            if integrity["stale_rgb"] or integrity["stale_lidar"]
            else self._visual_observations(
                visual_tracks,
                visual_lidar,
                pose_init,
                camera_boxes,
            )
        )
        object_observations = self._object_factor_observations(active_tracks, pose_init)
        stage["factor_build_ms"] = (time.perf_counter() - t) * 1000.0

        t = time.perf_counter()
        self.sliding_window.add_keyframe_data(
            frame_id=frame_id,
            timestamp=timestamp,
            init_pose=pose_init,
            init_vel=velocity_init,
            imu_preintegration=imu_preintegrated,
            planar_features=lidar_factors,
            # Never place a non-zero scan-registration factor into the graph
            # while the independent V19/V24 sensor consensus says the platform
            # is stationary.  ZUPT is authoritative in that state; admitting an
            # ICP noise factor only pollutes the marginalization prior.
            lidar_relative_pose=(
                lidar_se3
                if (not integrity["stale_lidar"] and not integrity["stationary_active"])
                else None
            ),
            visual_observations=visual_observations,
            object_observations=object_observations,
            prev_frame_id=frame_id - 1 if self._last_pose is not None else None,
            stationary_pose_prior=(
                self._stationary_anchor_pose.copy()
                if integrity["stationary_active"] and self._stationary_anchor_pose is not None
                else None
            ),
            stationary_velocity_prior=(
                np.zeros(3, dtype=np.float64)
                if integrity["stationary_active"]
                else None
            ),
            stationary_pose_weight=float(self.limits.stationary_pose_prior_weight),
            stationary_velocity_weight=float(self.limits.stationary_velocity_prior_weight),
        )
        estimate = self.sliding_window.get_current_estimate(frame_id)
        if estimate is None:
            estimate = {
                "frame_id": int(frame_id),
                "pose": pose_init.tolist(),
                "velocity": velocity_init.tolist(),
                "objects": {},
            }
        optimized_pose = np.asarray(estimate["pose"], dtype=np.float64)
        optimized_velocity = np.asarray(estimate.get("velocity", velocity_init), dtype=np.float64)
        if integrity["stationary_active"] and self._stationary_anchor_pose is not None:
            # Confident stationary mode is a zero-motion measurement. Holding the
            # state prevents moving traffic from leaking into ego motion factors.
            optimized_pose = self._stationary_anchor_pose.copy()
            optimized_velocity = np.zeros(3, dtype=np.float64)
            estimate["pose"] = optimized_pose.tolist()
            estimate["velocity"] = optimized_velocity.tolist()
        stage["graph_opt_ms"] = (time.perf_counter() - t) * 1000.0
        stage["optimization_ms"] = (time.perf_counter() - optimization_t0) * 1000.0

        t = time.perf_counter()
        optimized_pose_dict = self.pose_dict_from_vector(optimized_pose)
        map_stride = max(1, int(self.limits.map_point_stride))
        map_lidar_points = deskewed_lidar[::map_stride, :3]
        static_interval = max(1, int(self.limits.map_static_update_interval))
        planar_interval = max(1, int(self.limits.map_planar_update_interval))
        evict_interval = max(1, int(self.limits.map_evict_interval))
        update_static_map = (int(frame_id) % static_interval) == 0 and not integrity["stale_lidar"]
        update_planar_map = (int(frame_id) % planar_interval) == 0 and not integrity["stale_lidar"]
        evict_static_map = (int(frame_id) % evict_interval) == 0
        self.mapper_4d.add_frame_data(
            frame_id=frame_id,
            timestamp=timestamp,
            lidar_points=map_lidar_points,
            planar_features=planar_features,
            active_tracks=active_tracks,
            ego_pose=optimized_pose_dict,
            update_static=update_static_map,
            update_planar=update_planar_map,
            evict_static=evict_static_map,
        )
        map_summary = self.mapper_4d.export_summary()
        stage["map_static_updated"] = bool(update_static_map)
        stage["map_planar_updated"] = bool(update_planar_map)
        stage["mapping_ms"] = (time.perf_counter() - t) * 1000.0

        evaluation_frame_diag = {
            "raw_gt_objects": 0,
            "projected_gt_boxes": 0,
            "eligible_gt_objects": 0,
            "scored_tracks": 0,
            "effective_lidar_range_m": float(self._current_object_range()),
            "gt_lidar_support_total": 0,
        }

        # Evaluation path is strictly one-way and runs only after estimator
        # outputs are final. Localization is compared in frame-0 local world;
        # MOT is compared in the *current ego frame* on both sides.  This is
        # physically the sensor observation frame and removes arbitrary CARLA
        # world/frame-0 handedness from MOT scoring without feeding GT back to
        # the tracker.
        if ground_truth_ego_pose is not None:
            gt_local_pose = self._gt_pose_local(ground_truth_ego_pose)
            self.ground_truth_pose_history.append(gt_local_pose)
            self.estimated_pose_history.append(optimized_pose.copy())

            projected_gt_boxes = self._project_gt_camera_boxes(
                ground_truth_objects or [], ground_truth_ego_pose, image.shape
            )
            gt_eval_objects = self._gt_objects_current_lidar(
                ground_truth_objects or [], ground_truth_ego_pose
            )
            gt_lidar_support = self._gt_lidar_support_by_bbox(
                projected_gt_boxes, deskewed_lidar
            )
            gt_eval_objects = self._filter_gt_observable_current_lidar(
                gt_eval_objects, projected_gt_boxes, deskewed_lidar
            )
            track_eval_objects = self._tracks_current_lidar(active_tracks, optimized_pose)
            if self.detector_mode == "camera_lidar":
                # Apply the same image-resolution observability floor to sensor
                # tracks and GT.  This is sensor-only filtering (bbox geometry),
                # not GT feedback, and avoids scoring tiny far-range hypotheses
                # that are below the detector's declared evaluation resolution.
                cfg = self.camera_lidar_detector.config
                min_w = float(getattr(cfg, "min_eval_bbox_width_px", 0.0) or 0.0)
                min_h = float(getattr(cfg, "min_eval_bbox_height_px", 0.0) or 0.0)
                min_area = float(getattr(cfg, "min_eval_bbox_area_px", 0.0) or 0.0)
                filtered_tracks = []
                for tr in track_eval_objects:
                    bbox = tr.get("bbox")
                    if bbox is None or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw < min_w or bh < min_h or (bw * bh) < min_area:
                        continue
                    filtered_tracks.append(tr)
                track_eval_objects = filtered_tracks
            self.ground_truth_history.append(gt_eval_objects)
            self.tracking_history.append(track_eval_objects)
            evaluation_frame_diag = {
                "raw_gt_objects": int(len(ground_truth_objects or [])),
                "projected_gt_boxes": int(len(projected_gt_boxes)),
                "eligible_gt_objects": int(len(gt_eval_objects)),
                "scored_tracks": int(len(track_eval_objects)),
                "effective_lidar_range_m": float(self._current_object_range()),
                "gt_lidar_support_total": int(sum(gt_lidar_support.values())),
            }

        self._last_pose = optimized_pose.copy()
        self._last_velocity = optimized_velocity.copy()
        self._last_timestamp = timestamp
        self._last_imu_motion = imu_motion
        if not integrity["stale_lidar"]:
            self._previous_planar_ego = planar_ego
            self._previous_pose_for_lidar = optimized_pose.copy()
        self._last_image_hash = image_hash
        self._last_lidar_hash = lidar_hash

        stage["total_ms"] = (time.perf_counter() - t0) * 1000.0
        stage["rss_mb"] = self._memory_mb()

        result = {
            "frame": int(frame_id),
            "timestamp": timestamp,
            "estimator": {
                "ground_truth_used": False,
                "world_frame": "local_frame_0",
                "camera_lidar_imu_fusion": True,
                "stationary_mode": bool(integrity["stationary_active"]),
            },
            "sensor_integrity": integrity,
            "image": {"height": int(image.shape[0]), "width": int(image.shape[1])},
            "lidar": {
                "raw_points": int(len(lidar)),
                "deskewed_points": int(len(deskewed_lidar)),
                "planar_features": int(len(planar_features)),
                "temporal_features_used": int(len(planar_ego)),
            },
            "imu": {"motion": imu_motion, "preintegration": imu_preintegrated},
            "lidar_odometry": {
                "accepted": bool(lidar_se3.get("accepted", False)),
                "matches": int(lidar_se3.get("matches", 0)),
                "inliers": int(lidar_se3.get("inliers", 0)),
                "rmse_m": lidar_se3.get("rmse_m"),
                "translation_m": lidar_se3.get("translation_m"),
                "rotation_rad": lidar_se3.get("rotation_rad"),
                "point_rmse_m": lidar_se3.get("point_rmse_m"),
                "source": str(lidar_se3.get("source", "unknown")),
                "delta_t": np.asarray(lidar_se3.get("delta_t", np.zeros(3)), dtype=np.float64).tolist(),
                "delta_rpy": self.rot_to_euler(
                    np.asarray(lidar_se3.get("delta_R", np.eye(3)), dtype=np.float64)
                ).tolist(),
                "reason": str(lidar_se3.get("reason", "unknown")),
            },
            "camera": {
                "fast_keypoints_raw": int(len(fast_keypoints_all)),
                "fast_keypoints_used": int(len(fast_keypoints)),
                "brief_keypoints": int(len(brief_keypoints)),
                "descriptor_shape": list(descriptors.shape),
            },
            "objects": {
                "detector_mode": self.detector_mode,
                "sensor_proposals": int(len(sensor_objects)),
                "camera_supported_proposals": int(sum(bool(o.get("camera_evidence", False)) for o in sensor_objects)),
                "confirmed_tracks": int(len(active_tracks)),
                "effective_lidar_range_m": float(self._current_object_range()),
                "detector_diagnostics": (
                    dict(getattr(self.camera_lidar_detector, "last_diagnostics", {}))
                    if self.camera_lidar_detector is not None else {}
                ),
                "ground_truth_used_by_estimator": False,
                "evaluation": evaluation_frame_diag,
            },
            "multimodal_objects": multimodal_objects,
            "temporal_visual": {
                "active_tracks": visual_tracks,
                "long_tracks": int(len(self.visual_tracker.get_long_tracks())),
                "visual_factors": int(len(visual_observations)),
            },
            "temporal_lidar": {
                "active_planar_landmarks": temporal_lidar,
                "stable_landmarks": int(len(self.lidar_tracker.get_stable_landmarks())),
                "lidar_factors": int(len(lidar_factors)),
            },
            "tracking": {"active_tracks": active_tracks},
            "state_estimation": estimate,
            "graph": self.sliding_window.diagnostics(),
            "mapping_4d": map_summary,
            "performance": stage,
        }
        if include_lidar_points:
            result["lidar"]["points"] = deskewed_lidar[:, :3]
        return result

    def evaluation_summary(self) -> Dict:
        if not self.ground_truth_pose_history:
            return {"tracking": None, "localization": None}
        return {
            "tracking": self.evaluator.evaluate_tracking_sequence(
                self.tracking_history, self.ground_truth_history
            ),
            "localization": self.evaluator.evaluate_localization(
                self.estimated_pose_history, self.ground_truth_pose_history
            ),
        }
