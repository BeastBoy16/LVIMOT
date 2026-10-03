from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from carla_motion_classification import CarlaMotionClassifier, MotionState


class KalmanTrack:
    def __init__(
        self,
        track_id: int,
        observation: Dict,
        timestamp: float,
        world_position: Optional[np.ndarray] = None,
        history_size: int = 32,
    ):
        self.track_id = int(track_id)
        self.class_name = observation.get("type", observation.get("class", "object"))
        position = np.asarray(world_position if world_position is not None else observation["location"], dtype=np.float64)

        self.state = np.zeros(6, dtype=np.float64)
        self.state[:3] = position
        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 0.75
        self.P[3:, 3:] *= 8.0

        self.last_timestamp = float(timestamp)
        self.last_measurement_timestamp = float(timestamp)
        self.last_measurement_position = position.copy()
        self.age = 1
        self.hits = 1
        self.consecutive_hits = 1
        self.missed = 0
        self.observed_this_frame = True
        self.last_update_fallback = bool(observation.get("lidar_fallback", False))
        self.score = float(observation.get("score", 0.5))
        self.motion_state = MotionState.UNCERTAIN
        self.motion_confidence = 0.0
        self.dimensions = observation.get("dimensions", {"length": 4.5, "width": 1.8, "height": 1.5})
        self.rotation = observation.get("rotation", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
        self.bbox = list(observation.get("bbox", [])) if observation.get("bbox") is not None else None
        self.camera_confidence = float(observation.get("camera_confidence", 0.0))
        self.camera_evidence = bool(observation.get("camera_evidence", False))
        self.camera_support = int(observation.get("camera_support", observation.get("num_points", 0)) or 0)
        self.history = deque(maxlen=max(4, int(history_size)))
        self.history.append(
            {
                "timestamp": float(timestamp),
                "position": position.copy(),
                "velocity": self.state[3:].copy(),
            }
        )

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-4)
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Constant-velocity model with moderate acceleration process noise.
        q = 2.0
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[:3, :3] = np.eye(3) * (dt4 / 4.0) * q
        Q[:3, 3:] = np.eye(3) * (dt3 / 2.0) * q
        Q[3:, :3] = np.eye(3) * (dt3 / 2.0) * q
        Q[3:, 3:] = np.eye(3) * dt2 * q

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return self.state.copy()

    @staticmethod
    def _dims_vector(dimensions: Dict) -> np.ndarray:
        if isinstance(dimensions, dict):
            return np.array(
                [
                    float(dimensions.get("length", 0.0)),
                    float(dimensions.get("width", 0.0)),
                    float(dimensions.get("height", 0.0)),
                ],
                dtype=np.float64,
            )
        return np.asarray(dimensions, dtype=np.float64).reshape(-1)[:3]

    def update(
        self,
        observation: Dict,
        timestamp: float,
        world_position: Optional[np.ndarray] = None,
        position_noise: float = 0.20,
        velocity_noise: float = 1.50,
    ):
        z_pos = np.asarray(world_position if world_position is not None else observation["location"], dtype=np.float64)
        timestamp = float(timestamp)
        dt_meas = max(timestamp - self.last_measurement_timestamp, 1e-4)
        measured_v = (z_pos - self.last_measurement_position) / dt_meas

        # Position and finite-difference velocity are fused together. This
        # makes velocity converge quickly enough for short CARLA validation
        # sequences while remaining sensor-derived and GT-free.
        H = np.eye(6, dtype=np.float64)
        z = np.hstack([z_pos, measured_v])
        R = np.diag(
            [position_noise, position_noise, position_noise, velocity_noise, velocity_noise, velocity_noise]
        ).astype(np.float64)

        innovation = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        self.last_timestamp = timestamp
        self.last_measurement_timestamp = timestamp
        self.last_measurement_position = z_pos.copy()
        self.age += 1
        self.hits += 1
        self.consecutive_hits += 1
        self.missed = 0
        self.observed_this_frame = True
        self.last_update_fallback = bool(observation.get("lidar_fallback", False))
        self.score = 0.7 * self.score + 0.3 * float(observation.get("score", self.score))

        if "dimensions" in observation:
            self.dimensions = observation["dimensions"]
        if "rotation" in observation:
            self.rotation = observation["rotation"]
        if observation.get("bbox") is not None:
            self.bbox = list(observation["bbox"])
        self.camera_confidence = float(observation.get("camera_confidence", self.camera_confidence))
        self.camera_evidence = bool(observation.get("camera_evidence", self.camera_evidence))
        self.camera_support = int(observation.get("camera_support", observation.get("num_points", self.camera_support)) or 0)

        self.history.append(
            {
                "timestamp": timestamp,
                "position": self.state[:3].copy(),
                "velocity": self.state[3:].copy(),
            }
        )

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self.consecutive_hits = 0
        self.observed_this_frame = False
        self.last_update_fallback = False
        # A predicted-only hypothesis may remain internally for a short detector
        # gap, but its confidence decays and it is not published without current
        # sensor support.
        self.score *= 0.92

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.state[3:]))


class CarlaMultiObjectTracker:
    def __init__(
        self,
        distance_threshold: float = 5.0,
        max_missed: int = 5,
        min_hits: int = 1,
        use_ego_compensation: bool = True,
        motion_classifier: Optional[CarlaMotionClassifier] = None,
        history_size: int = 32,
        new_track_min_score: float = 0.0,
        publish_min_score: Optional[float] = None,
        publish_max_missed: int = 0,
        gap_publish_min_hits: int = 4,
        min_camera_support: int = 0,
        fast_confirm_hits: int = 3,
        fast_confirm_min_score: float = 0.42,
        fast_confirm_min_support: int = 8,
        weak_confirm_min_score: float = 0.22,
        weak_confirm_min_camera_confidence: float = 0.06,
        weak_confirm_min_support: int = 10,
        weak_confirm_hits: int = 2,
        static_suppress_min_hits: int = 5,
        static_suppress_min_confidence: float = 0.85,
        fallback_min_hits: int = 3,
        fallback_min_support: int = 18,
    ):
        self.distance_threshold = float(distance_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.use_ego_compensation = bool(use_ego_compensation)
        self.classifier = motion_classifier or CarlaMotionClassifier()
        self.history_size = int(history_size)
        self.new_track_min_score = float(new_track_min_score)
        self.publish_min_score = float(
            self.new_track_min_score if publish_min_score is None else publish_min_score
        )
        self.publish_max_missed = max(0, int(publish_max_missed))
        self.gap_publish_min_hits = max(1, int(gap_publish_min_hits))
        self.min_camera_support = max(0, int(min_camera_support))
        self.fast_confirm_hits = max(2, int(fast_confirm_hits))
        self.fast_confirm_min_score = float(fast_confirm_min_score)
        self.fast_confirm_min_support = max(0, int(fast_confirm_min_support))
        self.weak_confirm_min_score = float(weak_confirm_min_score)
        self.weak_confirm_min_camera_confidence = float(weak_confirm_min_camera_confidence)
        self.weak_confirm_min_support = max(0, int(weak_confirm_min_support))
        self.weak_confirm_hits = max(2, int(weak_confirm_hits))
        self.static_suppress_min_hits = max(2, int(static_suppress_min_hits))
        self.static_suppress_min_confidence = float(static_suppress_min_confidence)
        self.fallback_min_hits = max(2, int(fallback_min_hits))
        self.fallback_min_support = max(0, int(fallback_min_support))
        self.tracks: List[KalmanTrack] = []
        self.next_track_id = 0

    @staticmethod
    def _rotation_matrix(rotation: Dict[str, float]) -> np.ndarray:
        pitch = np.deg2rad(float(rotation.get("pitch", 0.0)))
        yaw = np.deg2rad(float(rotation.get("yaw", 0.0)))
        roll = np.deg2rad(float(rotation.get("roll", 0.0)))
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array(
            [
                [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
                [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
                [sp, -cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _dims_vector(dimensions) -> np.ndarray:
        if isinstance(dimensions, dict):
            return np.array(
                [
                    float(dimensions.get("length", 0.0)),
                    float(dimensions.get("width", 0.0)),
                    float(dimensions.get("height", 0.0)),
                ],
                dtype=np.float64,
            )
        arr = np.asarray(dimensions, dtype=np.float64).reshape(-1)
        out = np.zeros(3, dtype=np.float64)
        out[: min(3, len(arr))] = arr[:3]
        return out

    @staticmethod
    def _bbox_iou(a, b) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(a) != 4 or len(b) != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
            return 0.0
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0

    @staticmethod
    def _bbox_center_distance_normalized(a, b) -> float:
        if a is None or b is None:
            return float("inf")
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(a) != 4 or len(b) != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
            return float("inf")
        ac = np.array([(a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5], dtype=np.float64)
        bc = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)
        aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
        bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
        scale = max(8.0, 0.5 * (np.hypot(aw, ah) + np.hypot(bw, bh)))
        return float(np.linalg.norm(ac - bc) / scale)

    def _is_duplicate_of_observed_track(
        self,
        observation: Dict,
        world_position: np.ndarray,
        observed_tracks: List[KalmanTrack],
    ) -> bool:
        """Return True only for a very likely duplicate of a track matched now."""
        obs_bbox = observation.get("bbox")
        p = np.asarray(world_position, dtype=np.float64)
        for track in observed_tracks:
            iou = self._bbox_iou(track.bbox, obs_bbox)
            center_delta = self._bbox_center_distance_normalized(track.bbox, obs_bbox)
            distance = float(np.linalg.norm(track.position - p))
            if iou >= 0.62 and distance <= 3.25:
                return True
            if center_delta <= 0.16 and distance <= 2.25:
                return True
        return False

    def _merge_tentative_duplicates(self) -> None:
        """Remove only duplicate tentative identities; established IDs are preserved."""
        if len(self.tracks) < 2:
            return
        keep = [True] * len(self.tracks)
        for i, a in enumerate(self.tracks):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.tracks)):
                if not keep[j]:
                    continue
                b = self.tracks[j]
                if min(int(a.hits), int(b.hits)) > self.fast_confirm_hits:
                    continue
                if not (a.observed_this_frame and b.observed_this_frame):
                    continue
                iou = self._bbox_iou(a.bbox, b.bbox)
                center_delta = self._bbox_center_distance_normalized(a.bbox, b.bbox)
                distance = float(np.linalg.norm(a.position - b.position))
                duplicate = (iou >= 0.72 and distance <= 2.75) or (center_delta <= 0.12 and distance <= 2.0)
                if not duplicate:
                    continue
                quality_a = (int(a.hits), float(a.score), int(a.camera_support), -int(a.track_id))
                quality_b = (int(b.hits), float(b.score), int(b.camera_support), -int(b.track_id))
                if quality_a >= quality_b:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
        self.tracks = [track for flag, track in zip(keep, self.tracks) if flag]

    def established_track_count(self, min_hits: Optional[int] = None) -> int:
        threshold = self.fallback_min_hits if min_hits is None else max(1, int(min_hits))
        return sum(
            1
            for track in self.tracks
            if track.hits >= threshold and track.missed <= 1 and track.score >= self.publish_min_score
        )

    def _suppress_duplicate_tracks(self, tracks: List[KalmanTrack]) -> List[KalmanTrack]:
        """Keep one publication for duplicate hypotheses of the same observed object.

        This is deliberately conservative: two tracks are considered duplicates
        only when their *current camera boxes* overlap strongly and their 3-D
        positions are also close. Distinct adjacent vehicles are therefore kept.
        Internal tracker state is not deleted; suppression affects publication
        only, so a temporarily ambiguous hypothesis can still be reassociated.
        """
        ranked = sorted(
            tracks,
            key=lambda t: (
                float(t.score) + 0.015 * min(int(t.camera_support), 24)
                + 0.01 * min(int(t.hits), 20),
                int(t.hits),
            ),
            reverse=True,
        )
        kept: List[KalmanTrack] = []
        for track in ranked:
            duplicate = False
            for other in kept:
                iou = self._bbox_iou(track.bbox, other.bbox)
                dist = float(np.linalg.norm(track.position - other.position))
                if iou >= 0.68 and dist <= 3.5:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(track)
        return kept

    def _transform_to_world(self, position: np.ndarray, ego_pose: Optional[Dict]) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float64)
        if ego_pose is None or not self.use_ego_compensation:
            return pos
        loc = ego_pose.get("location", {"x": 0, "y": 0, "z": 0})
        t_w_e = np.array(
            [float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), float(loc.get("z", 0.0))],
            dtype=np.float64,
        )
        R_w_e = self._rotation_matrix(ego_pose.get("rotation", {}))
        return R_w_e @ pos + t_w_e

    def _create_track(self, observation: Dict, timestamp: float, world_position: np.ndarray) -> KalmanTrack:
        track = KalmanTrack(
            track_id=self.next_track_id,
            observation=observation,
            timestamp=timestamp,
            world_position=world_position,
            history_size=self.history_size,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def _build_cost_matrix(self, observations: List[Dict], world_positions: List[np.ndarray]) -> np.ndarray:
        if not self.tracks or not observations:
            return np.empty((len(self.tracks), len(observations)), dtype=np.float64)
        cost = np.full((len(self.tracks), len(observations)), np.inf, dtype=np.float64)
        for i, track in enumerate(self.tracks):
            track_dims = self._dims_vector(track.dimensions)
            for j, (observation, pos) in enumerate(zip(observations, world_positions)):
                distance = float(np.linalg.norm(track.position - pos))
                obs_dims = self._dims_vector(observation.get("dimensions", {}))
                denom = np.maximum(np.maximum(track_dims, obs_dims), 0.25)
                shape_delta = float(np.mean(np.abs(track_dims - obs_dims) / denom))
                if shape_delta > 0.90:
                    continue

                is_fallback = bool(observation.get("lidar_fallback", False))
                obs_bbox = observation.get("bbox")
                iou = self._bbox_iou(track.bbox, obs_bbox)
                center_delta = self._bbox_center_distance_normalized(track.bbox, obs_bbox)

                if is_fallback:
                    # LiDAR-only continuation is allowed only for existing tracks
                    # (allow_new_track=False upstream), so do not reject a valid
                    # 3-D continuation because its projected cluster box is noisy.
                    hard_gate = max(4.0, 1.65 * self.distance_threshold)
                    if distance > hard_gate:
                        continue
                    camera_cost = 0.0
                    fallback_penalty = 0.20
                else:
                    camera_consistent = (iou >= 0.04) or (center_delta <= 0.85)
                    strong_camera = (iou >= 0.12) or (center_delta <= 0.45)
                    hard_gate = max(5.5, 2.0 * self.distance_threshold)
                    if track.hits >= self.fast_confirm_hits + 1 and strong_camera:
                        hard_gate = max(hard_gate, 8.0)
                    if distance > self.distance_threshold:
                        if not camera_consistent or distance > hard_gate:
                            continue
                    if np.isfinite(center_delta):
                        camera_cost = 1.15 * (1.0 - iou) + 0.55 * min(center_delta, 2.0)
                    else:
                        camera_cost = 0.90
                    fallback_penalty = 0.0

                history_bonus = min(0.25, 0.015 * max(0, int(track.hits) - self.fast_confirm_hits))
                cost[i, j] = (
                    0.55 * distance
                    + 0.50 * shape_delta
                    + camera_cost
                    + fallback_penalty
                    - history_bonus
                )
        return cost

    def update(
        self,
        observations: List[Dict],
        timestamp: float,
        ego_pose: Optional[Dict] = None,
        suppress_static_publication: bool = False,
    ) -> List[Dict]:
        observations = list(observations)
        timestamp = float(timestamp)
        world_positions = []
        for obs in observations:
            loc = obs.get("location")
            if isinstance(loc, dict):
                p = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
            else:
                p = np.asarray(loc, dtype=np.float64)
            world_positions.append(self._transform_to_world(p, ego_pose))

        # Predict exactly once to this timestamp. V20 did not advance the state
        # timestamp on missed frames, so the same interval could be integrated
        # again on the following frame and fragment an otherwise stable ID.
        for track in self.tracks:
            dt = max(timestamp - track.last_timestamp, 0.0)
            track.predict(dt)
            track.last_timestamp = timestamp
            track.observed_this_frame = False
            track.last_update_fallback = False

        cost_matrix = self._build_cost_matrix(observations, world_positions)
        matched_tracks = set()
        matched_observations = set()
        if self.tracks and observations:
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost_matrix), cost_matrix, 1e9))
            for row, col in zip(rows, cols):
                if not np.isfinite(cost_matrix[row, col]):
                    continue
                matched_tracks.add(int(row))
                matched_observations.add(int(col))
                score = float(observations[col].get("score", 0.5))
                position_noise = float(np.clip(0.35 - 0.20 * score, 0.10, 0.35))
                self.tracks[row].update(
                    observations[col],
                    timestamp,
                    world_position=world_positions[col],
                    position_noise=position_noise,
                )

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        observed_tracks = [self.tracks[i] for i in sorted(matched_tracks) if i < len(self.tracks)]
        for j, observation in enumerate(observations):
            if j in matched_observations:
                continue
            if not bool(observation.get("allow_new_track", True)):
                continue
            if float(observation.get("score", 0.0)) < self.new_track_min_score:
                continue
            if bool(observation.get("camera_evidence", False)) and int(
                observation.get("camera_support", observation.get("num_points", 0)) or 0
            ) < self.min_camera_support:
                continue
            if self._is_duplicate_of_observed_track(observation, world_positions[j], observed_tracks):
                continue
            self._create_track(observation, timestamp, world_positions[j])

        self._merge_tentative_duplicates()

        for track in self.tracks:
            cls_res = self.classifier.classify_track(list(track.history), track.class_name)
            track.motion_state = cls_res["state"]
            track.motion_confidence = cls_res["confidence"]

        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return self.get_tracks(suppress_static_publication=suppress_static_publication)

    def get_tracks(
        self,
        include_history: bool = False,
        suppress_static_publication: bool = False,
    ) -> List[Dict]:
        results = []
        publishable: List[KalmanTrack] = []
        for track in self.tracks:
            if track.missed > self.publish_max_missed:
                continue
            if not track.observed_this_frame:
                # Backward-compatible prediction bridging is available only to
                # configurations that explicitly set publish_max_missed > 0.
                # The V21 runtime keeps publish_max_missed=0 and instead uses
                # current-frame LiDAR fallback support.
                if track.missed <= 0 or track.hits < self.gap_publish_min_hits:
                    continue
            support = int(track.camera_support)
            strong_fast_confirm = (
                track.consecutive_hits >= self.fast_confirm_hits
                and float(track.score) >= self.fast_confirm_min_score
                and support >= self.fast_confirm_min_support
            )
            weak_lidar_confirm = (
                not track.last_update_fallback
                and track.camera_evidence
                and track.consecutive_hits >= self.weak_confirm_hits
                and float(track.score) >= self.weak_confirm_min_score
                and float(track.camera_confidence) >= self.weak_confirm_min_camera_confidence
                and support >= self.weak_confirm_min_support
            )

            # Preserve the normal publication floor, but allow a distant weak
            # camera box to publish when LiDAR support and temporal consistency
            # independently confirm it. This is still camera+LiDAR LVIMOT, not a
            # LiDAR-only detector.
            if float(track.score) < self.publish_min_score and not weak_lidar_confirm:
                continue

            if track.last_update_fallback:
                if track.hits < self.fallback_min_hits:
                    continue
                if support < self.fallback_min_support:
                    continue
            else:
                if track.camera_evidence and support < self.min_camera_support:
                    continue
                if track.hits < self.min_hits and not (strong_fast_confirm or weak_lidar_confirm):
                    continue
                if self.fast_confirm_hits <= self.min_hits and track.hits < 3:
                    if not (strong_fast_confirm or weak_lidar_confirm):
                        continue

            # Dynamic-object MOT should not publish a persistent static world
            # hypothesis when the ego itself is sensor-confirmed stationary.
            # The track is kept internally and the static LiDAR map still keeps
            # the geometry; if the object later moves, it can return to MOT.
            if (
                suppress_static_publication
                and track.motion_state == MotionState.STATIC
                and track.hits >= self.static_suppress_min_hits
                and float(track.motion_confidence) >= self.static_suppress_min_confidence
            ):
                continue

            publishable.append(track)

        publishable = self._suppress_duplicate_tracks(publishable)
        publishable.sort(key=lambda t: int(t.track_id))
        for track in publishable:
            item = {
                "track_id": track.track_id,
                "class": track.class_name,
                "position": track.position.tolist(),
                "velocity": track.velocity.tolist(),
                "speed": track.speed,
                "score": float(track.score),
                "motion_state": track.motion_state.value,
                "motion_confidence": track.motion_confidence,
                "dimensions": track.dimensions,
                "rotation": track.rotation,
                "age": track.age,
                "hits": track.hits,
                "consecutive_hits": int(track.consecutive_hits),
                "missed": track.missed,
                "bbox": list(track.bbox) if track.bbox is not None else None,
                "camera_confidence": float(track.camera_confidence),
                "camera_evidence": bool(track.camera_evidence),
                "camera_support": int(track.camera_support),
                "lidar_fallback": bool(track.last_update_fallback),
            }
            if include_history:
                item["history"] = [
                    {
                        "timestamp": h["timestamp"],
                        "position": h["position"].tolist(),
                        "velocity": h["velocity"].tolist(),
                    }
                    for h in track.history
                ]
            results.append(item)
        return results

