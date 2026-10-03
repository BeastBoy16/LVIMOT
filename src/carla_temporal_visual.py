from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class VisualFeatureTrack:
    def __init__(
        self,
        track_id: int,
        initial_point: np.ndarray,
        descriptor: np.ndarray,
        frame_id: int,
        timestamp: float,
        history_size: int = 20,
    ):
        self.track_id = int(track_id)
        self.points = deque(maxlen=max(2, int(history_size)))
        self.descriptors = deque(maxlen=max(2, int(history_size)))
        self.frame_ids = deque(maxlen=max(2, int(history_size)))
        self.timestamps = deque(maxlen=max(2, int(history_size)))
        self.points.append(np.asarray(initial_point, dtype=np.float64))
        self.descriptors.append(np.asarray(descriptor, dtype=np.uint8))
        self.frame_ids.append(int(frame_id))
        self.timestamps.append(float(timestamp))
        self.age = 1
        self.observation_count = 1
        self.missed = 0
        self.object_id: Optional[int] = None

    def add_observation(
        self,
        point: np.ndarray,
        descriptor: np.ndarray,
        frame_id: int,
        timestamp: float,
    ):
        self.points.append(np.asarray(point, dtype=np.float64))
        self.descriptors.append(np.asarray(descriptor, dtype=np.uint8))
        self.frame_ids.append(int(frame_id))
        self.timestamps.append(float(timestamp))
        self.age += 1
        self.observation_count += 1
        self.missed = 0

    @property
    def latest_point(self) -> np.ndarray:
        return self.points[-1]

    @property
    def latest_descriptor(self) -> np.ndarray:
        return self.descriptors[-1]

    @property
    def track_length(self) -> int:
        return self.observation_count


class CarlaTemporalVisualTracker:
    """Bounded-memory FAST/BRIEF temporal tracker.

    The original implementation materialized a full N x M x descriptor_bytes
    XOR tensor. On real CARLA frames with thousands of features this could use
    many GB of RAM. This implementation caps the working set and uses OpenCV's
    compiled Hamming matcher when available.
    """

    def __init__(
        self,
        max_hamming_distance: int = 64,
        ratio_threshold: float = 0.80,
        max_pixel_displacement: float = 80.0,
        max_missed: int = 2,
        min_track_length_for_factor: int = 3,
        max_features: int = 1500,
        max_tracks: int = 2000,
        history_size: int = 20,
    ):
        self.max_hamming_distance = int(max_hamming_distance)
        self.ratio_threshold = float(ratio_threshold)
        self.max_pixel_displacement = float(max_pixel_displacement)
        self.max_missed = int(max_missed)
        self.min_track_length_for_factor = int(min_track_length_for_factor)
        self.max_features = max(16, int(max_features))
        self.max_tracks = max(16, int(max_tracks))
        self.history_size = max(2, int(history_size))
        self.tracks: List[VisualFeatureTrack] = []
        self.next_track_id = 0

    @staticmethod
    def compute_hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """Compatibility helper for small arrays used in tests/debugging."""
        desc1 = np.asarray(desc1, dtype=np.uint8)
        desc2 = np.asarray(desc2, dtype=np.uint8)
        if desc1.ndim == 1:
            desc1 = desc1[None, :]
        if desc2.ndim == 1:
            desc2 = desc2[None, :]
        # Chunk rows to prevent a large three-dimensional temporary tensor.
        result = np.empty((len(desc1), len(desc2)), dtype=np.uint16)
        chunk = 64
        for start in range(0, len(desc1), chunk):
            stop = min(start + chunk, len(desc1))
            xor_result = np.bitwise_xor(
                desc1[start:stop, None, :],
                desc2[None, :, :],
            )
            result[start:stop] = np.unpackbits(xor_result, axis=-1).sum(axis=-1)
        return result

    def _bounded_inputs(
        self,
        keypoints: np.ndarray,
        descriptors: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        points = np.asarray(keypoints, dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64), None if descriptors is None else np.empty((0, descriptors.shape[1]), dtype=np.uint8)
        points = points.reshape(-1, 2)
        if descriptors is None:
            return points[: self.max_features], None
        desc = np.asarray(descriptors, dtype=np.uint8)
        n = min(len(points), len(desc), self.max_features)
        return points[:n].copy(), desc[:n].copy()

    def match_descriptors(
        self,
        prev_pts: np.ndarray,
        prev_desc: np.ndarray,
        curr_pts: np.ndarray,
        curr_desc: np.ndarray,
    ) -> List[Tuple[int, int]]:
        if (
            len(prev_pts) == 0
            or len(curr_pts) == 0
            or prev_desc is None
            or curr_desc is None
        ):
            return []

        prev_pts = np.asarray(prev_pts, dtype=np.float64)
        curr_pts = np.asarray(curr_pts, dtype=np.float64)
        prev_desc = np.asarray(prev_desc, dtype=np.uint8)
        curr_desc = np.asarray(curr_desc, dtype=np.uint8)

        if cv2 is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            forward = matcher.knnMatch(prev_desc, curr_desc, k=2)
            reverse = matcher.match(curr_desc, prev_desc)
            reverse_best = {int(m.queryIdx): int(m.trainIdx) for m in reverse}
            matches: List[Tuple[int, int]] = []
            used_curr = set()
            for prev_i, pair in enumerate(forward):
                if not pair:
                    continue
                best = pair[0]
                if best.distance > self.max_hamming_distance:
                    continue
                if len(pair) >= 2:
                    second = pair[1]
                    if second.distance > 0 and (best.distance / second.distance) > self.ratio_threshold:
                        continue
                curr_j = int(best.trainIdx)
                if curr_j in used_curr:
                    continue
                if reverse_best.get(curr_j) != prev_i:
                    continue
                if np.linalg.norm(curr_pts[curr_j] - prev_pts[prev_i]) > self.max_pixel_displacement:
                    continue
                matches.append((prev_i, curr_j))
                used_curr.add(curr_j)
            return matches

        # Fallback path without OpenCV. Inputs are bounded before reaching here.
        dist_matrix = self.compute_hamming_distance(prev_desc, curr_desc)
        matches = []
        used_curr = set()
        for i in range(len(prev_pts)):
            spatial = np.linalg.norm(curr_pts - prev_pts[i], axis=1)
            valid = (spatial <= self.max_pixel_displacement) & (dist_matrix[i] <= self.max_hamming_distance)
            ids = np.flatnonzero(valid)
            if ids.size == 0:
                continue
            order = ids[np.argsort(dist_matrix[i, ids])]
            if len(order) >= 2:
                d0, d1 = float(dist_matrix[i, order[0]]), float(dist_matrix[i, order[1]])
                if d1 > 0 and d0 / d1 > self.ratio_threshold:
                    continue
            j = int(order[0])
            if j in used_curr:
                continue
            if int(np.argmin(dist_matrix[:, j])) != i:
                continue
            matches.append((i, j))
            used_curr.add(j)
        return matches

    def update(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        frame_id: int,
        timestamp: float,
        object_boxes_2d: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        keypoints, descriptors = self._bounded_inputs(keypoints, descriptors)

        # Retain a bounded number of live tracks. Prefer recently observed and
        # longer tracks so the factor graph receives stable features.
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda t: (t.missed, -t.track_length, -t.age))
            self.tracks = self.tracks[: self.max_tracks]

        if len(self.tracks) == 0:
            if descriptors is not None:
                for pt, desc in zip(keypoints[: self.max_tracks], descriptors[: self.max_tracks]):
                    track = VisualFeatureTrack(
                        self.next_track_id, pt, desc, frame_id, timestamp, self.history_size
                    )
                    self._assign_object_id(track, pt, object_boxes_2d)
                    self.tracks.append(track)
                    self.next_track_id += 1
            return self.get_active_tracks()

        prev_pts = np.asarray([t.latest_point for t in self.tracks], dtype=np.float64)
        prev_desc = np.asarray([t.latest_descriptor for t in self.tracks], dtype=np.uint8)
        matches = self.match_descriptors(prev_pts, prev_desc, keypoints, descriptors)
        matched_tracks = set()
        matched_obs = set()

        if descriptors is not None:
            for track_idx, obs_idx in matches:
                matched_tracks.add(track_idx)
                matched_obs.add(obs_idx)
                track = self.tracks[track_idx]
                track.add_observation(keypoints[obs_idx], descriptors[obs_idx], frame_id, timestamp)
                self._assign_object_id(track, keypoints[obs_idx], object_boxes_2d)

        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.missed += 1
                track.age += 1

        if descriptors is not None:
            capacity = max(0, self.max_tracks - len(self.tracks))
            if capacity:
                for j in range(len(keypoints)):
                    if j in matched_obs:
                        continue
                    track = VisualFeatureTrack(
                        self.next_track_id,
                        keypoints[j],
                        descriptors[j],
                        frame_id,
                        timestamp,
                        self.history_size,
                    )
                    self._assign_object_id(track, keypoints[j], object_boxes_2d)
                    self.tracks.append(track)
                    self.next_track_id += 1
                    capacity -= 1
                    if capacity <= 0:
                        break

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.get_active_tracks()

    def _assign_object_id(
        self,
        track: VisualFeatureTrack,
        point: np.ndarray,
        object_boxes_2d: Optional[List[Dict]],
    ):
        track.object_id = None
        if not object_boxes_2d:
            return
        u, v = float(point[0]), float(point[1])
        for obj in object_boxes_2d:
            box = obj.get("box_2d") or obj.get("bbox")
            if box is None or len(box) != 4:
                continue
            xmin, ymin, xmax, ymax = box
            if xmin <= u <= xmax and ymin <= v <= ymax:
                track.object_id = obj.get("track_id", obj.get("id"))
                return

    def get_active_tracks(self) -> List[Dict]:
        return [
            {
                "track_id": t.track_id,
                "point": t.latest_point.tolist(),
                "length": t.track_length,
                "age": t.age,
                "object_id": t.object_id,
                "history": [p.tolist() for p in t.points],
            }
            for t in self.tracks
            if t.missed == 0
        ]

    def get_long_tracks(self, min_len: Optional[int] = None) -> List[VisualFeatureTrack]:
        threshold = min_len if min_len is not None else self.min_track_length_for_factor
        return [t for t in self.tracks if t.track_length >= threshold and t.missed == 0]
