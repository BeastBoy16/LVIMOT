from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


class PlanarLandmark:
    def __init__(
        self,
        landmark_id: int,
        center: np.ndarray,
        normal: np.ndarray,
        planarity: float,
        frame_id: int,
        history_size: int = 12,
    ):
        self.landmark_id = int(landmark_id)
        self.center = np.asarray(center, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)
        self.planarity = float(planarity)
        self.frame_ids = deque([int(frame_id)], maxlen=max(2, int(history_size)))
        self.history_centers = deque([self.center.copy()], maxlen=max(2, int(history_size)))
        self.history_normals = deque([self.normal.copy()], maxlen=max(2, int(history_size)))
        self.hits = 1
        self.age = 1
        self.missed = 0

    def update(self, center: np.ndarray, normal: np.ndarray, planarity: float, frame_id: int):
        alpha = 0.3
        c = np.asarray(center, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        if np.dot(self.normal, n) < 0:
            n = -n
        self.center = (1.0 - alpha) * self.center + alpha * c
        self.normal = (1.0 - alpha) * self.normal + alpha * n
        self.normal /= max(np.linalg.norm(self.normal), 1e-6)
        self.planarity = (1.0 - alpha) * self.planarity + alpha * float(planarity)
        self.frame_ids.append(int(frame_id))
        self.history_centers.append(c)
        self.history_normals.append(n)
        self.hits += 1
        self.age += 1
        self.missed = 0


class CarlaTemporalLiDARMatcher:
    """Bounded-memory planar-feature matcher using compiled KD-tree queries."""

    def __init__(
        self,
        max_center_distance: float = 1.0,
        min_normal_cos: float = 0.85,
        max_planarity_diff: float = 0.20,
        max_missed: int = 3,
        min_hits_for_map: int = 3,
        max_features: int = 1500,
        max_landmarks: int = 2500,
        history_size: int = 12,
    ):
        self.max_center_distance = float(max_center_distance)
        self.min_normal_cos = float(min_normal_cos)
        self.max_planarity_diff = float(max_planarity_diff)
        self.max_missed = int(max_missed)
        self.min_hits_for_map = int(min_hits_for_map)
        self.max_features = max(32, int(max_features))
        self.max_landmarks = max(32, int(max_landmarks))
        self.history_size = max(2, int(history_size))
        self.landmarks: List[PlanarLandmark] = []
        self.next_landmark_id = 0

    def _select_features(self, features: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        if not features:
            return [], np.empty(0, dtype=np.int64)
        n = len(features)
        if n <= self.max_features:
            idx = np.arange(n, dtype=np.int64)
            return list(features), idx
        scores = np.asarray([float(f.get("planarity", 0.0)) for f in features], dtype=np.float64)
        idx = np.argpartition(scores, -self.max_features)[-self.max_features:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [features[int(i)] for i in idx], idx.astype(np.int64)

    def match_planar_features(
        self,
        prev_features: List[Dict],
        curr_features: List[Dict],
        R_pred: Optional[np.ndarray] = None,
        t_pred: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """Match planar features with bounded vectorized nearest-neighbor work.

        V11 used ``query_ball_point`` and then iterated over a Python list of
        neighbors for every previous feature.  The cost grew noticeably once
        the landmark bank filled.  For temporal LiDAR we only need a few local
        alternatives, so V12 queries the four nearest candidates inside the
        same geometric gate and evaluates them in one NumPy batch.
        """
        if not prev_features or not curr_features:
            return []

        prev_sel, prev_map = self._select_features(prev_features)
        curr_sel, curr_map = self._select_features(curr_features)

        R = np.asarray(R_pred if R_pred is not None else np.eye(3), dtype=np.float64)
        t = np.asarray(t_pred if t_pred is not None else np.zeros(3), dtype=np.float64)

        prev_centers = np.asarray([f["center"] for f in prev_sel], dtype=np.float64)
        prev_normals = np.asarray([f["normal"] for f in prev_sel], dtype=np.float64)
        prev_planarities = np.asarray([f.get("planarity", 0.0) for f in prev_sel], dtype=np.float64)
        curr_centers = np.asarray([f["center"] for f in curr_sel], dtype=np.float64)
        curr_normals = np.asarray([f["normal"] for f in curr_sel], dtype=np.float64)
        curr_planarities = np.asarray([f.get("planarity", 0.0) for f in curr_sel], dtype=np.float64)

        transformed_prev_centers = prev_centers @ R.T + t
        transformed_prev_normals = prev_normals @ R.T
        transformed_prev_normals /= np.maximum(
            np.linalg.norm(transformed_prev_normals, axis=1, keepdims=True), 1e-9
        )

        tree = cKDTree(curr_centers)
        k = min(4, len(curr_centers))
        distances, indices = tree.query(
            transformed_prev_centers,
            k=k,
            distance_upper_bound=self.max_center_distance,
            workers=-1,
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        valid = np.isfinite(distances) & (indices < len(curr_centers))
        if not np.any(valid):
            return []

        prev_ids, neighbor_slots = np.nonzero(valid)
        curr_ids = indices[prev_ids, neighbor_slots].astype(np.int64, copy=False)
        c_normals = curr_normals[curr_ids]
        p_normals = transformed_prev_normals[prev_ids]
        cos_sim = np.abs(np.einsum("ij,ij->i", c_normals, p_normals))
        planarity_diff = np.abs(curr_planarities[curr_ids] - prev_planarities[prev_ids])
        geometric = (cos_sim >= self.min_normal_cos) & (planarity_diff <= self.max_planarity_diff)
        if not np.any(geometric):
            return []

        prev_ids = prev_ids[geometric]
        curr_ids = curr_ids[geometric]
        c_normals = c_normals[geometric]
        cos_sim = cos_sim[geometric]
        delta = curr_centers[curr_ids] - transformed_prev_centers[prev_ids]
        center_dist = np.linalg.norm(delta, axis=1)
        point_plane = np.abs(np.einsum("ij,ij->i", c_normals, delta))
        cost = point_plane + 0.2 * center_dist + 0.5 * (1.0 - cos_sim)

        order = np.argsort(cost)
        used_prev = np.zeros(len(prev_sel), dtype=bool)
        used_curr = np.zeros(len(curr_sel), dtype=bool)
        matches: List[Tuple[int, int]] = []
        for idx in order:
            i = int(prev_ids[idx])
            j = int(curr_ids[idx])
            if used_prev[i] or used_curr[j]:
                continue
            used_prev[i] = True
            used_curr[j] = True
            matches.append((int(prev_map[i]), int(curr_map[j])))
        return matches

    def update(
        self,
        planar_features: List[Dict],
        frame_id: int,
        timestamp: float,
        R_world_ego: Optional[np.ndarray] = None,
        p_world_ego: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        selected, _ = self._select_features(planar_features)
        if not selected:
            for lm in self.landmarks:
                lm.missed += 1
                lm.age += 1
            self.landmarks = [lm for lm in self.landmarks if lm.missed <= self.max_missed]
            return self.get_active_landmarks()

        R = np.asarray(R_world_ego if R_world_ego is not None else np.eye(3), dtype=np.float64)
        p = np.asarray(p_world_ego if p_world_ego is not None else np.zeros(3), dtype=np.float64)

        centers = np.asarray([f["center"] for f in selected], dtype=np.float64)
        normals = np.asarray([f["normal"] for f in selected], dtype=np.float64)
        planarities = np.asarray([float(f.get("planarity", 0.0)) for f in selected], dtype=np.float64)
        centers_w = centers @ R.T + p
        normals_w = normals @ R.T
        normals_w /= np.maximum(np.linalg.norm(normals_w, axis=1, keepdims=True), 1e-9)
        world_features = [
            {"center": centers_w[i], "normal": normals_w[i], "planarity": planarities[i]}
            for i in range(len(selected))
        ]

        if not self.landmarks:
            for feat in world_features[: self.max_landmarks]:
                self.landmarks.append(
                    PlanarLandmark(
                        self.next_landmark_id,
                        feat["center"],
                        feat["normal"],
                        feat["planarity"],
                        frame_id,
                        self.history_size,
                    )
                )
                self.next_landmark_id += 1
            return self.get_active_landmarks()

        prev_feats = [
            {"center": lm.center, "normal": lm.normal, "planarity": lm.planarity}
            for lm in self.landmarks
        ]
        matches = self.match_planar_features(prev_feats, world_features)
        matched_lms, matched_obs = set(), set()
        for lm_idx, obs_idx in matches:
            if lm_idx >= len(self.landmarks) or obs_idx >= len(world_features):
                continue
            matched_lms.add(lm_idx)
            matched_obs.add(obs_idx)
            feat = world_features[obs_idx]
            self.landmarks[lm_idx].update(
                feat["center"], feat["normal"], feat["planarity"], frame_id
            )

        for i, lm in enumerate(self.landmarks):
            if i not in matched_lms:
                lm.missed += 1
                lm.age += 1

        self.landmarks = [lm for lm in self.landmarks if lm.missed <= self.max_missed]
        capacity = max(0, self.max_landmarks - len(self.landmarks))
        if capacity:
            for j, feat in enumerate(world_features):
                if j in matched_obs:
                    continue
                self.landmarks.append(
                    PlanarLandmark(
                        self.next_landmark_id,
                        feat["center"],
                        feat["normal"],
                        feat["planarity"],
                        frame_id,
                        self.history_size,
                    )
                )
                self.next_landmark_id += 1
                capacity -= 1
                if capacity <= 0:
                    break
        return self.get_active_landmarks()

    def get_active_landmarks(self) -> List[Dict]:
        return [
            {
                "landmark_id": lm.landmark_id,
                "center": lm.center.tolist(),
                "normal": lm.normal.tolist(),
                "planarity": lm.planarity,
                "hits": lm.hits,
                "age": lm.age,
            }
            for lm in self.landmarks
            if lm.missed == 0
        ]

    def get_stable_landmarks(self) -> List[PlanarLandmark]:
        return [lm for lm in self.landmarks if lm.hits >= self.min_hits_for_map and lm.missed == 0]
