from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class CarlaLiDARObjectDetector:
    """Bounded, GT-free LiDAR proposal generator.

    The detector is intentionally classical/self-contained. It removes the
    locally lowest surface (road) in coarse XY cells and clusters the remaining
    points in BEV. Using BEV connectivity is substantially more stable than
    connecting sparse 3-D LiDAR voxels and avoids producing several fragments
    for the same vehicle.
    """

    def __init__(
        self,
        voxel_size: float = 0.50,
        min_points: int = 18,
        max_points: int = 3500,
        max_range: float = 55.0,
        min_height: float = -3.0,
        max_height: float = 4.0,
        max_proposals: int = 12,
        ground_cell_size: float = 1.0,
        ground_clearance: float = 0.22,
        min_vehicle_score: float = 0.42,
    ):
        self.voxel_size = float(voxel_size)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.max_range = float(max_range)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.max_proposals = int(max_proposals)
        self.ground_cell_size = float(ground_cell_size)
        self.ground_clearance = float(ground_clearance)
        self.min_vehicle_score = float(min_vehicle_score)

    @staticmethod
    def _neighbors2(key: Tuple[int, int]):
        x, y = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    yield (x + dx, y + dy)

    def _remove_local_ground(self, xyz: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return xyz
        cells = np.floor(xyz[:, :2] / self.ground_cell_size).astype(np.int32)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        sorted_z = xyz[order, 2]
        change = np.ones(len(sorted_cells), dtype=bool)
        if len(change) > 1:
            change[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(sorted_cells)]
        keep_sorted = np.zeros(len(sorted_cells), dtype=bool)
        for s, e in zip(starts, ends):
            z_ref = float(np.percentile(sorted_z[s:e], 10.0))
            keep_sorted[s:e] = sorted_z[s:e] > (z_ref + self.ground_clearance)
        keep = np.zeros(len(xyz), dtype=bool)
        keep[order] = keep_sorted
        return xyz[keep]

    @staticmethod
    def _oriented_extent(cpts: np.ndarray):
        center_xy = np.mean(cpts[:, :2], axis=0)
        centered = cpts[:, :2] - center_xy
        if len(cpts) >= 3:
            cov = centered.T @ centered / max(len(cpts) - 1, 1)
            vals, vecs = np.linalg.eigh(cov)
            long_axis = vecs[:, int(np.argmax(vals))]
        else:
            long_axis = np.array([1.0, 0.0])
        short_axis = np.array([-long_axis[1], long_axis[0]])
        axes = np.column_stack([long_axis, short_axis])
        local = centered @ axes
        span = np.ptp(local, axis=0)
        if span[0] < span[1]:
            span = span[::-1]
            long_axis = short_axis
        length, width = float(span[0]), float(span[1])
        zmin, zmax = float(np.min(cpts[:, 2])), float(np.max(cpts[:, 2]))
        height = zmax - zmin
        center = np.array([center_xy[0], center_xy[1], 0.5 * (zmin + zmax)], dtype=np.float64)
        yaw = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
        return center, length, width, height, yaw

    @staticmethod
    def _vehicle_likeness(length: float, width: float, height: float, num_points: int, distance: float) -> float:
        """Geometry-only traffic-object prior; no CARLA labels are used.

        It is intentionally broad enough for motorcycles, cars, vans and
        trucks, but strongly penalizes poles, curbs, wall fragments and large
        vegetation patches which previously saturated the proposal cap.
        """
        area = max(length * width, 1e-6)
        aspect = length / max(width, 1e-6)

        # Soft centered ranges rather than a hard car template.
        length_score = np.exp(-0.5 * ((np.log(max(length, 0.2)) - np.log(3.8)) / 0.85) ** 2)
        width_score = np.exp(-0.5 * ((np.log(max(width, 0.15)) - np.log(1.65)) / 0.75) ** 2)
        height_score = np.exp(-0.5 * ((height - 1.55) / 1.20) ** 2)
        aspect_score = np.exp(-0.5 * ((np.log(max(aspect, 0.2)) - np.log(2.2)) / 0.95) ** 2)
        density = num_points / max(area, 0.25)
        density_score = float(np.clip(density / 18.0, 0.0, 1.0))
        points_score = float(np.clip(num_points / 55.0, 0.0, 1.0))
        range_score = float(np.clip(1.0 - distance / 70.0, 0.20, 1.0))
        return float(np.clip(
            0.22 * length_score
            + 0.18 * width_score
            + 0.16 * height_score
            + 0.14 * aspect_score
            + 0.15 * density_score
            + 0.10 * points_score
            + 0.05 * range_score,
            0.0,
            1.0,
        ))

    @staticmethod
    def _center_distance_xy(a: Dict, b: Dict) -> float:
        pa = np.asarray(a["location"], dtype=np.float64)[:2]
        pb = np.asarray(b["location"], dtype=np.float64)[:2]
        return float(np.linalg.norm(pa - pb))

    def _suppress_duplicate_fragments(self, proposals: List[Dict]) -> List[Dict]:
        """Suppress nearby fragments of the same physical object.

        Sparse scans can split one vehicle into multiple disconnected BEV
        components.  Keeping all fragments created dozens of stable false
        tracks.  This NMS is measurement-only and uses no ground truth.
        """
        kept: List[Dict] = []
        for proposal in proposals:
            duplicate = False
            p_dims = proposal.get("dimensions", {})
            p_scale = 0.35 * max(float(p_dims.get("length", 1.0)), float(p_dims.get("width", 1.0)), 1.0)
            for other in kept:
                o_dims = other.get("dimensions", {})
                o_scale = 0.35 * max(float(o_dims.get("length", 1.0)), float(o_dims.get("width", 1.0)), 1.0)
                if self._center_distance_xy(proposal, other) <= max(0.75, p_scale, o_scale):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(proposal)
            if len(kept) >= self.max_proposals:
                break
        return kept

    def detect(self, lidar_points: np.ndarray) -> List[Dict]:
        pts = np.asarray(lidar_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return []

        xyz = pts[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        ranges = np.linalg.norm(xyz[:, :2], axis=1)
        mask = (
            finite
            & (ranges >= 1.5)
            & (ranges <= self.max_range)
            & (xyz[:, 2] >= self.min_height)
            & (xyz[:, 2] <= self.max_height)
        )
        xyz = xyz[mask]
        if len(xyz) == 0:
            return []

        non_ground = self._remove_local_ground(xyz)
        if len(non_ground) < self.min_points:
            return []

        vox = np.floor(non_ground[:, :2] / self.voxel_size).astype(np.int32)
        voxel_points: Dict[Tuple[int, int], List[int]] = {}
        for idx, key_arr in enumerate(vox):
            key = (int(key_arr[0]), int(key_arr[1]))
            voxel_points.setdefault(key, []).append(idx)

        occupied = set(voxel_points)
        visited = set()
        proposals = []
        max_component_voxels = 240

        for seed in list(occupied):
            if seed in visited:
                continue
            queue = deque([seed])
            visited.add(seed)
            component = []
            oversized = False
            while queue:
                key = queue.popleft()
                component.append(key)
                if len(component) > max_component_voxels:
                    oversized = True
                    break
                for nb in self._neighbors2(key):
                    if nb in occupied and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if oversized:
                continue

            ids = []
            for key in component:
                ids.extend(voxel_points[key])
            if len(ids) < self.min_points or len(ids) > self.max_points:
                continue
            cpts = non_ground[np.asarray(ids, dtype=np.int64)]
            center, length, width, height, yaw = self._oriented_extent(cpts)

            # Traffic-object geometry gates.  These cover motorcycles through
            # medium trucks but reject the vast majority of poles, kerbs, wall
            # pieces and large vegetation clusters.
            if not (1.00 <= length <= 9.5):
                continue
            if not (0.45 <= width <= 3.4):
                continue
            if not (0.55 <= height <= 3.8):
                continue
            area = max(length * width, 1e-3)
            if not (0.65 <= area <= 28.0):
                continue
            aspect = length / max(width, 1e-6)
            if not (1.05 <= aspect <= 6.5):
                continue

            distance = float(np.linalg.norm(center[:2]))
            vehicle_score = self._vehicle_likeness(length, width, height, len(cpts), distance)
            if vehicle_score < self.min_vehicle_score:
                continue
            score = vehicle_score

            pmin = np.min(cpts, axis=0)
            pmax = np.max(cpts, axis=0)
            proposals.append(
                {
                    "location": center,
                    "dimensions": {"length": length, "width": width, "height": height},
                    "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": yaw},
                    "type": "object",
                    "class": "object",
                    "score": score,
                    "vehicle_likeness": score,
                    "num_points": int(len(cpts)),
                    "min_corner": pmin,
                    "max_corner": pmax,
                }
            )

        # Keep the strongest compact proposals. This is bounded and deterministic.
        proposals.sort(
            key=lambda p: (
                -float(p.get("score", 0.0)),
                -int(p.get("num_points", 0)),
                float(np.linalg.norm(np.asarray(p["location"])[:2])),
            )
        )
        proposals = [p for p in proposals if float(p.get("score", 0.0)) >= self.min_vehicle_score]
        return self._suppress_duplicate_fragments(proposals)

    def add_camera_evidence(self, proposals: List[Dict], keypoints, image_shape) -> List[Dict]:
        """Fuse weak camera evidence without requiring a learned detector.

        The camera evidence is deliberately only a confidence modifier; rear
        and side objects outside the RGB FOV remain valid LiDAR-only targets.
        """
        if not proposals:
            return []
        pts = []
        if keypoints is not None:
            for kp in keypoints:
                if hasattr(kp, "pt"):
                    pts.append(kp.pt)
                else:
                    arr = np.asarray(kp, dtype=np.float64).reshape(-1)
                    if len(arr) >= 2:
                        pts.append((arr[0], arr[1]))
        pts = np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)
        h, w = int(image_shape[0]), int(image_shape[1])
        out = []
        for proposal in proposals:
            item = dict(proposal)
            bbox = item.get("bbox")
            if bbox is None:
                item["camera_support"] = 0
                item["camera_evidence"] = False
                # Outside the RGB FOV we cannot demand camera confirmation,
                # but LiDAR-only tracks must be substantially more convincing
                # than camera-supported proposals. This suppresses persistent
                # roadside fragments while preserving strong side/rear actors.
                geom = float(item.get("score", 0.0))
                dims = item.get("dimensions", {})
                width = float(dims.get("width", 0.0))
                height = float(dims.get("height", 0.0))
                points = int(item.get("num_points", 0))
                if geom < 0.68 or points < 28 or width < 0.60 or height < 0.70:
                    continue
                out.append(item)
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            area_px = bw * bh
            if len(pts):
                inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
                count = int(np.count_nonzero(inside))
            else:
                count = 0
            # Feature density is normalized so very large background boxes do
            # not win merely by covering much of the image.
            density = count / max(area_px / 1000.0, 1.0)
            support = float(np.clip(density / 2.0, 0.0, 1.0))
            geom = float(item.get("score", 0.0))
            item["camera_support"] = count
            item["camera_evidence"] = bool(count >= 2)
            item["score"] = float(np.clip(0.72 * geom + 0.28 * support, 0.0, 1.0))
            # Camera-visible geometry should normally have visual structure.
            # Keep unsupported boxes only when LiDAR shape is exceptionally
            # vehicle-like; this is still sensor-only and uses no GT labels.
            if count < 2 and geom < 0.82:
                continue
            # Reject huge projected fragments which usually correspond to
            # roadside/building surfaces rather than compact traffic actors.
            if area_px > 0.35 * max(w * h, 1):
                continue
            out.append(item)
        out = [p for p in out if float(p.get("score", 0.0)) >= max(self.min_vehicle_score, 0.50)]
        out.sort(key=lambda p: (-float(p.get("score", 0.0)), -int(p.get("num_points", 0))))
        return out[: self.max_proposals]

    def attach_camera_boxes(self, proposals: List[Dict], calibration, image_shape) -> List[Dict]:
        enriched = []
        for proposal in proposals:
            pmin = np.asarray(proposal["min_corner"], dtype=np.float64)
            pmax = np.asarray(proposal["max_corner"], dtype=np.float64)
            corners = np.array(
                [[x, y, z] for x in (pmin[0], pmax[0]) for y in (pmin[1], pmax[1]) for z in (pmin[2], pmax[2])],
                dtype=np.float64,
            )
            uv, valid = calibration.project_lidar(corners)
            valid = valid & np.isfinite(uv).all(axis=1)
            item = dict(proposal)
            if np.any(valid):
                good = uv[valid]
                x1, y1 = np.min(good, axis=0)
                x2, y2 = np.max(good, axis=0)
                h, w = int(image_shape[0]), int(image_shape[1])
                x1, x2 = np.clip([x1, x2], 0, max(0, w - 1))
                y1, y2 = np.clip([y1, y2], 0, max(0, h - 1))
                if x2 > x1 and y2 > y1:
                    item["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
            enriched.append(item)
        return enriched


def build_sensor_multimodal_objects(
    proposals: List[Dict],
    planar_features: List[Dict],
    keypoints,
    descriptors: Optional[np.ndarray],
    max_features_per_object: int = 48,
) -> List[Dict]:
    """Attach actual bounded LiDAR/camera features to GT-free proposals."""
    result = []
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    for idx, proposal in enumerate(proposals):
        pmin = np.asarray(proposal["min_corner"], dtype=np.float64)
        pmax = np.asarray(proposal["max_corner"], dtype=np.float64)
        lidar_selected = []
        for feat in planar_features:
            p = np.asarray(feat.get("point", feat.get("center", [0, 0, 0])), dtype=np.float64)
            if np.all(p >= pmin) and np.all(p <= pmax):
                lidar_selected.append(
                    {
                        "point": p,
                        "normal": np.asarray(feat.get("normal", [0, 0, 1]), dtype=np.float64),
                        "planarity": float(feat.get("planarity", 0.0)),
                        "surface_variation": float(feat.get("surface_variation", 0.0)),
                    }
                )
                if len(lidar_selected) >= max_features_per_object:
                    break

        camera_selected = []
        bbox = proposal.get("bbox")
        if bbox is not None and keypoints is not None:
            x1, y1, x2, y2 = bbox
            for kp, desc in zip(keypoints, descriptors):
                x, y = kp.pt if hasattr(kp, "pt") else kp
                if x1 <= x <= x2 and y1 <= y <= y2:
                    camera_selected.append(
                        {
                            "keypoint": np.asarray([x, y], dtype=np.float64),
                            "descriptor": np.asarray(desc, dtype=np.uint8),
                        }
                    )
                    if len(camera_selected) >= max_features_per_object:
                        break

        result.append(
            {
                "proposal_id": int(idx),
                "type": proposal.get("type", "object"),
                "score": float(proposal.get("score", 0.0)),
                "location": np.asarray(proposal["location"], dtype=np.float64),
                "dimensions": proposal["dimensions"],
                "rotation": proposal.get("rotation", {}),
                "projected_bbox": bbox,
                "lidar_features": {"features": lidar_selected, "count": len(lidar_selected)},
                "camera_features": camera_selected,
            }
        )
    return result
