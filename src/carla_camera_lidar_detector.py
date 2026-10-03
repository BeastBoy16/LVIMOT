from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class CameraLiDARDetectionConfig:
    model_path: str = "yolov8n.pt"
    confidence: float = 0.12
    iou: float = 0.45
    image_size: int = 1280
    max_detections: int = 12
    min_lidar_points: int = 4
    max_range: float = 55.0
    device: str = "auto"
    half: Optional[bool] = None
    min_eval_bbox_width_px: float = 8.0
    min_eval_bbox_height_px: float = 8.0
    min_eval_bbox_area_px: float = 100.0


class CarlaCameraLiDARObjectDetector:
    """GT-free camera + LiDAR traffic-object detector.

    A pretrained COCO detector supplies *semantic 2-D boxes* from the RGB
    camera.  Raw LiDAR points are projected into each box and a robust near
    depth cluster provides the 3-D measurement used by MOT.  No CARLA actor
    pose, actor ID, semantic LiDAR, or simulator ground truth is consumed.

    The Ultralytics dependency is imported lazily so the rest of LVIMOT and
    the unit tests do not require it.  A test double may be passed as ``model``.
    """

    # COCO vehicle-like classes used by yolov8n/YOLO11 COCO weights.
    COCO_CLASS_MAP = {
        1: "bicycle",
        2: "vehicle",      # car
        3: "motorcycle",
        5: "vehicle",      # bus
        7: "vehicle",      # truck
    }

    def __init__(
        self,
        calibration,
        config: Optional[CameraLiDARDetectionConfig] = None,
        model=None,
    ):
        self.calibration = calibration
        self.config = config or CameraLiDARDetectionConfig()
        self._model = model
        self._model_load_error: Optional[Exception] = None
        self.last_camera_detections: List[Dict] = []
        self.last_diagnostics: Dict = {}

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._model_load_error is not None:
            raise RuntimeError(self._dependency_error_message()) from self._model_load_error
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.config.model_path)
            return self._model
        except Exception as exc:  # import error, missing model, download failure, etc.
            self._model_load_error = exc
            raise RuntimeError(self._dependency_error_message()) from exc

    def _dependency_error_message(self) -> str:
        return (
            "The GT-free MOT detector requires Ultralytics YOLO. Install it with: "
            "python -m pip install ultralytics ; then rerun. "
            f"Model: {self.config.model_path!r}. The model is downloaded once by Ultralytics "
            "if it is not already cached."
        )

    @staticmethod
    def _as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _resolve_runtime(self):
        requested = str(self.config.device).strip().lower()
        cuda_available = False
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        if requested in {"", "auto", "none"}:
            device = 0 if cuda_available else "cpu"
        elif requested in {"cuda", "cuda:0", "gpu", "0"}:
            device = 0 if cuda_available else "cpu"
        else:
            device = self.config.device

        if self.config.half is None:
            half = bool(cuda_available and device != "cpu")
        else:
            half = bool(self.config.half and cuda_available and device != "cpu")
        return device, half, cuda_available

    def _infer_boxes(self, image: np.ndarray) -> List[Dict]:
        model = self._ensure_model()
        device, half, cuda_available = self._resolve_runtime()
        kwargs = {
            "source": image,
            "conf": float(self.config.confidence),
            "iou": float(self.config.iou),
            "imgsz": int(self.config.image_size),
            "verbose": False,
            "device": device,
            "half": half,
        }
        self.last_diagnostics.update({
            "inference_device": str(device),
            "cuda_available": bool(cuda_available),
            "fp16": bool(half),
        })
        results = model.predict(**kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = self._as_numpy(getattr(boxes, "xyxy", None))
        cls = self._as_numpy(getattr(boxes, "cls", None))
        conf = self._as_numpy(getattr(boxes, "conf", None))
        if xyxy is None or cls is None or conf is None:
            return []
        xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
        cls = np.asarray(cls, dtype=np.int64).reshape(-1)
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)

        detections = []
        for bbox, class_id, score in zip(xyxy, cls, conf):
            mapped = self.COCO_CLASS_MAP.get(int(class_id))
            if mapped is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(class_id),
                    "class": mapped,
                    "type": mapped,
                    "camera_confidence": float(score),
                }
            )
        detections.sort(key=lambda d: -d["camera_confidence"])
        return detections[: int(self.config.max_detections)]

    @staticmethod
    def _depth_cluster(depths: np.ndarray, indices: np.ndarray, bin_size: float = 0.75) -> np.ndarray:
        """Return indices belonging to the most plausible foreground depth mode."""
        if len(indices) == 0:
            return indices
        d = np.asarray(depths[indices], dtype=np.float64)
        finite = np.isfinite(d) & (d > 0.1)
        if not np.any(finite):
            return np.empty(0, dtype=np.int64)
        indices = indices[finite]
        d = d[finite]
        if len(indices) <= 4:
            return indices

        bins = np.floor(d / float(bin_size)).astype(np.int32)
        unique, counts = np.unique(bins, return_counts=True)
        # Favor dense *nearer* modes. This suppresses wall/building points
        # behind a vehicle that happen to fall inside its 2-D box.
        centers = (unique.astype(np.float64) + 0.5) * float(bin_size)
        score = counts.astype(np.float64) / np.sqrt(np.maximum(centers, 1.0))
        # Discard one-point modes unless there is nothing better.
        score = np.where(counts >= 2, score, score * 0.15)
        best_bin = int(unique[int(np.argmax(score))])
        best_center = (best_bin + 0.5) * float(bin_size)
        # Keep a physically useful depth thickness around the visible surface.
        keep = np.abs(d - best_center) <= 2.25
        chosen = indices[keep]
        if len(chosen) < 3:
            order = np.argsort(d)
            chosen = indices[order[: min(8, len(order))]]
        return chosen

    @staticmethod
    def _robust_box(points: np.ndarray):
        pts = np.asarray(points, dtype=np.float64)
        center = np.median(pts, axis=0)
        p05 = np.percentile(pts, 5.0, axis=0)
        p95 = np.percentile(pts, 95.0, axis=0)
        span = np.maximum(p95 - p05, np.array([0.3, 0.3, 0.3]))
        # Raw CARLA LiDAR axes are X forward, Y right, Z up.  Partial visible
        # surfaces under-estimate vehicle extent, so only use dimensions for
        # tracker shape consistency, not for evaluation.
        length = float(max(span[0], span[1]))
        width = float(min(span[0], span[1]))
        height = float(span[2])
        return center, length, width, height, p05, p95

    @staticmethod
    def _bbox_iou(a, b) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if len(a) != 4 or len(b) != 4:
            return 0.0
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0.0 else 0.0

    def _suppress_duplicate_proposals(self, proposals: List[Dict]) -> List[Dict]:
        ranked = sorted(
            proposals,
            key=lambda p: (float(p.get("score", 0.0)), int(p.get("camera_support", 0))),
            reverse=True,
        )
        kept: List[Dict] = []
        for proposal in ranked:
            p = np.asarray(proposal.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
            duplicate = False
            for other in kept:
                q = np.asarray(other.get("location", [0.0, 0.0, 0.0]), dtype=np.float64)
                if self._bbox_iou(proposal.get("bbox"), other.get("bbox")) >= 0.80 and float(np.linalg.norm(p - q)) <= 4.0:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(proposal)
        return kept

    def detect(self, image: np.ndarray, lidar_points: np.ndarray) -> List[Dict]:
        self.last_diagnostics = {
            "camera_detections": 0,
            "max_range_m": float(self.config.max_range),
            "raw_lidar_points": int(len(lidar_points)) if hasattr(lidar_points, "__len__") else 0,
            "projection_valid_points": 0,
            "points_within_range": 0,
            "proposals": 0,
        }
        detections = self._infer_boxes(image)
        self.last_camera_detections = [dict(d) for d in detections]
        self.last_diagnostics["camera_detections"] = int(len(detections))
        if not detections:
            return []

        pts = np.asarray(lidar_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
            return []
        xyz = pts[:, :3]
        finite_xyz = np.isfinite(xyz).all(axis=1)

        camera_xyz = self.calibration.lidar_to_opencv(xyz)
        depth = camera_xyz[:, 2]
        uv, valid_projection = self.calibration.project_lidar(xyz)
        valid_projection_mask = finite_xyz & valid_projection & np.isfinite(uv).all(axis=1)
        self.last_diagnostics["projection_valid_points"] = int(np.count_nonzero(valid_projection_mask))
        valid = valid_projection_mask & (depth > 0.1) & (depth <= float(self.config.max_range))
        self.last_diagnostics["points_within_range"] = int(np.count_nonzero(valid))

        proposals: List[Dict] = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)

            # Use a central region to estimate foreground depth and avoid road
            # pixels at the very bottom of a 2-D box.
            cx1 = x1 + 0.08 * bw
            cx2 = x2 - 0.08 * bw
            cy1 = y1 + 0.08 * bh
            cy2 = y2 - 0.18 * bh
            central = (
                valid
                & (uv[:, 0] >= cx1)
                & (uv[:, 0] <= cx2)
                & (uv[:, 1] >= cy1)
                & (uv[:, 1] <= cy2)
            )
            idx = np.flatnonzero(central)
            if len(idx) < self.config.min_lidar_points:
                full = (
                    valid
                    & (uv[:, 0] >= x1)
                    & (uv[:, 0] <= x2)
                    & (uv[:, 1] >= y1)
                    & (uv[:, 1] <= y2)
                )
                idx = np.flatnonzero(full)
            if len(idx) < self.config.min_lidar_points:
                continue

            selected_idx = self._depth_cluster(depth, idx)
            if len(selected_idx) < self.config.min_lidar_points:
                continue
            selected = xyz[selected_idx]
            center, length, width, height, pmin, pmax = self._robust_box(selected)
            cls_name = det["class"]
            if cls_name == "motorcycle":
                length = float(np.clip(max(length, 1.6), 1.6, 3.2))
                width = float(np.clip(max(width, 0.55), 0.55, 1.2))
                height = float(np.clip(max(height, 1.0), 1.0, 2.2))
            elif cls_name == "bicycle":
                length = float(np.clip(max(length, 1.4), 1.4, 2.4))
                width = float(np.clip(max(width, 0.45), 0.45, 1.0))
                height = float(np.clip(max(height, 1.0), 1.0, 2.0))
            else:
                # Camera semantics make the dimensions a tracker stabilizer rather
                # than an object classifier.  Prevent sparse visible surfaces from
                # producing wildly different sizes on successive frames.
                length = float(np.clip(max(length, 3.2), 3.2, 9.0))
                width = float(np.clip(max(width, 1.4), 1.4, 3.2))
                height = float(np.clip(max(height, 1.2), 1.2, 3.8))

            r_xy = float(np.linalg.norm(center[:2]))
            if not np.isfinite(r_xy) or r_xy < 1.0 or r_xy > float(self.config.max_range):
                continue

            point_support = float(np.clip(len(selected_idx) / 18.0, 0.0, 1.0))
            # V19 recall repair: distant CARLA vehicles can have weak COCO
            # confidence while still carrying strong, geometrically coherent
            # LiDAR support.  Let LiDAR contribute enough to preserve those
            # candidates; temporal confirmation in the tracker suppresses noise.
            score = float(np.clip(0.55 * det["camera_confidence"] + 0.45 * point_support, 0.0, 1.0))
            proposals.append(
                {
                    "location": center,
                    "dimensions": {"length": length, "width": width, "height": height},
                    "rotation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "type": det["type"],
                    "class": det["class"],
                    "score": score,
                    "camera_confidence": float(det["camera_confidence"]),
                    "camera_evidence": True,
                    "camera_support": int(len(selected_idx)),
                    "num_points": int(len(selected_idx)),
                    "bbox": list(det["bbox"]),
                    "min_corner": pmin,
                    "max_corner": pmax,
                }
            )

        proposals = self._suppress_duplicate_proposals(proposals)
        proposals.sort(key=lambda p: -float(p.get("score", 0.0)))
        proposals = proposals[: int(self.config.max_detections)]
        self.last_diagnostics["proposals"] = int(len(proposals))
        return proposals
