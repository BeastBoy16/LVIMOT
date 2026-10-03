import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_pipeline_core import CarlaLVIMOTCore
from carla_evaluation import CarlaEvaluator


def _nearest_bev(proposals, gt_objects):
    if not proposals or not gt_objects:
        return None
    P = np.asarray([p["location"] for p in proposals], dtype=np.float64)
    G = np.asarray([g["location"] for g in gt_objects], dtype=np.float64)
    D = np.linalg.norm(P[:, None, :2] - G[None, :, :2], axis=2)
    return {
        "min_pair": float(np.min(D)),
        "mean_proposal_to_gt": float(np.mean(np.min(D, axis=1))),
        "mean_gt_to_proposal": float(np.mean(np.min(D, axis=0))),
    }


def _best_iou(proposals, gt_objects):
    best = None
    for pi, p in enumerate(proposals):
        for gi, g in enumerate(gt_objects):
            iou = CarlaEvaluator._bbox_iou(p.get("bbox"), g.get("bbox"))
            if iou is None:
                continue
            item = (float(iou), pi, gi)
            if best is None or item[0] > best[0]:
                best = item
    return best


def _save_overlay(image, raw_yolo, projected_gt, proposals, path):
    canvas = image.copy()
    for actor_id, box in projected_gt.items():
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(canvas, f"GT {actor_id}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    for i, d in enumerate(raw_yolo):
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(canvas, f"YOLO {i} {d.get('camera_confidence', 0):.2f}", (x1, min(canvas.shape[0]-4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    for i, p in enumerate(proposals):
        box = p.get("bbox")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(canvas, f"P{i}", (x1, min(canvas.shape[0]-4, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, canvas)


def main():
    ap = argparse.ArgumentParser(description="One-frame GT-free MOT detector/evaluation preflight")
    ap.add_argument("--carla-root", required=True)
    ap.add_argument("--sequence", default="0006")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--yolo-model", default="yolov8n.pt")
    ap.add_argument("--yolo-confidence", type=float, default=0.12)
    ap.add_argument("--yolo-image-size", type=int, default=1280)
    ap.add_argument("--yolo-device", default="auto")
    ap.add_argument("--overlay", default=None, help="Optional output path for GT/YOLO diagnostic overlay")
    args = ap.parse_args()

    reader = CarlaReader(args.carla_root, args.sequence)
    calibration = CarlaCalibration(os.path.join(args.carla_root, "calibration", "carla_calibration.json"))
    gt_detector = CarlaDetector(reader, calibration)
    core = CarlaLVIMOTCore(
        calibration,
        detector_mode="camera_lidar",
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_device=args.yolo_device,
    )

    image = reader.load_image(args.frame)
    lidar = reader.load_lidar(args.frame)
    gt_pose = reader.load_pose(args.frame)
    gt_objects = gt_detector.load_detections(args.frame)

    proposals = core.camera_lidar_detector.detect(image, lidar)
    projected_gt = core._project_gt_camera_boxes(gt_objects, gt_pose, image.shape)
    gt_lidar = core._gt_objects_current_lidar(gt_objects, gt_pose)
    gt_support = core._gt_lidar_support_by_bbox(projected_gt, lidar)
    gt_visible = core._filter_gt_observable_current_lidar(gt_lidar, projected_gt, lidar)
    raw_camera_detections = list(getattr(core.camera_lidar_detector, "last_camera_detections", []))
    detector_diag = dict(getattr(core.camera_lidar_detector, "last_diagnostics", {}))

    print("=" * 78)
    print("LVIMOT MOT PREFLIGHT V8")
    print("=" * 78)
    print(f"Sequence/frame: {str(args.sequence).zfill(4)} / {args.frame}")
    print(f"YOLO model/input/conf: {args.yolo_model} / {args.yolo_image_size} / {args.yolo_confidence:.2f}")
    print(f"Calibration LiDAR range: {float(calibration.lidar_range):.2f} m")
    print(f"Effective detector/evaluation range: {float(core.object_sensor_range):.2f} m")
    print(f"Raw YOLO vehicle detections: {len(raw_camera_detections)}")
    print(f"YOLO+LiDAR proposals: {len(proposals)}")
    print(f"GT actors raw: {len(gt_objects)}")
    print(f"GT vehicle boxes intersecting RGB image: {len(projected_gt)}")
    print(f"GT actors eligible after pixel-size + LiDAR-support gates: {len(gt_visible)}")
    print(
        "LiDAR projection diagnostics: "
        f"valid={detector_diag.get('projection_valid_points', 0)} | "
        f"within_range={detector_diag.get('points_within_range', 0)}"
    )
    print()

    if raw_camera_detections:
        print("Raw YOLO vehicle boxes:")
        for i, d in enumerate(raw_camera_detections[:20]):
            print(
                f"  Y{i}: class={d.get('class')} conf={float(d.get('camera_confidence', 0.0)):.3f} "
                f"bbox={np.round(d.get('bbox', []), 1).tolist()}"
            )
    if proposals:
        print("Proposal LiDAR centers [x,y,z] m:")
        for i, p in enumerate(proposals[:12]):
            loc = np.asarray(p["location"], dtype=np.float64)
            print(
                f"  P{i}: {loc.round(3).tolist()} class={p.get('class')} "
                f"score={p.get('score', 0):.3f} bbox={np.round(p.get('bbox', []), 1).tolist()}"
            )
    if gt_lidar:
        print("GT vehicle centres transformed to LiDAR frame (evaluation only):")
        visible_ids = {int(g.get("actor_id", -1)) for g in gt_visible}
        for i, g in enumerate(gt_lidar[:20]):
            loc = np.asarray(g["location"], dtype=np.float64)
            aid = int(g.get("actor_id", -1))
            r = float(np.linalg.norm(loc[:2]))
            bbox = projected_gt.get(aid)
            if bbox is None:
                dims = "N/A"
            else:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                dims = f"{x2-x1:.1f}x{y2-y1:.1f}px area={(x2-x1)*(y2-y1):.0f}"
            print(
                f"  G{i}: actor={aid} {loc.round(3).tolist()} range={r:.2f}m "
                f"rgb_box={'yes' if bbox is not None else 'no'} box={dims} "
                f"lidar_support={int(gt_support.get(aid, 0))} "
                f"eligible={'yes' if aid in visible_ids else 'no'}"
            )
    print()

    nearest = _nearest_bev(proposals, gt_visible)
    if nearest is None:
        print("3-D BEV comparison: N/A")
    else:
        print(
            "3-D BEV comparison: "
            f"min={nearest['min_pair']:.3f}m | "
            f"mean proposal->GT={nearest['mean_proposal_to_gt']:.3f}m | "
            f"mean GT->proposal={nearest['mean_gt_to_proposal']:.3f}m"
        )

    best = _best_iou(proposals, gt_visible)
    if best is None:
        print("Camera-box comparison: N/A")
    else:
        iou, pi, gi = best
        print(f"Camera-box comparison: best IoU={iou:.3f} (P{pi} vs eligible G{gi})")

    overlay = args.overlay
    if overlay is None:
        overlay = os.path.join("outputs_carla", "preflight", f"sequence_{str(args.sequence).zfill(4)}_frame_{args.frame:06d}.png")
    _save_overlay(image, raw_camera_detections, projected_gt, proposals, overlay)
    print(f"Overlay: {overlay}")

    print()
    if len(projected_gt) == 0:
        print("PREFLIGHT: FAIL - GT box projection produced zero camera-visible vehicles.")
    elif len(gt_visible) == 0:
        print("PREFLIGHT: FAIL - no GT vehicle is sufficiently observable by this RGB+LiDAR detector.")
        print("Use mot_scan.py to select a MOT-oriented sequence/window instead of running a full sequence.")
    elif len(proposals) == 0:
        print("PREFLIGHT: FAIL - observable GT exists but the RGB+LiDAR detector found no proposal.")
    elif best is not None and best[0] >= 0.10:
        if nearest is not None and nearest["min_pair"] <= 8.0:
            print("PREFLIGHT: PASS - camera and 3-D detector/GT geometry are plausibly aligned.")
        else:
            print("PREFLIGHT: PARTIAL PASS - camera association works, but 3-D geometry still needs attention.")
    else:
        print("PREFLIGHT: FAIL - detector has no overlap with the eligible labelled vehicles in this frame.")
        print("This is detector recall/domain coverage, not another coordinate-transform failure.")
        print("Use mot_scan.py to select a MOT-oriented sequence/window before changing localization code.")
    print("=" * 78)


if __name__ == "__main__":
    main()
