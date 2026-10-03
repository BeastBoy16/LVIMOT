"""Fast MOT validation-window selector.

This script does NOT run the factor graph, mapping, IMU integration, or full
LVIMOT pipeline.  It samples camera/LiDAR frames and uses CARLA labels only to
select a useful *evaluation window*.  Labels never enter the detector/tracker.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carla_reader import CarlaReader
from carla_calibration import CarlaCalibration
from carla_detection import CarlaDetector
from carla_pipeline_core import CarlaLVIMOTCore
from carla_evaluation import CarlaEvaluator


def best_iou(proposals, gt):
    best = 0.0
    for p in proposals:
        for g in gt:
            iou = CarlaEvaluator._bbox_iou(p.get("bbox"), g.get("bbox"))
            if iou is not None:
                best = max(best, float(iou))
    return best


def min_bev(proposals, gt):
    if not proposals or not gt:
        return None
    P = np.asarray([p["location"] for p in proposals], dtype=np.float64)
    G = np.asarray([g["location"] for g in gt], dtype=np.float64)
    D = np.linalg.norm(P[:, None, :2] - G[None, :, :2], axis=2)
    return float(np.min(D))


def main():
    ap = argparse.ArgumentParser(description="Quickly select a useful offline MOT validation window")
    ap.add_argument("--carla-root", required=True)
    ap.add_argument("--sequences", nargs="+", default=["0002", "0004", "0005"])
    ap.add_argument("--samples-per-sequence", type=int, default=4)
    ap.add_argument("--yolo-model", default="yolov8n.pt")
    ap.add_argument("--yolo-confidence", type=float, default=0.12)
    ap.add_argument("--yolo-image-size", type=int, default=1280)
    ap.add_argument("--yolo-device", default="auto")
    args = ap.parse_args()

    calibration = CarlaCalibration(os.path.join(args.carla_root, "calibration", "carla_calibration.json"))
    core = CarlaLVIMOTCore(
        calibration,
        detector_mode="camera_lidar",
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        yolo_device=args.yolo_device,
    )

    rows = []
    for sequence in args.sequences:
        reader = CarlaReader(args.carla_root, sequence)
        gt_detector = CarlaDetector(reader, calibration)
        n = reader.num_frames()
        k = max(1, min(int(args.samples_per_sequence), n))
        frames = np.unique(np.linspace(0, n - 1, num=k, dtype=int))
        for frame in frames:
            image = reader.load_image(int(frame))
            lidar = reader.load_lidar(int(frame))
            pose = reader.load_pose(int(frame))
            gt_objects = gt_detector.load_detections(int(frame))
            proposals = core.camera_lidar_detector.detect(image, lidar)
            projected = core._project_gt_camera_boxes(gt_objects, pose, image.shape)
            gt_lidar = core._gt_objects_current_lidar(gt_objects, pose)
            eligible = core._filter_gt_observable_current_lidar(gt_lidar, projected, lidar)
            iou = best_iou(proposals, eligible)
            dist = min_bev(proposals, eligible)
            # Prefer windows with multiple observable GT actors and at least one
            # detector/GT association.  This is only validation-set selection.
            score = (5.0 if iou >= 0.10 else 0.0) + 2.0 * min(len(eligible), 4) + min(len(proposals), 4) + 3.0 * iou
            rows.append({
                "sequence": str(sequence).zfill(4),
                "frame": int(frame),
                "frames": int(n),
                "eligible": int(len(eligible)),
                "proposals": int(len(proposals)),
                "best_iou": float(iou),
                "min_bev": dist,
                "score": float(score),
            })
            print(
                f"seq {str(sequence).zfill(4)} frame {int(frame):4d}: "
                f"eligibleGT={len(eligible):2d} proposals={len(proposals):2d} "
                f"bestIoU={iou:.3f} minBEV={'N/A' if dist is None else f'{dist:.2f}m'}"
            )

    rows.sort(key=lambda x: (-x["score"], -x["best_iou"], -x["eligible"], x["frame"]))
    print("\n" + "=" * 78)
    print("TOP MOT VALIDATION CANDIDATES")
    for row in rows[:10]:
        min_bev_text = "N/A" if row["min_bev"] is None else f"{row['min_bev']:.2f}m"
        print(
            f"  seq {row['sequence']} frame {row['frame']:4d} | "
            f"eligibleGT={row['eligible']} proposals={row['proposals']} "
            f"bestIoU={row['best_iou']:.3f} minBEV={min_bev_text}"
        )

    passing = [r for r in rows if r["eligible"] > 0 and r["proposals"] > 0 and r["best_iou"] >= 0.10]
    if not passing:
        print("\nSCAN RESULT: no sampled frame passed detector/GT overlap.")
        print("Do not run a 20-frame MOT benchmark yet. Increase model capacity or fine-tune the detector.")
        return 2

    best = passing[0]
    start = max(0, best["frame"] - 5)
    end = min(best["frames"] - 1, start + 19)
    start = max(0, end - 19)
    print("\nSCAN RESULT: PASS")
    print(f"Recommended MOT window: sequence {best['sequence']} frames {start}..{end}")
    print("Run:")
    print(
        f'{sys.executable} src/carla_main.py --carla-root "{args.carla_root}" '
        f'--sequence {best["sequence"]} --start-frame {start} --end-frame {end} '
        f'--yolo-confidence {args.yolo_confidence} --yolo-image-size {args.yolo_image_size}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
