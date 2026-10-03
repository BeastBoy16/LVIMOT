import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, "src")

from carla_visualizer import CarlaLVIMOTVisualizer
from carla_reader import CarlaReader


def visualize_frame_from_json(json_path: str, visualizer: CarlaLVIMOTVisualizer, carla_root: str = None, output_file: str = None) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)

    frame_id = data.get("frame", 0)
    timestamp = data.get("timestamp", 0.0)
    seq = str(data.get("sequence", "0006")).zfill(4)

    # Load camera image if CARLA dataset root is provided and exists
    image_data = None
    if carla_root and os.path.isdir(os.path.expanduser(carla_root)):
        try:
            reader = CarlaReader(carla_root, seq)
            image_data = reader.load_image(frame_id)
        except Exception:
            image_data = None

    # Extract tracks
    tracking_data = data.get("tracking", {})
    tracks = tracking_data.get("active_tracks", [])

    # If no tracking section in json, extract from multimodal_objects
    if not tracks:
        mm_objs = data.get("multimodal_objects", [])
        for i, obj in enumerate(mm_objs):
            tracks.append({
                "track_id": obj.get("id", i),
                "class": obj.get("type", "vehicle"),
                "position": obj.get("location", [0, 0, 0]),
                "velocity": obj.get("velocity", [0, 0, 0]),
                "speed": float(np.linalg.norm(obj.get("velocity", [0, 0, 0]))),
                "motion_state": "DYNAMIC" if float(np.linalg.norm(obj.get("velocity", [0, 0, 0]))) > 1.0 else "STATIC",
                "dimensions": obj.get("dimensions", {"length": 4.5, "width": 1.8}),
                "rotation": obj.get("rotation", {"yaw": 0.0}),
                "box_2d": obj.get("projected_box", obj.get("box_2d"))
            })

    # Planar features
    planar_features = data.get("temporal_lidar", {}).get("active_planar_landmarks", [])
    imu_motion = data.get("imu", {}).get("motion", {})
    map_summary = data.get("mapping_4d", {})

    out_path = visualizer.render_frame(
        frame_id=frame_id,
        timestamp=timestamp,
        image_data=image_data,
        tracks=tracks,
        planar_features=planar_features,
        imu_motion=imu_motion,
        map_summary=map_summary,
        save_path=output_file
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LVIMOT Output Visualization Tool")
    parser.add_argument("--results-dir", default="outputs_carla/sequence_0006", help="Path to output sequence JSON directory")
    parser.add_argument("--carla-root", default="~/catkin_ws/src/CARLA_LVIMOT", help="Path to raw CARLA dataset root (for camera images)")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame number to visualize (if None, visualizes sequence)")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum frames to render when processing sequence")
    parser.add_argument("--output-dir", default="visualizations", help="Directory to save rendered visualization images")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demonstration visualization")

    args = parser.parse_args()
    visualizer = CarlaLVIMOTVisualizer(output_dir=args.output_dir)

    if args.demo:
        print("\n--- Running LVIMOT Demonstration Visualizer ---")
        # Synthesize sample multi-object frame
        sample_tracks = [
            {
                "track_id": 1,
                "class": "vehicle",
                "position": [15.0, -2.5, 0.0],
                "velocity": [14.5, 0.0, 0.0],
                "speed": 14.5,
                "motion_state": "DYNAMIC",
                "dimensions": {"length": 4.8, "width": 2.0},
                "rotation": {"yaw": 0.0},
                "box_2d": [450, 160, 680, 290]
            },
            {
                "track_id": 2,
                "class": "vehicle",
                "position": [28.0, 4.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "speed": 0.0,
                "motion_state": "STATIC",
                "dimensions": {"length": 4.2, "width": 1.8},
                "rotation": {"yaw": 5.0},
                "box_2d": [780, 180, 920, 270]
            }
        ]
        sample_lidar = np.random.uniform(-20, 40, size=(1000, 3))
        sample_lidar[:, 1] = np.random.uniform(-15, 15, size=1000)
        sample_lidar[:, 2] = np.random.uniform(-1, 2, size=1000)

        out_img = visualizer.render_frame(
            frame_id=1,
            timestamp=0.1,
            tracks=sample_tracks,
            lidar_points=sample_lidar,
            imu_motion={"velocity": [10.0, 0.0, 0.0]},
            map_summary={"total_static_voxels": 4200, "total_dynamic_actors": 2},
            save_path=os.path.join(args.output_dir, "demo_visualization.png")
        )
        print(f"Demonstration rendered successfully: {out_img}")
        return

    # Check json frames
    pattern = os.path.join(args.results_dir, "frame_*.json")
    json_files = sorted(glob.glob(pattern))

    if not json_files:
        print(f"No output JSON frames found in: {args.results_dir}")
        print("Tip: Run  or ")
        return

    if args.frame is not None:
        target_file = os.path.join(args.results_dir, f"frame_{args.frame:06d}.json")
        if not os.path.exists(target_file):
            print(f"Frame file not found: {target_file}")
            return
        out_file = visualize_frame_from_json(target_file, visualizer, carla_root=args.carla_root)
        print(f"Rendered frame {args.frame} -> {out_file}")
    else:
        print(f"Rendering {min(len(json_files), args.max_frames)} frames from {args.results_dir}...")
        for i, jfile in enumerate(json_files[:args.max_frames]):
            out_file = visualize_frame_from_json(jfile, visualizer, carla_root=args.carla_root)
            if i % 10 == 0 or i == len(json_files[:args.max_frames]) - 1:
                print(f"Rendered frame [{i+1}/{min(len(json_files), args.max_frames)}]: {out_file}")

    print(f"\nVisualization complete! Images saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
