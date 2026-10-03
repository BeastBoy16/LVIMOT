import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless / server environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Union
from PIL import Image


class CarlaLVIMOTVisualizer:
    """
    Visualization Engine for LVIMOT.
    
    Generates synchronized multi-panel visualizations:
      1. Camera View: RGB image with 2D projected bounding boxes, track IDs,
         speed estimates, and motion classifications (DYNAMIC vs STATIC).
      2. Bird's-Eye View (BEV): Top-down spatial map showing ego vehicle position,
         LiDAR points/planar features, 3D object bounding boxes, and velocity vectors.
      3. Telemetry & State Dashboard: Real-time speed, acceleration, active tracks,
         sliding-window optimizer status, and 4D map statistics.
    """

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _draw_box_bev(ax, center_x, center_y, length, width, yaw_deg, color, label=None):
        yaw_rad = np.deg2rad(yaw_deg)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Local corners in vehicle frame (X=forward, Y=right)
        hl, hw = length / 2.0, width / 2.0
        corners_local = np.array([
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
            [-hl, -hw]
        ])
        
        # In BEV plot: X axis = Left/Right (Y in CARLA), Y axis = Forward (X in CARLA)
        # CARLA: X is forward, Y is right. Let BEV X = Y (lateral), BEV Y = X (longitudinal)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        rot_corners = (R @ corners_local.T).T
        
        bev_x = center_y + rot_corners[:, 1]
        bev_y = center_x + rot_corners[:, 0]
        
        poly = plt.Polygon(np.column_stack([bev_x, bev_y]), closed=True, edgecolor=color, facecolor=color, alpha=0.35, linewidth=2)
        ax.add_patch(poly)
        
        # Heading arrow
        arrow_len = hl * 0.8
        ax.arrow(center_y, center_x, arrow_len * sin_y, arrow_len * cos_y, head_width=0.6, head_length=0.8, fc=color, ec=color)
        
        if label:
            ax.text(center_y, center_x + hl + 0.8, label, color=color, fontsize=8, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec=color, alpha=0.7))

    def render_frame(
        self,
        frame_id: int,
        timestamp: float,
        image_data: Optional[np.ndarray] = None,
        tracks: Optional[List[Dict]] = None,
        lidar_points: Optional[np.ndarray] = None,
        planar_features: Optional[List[Dict]] = None,
        ego_pose: Optional[Dict] = None,
        imu_motion: Optional[Dict] = None,
        map_summary: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> str:
        tracks = tracks or []
        
        fig = plt.figure(figsize=(16, 10), facecolor="#121214")
        gs = GridSpec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.2, 0.8], figure=fig, hspace=0.25, wspace=0.2)

        # -----------------------------------------------------------------
        # Panel 1: Camera View (Top Full Width or Top-Left)
        # -----------------------------------------------------------------
        ax_cam = fig.add_subplot(gs[0, :])
        ax_cam.set_facecolor("#18181b")
        
        if image_data is not None:
            # Handle RGB vs BGR
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                # Assuming input is BGR from OpenCV / pipeline
                display_img = image_data[:, :, ::-1]
            else:
                display_img = image_data
            ax_cam.imshow(display_img)
        else:
            # Synthetic placeholder canvas
            ax_cam.set_xlim(0, 1242)
            ax_cam.set_ylim(375, 0)
            ax_cam.text(621, 187, "[Front Camera Stream]", color="#71717a", fontsize=14, ha="center")

        # Draw 2D projected bounding boxes on camera view
        for trk in tracks:
            box_2d = trk.get("box_2d", trk.get("bbox"))
            motion_state = trk.get("motion_state", "DYNAMIC")
            color = "#22c55e" if motion_state == "DYNAMIC" else ("#ef4444" if motion_state == "STATIC" else "#f59e0b")
            
            if box_2d is not None and len(box_2d) == 4:
                xmin, ymin, xmax, ymax = box_2d
                w = xmax - xmin
                h = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor=color, facecolor="none")
                ax_cam.add_patch(rect)
                
                speed = trk.get("speed", float(np.linalg.norm(trk.get("velocity", [0, 0, 0]))))
                tag = f"ID:{trk.get('track_id', 0)} | {motion_state} | {speed:.1f} m/s"
                ax_cam.text(xmin, max(ymin - 6, 12), tag, color="white", fontsize=8, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85))

        ax_cam.set_title(f"Camera Perception & Multimodal Detection | Frame #{frame_id:04d} (t={timestamp:.2f}s)", color="#f4f4f5", fontsize=12, pad=8)
        ax_cam.axis("off")

        # -----------------------------------------------------------------
        # Panel 2: Bird's-Eye View (BEV) Map (Bottom-Left)
        # -----------------------------------------------------------------
        ax_bev = fig.add_subplot(gs[1, 0])
        ax_bev.set_facecolor("#18181b")
        ax_bev.grid(True, linestyle="--", color="#27272a", alpha=0.6)

        # Draw LiDAR points in BEV
        if lidar_points is not None and len(lidar_points) > 0:
            # Subsample for smooth rendering
            sub_pts = lidar_points[::max(1, len(lidar_points) // 1500)]
            # BEV X = Y (lateral), BEV Y = X (forward)
            ax_bev.scatter(sub_pts[:, 1], sub_pts[:, 0], s=1.5, c="#38bdf8", alpha=0.4, label="LiDAR Points")

        # Draw Planar Landmarks
        if planar_features:
            for feat in planar_features[:30]:
                c = feat.get("center", [0, 0, 0])
                ax_bev.scatter(c[1], c[0], s=12, c="#e879f9", marker="s", alpha=0.7)

        # Draw Ego Vehicle at origin (0, 0)
        self._draw_box_bev(ax_bev, center_x=0.0, center_y=0.0, length=4.8, width=2.0, yaw_deg=0.0, color="#3b82f6", label="Ego Vehicle")

        # Draw Object Tracks in BEV
        for trk in tracks:
            pos = trk.get("position", [0, 0, 0])
            dims = trk.get("dimensions", {"length": 4.5, "width": 1.8})
            rot = trk.get("rotation", {"yaw": 0.0})
            motion_state = trk.get("motion_state", "DYNAMIC")
            speed = trk.get("speed", float(np.linalg.norm(trk.get("velocity", [0, 0, 0]))))
            
            # Color coding
            color = "#22c55e" if motion_state == "DYNAMIC" else ("#ef4444" if motion_state == "STATIC" else "#f59e0b")
            
            # Position relative to ego or world
            x, y = float(pos[0]), float(pos[1])
            l = float(dims.get("length", 4.5))
            w = float(dims.get("width", 1.8))
            yaw = float(rot.get("yaw", 0.0))
            
            label = f"ID:{trk.get('track_id', 0)} ({speed:.1f} m/s)"
            self._draw_box_bev(ax_bev, center_x=x, center_y=y, length=l, width=w, yaw_deg=yaw, color=color, label=label)

        ax_bev.set_xlim(-25, 25)
        ax_bev.set_ylim(-10, 50)
        ax_bev.set_xlabel("Lateral Y (meters)", color="#a1a1aa", fontsize=9)
        ax_bev.set_ylabel("Longitudinal X (meters)", color="#a1a1aa", fontsize=9)
        ax_bev.tick_params(colors="#71717a", labelsize=8)
        ax_bev.set_title("Bird's-Eye View (BEV) 3D Spatial Tracking & Occupancy", color="#f4f4f5", fontsize=11, pad=8)

        # -----------------------------------------------------------------
        # Panel 3: State Estimation & Telemetry Dashboard (Bottom-Right)
        # -----------------------------------------------------------------
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.set_facecolor("#18181b")
        ax_info.axis("off")

        # Construct status text
        ego_speed = 0.0
        if imu_motion and "velocity" in imu_motion:
            ego_speed = float(np.linalg.norm(imu_motion["velocity"]))
        elif ego_pose and "velocity" in ego_pose:
            v = ego_pose["velocity"]
            ego_speed = float(np.linalg.norm([v.get("x", 0), v.get("y", 0), v.get("z", 0)]))

        dynamic_count = sum(1 for t in tracks if t.get("motion_state") in ["DYNAMIC", "STOPPED_DYNAMIC"])
        static_count = sum(1 for t in tracks if t.get("motion_state") == "STATIC")
        uncertain_count = len(tracks) - dynamic_count - static_count

        info_lines = [
            "LVIMOT SYSTEM TELEMETRY",
            "=" * 38,
            f"Sequence Frame      : {frame_id:04d}",
            f"Timestamp           : {timestamp:.3f} s",
            f"Ego Vehicle Speed   : {ego_speed:.2f} m/s ({ego_speed * 3.6:.1f} km/h)",
            "",
            "MULTI-OBJECT TRACKING (MOT)",
            "-" * 38,
            f"Active Tracks Total : {len(tracks)}",
            f"  • Dynamic Objects : {dynamic_count}",
            f"  • Static Objects  : {static_count}",
            f"  • Uncertain/New   : {uncertain_count}",
            "",
            "SENSOR FUSION & STATE ESTIMATION",
            "-" * 38,
            "IMU Preintegration : Active (Gravity Comp.)",
            "LiDAR Planar Features: Active",
            "Sliding Window Opt  : Active (Ceres / LM)",
            "4D Map Static Voxels: " + str(map_summary.get("total_static_voxels", "N/A") if map_summary else "Active"),
            "4D Dynamic Tubes    : " + str(map_summary.get("total_dynamic_actors", dynamic_count) if map_summary else dynamic_count)
        ]

        text_block = "\n".join(info_lines)
        ax_info.text(0.05, 0.95, text_block, color="#38bdf8", fontfamily="monospace", fontsize=9.5, verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.6", fc="#09090b", ec="#27272a", linewidth=1.5))
        ax_info.set_title("Real-Time Fusion & Estimator Diagnostics", color="#f4f4f5", fontsize=11, pad=8)

        plt.tight_layout()
        
        target_path = save_path or os.path.join(self.output_dir, f"frame_{frame_id:06d}.png")
        plt.savefig(target_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        
        return target_path
