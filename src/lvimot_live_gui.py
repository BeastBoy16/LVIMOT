from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


APP_TITLE = "LVIMOT Live Dashboard"


def _default_carla_install() -> str:
    configured = os.environ.get("CARLA_LAUNCHER")
    if configured:
        return configured
    if os.name == "nt":
        return r"C:\CARLA\CarlaUE4.exe"
    return str(Path.home() / "CARLA" / "CarlaUE4.sh")


DEFAULT_CARLA_INSTALL = _default_carla_install()
DEFAULT_CALIBRATION = str(Path(__file__).resolve().parent.parent / "config" / "carla_calibration.json")
DEFAULT_OUTPUT_ROOT = "outputs_carla"
FRAME_RE = re.compile(
    r"frame\s+(?P<frame>\d+)\s+/\s+CARLA\s+(?P<carla>\d+)\s+\|\s+"
    r"(?P<ms>[0-9.]+)\s+ms\s+\|\s+RSS\s+(?P<rss>[0-9.]+)\s+MB\s+\|\s+"
    r"tracks\s+(?P<tracks>\d+)\s+\|\s+map\s+(?P<map>\d+)\s+\|\s+"
    r"stationary=(?P<stationary>True|False)\s+\|\s+sync=(?P<sync>[0-9.]+)\s+ms"
    r"(?:\s+\|\s+camlag=(?P<camlag>\d+)f)?"
    r"(?:\s+\|\s+pump=(?P<pump>\d+))?"
)


PRESETS = {
    "Validated": {
        "width": 1242,
        "height": 375,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.06,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "High Recall": {
        "width": 1242,
        "height": 375,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.035,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "Fast": {
        "width": 828,
        "height": 250,
        "fov": 90.0,
        "yolo_size": 640,
        "confidence": 0.08,
        "vehicles": 8,
        "fixed_delta": 0.10,
    },
    "Balanced": {
        "width": 1024,
        "height": 309,
        "fov": 90.0,
        "yolo_size": 960,
        "confidence": 0.06,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
    "Quality": {
        "width": 1656,
        "height": 500,
        "fov": 90.0,
        "yolo_size": 1280,
        "confidence": 0.05,
        "vehicles": 12,
        "fixed_delta": 0.10,
    },
}


class LVIMOTLiveGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        # Fit the dashboard to the actual desktop instead of assuming a tall
        # 900 px client area.  The control column is independently scrollable
        # and the START/STOP footer is always pinned on-screen.
        screen_w = max(1024, int(self.root.winfo_screenwidth()))
        screen_h = max(700, int(self.root.winfo_screenheight()))
        window_w = min(1500, max(1100, screen_w - 48))
        window_h = min(900, max(650, screen_h - 96))
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(1100, 650)

        self.project_root = Path(__file__).resolve().parent.parent
        self.python_exe = (
            self.project_root / ".venv_live" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        self.main_script = self.project_root / "src" / "carla_main.py"
        self.config_path = self.project_root / "lvimot_gui_config.json"

        self.backend_process: Optional[subprocess.Popen] = None
        self.carla_process: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self.events: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.latest_record: Dict = {}
        self.pose_history: List[Tuple[float, float]] = []
        self.map_history: List[Tuple[int, int]] = []
        self.track_history: Dict[int, Dict] = {}
        self.spatial_snapshot: Dict = {}
        self._spatial_snapshot_mtime_ns: Optional[int] = None
        self._camera_snapshot_mtime_ns: Optional[int] = None
        self._camera_photo = None
        self._camera_source_image = None
        self._camera_resize_after = None
        self.max_tracks_seen = 0
        self.processed_records = 0
        self.last_record_frame: Optional[int] = None
        self.log_file_offset = 0
        self.log_file_identity: Optional[Tuple[int, int]] = None
        self._control_widgets: List[tk.Widget] = []

        self._build_variables()
        self._build_ui()
        self._apply_preset("Validated")
        self._load_saved_config(silent=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_events)
        self.root.after(800, self._poll_carla_status)
        self.root.after(500, self._redraw_bev)
        self.root.after(350, self._poll_spatial_map_snapshot)
        self.root.after(400, self._poll_camera_snapshot)
        self.root.after(550, self._redraw_map_health)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    def _build_variables(self):
        self.var_carla_install = tk.StringVar(value=DEFAULT_CARLA_INSTALL)
        self.var_calibration = tk.StringVar(value=DEFAULT_CALIBRATION)
        self.var_host = tk.StringVar(value="127.0.0.1")
        self.var_port = tk.StringVar(value="2000")
        self.var_render_quality = tk.StringVar(value="Epic")

        self.var_preset = tk.StringVar(value="Validated")
        self.var_autopilot = tk.BooleanVar(value=True)
        self.var_gt_eval = tk.BooleanVar(value=False)
        # Dashboard camera/map streaming is always enabled in GUI mode so the
        # live ego RGB view remains visible regardless of the selected tab.
        self.var_processed_display = tk.BooleanVar(value=True)
        self.var_show_trajectory = tk.BooleanVar(value=False)
        self.var_candidate_hold = tk.StringVar(value="4")
        self.var_vehicles = tk.StringVar(value="12")
        self.var_duration = tk.StringVar(value="600")
        self.var_fixed_delta = tk.StringVar(value="0.10")
        self.var_seed = tk.StringVar(value="42")
        self.var_sensor_timeout = tk.StringVar(value="10")
        self.var_output_root = tk.StringVar(value=DEFAULT_OUTPUT_ROOT)

        self.var_width = tk.StringVar(value="1242")
        self.var_height = tk.StringVar(value="375")
        self.var_fov = tk.StringVar(value="90.0")
        self.var_yolo_model = tk.StringVar(value="yolov8n.pt")
        self.var_yolo_size = tk.StringVar(value="1280")
        self.var_confidence = tk.StringVar(value="0.06")
        self.var_yolo_device = tk.StringVar(value="auto")

        self.var_carla_status = tk.StringVar(value="OFFLINE")
        self.var_backend_status = tk.StringVar(value="STOPPED")
        self.var_frame = tk.StringVar(value="-")
        self.var_carla_frame = tk.StringVar(value="-")
        self.var_total_ms = tk.StringVar(value="-")
        self.var_fps = tk.StringVar(value="-")
        self.var_rss = tk.StringVar(value="-")
        self.var_tracks = tk.StringVar(value="0")
        self.var_map = tk.StringVar(value="0")
        self.var_stationary = tk.StringVar(value="-")
        self.var_sync = tk.StringVar(value="-")
        self.var_camlag = tk.StringVar(value="-")
        self.var_pump = tk.StringVar(value="-")
        self.var_pose = tk.StringVar(value="x=-  y=-  z=-  yaw=-")
        self.var_speed = tk.StringVar(value="-")
        self.var_sensor = tk.StringVar(value="Waiting for backend")
        self.var_tracks_note = tk.StringVar(value="No active published tracks yet.")
        self.var_map_note = tk.StringVar(value="4-D spatial map waiting for live data.")
        self.var_camera_note = tk.StringVar(value="Enable processed camera/map streaming before START LVIMOT to embed the live ego RGB view here.")
        self.var_health = tk.StringVar(value="WAITING")
        self.var_health_detail = tk.StringVar(value="Start LVIMOT to evaluate live system health.")
        self.var_last_error = tk.StringVar(value="")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left side: fixed-width shell.  Settings scroll in the middle while
        # the run controls stay pinned at the bottom and can never disappear.
        left_shell = ttk.Frame(self.root, padding=(8, 8, 6, 8), width=365)
        right = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        left_shell.grid(row=0, column=0, sticky="nsew")
        right.grid(row=0, column=1, sticky="nsew")
        left_shell.grid_propagate(False)
        left_shell.columnconfigure(0, weight=1)
        left_shell.rowconfigure(1, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=2)
        right.rowconfigure(2, weight=3)

        title = ttk.Label(
            left_shell,
            text="LVIMOT LIVE CONTROLS",
            font=("Segoe UI", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        scroll_host = ttk.Frame(left_shell)
        scroll_host.grid(row=1, column=0, sticky="nsew")
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        self.control_canvas = tk.Canvas(
            scroll_host,
            borderwidth=0,
            highlightthickness=0,
            background=self.root.cget("background"),
        )
        control_scrollbar = ttk.Scrollbar(
            scroll_host,
            orient="vertical",
            command=self.control_canvas.yview,
        )
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")

        self.control_content = ttk.Frame(self.control_canvas, padding=(0, 0, 4, 0))
        self._control_window = self.control_canvas.create_window(
            (0, 0),
            window=self.control_content,
            anchor="nw",
        )
        self.control_content.bind("<Configure>", self._on_control_content_configure)
        self.control_canvas.bind("<Configure>", self._on_control_canvas_configure)

        # Mouse wheel works anywhere over the settings column, including over
        # entries and labels.  The handler checks pointer position so the BEV,
        # console and track table keep their own scrolling behavior.
        self.root.bind_all("<MouseWheel>", self._on_control_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_control_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_control_mousewheel, add="+")

        self._build_controls(self.control_content)
        self._build_run_footer(left_shell)
        self._build_status_cards(right)
        self._build_dashboard(right)

    def _on_control_content_configure(self, _event=None):
        try:
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _on_control_canvas_configure(self, event):
        try:
            self.control_canvas.itemconfigure(self._control_window, width=max(1, event.width))
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
        except tk.TclError:
            pass

    def _pointer_over_control_canvas(self, event) -> bool:
        try:
            x = int(getattr(event, "x_root", self.root.winfo_pointerx()))
            y = int(getattr(event, "y_root", self.root.winfo_pointery()))
            left = self.control_canvas.winfo_rootx()
            top = self.control_canvas.winfo_rooty()
            right = left + self.control_canvas.winfo_width()
            bottom = top + self.control_canvas.winfo_height()
            return left <= x <= right and top <= y <= bottom
        except tk.TclError:
            return False

    def _on_control_mousewheel(self, event):
        if not self._pointer_over_control_canvas(event):
            return

        try:
            first, last = self.control_canvas.yview()
        except tk.TclError:
            return
        if first <= 0.0 and last >= 1.0:
            return

        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return
            direction = -1 if delta > 0 else 1

        self.control_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _build_controls(self, parent):
        server = ttk.LabelFrame(parent, text="CARLA Server", padding=8)
        server.pack(fill="x", pady=(0, 4))
        self._entry_row(server, "Install", self.var_carla_install, browse=self._browse_carla)
        self._entry_row(server, "Host", self.var_host)
        self._entry_row(server, "Port", self.var_port)
        self._combo_row(server, "Render quality", self.var_render_quality, ["Low", "Epic"])

        server_buttons = ttk.Frame(server)
        server_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(server_buttons, text="Check CARLA", command=self._check_carla_now).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(server_buttons, text="Start CARLA", command=self._start_carla).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(server_buttons, text="Stop CARLA", command=self._stop_carla).pack(side="left", expand=True, fill="x", padx=(3, 0))

        run = ttk.LabelFrame(parent, text="Scenario", padding=8)
        run.pack(fill="x", pady=4)
        self._combo_row(run, "Preset", self.var_preset, list(PRESETS.keys()), callback=self._preset_changed)
        self._check_row(run, "Ego autopilot", self.var_autopilot)
        self._check_row(run, "GT evaluation only", self.var_gt_eval)
        always_stream = ttk.Checkbutton(
            run, text="Live camera + spatial 4-D map stream (always on)",
            variable=self.var_processed_display, state="disabled"
        )
        always_stream.pack(anchor="w", pady=2)
        self._check_row(run, "Show estimated ego trail in Overview", self.var_show_trajectory)
        self._entry_row(run, "NPC vehicles", self.var_vehicles)
        self._entry_row(run, "Run seconds", self.var_duration)
        self._entry_row(run, "Random seed", self.var_seed)

        perception = ttk.LabelFrame(parent, text="Camera / Perception", padding=8)
        perception.pack(fill="x", pady=4)
        self._entry_row(perception, "Calibration", self.var_calibration, browse=self._browse_calibration)
        self._entry_row(perception, "Width", self.var_width)
        self._entry_row(perception, "Height", self.var_height)
        self._entry_row(perception, "Horizontal FOV", self.var_fov)
        self._entry_row(perception, "YOLO model", self.var_yolo_model)
        self._combo_row(perception, "YOLO imgsz", self.var_yolo_size, ["640", "832", "960", "1280"])
        self._entry_row(perception, "Confidence", self.var_confidence)
        self._combo_row(perception, "Candidate hold", self.var_candidate_hold, ["0", "2", "3", "4", "5", "6", "8"])
        self._combo_row(perception, "YOLO device", self.var_yolo_device, ["auto", "0", "cpu"])

        advanced = ttk.LabelFrame(parent, text="Advanced", padding=8)
        advanced.pack(fill="x", pady=4)
        self._combo_row(advanced, "Fixed delta", self.var_fixed_delta, ["0.05", "0.10", "0.20"])
        self._entry_row(advanced, "Sensor timeout", self.var_sensor_timeout)
        self._entry_row(advanced, "Output root", self.var_output_root)

        note = ttk.Label(
            parent,
            text=(
                "Validated baseline: 1242×375, FOV 90°, YOLOv8n, imgsz 1280, "
                "conf 0.06, 10 Hz.\nThe annotated live ego camera is permanently visible in the dashboard. "
                "Candidate hold is display-only continuity for brief fused-detection gaps; it never creates a MOT track. "
                "CARLA itself remains a separate simulation window.\n"
                "Scroll this panel for settings; START/STOP stays pinned below."
            ),
            wraplength=325,
            justify="left",
        )
        note.pack(fill="x", pady=(6, 8))

    def _build_run_footer(self, parent):
        footer = ttk.Frame(parent)
        footer.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        footer.columnconfigure(0, weight=1)

        launch = ttk.LabelFrame(footer, text="Run", padding=8)
        launch.grid(row=0, column=0, sticky="ew")

        row1 = ttk.Frame(launch)
        row1.pack(fill="x")
        self.btn_start = ttk.Button(
            row1,
            text="START LVIMOT",
            command=self._start_backend,
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.btn_stop = ttk.Button(
            row1,
            text="STOP",
            command=self._stop_backend,
            state="disabled",
        )
        self.btn_stop.pack(side="left", expand=True, fill="x")

        row2 = ttk.Frame(launch)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Button(row2, text="Save", command=self._save_config).pack(
            side="left", expand=True, fill="x", padx=(0, 3)
        )
        ttk.Button(row2, text="Load", command=self._load_saved_config).pack(
            side="left", expand=True, fill="x", padx=3
        )
        ttk.Button(row2, text="Outputs", command=self._open_outputs).pack(
            side="left", expand=True, fill="x", padx=(3, 0)
        )

        ttk.Label(
            footer,
            textvariable=self.var_last_error,
            foreground="#b00020",
            wraplength=335,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _build_status_cards(self, parent):
        """Compact two-line status strip to preserve space for the live camera."""
        strip = ttk.Frame(parent)
        strip.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for c in range(7):
            strip.columnconfigure(c, weight=1)

        cards = [
            ("CARLA", self.var_carla_status),
            ("Backend", self.var_backend_status),
            ("Frame", self.var_frame),
            ("Pipe", self.var_total_ms),
            ("FPS", self.var_fps),
            ("Tracks", self.var_tracks),
            ("Map", self.var_map),
            ("RSS", self.var_rss),
            ("Move", self.var_stationary),
            ("Sync", self.var_sync),
            ("Cam lag", self.var_camlag),
            ("Pump", self.var_pump),
            ("Speed", self.var_speed),
        ]
        for i, (name, variable) in enumerate(cards):
            row = i // 7
            col = i % 7
            cell = ttk.Frame(strip, padding=(3, 1))
            cell.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 2, 0), pady=(0 if row == 0 else 2, 0))
            ttk.Label(cell, text=name, font=("Segoe UI", 8)).pack()
            ttk.Label(cell, textvariable=variable, font=("Segoe UI", 9, "bold")).pack()

    def _build_dashboard(self, parent):
        # The camera is intentionally NOT a tab.  It is always visible while the
        # backend runs; the notebook below is for secondary views/diagnostics.
        camera_shell = ttk.LabelFrame(parent, text="Live Ego RGB", padding=(5, 4))
        camera_shell.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        camera_shell.columnconfigure(0, weight=1)
        camera_shell.rowconfigure(1, weight=1)
        ttk.Label(
            camera_shell,
            textvariable=self.var_camera_note,
            wraplength=1100,
            justify="left",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, sticky="ew", pady=(0, 3))
        self.camera_canvas = tk.Canvas(camera_shell, background="#0b1117", highlightthickness=0)
        self.camera_canvas.grid(row=1, column=0, sticky="nsew")
        self.camera_canvas.bind("<Configure>", self._on_camera_canvas_configure, add="+")

        notebook = ttk.Notebook(parent)
        notebook.grid(row=2, column=0, sticky="nsew")

        overview = ttk.Frame(notebook, padding=6)
        tracks = ttk.Frame(notebook, padding=6)
        map_health = ttk.Frame(notebook, padding=6)
        console = ttk.Frame(notebook, padding=6)
        settings = ttk.Frame(notebook, padding=6)
        notebook.add(overview, text="Overview / BEV")
        notebook.add(tracks, text="Tracks")
        notebook.add(map_health, text="4-D Map / Health")
        notebook.add(console, text="Console")
        notebook.add(settings, text="Current Command")

        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(1, weight=1)
        info = ttk.Frame(overview)
        info.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        ttk.Label(info, textvariable=self.var_pose, font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self.var_sensor, font=("Consolas", 9)).grid(row=0, column=1, sticky="e")
        self.bev_canvas = tk.Canvas(overview, background="#111820", highlightthickness=0)
        self.bev_canvas.grid(row=1, column=0, sticky="nsew")

        tracks.columnconfigure(0, weight=1)
        tracks.rowconfigure(1, weight=1)
        ttk.Label(
            tracks,
            textvariable=self.var_tracks_note,
            wraplength=1000,
            justify="left",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        columns = ("id", "status", "class", "distance", "speed", "x", "y", "state", "hits", "last_seen")
        self.track_table = ttk.Treeview(tracks, columns=columns, show="headings")
        headings = {
            "id": "ID", "status": "Status", "class": "Class",
            "distance": "Distance m", "speed": "Speed m/s",
            "x": "X", "y": "Y", "state": "Motion",
            "hits": "Hits", "last_seen": "Last seen",
        }
        widths = {
            "id": 50, "status": 78, "class": 95, "distance": 90, "speed": 85,
            "x": 75, "y": 75, "state": 95, "hits": 60, "last_seen": 80,
        }
        for col in columns:
            self.track_table.heading(col, text=headings[col])
            self.track_table.column(col, width=widths[col], anchor="center")
        scroll = ttk.Scrollbar(tracks, orient="vertical", command=self.track_table.yview)
        self.track_table.configure(yscrollcommand=scroll.set)
        self.track_table.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")

        map_health.columnconfigure(0, weight=1)
        map_health.rowconfigure(2, weight=1)
        health_header = ttk.Frame(map_health)
        health_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        health_header.columnconfigure(1, weight=1)
        ttk.Label(health_header, text="Live system health:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(health_header, textvariable=self.var_health, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            map_health, textvariable=self.var_health_detail, wraplength=1000,
            justify="left", font=("Segoe UI", 8),
        ).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.map_canvas = tk.Canvas(map_health, background="#111820", highlightthickness=0)
        self.map_canvas.grid(row=2, column=0, sticky="nsew")
        ttk.Label(
            map_health, textvariable=self.var_map_note, wraplength=1000,
            justify="left", font=("Segoe UI", 8),
        ).grid(row=3, column=0, sticky="ew", pady=(4, 0))

        console.columnconfigure(0, weight=1)
        console.rowconfigure(0, weight=1)
        self.console = tk.Text(console, wrap="none", background="#0d1117", foreground="#d9e1e8", insertbackground="white", font=("Consolas", 8))
        cscroll_y = ttk.Scrollbar(console, orient="vertical", command=self.console.yview)
        cscroll_x = ttk.Scrollbar(console, orient="horizontal", command=self.console.xview)
        self.console.configure(yscrollcommand=cscroll_y.set, xscrollcommand=cscroll_x.set)
        self.console.grid(row=0, column=0, sticky="nsew")
        cscroll_y.grid(row=0, column=1, sticky="ns")
        cscroll_x.grid(row=1, column=0, sticky="ew")

        settings.columnconfigure(0, weight=1)
        settings.rowconfigure(0, weight=1)
        self.command_text = tk.Text(settings, wrap="word", font=("Consolas", 9))
        self.command_text.grid(row=0, column=0, sticky="nsew")
        self.command_text.insert("1.0", "The exact backend command will appear here before launch.")
        self.command_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Reusable control rows
    # ------------------------------------------------------------------

    def _entry_row(self, parent, label, variable, browse=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)
        self._control_widgets.append(entry)
        if browse is not None:
            ttk.Button(row, text="...", width=3, command=browse).pack(side="left", padx=(4, 0))
        return entry

    def _combo_row(self, parent, label, variable, values, callback=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=15).pack(side="left")
        combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly")
        combo.pack(side="left", fill="x", expand=True)
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", callback)
        self._control_widgets.append(combo)
        return combo

    def _check_row(self, parent, label, variable):
        check = ttk.Checkbutton(parent, text=label, variable=variable)
        check.pack(anchor="w", pady=2)
        self._control_widgets.append(check)
        return check

    # ------------------------------------------------------------------
    # Presets / validation
    # ------------------------------------------------------------------

    def _preset_changed(self, _event=None):
        self._apply_preset(self.var_preset.get())

    def _apply_preset(self, name):
        preset = PRESETS.get(name)
        if not preset:
            return
        self.var_width.set(str(preset["width"]))
        self.var_height.set(str(preset["height"]))
        self.var_fov.set(str(preset["fov"]))
        self.var_yolo_size.set(str(preset["yolo_size"]))
        self.var_confidence.set(str(preset["confidence"]))
        self.var_vehicles.set(str(preset["vehicles"]))
        self.var_fixed_delta.set(f"{preset['fixed_delta']:.2f}")

    def _numeric_config(self) -> Dict:
        try:
            cfg = {
                "port": int(self.var_port.get()),
                "vehicles": int(self.var_vehicles.get()),
                "seconds": float(self.var_duration.get()),
                "fixed_delta": float(self.var_fixed_delta.get()),
                "seed": int(self.var_seed.get()),
                "sensor_timeout": float(self.var_sensor_timeout.get()),
                "width": int(self.var_width.get()),
                "height": int(self.var_height.get()),
                "fov": float(self.var_fov.get()),
                "yolo_size": int(self.var_yolo_size.get()),
                "confidence": float(self.var_confidence.get()),
                "candidate_hold": int(self.var_candidate_hold.get()),
            }
        except ValueError as exc:
            raise ValueError("One or more numeric settings are invalid") from exc

        if cfg["vehicles"] < 0 or cfg["vehicles"] > 200:
            raise ValueError("NPC vehicles must be between 0 and 200")
        if cfg["seconds"] <= 0:
            raise ValueError("Run seconds must be > 0")
        if cfg["fixed_delta"] <= 0:
            raise ValueError("Fixed delta must be > 0")
        if cfg["sensor_timeout"] < 1:
            raise ValueError("Sensor timeout should be at least 1 second")
        if cfg["width"] < 320 or cfg["height"] < 120:
            raise ValueError("Camera resolution is too small")
        if cfg["width"] > 3840 or cfg["height"] > 2160:
            raise ValueError("Camera resolution is too large for this GUI")
        if not 30.0 <= cfg["fov"] <= 140.0:
            raise ValueError("Camera FOV must be between 30 and 140 degrees")
        if not 0.001 <= cfg["confidence"] <= 1.0:
            raise ValueError("YOLO confidence must be between 0.001 and 1.0")
        if cfg["candidate_hold"] < 0 or cfg["candidate_hold"] > 12:
            raise ValueError("Candidate hold must be between 0 and 12 frames")
        return cfg

    # ------------------------------------------------------------------
    # Calibration / command building
    # ------------------------------------------------------------------

    def _runtime_calibration_path(self) -> Path:
        output_root = Path(self.var_output_root.get())
        if not output_root.is_absolute():
            output_root = self.project_root / output_root
        path = output_root / "gui" / "runtime_calibration.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_runtime_calibration(self, cfg: Dict) -> Path:
        source = Path(os.path.expandvars(os.path.expanduser(self.var_calibration.get())))
        if not source.exists():
            raise FileNotFoundError(f"Calibration file not found: {source}")
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)

        camera = data.get("camera")
        if not isinstance(camera, dict):
            raise ValueError("Calibration JSON does not contain a camera section")

        width = cfg["width"]
        height = cfg["height"]
        fov = cfg["fov"]
        focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        camera["width"] = int(width)
        camera["height"] = int(height)
        camera["fov"] = float(fov)
        camera["intrinsic"] = [
            [float(focal), 0.0, float(width) / 2.0],
            [0.0, float(focal), float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ]

        path = self._runtime_calibration_path()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def _build_backend_command(self) -> Tuple[List[str], Path]:
        cfg = self._numeric_config()
        if not self.python_exe.exists():
            raise FileNotFoundError(
                f"Live Python environment not found: {self.python_exe}\nRun setup_live_carla_env.sh first."
            )
        if not self.main_script.exists():
            raise FileNotFoundError(f"carla_main.py not found: {self.main_script}")

        runtime_calibration = self._write_runtime_calibration(cfg)
        command = [
            str(self.python_exe),
            "-u",
            str(self.main_script),
            "--live",
            "--carla-install",
            str(self._carla_install_root()),
            "--calibration-file",
            str(runtime_calibration),
            "--host",
            self.var_host.get(),
            "--port",
            str(cfg["port"]),
            "--seconds",
            str(cfg["seconds"]),
            "--traffic-vehicles",
            str(cfg["vehicles"]),
            "--fixed-delta",
            str(cfg["fixed_delta"]),
            "--seed",
            str(cfg["seed"]),
            "--sensor-timeout",
            str(cfg["sensor_timeout"]),
            "--output-root",
            self.var_output_root.get(),
            "--yolo-model",
            self.var_yolo_model.get(),
            "--yolo-confidence",
            str(cfg["confidence"]),
            "--yolo-image-size",
            str(cfg["yolo_size"]),
            "--yolo-device",
            self.var_yolo_device.get(),
        ]
        # GUI mode always enables the backend visualizer stream because the
        # live camera and spatial map are permanent dashboard panels.
        self.var_processed_display.set(True)
        if self.var_autopilot.get():
            command.append("--autopilot")
        if self.var_gt_eval.get():
            command.append("--evaluate-ground-truth")
        return command, runtime_calibration

    def _format_command(self, command: List[str]) -> str:
        return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)

    def _display_command(self, command: List[str]):
        text = self._format_command(command)
        self.command_text.configure(state="normal")
        self.command_text.delete("1.0", "end")
        self.command_text.insert("1.0", text)
        self.command_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # CARLA server controls
    # ------------------------------------------------------------------

    def _carla_install_root(self) -> Path:
        raw = Path(os.path.expandvars(os.path.expanduser(self.var_carla_install.get())))
        if raw.name.lower() in {"carlaue4.sh", "carlaue4.exe"} or raw.is_file():
            return raw.parent
        return raw

    def _carla_executable(self) -> Path:
        raw = Path(os.path.expandvars(os.path.expanduser(self.var_carla_install.get())))
        if raw.name.lower() in {"carlaue4.sh", "carlaue4.exe"}:
            return raw
        return raw / ("CarlaUE4.exe" if os.name == "nt" else "CarlaUE4.sh")

    def _is_carla_reachable(self) -> bool:
        try:
            port = int(self.var_port.get())
            with socket.create_connection((self.var_host.get(), port), timeout=0.35):
                return True
        except Exception:
            return False

    def _poll_carla_status(self):
        reachable = self._is_carla_reachable()
        self.var_carla_status.set("CONNECTED" if reachable else "OFFLINE")
        self.root.after(1200, self._poll_carla_status)

    def _check_carla_now(self):
        if self._is_carla_reachable():
            messagebox.showinfo("CARLA", "CARLA server is reachable on the configured host/port.")
        else:
            messagebox.showwarning("CARLA", "CARLA server is not reachable yet.")

    def _start_carla(self):
        if self._is_carla_reachable():
            messagebox.showinfo("CARLA", "CARLA is already running.")
            return
        exe = self._carla_executable()
        if not exe.exists():
            messagebox.showerror("CARLA", f"CARLA launcher not found:\n{exe}")
            return
        quality = self.var_render_quality.get()
        args = [str(exe), f"-quality-level={quality}"]
        if os.name != "nt" and exe.suffix == ".sh":
            args = ["bash", str(exe), f"-quality-level={quality}"]
        try:
            self.carla_process = subprocess.Popen(
                args, cwd=str(exe.parent), start_new_session=(os.name != "nt")
            )
            self._append_console("[GUI] Started CARLA: " + self._format_command(args))
        except Exception as exc:
            messagebox.showerror("CARLA", str(exc))

    def _stop_carla(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            messagebox.showwarning("CARLA", "Stop LVIMOT before stopping CARLA.")
            return
        if self.carla_process is None or self.carla_process.poll() is not None:
            messagebox.showinfo("CARLA", "The GUI did not start a CARLA process that it can stop.")
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self.carla_process.pid), signal.SIGTERM)
            else:
                self.carla_process.terminate()
            self._append_console("[GUI] Requested CARLA shutdown.")
        except Exception as exc:
            messagebox.showerror("CARLA", str(exc))

    # ------------------------------------------------------------------
    # Backend controls
    # ------------------------------------------------------------------

    def _start_backend(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            messagebox.showinfo(APP_TITLE, "LVIMOT is already running.")
            return
        if not self._is_carla_reachable():
            messagebox.showwarning(APP_TITLE, "Start CARLA and wait until it is reachable before starting LVIMOT.")
            return

        try:
            command, runtime_calibration = self._build_backend_command()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._save_config(silent=True)
        self._display_command(command)
        self._archive_previous_live_outputs()
        self._reset_runtime_dashboard()
        self.stop_requested = False
        self.var_backend_status.set("STARTING")
        self.var_last_error.set("")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_controls_enabled(False)

        env = os.environ.copy()
        env["PYTHONFAULTHANDLER"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["LVIMOT_LIVE_MAP_SNAPSHOT"] = str(self._map_snapshot_path())
        env["LVIMOT_LIVE_CAMERA_SNAPSHOT"] = str(self._camera_snapshot_path())
        env["LVIMOT_CANDIDATE_HOLD_FRAMES"] = str(self._numeric_config()["candidate_hold"])

        popen_kwargs = {}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        elif os.name != "nt":
            popen_kwargs["start_new_session"] = True

        try:
            self.backend_process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                **popen_kwargs,
            )
        except Exception as exc:
            self.var_backend_status.set("FAILED")
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self._set_controls_enabled(True)
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self.var_backend_status.set("RUNNING")
        self._append_console(f"[GUI] Runtime calibration: {runtime_calibration}")
        self._append_console("[GUI] " + self._format_command(command))

        threading.Thread(target=self._stdout_worker, daemon=True).start()
        threading.Thread(target=self._json_tail_worker, daemon=True).start()

    def _stdout_worker(self):
        process = self.backend_process
        if process is None or process.stdout is None:
            return
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                self.events.put(("stdout", line.rstrip("\r\n")))
        except Exception as exc:
            self.events.put(("error", f"stdout reader: {exc}"))
        finally:
            rc = process.wait()
            self.events.put(("backend_exit", rc))

    def _map_snapshot_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        path = output / "live" / "live_map_snapshot.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _camera_snapshot_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        path = output / "live" / "live_camera_preview.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _live_jsonl_path(self) -> Path:
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        return output / "live" / "live_frames.jsonl"

    def _json_tail_worker(self):
        path = self._live_jsonl_path()
        offset = 0
        last_size = 0
        while self.backend_process is not None and self.backend_process.poll() is None:
            try:
                if path.exists():
                    size = path.stat().st_size
                    if size < offset or size < last_size:
                        offset = 0
                    last_size = size
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            offset = f.tell()
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            self.events.put(("record", record))
                time.sleep(0.15)
            except Exception as exc:
                self.events.put(("error", f"JSON monitor: {exc}"))
                time.sleep(0.5)

    def _stop_backend(self):
        process = self.backend_process
        if process is None or process.poll() is not None:
            return
        self.stop_requested = True
        self.var_backend_status.set("STOPPING")
        self._append_console("[GUI] Stop requested...")

        def worker():
            try:
                if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                elif os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                deadline = time.monotonic() + 5.0
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if process.poll() is None:
                    process.terminate()
            except Exception:
                try:
                    process.terminate()
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Event processing / dashboard updates
    # ------------------------------------------------------------------

    def _process_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "stdout":
                    self._handle_stdout(str(payload))
                elif kind == "record":
                    self._handle_record(payload if isinstance(payload, dict) else {})
                elif kind == "error":
                    self.var_last_error.set(str(payload))
                    self._append_console("[GUI ERROR] " + str(payload))
                elif kind == "backend_exit":
                    self._handle_backend_exit(int(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._process_events)

    def _handle_stdout(self, line: str):
        self._append_console(line)
        match = FRAME_RE.search(line)
        if not match:
            if "[LVIMOT LIVE ERROR" in line or "Traceback" in line:
                self.var_last_error.set("Backend reported an error. See Console tab.")
            return

        groups = match.groupdict()
        ms = float(groups["ms"])
        self.var_frame.set(groups["frame"])
        self.var_carla_frame.set(groups["carla"])
        self.var_total_ms.set(f"{ms:.0f} ms")
        self.var_fps.set(f"{1000.0 / ms:.2f}" if ms > 0 else "-")
        self.var_rss.set(f"{float(groups['rss']):.0f} MB")
        self.var_tracks.set(groups["tracks"])
        self.var_map.set(groups["map"])
        self.var_stationary.set("YES" if groups["stationary"] == "True" else "NO")
        self.var_sync.set(f"{float(groups['sync']):.2f} ms")
        self.var_camlag.set((groups.get("camlag") or "0") + " f")
        self.var_pump.set(groups.get("pump") or "-")

    def _handle_record(self, record: Dict):
        self.latest_record = record
        self.processed_records += 1
        try:
            self.last_record_frame = int(record.get("frame"))
        except Exception:
            self.last_record_frame = None
        self.var_frame.set(str(record.get("frame", self.var_frame.get())))
        self.var_carla_frame.set(str(record.get("carla_frame", self.var_carla_frame.get())))

        estimate = record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        velocity = estimate.get("velocity") or []
        if isinstance(pose, list) and len(pose) >= 6:
            roll, pitch, yaw, x, y, z = [self._safe_float(v) for v in pose[:6]]
            self.var_pose.set(f"x={x:8.2f} m   y={y:8.2f} m   z={z:7.2f} m   yaw={math.degrees(yaw):7.2f}°")
            self.pose_history.append((x, y))
            if len(self.pose_history) > 500:
                self.pose_history = self.pose_history[-500:]

        if isinstance(velocity, list) and len(velocity) >= 3:
            vx, vy, vz = [self._safe_float(v) for v in velocity[:3]]
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            self.var_speed.set(f"{speed:.2f} m/s")

        perf = record.get("performance") or {}
        total_ms = perf.get("total_ms")
        if total_ms is not None:
            total_ms = self._safe_float(total_ms)
            self.var_total_ms.set(f"{total_ms:.0f} ms")
            if total_ms > 0:
                self.var_fps.set(f"{1000.0 / total_ms:.2f}")
        rss = perf.get("rss_mb")
        if rss is not None:
            self.var_rss.set(f"{self._safe_float(rss):.0f} MB")

        tracks = record.get("tracks") or []
        active_tracks = len(tracks)
        self.max_tracks_seen = max(self.max_tracks_seen, active_tracks)
        self.var_tracks.set(str(active_tracks))

        current_frame = self.last_record_frame if self.last_record_frame is not None else self.processed_records
        active_ids = set()
        for track in tracks:
            try:
                tid = int(track.get("track_id"))
            except Exception:
                continue
            active_ids.add(tid)
            self.track_history[tid] = {
                "track": dict(track),
                "last_seen": int(current_frame),
                "active": True,
            }
        for tid, item in list(self.track_history.items()):
            item["active"] = tid in active_ids
            age = int(current_frame) - int(item.get("last_seen", current_frame))
            if age > 120:
                self.track_history.pop(tid, None)

        if active_tracks > 0:
            self.var_tracks_note.set(
                f"{active_tracks} active published track(s) now. The table also retains recent tracks "
                f"for up to 120 processed frames; maximum active at once this run: {self.max_tracks_seen}."
            )
        elif self.track_history:
            self.var_tracks_note.set(
                "No active published tracks in the current frame. Recent tracks are retained below as RECENT "
                "instead of making the table appear empty."
            )
        else:
            self.var_tracks_note.set(
                "No published tracks have been observed yet. Tracks appear only after camera+LiDAR evidence "
                "passes the tracker publication rules."
            )

        mapping = record.get("mapping_4d") or {}
        map_count = mapping.get("total_static_voxels")
        if map_count is not None:
            map_count_int = int(self._safe_float(map_count))
            self.var_map.set(str(map_count_int))
            frame_for_map = self.last_record_frame if self.last_record_frame is not None else self.processed_records
            self.map_history.append((int(frame_for_map), map_count_int))
            if len(self.map_history) > 1200:
                self.map_history = self.map_history[-1200:]
            self.var_map_note.set(
                f"4-D mapper is active with {map_count_int:,} static voxels. "
                "The 4-D Map tab renders a cleaned occupancy-style rolling local BEV; the underlying mapper data is unchanged."
            )

        integrity = record.get("sensor_integrity") or {}
        if "stationary_active" in integrity:
            self.var_stationary.set("YES" if bool(integrity.get("stationary_active")) else "NO")

        sync = record.get("live_sync") or {}
        skew = sync.get("timestamp_skew_s")
        camlag = sync.get("simulation_lag_frames", sync.get("camera_lag_frames"))
        pump = sync.get("world_ticks_last_call")
        if skew is not None:
            self.var_sync.set(f"{1000.0 * self._safe_float(skew):.2f} ms")
        if camlag is not None:
            self.var_camlag.set(f"{int(camlag)} f")
        if pump is not None:
            self.var_pump.set(str(pump))

        camera_ok = sync.get("camera_last_frame") is not None or "camera_dropped" in sync
        lidar_ok = sync.get("lidar_last_frame") is not None or "lidar_dropped" in sync
        imu_ok = sync.get("imu_last_frame") is not None or "imu_dropped" in sync
        dropped = (
            int(sync.get("camera_dropped", 0)),
            int(sync.get("lidar_dropped", 0)),
            int(sync.get("imu_dropped", 0)),
        )
        self.var_sensor.set(
            "RGB={}  LiDAR={}  IMU={}  dropped={}/{}/{}".format(
                "OK" if camera_ok else "?",
                "OK" if lidar_ok else "?",
                "OK" if imu_ok else "?",
                dropped[0], dropped[1], dropped[2],
            )
        )

        self._update_track_table(tracks, pose)
        self._update_health(record)

    def _handle_backend_exit(self, rc: int):
        if self.stop_requested:
            self.var_backend_status.set("STOPPED")
        elif rc == 0:
            self.var_backend_status.set("COMPLETE")
        else:
            self.var_backend_status.set(f"FAILED ({rc})")
            self.var_last_error.set(f"LVIMOT exited with code {rc}. See Console tab.")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_controls_enabled(True)
        self.backend_process = None

    def _update_track_table(self, tracks: List[Dict], pose):
        for item in self.track_table.get_children():
            self.track_table.delete(item)

        ego_x = self._safe_float(pose[3]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        ego_y = self._safe_float(pose[4]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        current_frame = self.last_record_frame if self.last_record_frame is not None else self.processed_records

        rows = []
        for tid, item in self.track_history.items():
            track = item.get("track") or {}
            position = track.get("position") or [0.0, 0.0, 0.0]
            x = self._safe_float(position[0]) if len(position) > 0 else 0.0
            y = self._safe_float(position[1]) if len(position) > 1 else 0.0
            distance = math.hypot(x - ego_x, y - ego_y)
            speed = track.get("speed")
            if speed is None:
                vel = track.get("velocity") or [0.0, 0.0, 0.0]
                speed = math.sqrt(sum(self._safe_float(v) ** 2 for v in vel[:3]))
            last_seen = int(item.get("last_seen", current_frame))
            age = max(0, int(current_frame) - last_seen)
            status = "ACTIVE" if bool(item.get("active")) else "RECENT"
            rows.append((0 if status == "ACTIVE" else 1, age, tid, track, distance, speed, x, y, status))

        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        for _, age, tid, track, distance, speed, x, y, status in rows:
            self.track_table.insert(
                "",
                "end",
                values=(
                    tid,
                    status,
                    track.get("class", "object"),
                    f"{distance:.1f}",
                    f"{self._safe_float(speed):.1f}",
                    f"{x:.1f}",
                    f"{y:.1f}",
                    track.get("motion_state", "-"),
                    track.get("hits", "-"),
                    "now" if age == 0 else f"{age} f ago",
                ),
            )

    # ------------------------------------------------------------------
    # BEV drawing
    # ------------------------------------------------------------------

    def _redraw_bev(self):
        canvas = self.bev_canvas
        if not canvas.winfo_exists():
            return
        width = max(300, canvas.winfo_width())
        height = max(300, canvas.winfo_height())
        canvas.delete("all")

        cx = width / 2.0
        cy = height * 0.72
        px_per_m = min(width / 120.0, height / 100.0)

        for meters in (10, 20, 30, 40, 50):
            radius = meters * px_per_m
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#25384a")
            canvas.create_text(cx + 5, cy - radius + 10, text=f"{meters}m", fill="#6c879d", anchor="w")
        canvas.create_line(0, cy, width, cy, fill="#25384a")
        canvas.create_line(cx, 0, cx, height, fill="#25384a")

        estimate = self.latest_record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        ego_x = self._safe_float(pose[3]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        ego_y = self._safe_float(pose[4]) if isinstance(pose, list) and len(pose) >= 6 else 0.0
        yaw = self._safe_float(pose[2]) if isinstance(pose, list) and len(pose) >= 6 else 0.0

        def world_delta_to_screen(dx: float, dy: float):
            # Heading-up ego-centric display.  Forward is up; left/right is horizontal.
            forward = math.cos(yaw) * dx + math.sin(yaw) * dy
            lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
            return cx + lateral * px_per_m, cy - forward * px_per_m

        # Optional blue trail = recent estimated ego path only.  It is OFF by
        # default because the spatial map is the primary visualization.  Only
        # points inside the visible local radius are drawn, and discontinuous
        # pose jumps break the line instead of stretching across the screen.
        if self.var_show_trajectory.get() and len(self.pose_history) >= 2:
            segment = []
            previous = None
            for x, y in self.pose_history[-120:]:
                dx = x - ego_x
                dy = y - ego_y
                if math.hypot(dx, dy) > 52.0:
                    if len(segment) >= 4:
                        canvas.create_line(*segment, fill="#4ba3ff", width=2)
                    segment = []
                    previous = (x, y)
                    continue
                if previous is not None and math.hypot(x - previous[0], y - previous[1]) > 5.0:
                    if len(segment) >= 4:
                        canvas.create_line(*segment, fill="#4ba3ff", width=2)
                    segment = []
                px, py = world_delta_to_screen(dx, dy)
                segment.extend([px, py])
                previous = (x, y)
            if len(segment) >= 4:
                canvas.create_line(*segment, fill="#4ba3ff", width=2)

        # In heading-up ego coordinates the ego symbol always points upward.
        triangle = [cx, cy - 18, cx - 10, cy + 12, cx + 10, cy + 12]
        canvas.create_polygon(*triangle, fill="#00d084", outline="white")
        canvas.create_text(cx, cy + 25, text="EGO", fill="white")

        tracks = self.latest_record.get("tracks") or []
        for track in tracks:
            pos = track.get("position") or []
            if len(pos) < 2:
                continue
            tx = self._safe_float(pos[0]) - ego_x
            ty = self._safe_float(pos[1]) - ego_y
            px, py = world_delta_to_screen(tx, ty)
            if px < -20 or px > width + 20 or py < -20 or py > height + 20:
                continue
            r = 6
            canvas.create_oval(px - r, py - r, px + r, py + r, fill="#ffb020", outline="white")
            canvas.create_text(px + 9, py - 8, text=f"ID {track.get('track_id', '?')}", fill="white", anchor="w")

        canvas.create_text(
            12, 12,
            text=(
                "Heading-up ego BEV | orange = active tracks"
                + (" | blue = optional recent estimated trail" if self.var_show_trajectory.get() else "")
            ),
            fill="#c8d6e5", anchor="nw", font=("Segoe UI", 10, "bold")
        )
        self.root.after(700, self._redraw_bev)

    def _poll_spatial_map_snapshot(self):
        path = self._map_snapshot_path()
        try:
            if path.exists():
                stat = path.stat()
                mtime_ns = int(stat.st_mtime_ns)
                if self._spatial_snapshot_mtime_ns != mtime_ns:
                    with path.open("r", encoding="utf-8") as handle:
                        snapshot = json.load(handle)
                    if isinstance(snapshot, dict):
                        self.spatial_snapshot = snapshot
                        self._spatial_snapshot_mtime_ns = mtime_ns
        except (OSError, json.JSONDecodeError):
            # The visualizer writes atomically, but tolerate a transient read
            # failure without disturbing the backend.
            pass
        self.root.after(300, self._poll_spatial_map_snapshot)

    def _on_camera_canvas_configure(self, _event=None):
        """Keep the live camera frame fitted to the entire available canvas."""
        if self._camera_source_image is None:
            return
        if self._camera_resize_after is not None:
            try:
                self.root.after_cancel(self._camera_resize_after)
            except tk.TclError:
                pass
        self._camera_resize_after = self.root.after(40, self._render_camera_frame)

    def _render_camera_frame(self):
        """Stretch the current RGB preview edge-to-edge across the camera panel."""
        self._camera_resize_after = None
        if (
            self._camera_source_image is None
            or not hasattr(self, "camera_canvas")
            or not self.camera_canvas.winfo_exists()
        ):
            return

        canvas_w = max(1, int(self.camera_canvas.winfo_width()))
        canvas_h = max(1, int(self.camera_canvas.winfo_height()))
        if canvas_w <= 2 or canvas_h <= 2:
            return

        try:
            resampling = getattr(Image, "Resampling", Image)
            resized = self._camera_source_image.resize(
                (canvas_w, canvas_h),
                resampling.BILINEAR,
            )
            self._camera_photo = ImageTk.PhotoImage(resized)
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(
                0,
                0,
                image=self._camera_photo,
                anchor="nw",
            )
            self.camera_canvas.create_text(
                12,
                12,
                text="LIVE EGO RGB | cyan=current fused candidate | dashed cyan=short candidate hold | green=published MOT | amber=track hold",
                fill="white",
                anchor="nw",
                font=("Segoe UI", 10, "bold"),
            )
        except (tk.TclError, OSError, ValueError):
            pass

    def _poll_camera_snapshot(self):
        if not hasattr(self, "camera_canvas") or not self.camera_canvas.winfo_exists():
            return

        path = self._camera_snapshot_path()
        try:
            if path.exists():
                stat = path.stat()
                mtime_ns = int(stat.st_mtime_ns)
                if self._camera_snapshot_mtime_ns != mtime_ns:
                    with Image.open(path) as image:
                        self._camera_source_image = image.convert("RGB").copy()

                    self._camera_snapshot_mtime_ns = mtime_ns
                    self._render_camera_frame()
                    self.var_camera_note.set(
                        "Live CARLA ego RGB is permanently visible. Solid cyan is a current fused camera+LiDAR candidate; "
                        "dashed cyan is a short display-only candidate hold. Green is a real published MOT track; amber is track hold."
                    )
        except (OSError, tk.TclError, ValueError):
            # The preview file can be replaced while the GUI is polling it.
            # Ignore that transient read and retry on the next poll.
            pass

        if not self.var_processed_display.get() and self._camera_photo is None:
            self.camera_canvas.delete("all")
            self.camera_canvas.create_text(
                max(1, self.camera_canvas.winfo_width()) / 2.0,
                max(1, self.camera_canvas.winfo_height()) / 2.0,
                text="Waiting for the live ego RGB stream...\nStart LVIMOT to populate this permanent camera panel.",
                fill="#8aa0b5",
                justify="center",
                width=700,
            )

        self.root.after(350, self._poll_camera_snapshot)

    def _redraw_map_health(self):
        if not hasattr(self, "map_canvas") or not self.map_canvas.winfo_exists():
            return
        canvas = self.map_canvas
        width = max(320, canvas.winfo_width())
        height = max(240, canvas.winfo_height())
        canvas.delete("all")

        snapshot = self.spatial_snapshot or {}
        points = snapshot.get("map_points_local") or []
        ground_points = snapshot.get("ground_points_local") or []
        structure_points = snapshot.get("structure_points_local") or []
        tracks = snapshot.get("tracks_local") or []
        meters = max(20.0, self._safe_float(snapshot.get("meters", 60.0)))

        cx = width / 2.0
        cy = height * 0.58
        px_per_m = min((width - 40) / (2.0 * meters), (height - 70) / (1.25 * meters))

        for radius_m in (10, 20, 30, 40, 50, 60):
            if radius_m > meters:
                continue
            r = radius_m * px_per_m
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#25384a")
            canvas.create_text(cx + 4, cy - r + 10, text=f"{radius_m}m", fill="#6c879d", anchor="w")
        canvas.create_line(0, cy, width, cy, fill="#25384a")
        canvas.create_line(cx, 0, cx, height, fill="#25384a")

        def draw_map_cells(items, color, size):
            if not items:
                return
            stride = max(1, len(items) // 5000)
            for point in items[::stride]:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                forward = self._safe_float(point[0])
                lateral = self._safe_float(point[1])
                px = cx + lateral * px_per_m
                py = cy - forward * px_per_m
                if 1 <= px < width - 1 and 1 <= py < height - 1:
                    canvas.create_rectangle(
                        px - size, py - size, px + size, py + size,
                        outline=color, fill=color
                    )

        if ground_points or structure_points:
            # V4 snapshot is already cleaned/rasterized by the visualizer.
            # Ground is intentionally dim; walls, poles and other structure are bright.
            draw_map_cells(ground_points, "#4c5965", 1)
            draw_map_cells(structure_points, "#b7c1ca", 1)
        elif points:
            # Backward compatibility with a V3 visualizer snapshot.
            draw_map_cells(points, "#9eabb7", 0)

        # Ego at the local origin, facing up.
        triangle = [cx, cy - 18, cx - 10, cy + 12, cx + 10, cy + 12]
        canvas.create_polygon(*triangle, fill="#00d084", outline="white")
        canvas.create_text(cx, cy + 25, text="EGO", fill="white")

        for track in tracks:
            forward = self._safe_float(track.get("forward", 0.0))
            lateral = self._safe_float(track.get("lateral", 0.0))
            px = cx + lateral * px_per_m
            py = cy - forward * px_per_m
            if -20 <= px <= width + 20 and -20 <= py <= height + 20:
                predicted = str(track.get("status", "ACTIVE")) != "ACTIVE"
                r = 5 if predicted else 6
                color = "#f59e0b" if predicted else "#ffb020"
                canvas.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="white")
                suffix = " P" if predicted else ""
                canvas.create_text(
                    px + 9, py - 8,
                    text=f"ID {track.get('track_id', '?')}{suffix}",
                    fill="white", anchor="w"
                )

        canvas.create_text(
            12, 12,
            text="CLEAN ROLLING 4-D LOCAL MAP | dark = ground/road | light = structures | orange = tracks",
            fill="#c8d6e5", anchor="nw", font=("Segoe UI", 10, "bold")
        )

        if not points:
            message = (
                "Spatial map stream is not available yet. Enable 'Open processed camera + spatial 4-D map' "
                "before START LVIMOT, then wait for the mapper to accumulate points."
                if not self.var_processed_display.get()
                else "Waiting for the processed visualizer to stream the first spatial map snapshot..."
            )
            canvas.create_text(width / 2, height / 2, text=message, fill="#8aa0b5", width=max(300, width - 120), justify="center")
        else:
            frame = snapshot.get("frame", "-")
            count = snapshot.get("map_voxels", len(points))
            stats = snapshot.get("map_render_stats") or {}
            clean_cells = int(self._safe_float(stats.get("clean_cells", len(points))))
            canvas.create_text(
                12, height - 16,
                text=f"frame {frame} | mapper voxels {int(count):,} | cleaned BEV cells {clean_cells:,}",
                fill="#8aa0b5", anchor="sw"
            )

        self.root.after(550, self._redraw_map_health)

    def _update_health(self, record: Dict):
        sync = record.get("live_sync") or {}
        mapping = record.get("mapping_4d") or {}
        estimate = record.get("state_estimation") or {}
        pose = estimate.get("pose") or []
        skew = self._safe_float(sync.get("timestamp_skew_s", 999.0))
        drops = sum(int(sync.get(k, 0) or 0) for k in ("camera_dropped", "lidar_dropped", "imu_dropped"))
        map_count = int(self._safe_float(mapping.get("total_static_voxels", 0)))
        pose_ok = isinstance(pose, list) and len(pose) >= 6
        frame_ok = self.processed_records >= 5
        sync_ok = skew <= 0.002
        map_ok = map_count > 0
        backend_ok = self.var_backend_status.get() in ("RUNNING", "COMPLETE")
        core_ok = backend_ok and frame_ok and sync_ok and map_ok and pose_ok

        if core_ok and drops == 0:
            self.var_health.set("CORE LIVE: PASS")
        elif core_ok:
            self.var_health.set("CORE LIVE: PASS / DROP WARNING")
        else:
            self.var_health.set("CHECKING")

        mot_text = "MOT observed" if self.max_tracks_seen > 0 else "MOT not yet exercised"
        self.var_health_detail.set(
            f"Frames received: {self.processed_records} | exact sensor skew: {1000.0 * skew:.2f} ms | "
            f"sensor drops: {drops} | mapper: {map_count:,} voxels | pose output: {'YES' if pose_ok else 'NO'} | "
            f"{mot_text} (max active tracks this run: {self.max_tracks_seen}). "
            "This proves the live dataflow is operating; quantitative accuracy still requires GT evaluation."
        )

    # ------------------------------------------------------------------
    # Config / file helpers
    # ------------------------------------------------------------------

    def _config_dict(self) -> Dict:
        return {
            "gui_version": 6,
            "carla_install": self.var_carla_install.get(),
            "calibration": self.var_calibration.get(),
            "host": self.var_host.get(),
            "port": self.var_port.get(),
            "render_quality": self.var_render_quality.get(),
            "preset": self.var_preset.get(),
            "autopilot": self.var_autopilot.get(),
            "gt_eval": self.var_gt_eval.get(),
            "processed_display": True,
            "show_trajectory": self.var_show_trajectory.get(),
            "candidate_hold": self.var_candidate_hold.get(),
            "vehicles": self.var_vehicles.get(),
            "duration": self.var_duration.get(),
            "fixed_delta": self.var_fixed_delta.get(),
            "seed": self.var_seed.get(),
            "sensor_timeout": self.var_sensor_timeout.get(),
            "output_root": self.var_output_root.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "fov": self.var_fov.get(),
            "yolo_model": self.var_yolo_model.get(),
            "yolo_size": self.var_yolo_size.get(),
            "confidence": self.var_confidence.get(),
            "yolo_device": self.var_yolo_device.get(),
        }

    def _apply_config_dict(self, data: Dict):
        mapping = {
            "carla_install": self.var_carla_install,
            "calibration": self.var_calibration,
            "host": self.var_host,
            "port": self.var_port,
            "render_quality": self.var_render_quality,
            "preset": self.var_preset,
            "autopilot": self.var_autopilot,
            "gt_eval": self.var_gt_eval,
            "processed_display": self.var_processed_display,
            "show_trajectory": self.var_show_trajectory,
            "candidate_hold": self.var_candidate_hold,
            "vehicles": self.var_vehicles,
            "duration": self.var_duration,
            "fixed_delta": self.var_fixed_delta,
            "seed": self.var_seed,
            "sensor_timeout": self.var_sensor_timeout,
            "output_root": self.var_output_root,
            "width": self.var_width,
            "height": self.var_height,
            "fov": self.var_fov,
            "yolo_model": self.var_yolo_model,
            "yolo_size": self.var_yolo_size,
            "confidence": self.var_confidence,
            "yolo_device": self.var_yolo_device,
        }
        for key, var in mapping.items():
            if key in data:
                var.set(data[key])
        self.var_processed_display.set(True)

    def _save_config(self, silent=False):
        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self._config_dict(), f, indent=2)
            if not silent:
                messagebox.showinfo(APP_TITLE, f"Saved configuration:\n{self.config_path}")
        except Exception as exc:
            if not silent:
                messagebox.showerror(APP_TITLE, str(exc))

    def _load_saved_config(self, silent=False):
        if not self.config_path.exists():
            if not silent:
                messagebox.showinfo(APP_TITLE, "No saved GUI configuration exists yet.")
            return
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._apply_config_dict(data)
            if os.name != "nt":
                carla_value = self.var_carla_install.get()
                calibration_value = self.var_calibration.get()
                if re.match(r"^[A-Za-z]:[\\/]", carla_value):
                    self.var_carla_install.set(DEFAULT_CARLA_INSTALL)
                if re.match(r"^[A-Za-z]:[\\/]", calibration_value) or not Path(calibration_value).expanduser().exists():
                    self.var_calibration.set(DEFAULT_CALIBRATION)
            if int(data.get("gui_version", 0) or 0) < 4:
                # V3 could save the optional blue trail as enabled.  V4 starts
                # old configurations with it off so the spatial map remains the
                # primary visualization unless the user explicitly re-enables it.
                self.var_show_trajectory.set(False)
            if not silent:
                messagebox.showinfo(APP_TITLE, "Configuration loaded.")
        except Exception as exc:
            if not silent:
                messagebox.showerror(APP_TITLE, str(exc))

    def _browse_carla(self):
        current = Path(os.path.expanduser(self.var_carla_install.get())) if self.var_carla_install.get() else self.project_root
        initialdir = str(current.parent if current.suffix else current)
        if os.name == "nt":
            path = filedialog.askdirectory(initialdir=initialdir)
        else:
            path = filedialog.askopenfilename(
                title="Select CarlaUE4.sh",
                initialdir=initialdir,
                filetypes=[("CARLA launcher", "CarlaUE4.sh"), ("Shell scripts", "*.sh"), ("All files", "*")],
            )
        if path:
            self.var_carla_install.set(path)

    def _browse_calibration(self):
        path = filedialog.askopenfilename(
            initialdir=str(Path(self.var_calibration.get()).parent) if self.var_calibration.get() else str(self.project_root),
            filetypes=[("JSON calibration", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.var_calibration.set(path)

    def _archive_previous_live_outputs(self):
        """Prevent a previous run from being replayed into a new GUI session."""
        output = Path(self.var_output_root.get())
        if not output.is_absolute():
            output = self.project_root / output
        live_dir = output / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for name in ("live_frames.jsonl", "live_results.json", "live_map_snapshot.json", "live_camera_preview.png"):
            path = live_dir / name
            if not path.exists():
                continue
            archived = live_dir / f"{path.stem}_{stamp}{path.suffix}"
            try:
                path.replace(archived)
            except Exception:
                try:
                    path.unlink()
                except Exception:
                    pass

    def _open_outputs(self):
        path = Path(self.var_output_root.get())
        if not path.is_absolute():
            path = self.project_root / path
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        else:
            opener = shutil.which("xdg-open") or shutil.which("gio")
            if opener:
                args = [opener, str(path)] if Path(opener).name == "xdg-open" else [opener, "open", str(path)]
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                messagebox.showinfo(APP_TITLE, str(path))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        for widget in self._control_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="readonly" if enabled else "disabled")
                else:
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _append_console(self, line: str):
        self.console.insert("end", line + "\n")
        # Bound GUI memory during long demonstrations.
        try:
            line_count = int(self.console.index("end-1c").split(".")[0])
            if line_count > 2500:
                self.console.delete("1.0", "500.0")
        except Exception:
            pass
        self.console.see("end")

    def _reset_runtime_dashboard(self):
        self.latest_record = {}
        self.pose_history = []
        self.map_history = []
        self.track_history = {}
        self.spatial_snapshot = {}
        self._spatial_snapshot_mtime_ns = None
        self._camera_snapshot_mtime_ns = None
        self._camera_photo = None
        self._camera_source_image = None
        if self._camera_resize_after is not None:
            try:
                self.root.after_cancel(self._camera_resize_after)
            except tk.TclError:
                pass
            self._camera_resize_after = None
        self.max_tracks_seen = 0
        self.processed_records = 0
        self.last_record_frame = None
        self.var_frame.set("-")
        self.var_carla_frame.set("-")
        self.var_total_ms.set("-")
        self.var_fps.set("-")
        self.var_rss.set("-")
        self.var_tracks.set("0")
        self.var_map.set("0")
        self.var_stationary.set("-")
        self.var_sync.set("-")
        self.var_camlag.set("-")
        self.var_pump.set("-")
        self.var_pose.set("x=-  y=-  z=-  yaw=-")
        self.var_speed.set("-")
        self.var_tracks_note.set("No active published tracks yet.")
        self.var_map_note.set("4-D spatial map waiting for live data.")
        self.var_camera_note.set("Enable processed camera/map streaming before START LVIMOT to embed the live ego RGB view here.")
        self.var_health.set("WAITING")
        self.var_health_detail.set("Start LVIMOT to evaluate live system health.")
        for item in self.track_table.get_children():
            self.track_table.delete(item)
        if hasattr(self, "camera_canvas") and self.camera_canvas.winfo_exists():
            self.camera_canvas.delete("all")

    @staticmethod
    def _safe_float(value) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def _on_close(self):
        if self.backend_process is not None and self.backend_process.poll() is None:
            if not messagebox.askyesno(APP_TITLE, "LVIMOT is still running. Stop it and close the GUI?"):
                return
            self._stop_backend()
        self.root.after(250, self.root.destroy)


def main():
    root = tk.Tk()
    app = LVIMOTLiveGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()