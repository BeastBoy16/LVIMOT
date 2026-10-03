# LVIMOT

**LiDAR-Visual-Inertial Localization, Multi-Object Tracking and 4-D Mapping in CARLA**

LVIMOT is a research-oriented autonomous-driving perception and estimation pipeline that combines RGB camera, LiDAR and IMU data for ego-state estimation, multimodal object detection/tracking and a rolling local 4-D map.

## Features

- RGB + LiDAR + IMU live CARLA pipeline
- Camera/LiDAR fusion for object candidates
- Multi-object tracking with temporal state estimation
- Ego-motion estimation and sensor integrity reporting
- LiDAR deskewing and temporal/geometric feature processing
- Sliding-window/factor-graph estimation components
- Rolling local 4-D spatial map
- Live Tk dashboard with permanent ego RGB view, BEV, tracks, map health and console
- Optional CARLA ground-truth evaluation path kept separate from sensor-based estimation
- Offline CARLA-sequence processing and test suite

## Releases

Two platform releases are provided because the validated CARLA/Python combinations differ:

| Release | CARLA | Python | Setup |
| --- | --- | --- | --- |
| LVIMOT Windows | 0.9.13 | 3.8 x64 | `setup_live_carla_env.bat` |
| LVIMOT Linux | 0.9.16 | 3.12 | `./setup_live_carla_env.sh` |

See [`docs/WINDOWS.md`](docs/WINDOWS.md) and [`docs/LINUX.md`](docs/LINUX.md).

## Quick start

### Windows

```bat
setup_live_carla_env.bat
set CARLA_LAUNCHER=D:\path\to\CARLA\CarlaUE4.exe
run_live_lvimot.bat
```

### Linux

```bash
chmod +x *.sh
./setup_live_carla_env.sh
export CARLA_LAUNCHER=/path/to/CarlaUE4.sh
./run_live_lvimot.sh
```

CARLA may be started separately or from the dashboard after selecting the launcher path.

## Repository structure

```text
LVIMOT/
├── config/                  Calibration and runtime configuration
├── docs/                    Platform guides and verification history
├── src/                     LVIMOT implementation
├── tests/                   Automated and diagnostic tests
├── setup_live_carla_env.*   Platform environment bootstrap
├── run_live_lvimot.*        Dashboard launchers
├── start_carla.*            Convenience CARLA launchers
└── requirements-*.txt       Platform dependency baselines
```

## Data and generated files

CARLA installations, datasets, virtual environments, model weights, generated outputs and user-local GUI configuration are intentionally excluded from Git. The CARLA dataset used during development is generated separately.

## Notes

This repository is intended for research/development use. Quantitative claims should be based on the evaluation outputs for the exact configuration being tested. CARLA ground truth is used only in explicit evaluation paths and is not intended as an input to the live sensor-estimation pipeline.
