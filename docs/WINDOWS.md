# Windows release

## Supported stack

- Windows 10/11 x64
- CARLA 0.9.13
- Python 3.8 x64
- NVIDIA CUDA-capable GPU recommended

## Setup

1. Install CARLA 0.9.13 and Python 3.8 x64.
2. Extract the Windows release to a path without unusual permission restrictions.
3. Run `setup_live_carla_env.bat`.
4. Set `CARLA_LAUNCHER` to the full path of `CarlaUE4.exe`, or browse to it from the dashboard.
5. Start CARLA and then run `run_live_lvimot.bat`.

Example for the current terminal:

```bat
set CARLA_LAUNCHER=D:\CARLA_0.9.13\WindowsNoEditor\CarlaUE4.exe
start_carla.bat
run_live_lvimot.bat
```

The dataset is not included in the repository or release.
