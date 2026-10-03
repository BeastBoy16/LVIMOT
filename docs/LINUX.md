# Linux release

## Supported stack

- Ubuntu x86_64
- CARLA 0.9.16
- Python 3.12
- NVIDIA CUDA-capable GPU recommended

## Setup

```bash
chmod +x *.sh
./setup_live_carla_env.sh
export CARLA_LAUNCHER=/path/to/CarlaUE4.sh
./start_carla.sh
```

In another terminal:

```bash
./run_live_lvimot.sh
```

For remote GPU VMs, CARLA can be launched off-screen:

```bash
./start_carla.sh -RenderOffScreen
```

Avoid project-directory names containing spaces when using third-party environment/bootstrap tools.
The dataset is not included in the repository or release.
