# GitHub release plan

The repository contains one cross-platform LVIMOT codebase. Publish two platform-specific binary/source bundles as separate GitHub releases.

## Release 1: LVIMOT Windows v1.0.0

Suggested tag: `windows-v1.0.0`

Asset: `LVIMOT-Windows-v1.0.0.zip`

Runtime baseline: Windows + CARLA 0.9.13 + Python 3.8.

## Release 2: LVIMOT Linux v1.0.0

Suggested tag: `linux-v1.0.0`

Asset: `LVIMOT-Linux-v1.0.0.zip`

Runtime baseline: Ubuntu + CARLA 0.9.16 + Python 3.12.

Do not upload local virtual environments, CARLA itself, generated outputs, datasets, or user-specific `lvimot_gui_config.json` files.
