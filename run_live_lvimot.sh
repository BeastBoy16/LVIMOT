#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
VPY="$PROJECT_DIR/.venv_live/bin/python"
if [[ ! -x "$VPY" ]]; then
    "$PROJECT_DIR/setup_live_carla_env.sh"
fi
exec "$VPY" -u "$PROJECT_DIR/src/lvimot_live_gui.py"
