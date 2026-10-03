#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
VPY="$PROJECT_DIR/.venv_live/bin/python"
[[ -x "$VPY" ]] || "$PROJECT_DIR/setup_live_carla_env.sh"
exec "$VPY" -u "$PROJECT_DIR/src/carla_main.py" --live --evaluate-ground-truth "$@"
