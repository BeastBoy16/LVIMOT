#!/usr/bin/env bash
set -euo pipefail
CARLA_LAUNCHER="${CARLA_LAUNCHER:-$HOME/CARLA/CarlaUE4.sh}"
if [[ ! -f "$CARLA_LAUNCHER" ]]; then
    echo "[ERROR] CARLA launcher not found: $CARLA_LAUNCHER" >&2
    echo 'Set CARLA_LAUNCHER to your CarlaUE4.sh path, for example:' >&2
    echo '  export CARLA_LAUNCHER=/opt/CARLA/CarlaUE4.sh' >&2
    exit 2
fi
chmod +x "$CARLA_LAUNCHER" 2>/dev/null || true
cd "$(dirname "$CARLA_LAUNCHER")"
exec "$CARLA_LAUNCHER" -quality-level=Low -nosound -carla-rpc-port=2000 "$@"
