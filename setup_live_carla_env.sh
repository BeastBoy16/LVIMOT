#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="$PROJECT_DIR/.venv_live"

printf '%s\n' '======================================================================'
printf '%s\n' 'LVIMOT Linux environment - CARLA 0.9.16 / Python 3.12'
printf '%s\n' '======================================================================'

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] $PYTHON_BIN was not found. Install Python 3.12, python3.12-venv and python3-tk." >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)' >/dev/null 2>&1; then
    echo "[ERROR] Linux release expects Python 3.12. Got: $($PYTHON_BIN --version 2>&1)" >&2
    exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo '[1/6] Creating .venv_live ...'
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo '[1/6] Reusing .venv_live'
fi
VPY="$VENV_DIR/bin/python"

echo '[2/6] Updating pip tooling ...'
"$VPY" -m pip install --upgrade pip setuptools wheel

echo '[3/6] Installing LVIMOT dependencies ...'
"$VPY" -m pip install --upgrade --force-reinstall -r requirements-linux.txt

echo '[4/6] Installing PyTorch ...'
if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_INDEX="${LVIMOT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
else
    TORCH_INDEX="${LVIMOT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
fi
"$VPY" -m pip install --upgrade --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url "$TORCH_INDEX"

echo '[5/6] Verifying packages ...'
"$VPY" - <<'PY'
import sys, carla, cv2, numpy, scipy, torch
from PIL import Image
print('Python:', sys.version.replace('\n', ' '))
print('CARLA module:', getattr(carla, '__file__', 'unknown'))
print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('SciPy:', scipy.__version__)
print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY

echo '[6/6] Ready.'
echo 'Run: ./run_live_lvimot.sh'
