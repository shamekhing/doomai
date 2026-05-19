#!/usr/bin/env bash
# One-time setup for the DOOM assignment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
VENV="$ROOT/.venv"

echo "=== [1/3] Creating virtual environment ==="
python3 -m venv --system-site-packages "$VENV"

echo "=== [2/3] Installing Python dependencies ==="
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install \
  numpy \
  torch \
  torchvision \
  matplotlib \
  tqdm \
  einops \
  pillow \
  imageio \
  imageio-ffmpeg \
  vizdoom \
  portpicker \
  onnx \
  onnxscript \
  onnx2pytorch \
  gymnasium \
  pygame \
  ipykernel

echo "=== [3/3] Registering Jupyter kernel ==="
"$VENV/bin/python" -m ipykernel install --user \
  --name=drl-doom \
  --display-name "Python (DRL-Doom)" 2>/dev/null || true

echo ""
echo "Done."
echo "Activate with : source .venv/bin/activate"
echo "Collect demos : bash scripts/run.sh --collect-demos --episodes 5"
echo "Train with    : bash scripts/run.sh --hybrid-demos --demo-dir demos/human"
