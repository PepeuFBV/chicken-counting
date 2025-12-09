#!/usr/bin/env bash

# Lightweight runner for training inside the project's venv.
# Usage: ./scripts/run_train.sh [extra python args]
# Examples:
#   ./scripts/run_train.sh                      # runs default training cmd
#   ./scripts/run_train.sh --epochs 30 --batch_size 4
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_train.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/.venv/bin/activate"

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH"
else
  echo "Warning: virtualenv activate not found at $VENV_PATH" >&2
fi

: ${CUDA_VISIBLE_DEVICES:=0}

python src/train.py \
  --data_dir data/dataset \
  --out_dir checkpoints \
  --epochs 50 \
  --batch_size 6 \
  --device auto "$@"
