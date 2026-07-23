#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:?Usage: bash scripts/train_two_stage.sh <train.yaml> <run_name>}"
RUN_NAME="${2:?Usage: bash scripts/train_two_stage.sh <train.yaml> <run_name>}"

cd "$ROOT_DIR"
"$PYTHON_BIN" main.py --fname "$CONFIG_PATH" --task both --run-name "$RUN_NAME"
