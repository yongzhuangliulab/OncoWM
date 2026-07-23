#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:?Usage: bash scripts/run_inference.sh <inference.yaml>}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m src.inference --fname "$CONFIG_PATH"
