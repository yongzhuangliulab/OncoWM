#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:?Usage: bash scripts/evaluate_survival.sh <inference.yaml> <risk_scores.csv>}"
SCORES_PATH="${2:?Usage: bash scripts/evaluate_survival.sh <inference.yaml> <risk_scores.csv>}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m src.evaluate --yaml "$CONFIG_PATH" --scores "$SCORES_PATH"
