#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-pretrained_models/medgemma-1.5-4b-it}"
INPUT_CSV="${INPUT_CSV:-data/raw/clinical.csv}"
OUTPUT_PT="${OUTPUT_PT:-data/processed/clinical.pt}"

cd "$ROOT_DIR"
"$PYTHON_BIN" data_processing/process_clinical_data_qu.py \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_PT" \
  --model-path "$MODEL_PATH" \
  --batch-size "${BATCH_SIZE:-16}" \
  --max-length "${MAX_LENGTH:-512}"
