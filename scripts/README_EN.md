# Command Wrappers

[中文](README.md)

The scripts can be invoked from any directory and automatically switch to the repository root. They use the active environment's `python` by default. Override it with `PYTHON_BIN=/path/to/python`.

```bash
bash scripts/train_two_stage.sh configs/tcga/train_BRCA_OS.yaml BRCA_OS

cp configs/inference.example.yaml configs/inference_BRCA_OS_full_cohort.yaml
bash scripts/run_inference.sh configs/inference_BRCA_OS_full_cohort.yaml

bash scripts/evaluate_survival.sh \
  configs/inference_BRCA_OS_full_cohort.yaml \
  outputs/inference/BRCA_OS_full_cohort/prognosis_risk_scores.csv
```
