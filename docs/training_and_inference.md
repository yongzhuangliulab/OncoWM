# Training, Inference, and Survival Evaluation


For installation and file placement, see the [Usage Guide](../USAGE_EN.md).

## Current pipeline

| File | Responsibility |
| --- | --- |
| `main.py` | Two-stage training entry point and completion markers |
| `src/train.py` | Five-fold training loop |
| `src/models/mjepa.py` | M-JEPA model, alignment loss, and prognosis head |
| `src/datasets/medical.py` | Multimodal dataset and stratified folds |
| `src/inference.py` | Checkpoint inference, embeddings, and risk scores |
| `src/evaluate.py` | C-index, time-dependent AUC, and integrated AUC |

## 1. Two-stage training

Training must run in order:

1. `align` learns the multimodal representation and saves `align-best.pth.tar` per fold.
2. `prognosis` loads the matching alignment checkpoint, freezes the backbone, and trains the prognosis head.

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

For a custom cohort:

```bash
cp configs/train.example.yaml configs/train_my_cohort.yaml
python main.py \
  --fname configs/train_my_cohort.yaml \
  --task both \
  --run-name my_cohort
```

Equivalent wrapper:

```bash
bash scripts/train_two_stage.sh configs/tcga/train_BRCA_OS.yaml BRCA_OS
```

Example output:

```text
outputs/train/BRCA/BRCA_OS/
├── fold_0/
│   ├── align-best.pth.tar
│   ├── prognosis-best.pth.tar
│   ├── train.csv
│   └── val.csv
├── fold_1/ ... fold_4/
├── align-best-overall.pth.tar
├── prognosis-best-overall.pth.tar
└── *_cross_validation_results.csv
```

Training requires pathology, clinical, mutation, and transcriptome PT dictionaries. It keeps only sample IDs shared with the survival CSV. `geno_raw_dim`, `path_dim`, and `tran_dim` must match the actual tensors.

Prognosis training requires the matching alignment checkpoint unless `allow_prognosis_from_scratch: true` is explicitly set.

## 2. Full-cohort inference

Published weights use:

```text
pretrained_models/mjepa/<CANCER>_<ENDPOINT>/full_cohort/prognosis_full_final.pth.tar
```

```bash
cp configs/inference.example.yaml configs/inference_BRCA_OS_full_cohort.yaml
python -m src.inference \
  --fname configs/inference_BRCA_OS_full_cohort.yaml
```

Or:

```bash
bash scripts/run_inference.sh configs/inference_BRCA_OS_full_cohort.yaml
```

Inference permits unavailable modalities to be YAML `null`. The dimensions must still match the checkpoint architecture.

Prognosis inference writes:

```text
outputs/inference/<run_name>/
├── prog_embeddings.npz
└── prognosis_risk_scores.csv
```

## 3. Survival evaluation

```bash
python -m src.evaluate \
  --scores outputs/inference/BRCA_OS_full_cohort/prognosis_risk_scores.csv \
  --yaml configs/inference_BRCA_OS_full_cohort.yaml
```

Labels are joined by `sample_id`, and `evaluation_metrics.csv` is written next to the risk scores. Larger `risk_score` values represent higher risk. Event coding is `0=censored` and `1=event`; survival time is measured in months.

The current time-dependent AUC implementation uses the same cohort to construct the survival distribution and compute the metric. Treat it as a within-cohort descriptive result unless an external statistical protocol is provided.
