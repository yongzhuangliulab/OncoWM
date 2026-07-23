# M-JEPA Usage Guide


This guide explains how to install M-JEPA, prepare processed inputs and model weights, train a model, run inference with a published full-cohort checkpoint, and evaluate survival predictions.

## 1. Installation

```bash
git clone https://github.com/yongzhuangliulab/Multimoda_JEPA.git
cd Multimoda_JEPA

conda env create -f environment.yml
conda activate mjepa
```

Update an existing environment:

```bash
git pull origin main
conda env update -n mjepa -f environment.yml --prune
```

Check Python, PyTorch, and CUDA:

```bash
python -c "
import sys
import torch
print('python:', sys.executable)
print('torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
"
```

The current dependency set requires PyTorch 2.6 or newer.

## 2. Prepare processed data

M-JEPA does not directly consume hospital source tables, SVS files, or raw expression matrices. Convert them to patient-level model inputs first.

For a single cohort:

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

For the provided TCGA configurations:

```text
data/processed/tcga/
├── survival/<CANCER>_<ENDPOINT>_dataset.csv
├── pathology/TCGA_TITAN_first_slice_dict.pt
├── clinical/TCGA_data_clinical_patient_prompt.pt
├── mutation/TCGA-<CANCER>.pt
└── transcriptome/TCGA-<CANCER>.pt
```

| Input | Format |
| --- | --- |
| Survival labels | CSV with `sample_id,cancer_type,survival_time,survival_event` |
| Clinical features | `dict[sample_id, record]` containing token tensors |
| Mutation features | Equal-length `dict[sample_id, Tensor]` in the published gene order |
| Pathology features | `dict[sample_id, Tensor]`; default dimension 768 |
| Transcriptome features | `dict[sample_id, Tensor]`; dimension must match YAML |

All inputs must use identical `sample_id` rules. Training keeps the intersection of the survival table and all four modalities.

Read:

- [Model Input Data Contract](docs/data_contract_EN.md)
- [PyTorch PT Formats](docs/pt_formats_EN.md)
- [Data Conversion Scripts](data_processing/README_EN.md)

Inspect trusted PT files:

```bash
python data_processing/inspect_pt.py data/processed/clinical.pt --max-items 3
python data_processing/inspect_pt.py data/processed/mutation.pt --max-items 3
```

## 3. Prepare model weights

```text
pretrained_models/
├── medgemma-1.5-4b-it/
├── conch-v1.5/
├── titan/
├── bulkformer/
└── mjepa/
    └── <CANCER>_<ENDPOINT>/
        └── full_cohort/
            └── prognosis_full_final.pth.tar
```

- MedGemma is the clinical text encoder used during training and inference.
- CONCH and TITAN are used by an external preprocessing pipeline to produce pathology embeddings.
- BulkFormer is used by an external preprocessing pipeline to produce transcriptome embeddings.
- Published M-JEPA checkpoints belong under `pretrained_models/mjepa/`.

Keep the complete original directory structure of each third-party model, including configuration, tokenizer, and weight files. See [Downloaded Model Layout](pretrained_models/README_EN.md).

Once `pathology.pt` has been produced, TITAN is no longer loaded by M-JEPA training or inference.

## 4. Configuration files

```text
configs/
├── train.example.yaml
├── inference.example.yaml
├── inference.ALL_DFI_full_cohort.yaml
└── tcga/
```

For your own cohort:

```bash
cp configs/train.example.yaml configs/train_my_cohort.yaml
cp configs/inference.example.yaml configs/inference_my_cohort_full_cohort.yaml
```

Verify:

- all CSV and PT paths under `data`;
- `models.text_encoder_name`;
- `path_dim`, `geno_raw_dim`, `geno_dim`, and `tran_dim`;
- batch sizes;
- checkpoint and output paths;
- the survival endpoint.

`geno_raw_dim` must equal the true mutation-vector length. Do not mix OS, DSS, DFI, or PFI labels and checkpoints.

## 5. Training

Training has two stages:

1. `align` learns the multimodal representation;
2. `prognosis` loads the matching alignment checkpoint, freezes the backbone, and trains the survival head.

Run both stages in order:

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

Or use:

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

Use `--force` to rerun a task that already has a completion marker.

## 6. Inference

Run a copied and edited inference configuration:

```bash
python -m src.inference \
  --fname configs/inference_my_cohort_full_cohort.yaml
```

Or use the provided full-cohort example:

```bash
python -m src.inference \
  --fname configs/inference.ALL_DFI_full_cohort.yaml
```

Prognosis inference produces:

```text
outputs/inference/<experiment>/<run_name>/
├── prog_embeddings.npz
└── prognosis_risk_scores.csv
```

Inference permits a disabled modality path to be YAML `null`. The loader still intersects the sample IDs of all enabled modalities.

Published M-JEPA checkpoints omit the frozen MedGemma backbone. Missing keys that belong only to the base `text_model` may therefore be reported while the backbone is loaded from `models.text_encoder_name`.

## 7. Survival evaluation

```bash
python -m src.evaluate \
  --scores outputs/inference/<experiment>/<run_name>/prognosis_risk_scores.csv \
  --yaml configs/inference_my_cohort_full_cohort.yaml
```

The evaluator joins labels by `sample_id` and writes:

```text
evaluation_metrics.csv
```

Reported metrics include the C-index, 1/3/5-year time-dependent AUC, and integrated AUC.

## 8. Troubleshooting

### PyTorch version error

If an error says that PyTorch 2.6 or newer is required:

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Use the same Conda environment for dependency installation and execution.

### CUDA out of memory

Reduce `align_batch_size`, `prognosis_batch_size`, or the inference `batch_size`.

### No shared sample IDs

Compare the CSV `sample_id` values with the dictionary keys in every enabled PT file. Equal sample counts do not imply matching IDs.

### Missing alignment checkpoint

Run alignment first or use `--task both`. Each prognosis fold requires the corresponding `align-best.pth.tar`.

## 9. Data safety

Do not commit patient data, identifier maps, `.pt/.h5` features, SVS files, third-party weights, M-JEPA checkpoints, access tokens, or server logs. Review:

```bash
git status --short
```
