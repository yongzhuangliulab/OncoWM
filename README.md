# OncoWM


OncoWM is a multimodal survival-prediction research project. The repository defines a standard interface for the **processed patient-level inputs consumed by the model**. Raw tables may differ across hospitals, cohorts, and public datasets; users should convert their own data to the documented result formats.

The repository contains the v5.3 training, inference, and survival-evaluation pipeline, together with example conversion scripts for clinical data, survival labels, and mutation features. Whole-slide image and transcriptome feature extraction are external preprocessing steps.

For installation and end-to-end commands, start with the [Usage Guide](USAGE.md).

## Repository layout

```text
OncoWM/
├── README.md
├── USAGE.md
├── environment.yml
├── requirements.txt
├── main.py
├── configs/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── models/OncoWM.py
│   └── datasets/medical.py
├── scripts/
├── data_processing/
├── list/
├── data/
│   ├── raw/
│   └── processed/
├── pretrained_models/
├── outputs/
├── docs/
└── tests/
```

## Processed model inputs

All modalities must use the same de-identified `sample_id`.

| Suggested file | Content | Minimum requirement |
| --- | --- | --- |
| `survival.csv` | Survival labels | `sample_id,cancer_type,survival_time,survival_event` |
| `clinical.pt` | Clinical text tokens | `dict[sample_id, record]` with text and token tensors |
| `mutation.pt` | Ordered mutation vectors | `dict[sample_id, Tensor]`; fixed gene order and equal length |
| `pathology.pt` | Patient-level pathology embeddings | `dict[sample_id, Tensor]`; default dimension 768 |
| `transcriptome.pt` | Patient-level transcriptome embeddings | Dimension must match the YAML configuration |

For a single cohort:

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

The TCGA configurations use modality-specific subdirectories under `data/processed/tcga/`. See [TCGA configuration documentation](configs/tcga/README.md).

## Downloaded model weights

Place downloaded third-party models and published OncoWM checkpoints under:

```text
pretrained_models/
├── medgemma-1.5-4b-it/
├── conch-v1.5/
├── titan/
├── bulkformer/
└── oncowm/<CANCER>_<ENDPOINT>/full_cohort/prognosis_full_final.pth.tar
```

The training code does not read SVS files directly. A whole-slide image must first be processed into patch features and then a patient-level pathology embedding, for example:

```text
SVS → CONCH patch features → TITAN embedding → pathology.pt
```

See [Downloaded Model Layout](pretrained_models/README.md).

## Installation

```bash
conda env create -f environment.yml
conda activate oncowm
```

Alternatively, in an environment with a compatible CUDA stack:

```bash
pip install -r requirements.txt
```

Put the complete MedGemma model directory at:

```text
pretrained_models/medgemma-1.5-4b-it/
```

## Training

Run alignment and prognosis training in sequence:

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

The repository includes 84 TCGA five-fold configurations covering 21 cohorts and four endpoints: OS, DSS, DFI, and PFI.

## Inference and evaluation

```bash
python -m src.inference \
  --fname configs/inference.ALL_DFI_full_cohort.yaml

python -m src.evaluate \
  --scores outputs/inference/ALL/ALL_DFI/prognosis/Full_modal/full_cohort/prognosis_risk_scores.csv \
  --yaml configs/inference.ALL_DFI_full_cohort.yaml
```

For details, see the [Usage Guide](USAGE.md) and [Training, Inference, and Survival Evaluation](docs/training_and_inference.md).

## Adapting your own data

The scripts in `data_processing/` are adaptation examples rather than mandatory raw-data formats. Users may map different column names through CLI options or modify the scripts for their own cohort, but the outputs must satisfy [the model input contract](docs/data_contract.md).

## Data safety

Do not commit patient data, identifier mapping tables, SVS files, generated `.pt/.h5` features, third-party model weights, OncoWM checkpoints, tokens, or server logs. These paths are ignored by Git, but always review:

```bash
git status --short
```

## License

The code is distributed under the [Apache License 2.0](LICENSE). Users must also follow the licenses and access requirements of MedGemma, external pretrained models, and source datasets.
