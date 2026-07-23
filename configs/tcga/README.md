# TCGA Five-Fold Training Configurations


This directory contains 84 configurations accepted by the  five-fold training pipeline: 21 cohorts × four endpoints.

- `OS`: Overall Survival
- `DSS`: Disease-Specific Survival
- `DFI`: Disease-Free Interval
- `PFI`: Progression-Free Interval

## Included cohorts and key dimensions

| Cohort | `geno_raw_dim` | Align batch | Prognosis batch |
| --- | ---: | ---: | ---: |
| ALL | 19766 | 90 | 90 |
| BLCA | 17533 | 64 | 64 |
| BRCA | 16658 | 90 | 128 |
| CESC | 16092 | 64 | 64 |
| COAD | 18636 | 64 | 64 |
| ESCA | 11810 | 32 | 32 |
| HNSC | 16509 | 64 | 90 |
| KIRC | 10923 | 90 | 128 |
| KIRP | 9920 | 64 | 64 |
| LGG | 11975 | 64 | 64 |
| LIHC | 14338 | 64 | 64 |
| LUAD | 17699 | 64 | 90 |
| LUSC | 17621 | 64 | 90 |
| PAAD | 11207 | 32 | 32 |
| PCPG | 1680 | 32 | 200 |
| PRAD | 11066 | 64 | 500 |
| SARC | 8877 | 32 | 64 |
| SKCM | 18332 | 64 | 64 |
| STAD | 18003 | 64 | 64 |
| THCA | 4163 | 64 | 500 |
| UCEC | 18993 | 64 | 64 |

These values come from the original experiment configurations and were not normalized. `geno_raw_dim` must equal the true length of `TCGA-<CANCER>.pt`.

## Data layout

For example, `train_BRCA_OS.yaml` reads:

```text
data/processed/tcga/
├── survival/BRCA_OS_dataset.csv
├── pathology/TCGA_TITAN_first_slice_dict.pt
├── clinical/TCGA_data_clinical_patient_prompt.pt
├── mutation/TCGA-BRCA.pt
└── transcriptome/TCGA-BRCA.pt
```

MedGemma is expected at:

```text
pretrained_models/medgemma-1.5-4b-it/
```

## Running

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

Output:

```text
outputs/train/BRCA/BRCA_OS/
```

## Configurations not included

GBM, OV, READ, TGCT, and THYM source YAML files contain unresolved empty hyperparameters and are not published as runnable configurations. Full-cohort and unimodal training configurations are also excluded because the streamlined training entry point does not implement their original experiment semantics.
