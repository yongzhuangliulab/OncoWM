# Local Data Directory


- `raw/`: user-provided source data; formats may differ by cohort.
- `processed/`: model inputs that satisfy `docs/data_contract.md`.

TCGA configurations use:

```text
processed/tcga/
├── survival/
├── pathology/
├── clinical/
├── mutation/
└── transcriptome/
```

Patient-level files in both directories are ignored by Git. The repository tracks only directory placeholders.

Do not place TITAN, MedGemma, or other model weights here. Downloaded weights belong under the repository-level `pretrained_models/` directory.
