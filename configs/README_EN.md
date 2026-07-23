# Configuration Files

[中文](README.md)

- `train.example.yaml`: generic five-fold, two-stage training template.
- `inference.example.yaml`: generic full-cohort checkpoint inference template.
- `inference.ALL_DFI_full_cohort.yaml`: ALL_DFI full-cohort, full-modality inference example.
- `tcga/`: 84 training configurations covering 21 cohorts and OS, DSS, DFI, and PFI endpoints.

Copy a template and edit it for your cohort and local paths. Tensor dimensions and checkpoint architecture must match, especially `geno_raw_dim`, `path_dim`, and `tran_dim`.

See the [Usage Guide](../USAGE_EN.md) and [TCGA Configuration Guide](tcga/README_EN.md).
