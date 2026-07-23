# Model Input Data Contract


This document defines the **processed results consumed by M-JEPA**. Raw CSV/TSV columns and file organization may differ between cohorts; data providers are responsible for local adaptation.

## 1. General conventions

- Organize every modality by patient or sample with the unique key `sample_id`.
- `sample_id` must be de-identified, non-empty, and read as a string.
- The same patient must use exactly the same ID in every result file.
- Save PT files with `torch.save` and load only trusted files.
- Before training or inference, report each modality count, the ID intersection, and excluded counts.

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

## 2. Survival labels

`survival.csv` must provide:

| Column | Type | Constraint |
| --- | --- | --- |
| `sample_id` | string | Unique, non-empty, de-identified |
| `cancer_type` | string | Standard abbreviation such as `LUAD` |
| `survival_time` | float | Valid numeric value in months |
| `survival_event` | int | `0=censored/alive`, `1=event/death` |

```csv
sample_id,cancer_type,survival_time,survival_event
P001,LUAD,25.3,0
P002,LUSC,13.0,1
```

## 3. Clinical features

`clinical.pt` must be a dictionary keyed by `sample_id`:

```python
{
    "P001": {
        "text": "The patient ...",
        "input_ids": torch.Tensor,
        "attention_mask": torch.Tensor,
        "token_type_ids": torch.Tensor,
    }
}
```

- Token tensors are one-dimensional and equal length; the current default is 512.
- `text` is the standardized clinical summary generated with the project prompt.
- If a tokenizer does not return `token_type_ids`, store an all-zero tensor.
- Missing source fields must be represented as `Unknown`, not guessed by the model.

## 4. Mutation features

`mutation.pt` must contain equal-length binary vectors:

```python
{
    "P001": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "P002": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
}
```

- Shape is `[G]` and dtype is `float32`.
- Position `i` corresponds exactly to line `i` of `list/TCGA-<CANCER>_gene_list.txt`.
- A missing listed gene is encoded as `0.0`; unlisted genes are ignored.
- Publish the matching gene list with a model release and record its SHA-256 when possible.

## 5. Pathology features

```python
{"P001": torch.Tensor}  # [D_path] or [1, D_path]
```

- Default `D_path=768`.
- The dimension must equal `models.path_dim`.
- Multiple slides for one patient must be aggregated during preprocessing.
- This repository does not include the complete SVS feature-extraction pipeline.

## 6. Transcriptome features

```python
{"P001": torch.Tensor}  # [D_tran]
```

- Every sample must have the same length.
- The dimension must equal `models.tran_dim`.
- This repository does not include raw-expression feature extraction.

## 7. Training and inference modalities

- Training requires valid paths for all four PT modalities and keeps their intersection with the survival CSV.
- Inference may set a missing modality to YAML `null`, but enabled modalities are still intersected.
- Model dimensions are determined by the checkpoint and must not change when a modality is disabled.

## 8. Minimum quality checks

1. Counts, empty IDs, and duplicate IDs.
2. Cross-modality ID intersections and missing counts.
3. Numeric survival times and binary event values.
4. Equal clinical token fields, dtypes, and lengths.
5. Equal binary mutation vectors matching the gene list length.
6. No NaN or Inf in floating-point tensors.
7. Pathology and transcriptome dimensions matching YAML.

See [PyTorch PT Formats](pt_formats_EN.md).
