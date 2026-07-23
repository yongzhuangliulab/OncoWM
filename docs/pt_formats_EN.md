# PyTorch `.pt` Result Formats

[中文](pt_formats.md)

M-JEPA PT inputs are patient-level Python dictionaries saved with `torch.save`, not anonymous standalone tensors. Load only files from trusted sources.

## Clinical results

Suggested path: `data/processed/clinical.pt`

```python
{
    "P001": {
        "text": "The patient ...",
        "input_ids": torch.tensor([...], dtype=torch.int64),
        "attention_mask": torch.tensor([...], dtype=torch.int64),
        "token_type_ids": torch.tensor([...], dtype=torch.int64),
    }
}
```

The top-level key is `sample_id`. All three token tensors must be one-dimensional and equal length. When the tokenizer does not return token type IDs, store an all-zero tensor.

## Mutation results

Suggested path: `data/processed/mutation.pt`

```python
{
    "P001": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "P002": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
}
```

Each value is a one-dimensional `float32` tensor of shape `[G]`. Every vector must be the same length, contain only `0.0` or `1.0`, and follow the exact published gene-list order.

## Pathology and transcriptome results

```python
{"P001": torch.Tensor}
```

Pathology accepts `[D_path]` or `[1, D_path]`; inference squeezes a leading singleton dimension. Transcriptome features are normally `[D_tran]`. All samples must be equal length and match `models.path_dim` or `models.tran_dim`.

## Inspecting files

```bash
python data_processing/inspect_pt.py data/processed/clinical.pt --max-items 3
python data_processing/inspect_pt.py data/processed/mutation.pt --max-items 3
```

Inspect the top-level type, sample keys, record fields, tensor shape, dtype, NaN, and Inf. See the full [Model Input Data Contract](data_contract_EN.md).
