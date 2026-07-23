# Custom Data Conversion Scripts


These scripts convert cohort-specific raw data into standard M-JEPA inputs. They form a **data adaptation layer** and are separate from the model-side dataset in `src/datasets/`.

## Principles

- Raw table formats do not need to match the QU examples.
- Use CLI column options when only column names differ.
- Copy and adapt a script when the source layout differs substantially.
- Every output must satisfy the [Model Input Data Contract](../docs/data_contract_EN.md).

This directory currently produces `survival.csv`, `clinical.pt`, and `mutation.pt`. Training also requires precomputed pathology and transcriptome features.

| File | Purpose |
| --- | --- |
| `process_survival_data.py` | Create the standard four-column survival table |
| `process_clinical_data_qu.py` | Map QU clinical fields to the MedGemma prompt and save token PT |
| `process_mutation_qu.py` | Convert `mut_*` wide-table columns to ordered binary mutation vectors |
| `inspect_pt.py` | Inspect trusted PT types, keys, shapes, dtypes, NaN, and Inf |
| `process_to_pt.sh` | Bash example for clinical conversion |
| `templates/` | Synthetic input and output examples |

Default outputs:

```text
data/processed/survival.csv
data/processed/clinical.pt
data/processed/mutation.pt
```

Example with custom column names:

```bash
python data_processing/process_survival_data.py \
  --clinical data/raw/my_followup.csv \
  --pathology data/raw/my_pathology.csv \
  --output data/processed/survival.csv \
  --sample-col patient_id \
  --time-col OS_months \
  --event-col OS_event \
  --group-col histology
```

View all parameters:

```bash
python data_processing/process_survival_data.py --help
python data_processing/process_clinical_data_qu.py --help
python data_processing/process_mutation_qu.py --help
```
