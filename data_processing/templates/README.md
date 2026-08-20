# Editable Data Templates

The CSV/TSV files in this directory contain synthetic identifiers and no real patient data. Edit them with Excel, LibreOffice, or a text editor.

| Template | Purpose |
| --- | --- |
| `clinical_input.csv` | Clinical-summary input |
| `mutation_input.csv` | Wide mutation table with `mut_<gene>` columns |
| `survival_clinical_input.csv` | Follow-up time and event |
| `pathology_group_input.csv` | Cancer or pathology grouping |
| `survival_output_example.csv` | Standard survival-label example |
| `column_mapping.csv` | Record source-to-standard column mappings |

Spreadsheet precautions:

- Format patient IDs, pathology IDs, and Ensembl IDs as text.
- Store survival time as numeric values with one unit across the cohort; the default is months.
- Save CSV files as UTF-8.
- Never commit real patient identifiers.

See [Model Input Data Contract](../../docs/data_contract.md) and [PyTorch PT Formats](../../docs/pt_formats.md).
