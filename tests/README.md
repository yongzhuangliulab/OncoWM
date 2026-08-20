# Tests

The test suite covers core data-conversion rules, YAML structure, portable path conventions, and the absence of original absolute server paths in packaged files.

```bash
python -m pytest -q
```

GPU integration tests requiring model weights and patient-level features are intentionally not included in the repository test suite.
