# Fixed Gene Lists

This directory contains 26 model-versioned gene lists: one pan-cancer list and cohort-specific lists for BLCA, BRCA, CESC, COAD, ESCA, GBM, HNSC, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, OV, PAAD, PCPG, PRAD, READ, SARC, SKCM, STAD, TGCT, THCA, THYM, and UCEC. They are named:

```text
TCGA-<CANCER>_gene_list.txt
```

Each line contains one gene symbol. Line order is the mutation-vector feature order. Do not sort, deduplicate, add, or remove entries. Use `TCGA-ALL_gene_list.txt` for pan-cancer tasks and the matching `TCGA-<CANCER>_gene_list.txt` for a single cohort. The resulting feature dimension must match `models.geno_raw_dim` in the YAML configuration.

A model release should include its matching gene list and preferably record the SHA-256 checksum.
