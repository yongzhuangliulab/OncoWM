# Fixed gene lists

[English](README_EN.md)

本目录保存由项目作者提供、与模型版本绑定的基因列表。仓库共包含 26 份文件：一份全癌种列表，以及 BLCA、BRCA、CESC、COAD、ESCA、GBM、HNSC、KIRC、KIRP、LGG、LIHC、LUAD、LUSC、OV、PAAD、PCPG、PRAD、READ、SARC、SKCM、STAD、TGCT、THCA、THYM 和 UCEC 的癌种列表。命名规则为：

```text
TCGA-<CANCER>_gene_list.txt
```

每行一个基因符号，行顺序就是突变向量的特征顺序。使用者不应排序、去重或自行增删。全癌种任务使用 `TCGA-ALL_gene_list.txt`，单癌种任务使用对应的 `TCGA-<CANCER>_gene_list.txt`；最终特征维度必须与 YAML 中的 `models.geno_raw_dim` 一致。

需要支持新癌种时，应由项目作者发布对应列表并记录版本或 SHA-256。
