# Fixed gene lists

本目录保存由项目作者提供、与模型版本绑定的癌种基因列表，命名规则为：

```text
TCGA-<CANCER>_gene_list.txt
```

每行一个基因符号，行顺序就是突变向量的特征顺序。使用者不应排序、去重或自行增删；需要支持新癌种时，应由项目作者发布对应列表并记录版本或 SHA-256。

当前仓库已包含已有的 `TCGA-KIRP_gene_list.txt`：9,920 个基因，SHA-256 为 `A2848D1A3025078F19580648CDCFA592B3D6A0E4E9CC16F12917DDE8AA888C3F`。
