# 配置文件

- `train.example.yaml`：五折两阶段训练模板。
- `inference.example.yaml`：单个 checkpoint 推理模板。
- `tcga/`：21 个癌种（含 `ALL` 汇总队列）× OS/DSS/DFI/PFI，共 84 份可运行五折训练配置。

复制模板并按自己的癌种、终点和文件路径修改。配置中的维度必须与 PT Tensor 和训练 checkpoint 一致，尤其是 `geno_raw_dim`、`path_dim` 和 `tran_dim`。

TCGA 配置详情、目录约定和未纳入配置见 [`tcga/README.md`](tcga/README.md)。
