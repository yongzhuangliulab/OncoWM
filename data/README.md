# 本地数据目录

- `raw/`：使用者自己的原始数据，格式可因队列而异。
- `processed/`：满足 `docs/data_contract.md` 的模型输入结果。

TCGA 批量配置采用以下本地结构：

```text
processed/tcga/
├── survival/
├── pathology/
├── clinical/
├── mutation/
└── transcriptome/
```

两个目录中的患者级文件均被 Git 忽略。仓库只保留目录占位，不提交真实数据或生成特征。
