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

注意：这里放的是已经处理好的患者级结果，不放 TITAN、MedGemma 等模型权重。下载权重统一放在仓库根目录的 `pretrained_models/`。
