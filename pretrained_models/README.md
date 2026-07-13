# 下载模型权重放置目录

所有从外部下载、而不是由本次训练产生的模型权重统一放在本目录。权重文件本身已被 Git 忽略，只提交本说明。

推荐结构：

```text
pretrained_models/
├── medgemma-1.5-4b-it/        # 临床摘要生成和临床文本编码器
├── conch-v1.5/                # SVS patch-level 特征提取
├── titan/                     # patch 特征聚合为患者/切片级病理 embedding
├── bulkformer/                # 转录组 embedding 提取
└── mjepa/                     # 下载的 M-JEPA checkpoint
    └── <CANCER>_<ENDPOINT>/
        └── fold_0/
            └── prognosis-best.pth.tar
```

第三方模型目录内部的文件名应保持其原始发布结构，不要只复制单个权重文件而丢失 `config.json`、tokenizer 或其他配套文件。

## 两类权重不要混淆

- `pretrained_models/`：别人下载后放置的第三方模型和已发布 M-JEPA checkpoint。
- `outputs/train/`：在本仓库中运行训练后新生成的 checkpoint、日志和划分文件。

## 病理切片特别说明

当前训练代码不直接读取 `.svs`，而是读取 `data/processed/.../pathology.pt`。使用者需要先通过 CONCH 等 patch encoder 提取 patch 特征，再使用 TITAN 聚合成病理 embedding，最后按 `sample_id -> Tensor` 字典保存到处理结果目录。

因此，处理 SVS 时至少需要准备相应的 CONCH 与 TITAN 权重，并在所使用的外部病理预处理脚本中指向：

```text
pretrained_models/conch-v1.5/
pretrained_models/titan/
```

本仓库当前尚未包含完整的 SVS → CONCH → TITAN 流程；这两个目录是统一的权重放置约定，不代表仅下载权重即可自动完成切片处理。
