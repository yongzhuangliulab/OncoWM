# M-JEPA 使用指南

本文说明如何安装 M-JEPA、准备模型输入、训练模型，以及使用已有 checkpoint 完成推理和生存评估。

## 1. 安装

克隆仓库：

```bash
git clone https://github.com/qazlcy/mjepa.git
cd mjepa
```

推荐使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate mjepa
```

如果已经创建过环境：

```bash
git pull origin main
conda env update -n mjepa -f environment.yml --prune
```

检查 Python、PyTorch 和 GPU：

```bash
python -c "
import sys
import torch
print('python:', sys.executable)
print('torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
"
```

当前依赖要求 PyTorch 2.6 或以上。

## 2. 准备处理后的数据

M-JEPA 不直接读取医院原始表格、SVS 或表达矩阵。使用者需要先把自己的数据转换成模型规定的患者级结果格式。

普通队列建议放在：

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

TCGA 配置采用按模态分目录的结构：

```text
data/processed/tcga/
├── survival/<CANCER>_<ENDPOINT>_dataset.csv
├── pathology/TCGA_TITAN_first_slice_dict.pt
├── clinical/TCGA_data_clinical_patient_prompt.pt
├── mutation/TCGA-<CANCER>.pt
└── transcriptome/TCGA-<CANCER>.pt
```

五类文件的最低要求：

| 文件 | 格式 |
| --- | --- |
| 生存标签 | CSV 列为 `sample_id,cancer_type,survival_time,survival_event` |
| 临床特征 | `dict[sample_id, record]`，包含 `input_ids`、`attention_mask` 等 |
| 突变特征 | `dict[sample_id, Tensor]`，所有向量长度和基因顺序一致 |
| 病理特征 | `dict[sample_id, Tensor]`，默认维度 768 |
| 转录组特征 | `dict[sample_id, Tensor]`，维度与 YAML 配置一致 |

所有模态必须使用相同的 `sample_id` 规则。训练会取生存标签与四种模态的样本交集。

完整字段规范见：

- [数据接口规范](docs/data_contract.md)
- [PT 文件格式](docs/pt_formats.md)
- [数据处理脚本说明](data_processing/README.md)

检查可信来源的 `.pt` 文件：

```bash
python data_processing/inspect_pt.py data/processed/clinical.pt --max-items 3
python data_processing/inspect_pt.py data/processed/mutation.pt --max-items 3
```

## 3. 准备模型权重

下载的第三方模型和 M-JEPA checkpoint 统一放在：

```text
pretrained_models/
├── medgemma-1.5-4b-it/
├── conch-v1.5/
├── titan/
├── bulkformer/
└── mjepa/
    └── <CANCER>_<ENDPOINT>/
        └── fold_0/
            └── prognosis-best.pth.tar
```

用途：

- MedGemma：训练和推理时的临床文本编码器；
- CONCH/TITAN：在外部预处理流程中把 SVS 转换成病理 embedding；
- BulkFormer：在外部预处理流程中生成转录组 embedding；
- `pretrained_models/mjepa/`：放置下载的 M-JEPA checkpoint。

第三方模型目录需要保留完整发布结构，包括配置、tokenizer 和权重文件。详细说明见 [模型权重目录](pretrained_models/README.md)。

如果已经得到 `pathology.pt`，训练和推理阶段不再需要加载 TITAN。

## 4. 配置文件

仓库提供：

```text
configs/
├── train.example.yaml
├── inference.example.yaml
├── inference.ALL_DFI_fold_0.yaml
└── tcga/                       # 84 份 TCGA 训练配置
```

使用自有数据时先复制模板：

```bash
cp configs/train.example.yaml configs/train_my_cohort.yaml
cp configs/inference.example.yaml configs/inference_my_cohort_fold_0.yaml
```

重点检查：

- `data` 中的 CSV 和四种 `.pt` 路径；
- `models.text_encoder_name`；
- `path_dim`、`geno_raw_dim`、`geno_dim`、`tran_dim`；
- batch size；
- checkpoint 和输出目录；
- 生存终点是否匹配。

`geno_raw_dim` 必须等于突变向量的真实长度。OS、DSS、DFI、PFI 标签和 checkpoint 不能混用。

## 5. 训练

训练分为两个阶段：

1. `align`：学习多模态联合表征；
2. `prognosis`：加载同一折的 align checkpoint，冻结骨干并训练预后头。

推荐使用 `--task both` 按顺序执行两个阶段：

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

也可以使用脚本：

```bash
bash scripts/train_two_stage.sh configs/tcga/train_BRCA_OS.yaml BRCA_OS
```

输出示例：

```text
outputs/train/BRCA/BRCA_OS/
├── fold_0/
│   ├── align-best.pth.tar
│   ├── prognosis-best.pth.tar
│   ├── train.csv
│   └── val.csv
├── fold_1/ ... fold_4/
├── align-best-overall.pth.tar
├── prognosis-best-overall.pth.tar
└── *_cross_validation_results.csv
```

重新执行已完成的任务时，可增加 `--force`。

## 6. 推理

复制并修改推理模板后运行：

```bash
python -m src.inference \
  --fname configs/inference_my_cohort_fold_0.yaml
```

或者使用仓库提供的 TCGA ALL_DFI 配置示例：

```bash
python -m src.inference \
  --fname configs/inference.ALL_DFI_fold_0.yaml
```

prognosis 推理生成：

```text
outputs/inference/<experiment>/<run_name>/
├── prog_embeddings.npz
└── prognosis_risk_scores.csv
```

推理允许把缺失模态路径写成 YAML `null`。所有启用模态仍会取 `sample_id` 交集。

M-JEPA checkpoint 不重复保存冻结的 MedGemma 主干，因此加载时可能显示属于 `text_model` 主干的 `missing keys`。主干会从 `models.text_encoder_name` 指定的 MedGemma 目录加载。

## 7. 生存评估

```bash
python -m src.evaluate \
  --scores outputs/inference/<experiment>/<run_name>/prognosis_risk_scores.csv \
  --yaml configs/inference_my_cohort_fold_0.yaml
```

评估脚本会根据 `sample_id` 合并 YAML 中生存 CSV 的真实标签，并生成：

```text
evaluation_metrics.csv
```

输出指标包括 C-index、1/3/5 年时间依赖 AUC 和 iAUC。

## 8. 常见问题

### PyTorch 版本不足

如果出现 `require torch>=2.6`：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

确保安装依赖和运行程序使用同一个 Conda 环境中的 Python。

### CUDA 显存不足

降低 YAML 中的：

- `align_batch_size`；
- `prognosis_batch_size`；
- 推理 `batch_size`。

### 没有共同样本

检查 CSV 的 `sample_id` 与所有启用 `.pt` 字典的键。样本数量相同不代表样本键相同。

### prognosis 找不到 align checkpoint

先运行 align，或使用 `--task both`。每一折的 prognosis 都需要对应的 `align-best.pth.tar`。

## 9. 数据安全

不要向 GitHub 提交：

- 患者数据或患者映射表；
- `.pt`、`.h5` 和 SVS；
- 第三方模型权重；
- M-JEPA checkpoint；
- 访问令牌和服务器日志。

这些路径默认已写入 `.gitignore`。提交前仍应检查：

```bash
git status --short
```
