# M-JEPA 复现指南

本文给出从克隆仓库到完成训练、推理和生存评估的完整流程。命令均从仓库根目录执行，数据和模型权重使用仓库相对路径。

## 1. 已验证范围

以下流程已在 NVIDIA GPU 服务器上完成端到端验证：

- Python 3.10；
- PyTorch 2.6.0 + CUDA 11.8 wheel；
- MedGemma 1.5 4B IT 本地模型；
- 四模态患者级输入：临床、病理、突变和转录组；
- M-JEPA prognosis checkpoint 推理；
- C-index、1/3/5 年时间依赖 AUC 和 iAUC 评估。

ALL_DFI fold 0 验证集共 856 个样本，四模态交集仍为 856。一次已验证运行得到：

| 指标 | 数值 |
| --- | ---: |
| C-index | 0.751374 |
| AUC 1-Year | 0.767750 |
| AUC 3-Year | 0.763602 |
| AUC 5-Year | 0.778003 |
| iAUC | 0.768341 |

这些数值用于检查复现链路，不代表独立外部验证。硬件、依赖版本、checkpoint、数据划分或数据预处理发生变化时，结果可能不同。

## 2. 克隆与环境

本仓库为私有仓库时，服务器需要先配置 GitHub SSH key 或登录 GitHub CLI。

```bash
git clone git@github.com:qazlcy/mjepa.git
cd mjepa

conda env create -f environment.yml
conda activate mjepa
```

如果环境已经创建，更新已有环境：

```bash
git pull origin main
conda activate mjepa
conda env update -n mjepa -f environment.yml --prune
```

确认当前命令使用的是环境内 Python：

```bash
PY="$(which python)"

"$PY" -c "
import sys
import torch
import transformers
import peft
print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
"
```

当前 Gemma 3 实现要求 PyTorch 2.6 或以上。`CUDA available` 应为 `True`。

## 3. 本地文件布局

真实数据、第三方权重和运行输出均被 Git 忽略，不应上传到 GitHub。

### 3.1 TCGA 训练数据

以 BRCA_OS 为例：

```text
data/processed/tcga/
├── survival/BRCA_OS_dataset.csv
├── pathology/TCGA_TITAN_first_slice_dict.pt
├── clinical/TCGA_data_clinical_patient_prompt.pt
├── mutation/TCGA-BRCA.pt
└── transcriptome/TCGA-BRCA.pt
```

训练会取生存 CSV 和四个 `.pt` 字典的 `sample_id` 交集。缺少任一模态的患者不会进入训练。

### 3.2 模型权重

```text
pretrained_models/
├── medgemma-1.5-4b-it/
└── mjepa/
    └── ALL_DFI/
        └── fold_0/
            └── prognosis-best.pth.tar
```

- 从头训练必须准备完整的 `medgemma-1.5-4b-it/` 目录。
- 使用已训练 checkpoint 推理时，再把 checkpoint 放入 `pretrained_models/mjepa/`。
- 已经有 `pathology.pt` 时不需要 TITAN 权重；只有重新从 SVS 提取病理 embedding 时才需要 CONCH/TITAN。

### 3.3 使用软链接

服务器已有数据时，建议建立软链接，避免重复复制大文件。下面的 `SOURCE_ROOT` 由使用者替换成自己的数据根目录：

```bash
ROOT="$(pwd)"
SOURCE_ROOT=/path/to/existing/Multimodal_JEPA

mkdir -p "$ROOT/data/processed/tcga"/{survival,pathology,clinical,mutation,transcriptome}
mkdir -p "$ROOT/pretrained_models/mjepa/ALL_DFI/fold_0"

ln -sfn "$SOURCE_ROOT/Dataset/TCGA/TCGA_phenotype/BRCA_OS_dataset.csv" \
  "$ROOT/data/processed/tcga/survival/BRCA_OS_dataset.csv"
ln -sfn "$SOURCE_ROOT/Dataset/TCGA/TCGA_SVS/TCGA_TITAN_first_slice_dict.pt" \
  "$ROOT/data/processed/tcga/pathology/TCGA_TITAN_first_slice_dict.pt"
ln -sfn "$SOURCE_ROOT/Dataset/TCGA/TCGA_phenotype/TCGA_data_clinical_patient_prompt.pt" \
  "$ROOT/data/processed/tcga/clinical/TCGA_data_clinical_patient_prompt.pt"
ln -sfn "$SOURCE_ROOT/Dataset/TCGA/TCGA_genome/TCGA-BRCA.pt" \
  "$ROOT/data/processed/tcga/mutation/TCGA-BRCA.pt"
ln -sfn "$SOURCE_ROOT/Dataset/TCGA/TCGA_transcriptome/TCGA-BRCA.pt" \
  "$ROOT/data/processed/tcga/transcriptome/TCGA-BRCA.pt"
ln -sfn "$SOURCE_ROOT/medgemma-1.5-4b-it" \
  "$ROOT/pretrained_models/medgemma-1.5-4b-it"
```

检查失效软链接；命令没有输出才表示全部有效：

```bash
find data/processed pretrained_models -type l ! -exec test -e {} \; -print
```

## 4. 两阶段五折训练

仓库提供 84 份 TCGA 训练配置。BRCA_OS 的原始实验参数已经转换为相对路径：

```bash
"$PY" main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

`--task both` 严格依次执行：

1. `align`：训练多模态联合表征；
2. `prognosis`：读取每折的 `align-best.pth.tar`，冻结骨干并训练预后头。

输出：

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

长时间训练建议在 `tmux` 中运行，或使用：

```bash
mkdir -p logs
nohup "$PY" main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS \
  > logs/train_BRCA_OS.log 2>&1 &

tail -f logs/train_BRCA_OS.log
```

已有成功标记时任务会跳过。确定需要重跑可追加 `--force`。

## 5. 已训练模型推理

ALL_DFI fold 0 的已验证配置是：

```text
configs/inference.ALL_DFI_fold_0.yaml
```

需要准备：

```text
data/processed/tcga/survival/ALL_DFI_fold_0_val.csv
data/processed/tcga/pathology/TCGA_TITAN_first_slice_dict.pt
data/processed/tcga/clinical/TCGA_data_clinical_patient_prompt.pt
data/processed/tcga/mutation/TCGA-ALL.pt
data/processed/tcga/transcriptome/TCGA-ALL.pt
pretrained_models/medgemma-1.5-4b-it/
pretrained_models/mjepa/ALL_DFI/fold_0/prognosis-best.pth.tar
```

运行：

```bash
"$PY" -m src.inference \
  --fname configs/inference.ALL_DFI_fold_0.yaml
```

输出：

```text
outputs/inference/ALL/ALL_DFI/prognosis/Full_modal/fold_0/
├── prog_embeddings.npz
└── prognosis_risk_scores.csv
```

checkpoint 为节省空间不会重复保存冻结的 MedGemma 主干。因此加载时出现大量只属于 `text_model` 主干的 `missing keys` 提示是预期现象；MedGemma 主干会从 `pretrained_models/medgemma-1.5-4b-it/` 加载。

## 6. 生存评估

```bash
"$PY" -m src.evaluate \
  --scores outputs/inference/ALL/ALL_DFI/prognosis/Full_modal/fold_0/prognosis_risk_scores.csv \
  --yaml configs/inference.ALL_DFI_fold_0.yaml
```

评估程序按 `sample_id` 从配置指定的 CSV 合并真实标签，并在风险评分目录生成：

```text
evaluation_metrics.csv
```

## 7. 使用自己的数据

复制通用模板：

```bash
cp configs/train.example.yaml configs/train_my_cohort.yaml
cp configs/inference.example.yaml configs/inference_my_cohort_fold_0.yaml
```

至少修改：

- 生存 CSV 与四个 `.pt` 路径；
- `text_encoder_name`；
- `geno_raw_dim`、`path_dim`、`tran_dim`；
- batch size、输出目录和 run name；
- 推理 checkpoint 路径。

训练要求四模态齐全；推理允许把未启用模态写成 YAML `null`，但所有启用模态仍会取 `sample_id` 交集。不同终点不能混用，例如 PFI 标签不能用 OS checkpoint 做有意义的评估。

## 8. 常见问题

### `require torch>=2.6`

说明执行推理的 Python 仍使用旧 PyTorch。检查：

```bash
"$PY" -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

安装和运行必须使用同一个环境的 `"$PY"`。

### CUDA 显存不足

降低配置中的 `align_batch_size`、`prognosis_batch_size` 或推理 `batch_size`，例如从 90 降为 16 或 8。

### `No shared sample_id`

CSV 和 `.pt` 字典的患者键不一致。不能只比较样本数量，必须检查字符串键的真实交集。

### prognosis 找不到 align checkpoint

先运行 `align`，或直接使用 `--task both`。不建议在没有明确实验设计的情况下设置 `allow_prognosis_from_scratch`。

## 9. 数据与发布安全

发布前运行：

```bash
git status --short
```

不要提交患者数据、`.pt/.h5` 特征、SVS、模型权重、checkpoint、访问令牌或服务器日志。只提交代码、配置模板、合成示例和文档。
