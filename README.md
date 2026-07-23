# M-JEPA

[English](README_EN.md)

M-JEPA 是一个多模态生存预测研究项目。本仓库以**模型实际接收的标准化结果数据**为接口；不同医院、队列或公开数据集的原始表头可以不同，使用者需要先把自己的数据转换成本文规定的结果格式。

> 当前仓库包含 v5.3 模型训练、推理、C-index 评估，以及临床、生存标签和突变三类数据的示例转换脚本。病理切片（SVS）与转录组的特征提取流程暂未纳入本轮整理。

首次使用请阅读 [使用指南](USAGE.md)，其中包含安装、数据与权重准备、配置、训练、推理、评估和常见问题。

## 仓库结构

```text
mjepa/
├── README.md
├── USAGE.md                 # 安装、训练、推理和评估说明
├── environment.yml
├── requirements.txt
├── main.py                  # align/prognosis 两阶段训练入口
├── configs/                 # 训练和推理 YAML 模板
├── src/
│   ├── train.py             # 五折训练
│   ├── inference.py         # embedding 与风险分数推理
│   ├── evaluate.py          # C-index/AUC 评估
│   ├── models/mjepa.py      # v5.3 模型
│   └── datasets/medical.py  # 标准结果 Dataset
├── scripts/                 # 简化后的训练、推理和评估命令
├── data_processing/         # 自有数据到标准结果的转换脚本
│   ├── process_survival_data.py
│   ├── process_clinical_data_qu.py
│   ├── process_mutation_qu.py
│   ├── inspect_pt.py
│   ├── process_to_pt.sh
│   └── templates/           # 合成示例，不含真实患者数据
├── list/                    # 作者提供的固定癌种基因列表
├── data/
│   ├── raw/                 # 使用者自己的原始数据；Git 忽略
│   └── processed/           # 提交给模型的标准结果；Git 忽略
├── pretrained_models/       # 下载的 MedGemma/TITAN/CONCH/BulkFormer/M-JEPA 权重
├── outputs/                 # 本仓库训练和推理生成的结果；Git 忽略
├── docs/
│   ├── data_contract.md     # 首要文档：模型输入契约
│   ├── pt_formats.md        # PT 内部对象说明
│   └── training_and_inference.md
└── tests/                   # 数据转换回归测试
```

## 下载后怎样放置文件

使用者下载仓库后只需准备两类本地文件：处理好的患者级数据和模型权重。二者都不会上传到 GitHub。

### 1. 处理好的数据

无论原始数据来自哪里，处理后请放入 `data/processed/`，并保证所有模态使用相同的 `sample_id`：

| 建议文件名 | 内容 | 模型侧最低要求 |
| --- | --- | --- |
| `survival.csv` | 生存标签 | 四列：`sample_id,cancer_type,survival_time,survival_event` |
| `clinical.pt` | 临床文本 token | `dict[sample_id, record]`；record 含 `text/input_ids/attention_mask/token_type_ids` |
| `mutation.pt` | 固定基因顺序的突变向量 | `dict[sample_id, Tensor]`；所有向量等长，顺序与癌种 gene list 完全一致 |
| `pathology.pt` | 患者级病理 embedding | `dict[sample_id, Tensor]`；当前默认维度 768 |
| `transcriptome.pt` | 患者级转录组 embedding | `dict[sample_id, Tensor]`；维度必须与配置一致 |

前三类已有转换示例；病理和转录组目前要求使用者提供预先提取的结果。完整约束见 [数据接口规范](docs/data_contract.md)。模型训练或推理只依赖处理结果，不依赖某个队列的原始表头。

普通单队列可以直接使用：

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

批量 TCGA 配置使用 `data/processed/tcga/` 下按模态分目录的结构，详见 [TCGA 配置说明](configs/tcga/README.md)。

### 2. 下载的模型权重

所有下载权重统一放入 `pretrained_models/`：

```text
pretrained_models/
├── medgemma-1.5-4b-it/
├── conch-v1.5/
├── titan/
├── bulkformer/
└── mjepa/<CANCER>_<ENDPOINT>/full_cohort/prognosis_full_final.pth.tar
```

用途：

| 目录 | 用途 |
| --- | --- |
| `medgemma-1.5-4b-it/` | 生成临床摘要，并作为训练/推理的临床文本编码器 |
| `conch-v1.5/` | 从 SVS patch 提取特征 |
| `titan/` | 将 patch 特征聚合为病理 embedding |
| `bulkformer/` | 从表达矩阵提取转录组 embedding |
| `mjepa/` | 放置别人下载的训练好 M-JEPA checkpoint，用于推理 |

训练代码不会直接读取 SVS。病理切片必须先经过 `SVS → CONCH patch 特征 → TITAN embedding → pathology.pt`，再把最终 `pathology.pt` 放入 `data/processed/`。详细说明见 [pretrained_models/README.md](pretrained_models/README.md)。

## 安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate mjepa
```

也可以在已正确安装 PyTorch/CUDA 的环境中运行：

```bash
pip install -r requirements.txt
```

临床处理还需要用户把 MedGemma 1.5 4B IT 放到 `pretrained_models/medgemma-1.5-4b-it/`，并遵守其许可证和访问要求。

## 训练、推理和评估

复制并修改 YAML 模板后，两阶段训练可用一条命令完成：

```bash
python main.py --fname configs/tcga/train_BRCA_OS.yaml --task both --run-name BRCA_OS
```

推理与评估：

```bash
python -m src.inference --fname configs/inference.ALL_DFI_full_cohort.yaml

python -m src.evaluate \
  --scores outputs/inference/ALL/ALL_DFI/prognosis/Full_modal/full_cohort/prognosis_risk_scores.csv \
  --yaml configs/inference.ALL_DFI_full_cohort.yaml
```

完整使用步骤见 [使用指南](USAGE.md)，阶段关系和实现注意事项见 [训练与推理说明](docs/training_and_inference.md)。

仓库还提供 [84 份 TCGA 五折配置](configs/tcga/README.md)，覆盖 21 个癌种/队列和 OS、DSS、DFI、PFI 四类终点。配置使用仓库相对路径，不包含原服务器目录。

## 如何处理自己的数据

`data_processing/` 中的脚本是 **QU 队列的适配示例**，不是强制所有用户采用相同原始字段。优先阅读 [data_processing/README.md](data_processing/README.md)，再选择以下方式：

1. 原始字段相同：直接运行脚本。
2. 只是列名不同：使用脚本提供的 `--sample-col`、`--time-col` 等参数映射。
3. 数据组织方式不同：复制并修改转换脚本，但最终输出必须满足 `docs/data_contract.md`。

示例命令：

```bash
python data_processing/process_survival_data.py \
  --clinical data/raw/followup.csv \
  --pathology data/raw/pathology.csv \
  --output data/processed/survival.csv

python data_processing/process_clinical_data_qu.py \
  --input data/raw/clinical.csv \
  --output data/processed/clinical.pt \
  --model-path pretrained_models/medgemma-1.5-4b-it

python data_processing/process_mutation_qu.py \
  --input data/raw/molecular.tsv \
  --cancer-type KIRP \
  --output data/processed/mutation.pt
```

突变命令中的 `--cancer-type KIRP` 会读取 `list/TCGA-KIRP_gene_list.txt`。其他癌种请由项目作者在 `list/` 中提供相应固定列表。

检查可信的 PT 文件：

```bash
python data_processing/inspect_pt.py data/processed/clinical.pt --max-items 3
python data_processing/inspect_pt.py data/processed/mutation.pt --max-items 3
```

## 数据对齐与提交前检查

- `sample_id` 必须是去标识化的非空字符串，并在临床、突变和生存标签中使用同一规则。
- `survival_time` 必须为数值且全队列单位一致；当前约定单位为月。
- `survival_event` 固定为 `0=删失/存活`、`1=事件发生/死亡`。
- 突变向量长度必须等于所用 gene list 的行数，且不得自行改变顺序。
- 浮点 Tensor 不得包含 NaN 或 Inf。
- 训练前应记录每个模态样本数以及全部启用模态的样本 ID 交集。

## 数据安全

不要向 GitHub 提交真实临床数据、患者映射表、病理切片、生成的 `.pt/.h5`、模型权重、访问令牌或服务器日志。`.gitignore` 已忽略常见数据与产物，但发布前仍需人工检查 `git status`。

## 当前范围

本次整理保留了当前 v5.3 核心调用链，并去除了历史版本、服务器绝对路径、批量癌种命令、日志和患者级大文件。训练代码目前要求病理、临床、突变和转录组四模态齐全；推理允许缺失模态。模型权重不包含在 GitHub 中；SVS/转录组特征提取流程仍需使用对应外部项目，完成后只把标准结果放进 `data/processed/`。

## License

代码附带 [Apache License 2.0](LICENSE)。模型代码源自 M3-JEPA 研究代码的当前临床 v5.3 分支；公开发布前仍需核对所有作者归属、MedGemma、预训练权重和数据集的使用与再发布条款。
