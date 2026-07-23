# TCGA 五折训练配置

本目录包含当前 v5.3 五折训练代码可直接读取的配置：21 个癌种/队列 × 4 个生存终点，共 84 份。

终点：

- `OS`：Overall Survival
- `DSS`：Disease-Specific Survival
- `DFI`：Disease-Free Interval
- `PFI`：Progression-Free Interval

## 已纳入癌种与关键参数

| 癌种 | `geno_raw_dim` | Align batch | Prognosis batch |
| --- | ---: | ---: | ---: |
| ALL | 19766 | 90 | 90 |
| BLCA | 17533 | 64 | 64 |
| BRCA | 16658 | 90 | 128 |
| CESC | 16092 | 64 | 64 |
| COAD | 18636 | 64 | 64 |
| ESCA | 11810 | 32 | 32 |
| HNSC | 16509 | 64 | 90 |
| KIRC | 10923 | 90 | 128 |
| KIRP | 9920 | 64 | 64 |
| LGG | 11975 | 64 | 64 |
| LIHC | 14338 | 64 | 64 |
| LUAD | 17699 | 64 | 90 |
| LUSC | 17621 | 64 | 90 |
| PAAD | 11207 | 32 | 32 |
| PCPG | 1680 | 32 | 200 |
| PRAD | 11066 | 64 | 500 |
| SARC | 8877 | 32 | 64 |
| SKCM | 18332 | 64 | 64 |
| STAD | 18003 | 64 | 64 |
| THCA | 4163 | 64 | 500 |
| UCEC | 18993 | 64 | 64 |

以上数值来自补充的原始配置，并未擅自统一。`geno_raw_dim` 必须等于对应 `TCGA-<CANCER>.pt` 的真实向量长度。

## 数据目录约定

每份配置使用仓库相对路径。例如 `train_BRCA_OS.yaml` 读取：

```text
data/processed/tcga/
├── survival/BRCA_OS_dataset.csv
├── pathology/TCGA_TITAN_first_slice_dict.pt
├── clinical/TCGA_data_clinical_patient_prompt.pt
├── mutation/TCGA-BRCA.pt
└── transcriptome/TCGA-BRCA.pt
```

MedGemma 默认放在：

```text
pretrained_models/medgemma-1.5-4b-it/
```

数据、模型权重和输出均被 Git 忽略，不应上传到公开仓库。

## 运行示例

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

输出写入：

```text
outputs/train/BRCA/BRCA_OS/
```

## 未纳入的配置

补充目录中的以下五个癌种虽然各有四个 YAML，但关键数值为 YAML 空列表 `[]`，不能传给当前训练代码：

```text
GBM, OV, READ, TGCT, THYM
```

缺失项包括 `align_batch_size`、`prognosis_batch_size`、`t2i_weight_decay`、`t2i_final_weight_decay` 和 prognosis `lr`。在这些参数由实验负责人确认前，不把它们发布成“可运行配置”。

`full_cohort/` 与 `unimodal/` 配置也未迁入：当前精简仓库没有对应的全队列和单模态训练入口，直接交给 `main.py` 会改变原实验含义。
