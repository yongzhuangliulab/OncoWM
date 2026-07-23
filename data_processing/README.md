# 自有数据转换脚本

本目录中的脚本用于把一个具体队列的原始数据转换成 M-JEPA 的标准结果。它们属于**数据适配层**，不属于模型侧的 `src/datasets/`。

## 使用原则

- 原始表格格式可以不同，不要求所有用户复刻 QU 数据表。
- 能通过 CLI 列名参数适配时，无需修改代码。
- 数据布局差异较大时，可以复制脚本并为自己的队列改写。
- 无论如何改写，最终结果必须符合 [`../docs/data_contract.md`](../docs/data_contract.md)。本目录目前只负责 `survival.csv`、`clinical.pt` 和 `mutation.pt`；训练还需要预先提取的病理和转录组特征。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `process_survival_data.py` | 合并随访与病理分组，生成标准四列生存标签 |
| `process_clinical_data_qu.py` | 将 QU 临床字段映射到既定 MedGemma prompt，生成临床 token PT |
| `process_mutation_qu.py` | 读取 `mut_*` 宽表并按固定 gene list 生成二值向量 PT |
| `inspect_pt.py` | 查看可信 PT 的顶层类型、键、Tensor shape、dtype、NaN/Inf |
| `process_to_pt.sh` | 从仓库根目录启动临床转换的 Bash 示例 |
| `templates/` | 合成输入与标准输出示例，不含真实患者信息 |

## 输出位置

默认把本地结果写到：

```text
data/processed/survival.csv
data/processed/clinical.pt
data/processed/mutation.pt
```

这些结果包含患者级数据，已被 Git 忽略，不应提交到公开仓库。

## 列名不同的示例

```bash
python data_processing/process_survival_data.py \
  --clinical data/raw/my_followup.csv \
  --pathology data/raw/my_pathology.csv \
  --output data/processed/survival.csv \
  --sample-col patient_id \
  --time-col OS_months \
  --event-col OS_event \
  --group-col histology
```

临床脚本还支持 `--sample-col`、`--group-col`、`--sex-col`、`--age-col`、`--stage-col`；突变脚本支持 `--sample-col` 和 `--gene-list`。

完整参数可运行：

```bash
python data_processing/process_survival_data.py --help
python data_processing/process_clinical_data_qu.py --help
python data_processing/process_mutation_qu.py --help
```
