# 模型输入数据契约

[English](data_contract_EN.md)

本文只定义 **M-JEPA 模型应接收的处理结果**。原始 CSV/TSV 的列名、文件数量和组织方式可以因队列而异；数据提供者负责完成本地适配。

## 1. 通用约定

- 所有模态按患者/样本组织，唯一键统一命名为 `sample_id`。
- `sample_id` 必须去标识化，读取为字符串，不能依赖 Excel 数值格式。
- 同一个患者在所有结果文件中必须使用完全相同的 ID。
- `.pt` 使用 `torch.save` 保存，只加载可信来源的文件。
- 训练/推理前使用各模态 ID 的交集，并报告各自样本数、交集数和被排除数量。

推荐结果目录：

```text
data/processed/
├── survival.csv
├── clinical.pt
├── mutation.pt
├── pathology.pt
└── transcriptome.pt
```

## 2. 生存标签 `survival.csv`

必须恰好提供以下语义（列顺序建议保持一致）：

| 列 | 类型 | 约束 |
| --- | --- | --- |
| `sample_id` | string | 非空、唯一、去标识化 |
| `cancer_type` | string | 标准癌种缩写，如 `LUAD`、`LUSC`；无法确定时为 `unknow` |
| `survival_time` | float | 单位为月，必须是有效数字；缺失或非数字记录不进入结果 |
| `survival_event` | int | 只能是 `0` 或 `1`；`0=删失/存活`，`1=死亡/事件发生` |

示例：

```csv
sample_id,cancer_type,survival_time,survival_event
P001,LUAD,25.3,0
P002,LUSC,13.0,1
```

当前 QU 示例的分组映射为：肺癌/腺癌相关值映射为 `LUAD`，鳞癌相关值映射为 `LUSC`，其他映射为 `unknow`。这是转换规则，不是模型输入接口的一部分；其他队列可自行映射到标准癌种缩写。

## 3. 临床特征 `clinical.pt`

顶层对象必须是以 `sample_id` 为键的字典：

```python
{
    "P001": {
        "text": "The patient ...",
        "input_ids": torch.Tensor,       # torch.int64, [L]
        "attention_mask": torch.Tensor,  # torch.int64, [L]
        "token_type_ids": torch.Tensor,  # torch.int64, [L]
    }
}
```

约束：

- 所有样本的三个 token Tensor 长度一致，当前默认 `L=512`。
- `text` 是 MedGemma 按项目既定提示词生成的标准临床摘要。
- tokenizer 不返回 `token_type_ids` 时，以全零 Tensor 补齐。
- 缺失临床字段应明确表示为 `Unknown`，不得由生成模型推测。
- QU 示例脚本只将本地列映射到原提示词的四个输入：`Cancer Subtype`、`Age`、`Sex`、`AJCC Pathologic Stage`；提示词工程本身保持不变。

## 4. 突变特征 `mutation.pt`

顶层对象必须是以 `sample_id` 为键的二值向量字典：

```python
{
    "P001": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "P002": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
}
```

约束：

- Tensor dtype 为 `torch.float32`，shape 为 `[G]`。
- 同一癌种所有样本的 `G` 相同。
- 第 `i` 个值严格对应 `list/TCGA-<CANCER>_gene_list.txt` 的第 `i` 行。
- 输入缺少列表中的基因时填 `0.0`；输入中列表外的基因忽略。
- 当前示例宽表只读取 `mut_` 前缀列，去掉前缀后与 gene list 对齐；非零数值记为 `1.0`。
- 发布模型版本时，必须同时发布对应 gene list，并建议记录其 SHA-256 校验值。

## 5. 病理特征 `pathology.pt`

当前训练代码要求患者级病理 embedding：

```python
{"P001": torch.Tensor}  # float32/bfloat16, [D_path] 或 [1, D_path]
```

- 默认 `D_path=768`，实际维度必须等于 YAML 的 `models.path_dim`。
- 同一患者多张切片需要在数据处理阶段合并成一个患者级表示。
- 当前仓库尚未包含 SVS 到该结果的处理流程。

## 6. 转录组特征 `transcriptome.pt`

当前训练代码要求患者级转录组 embedding：

```python
{"P001": torch.Tensor}  # float32/bfloat16, [D_tran]
```

- 所有样本等长，实际维度必须等于 YAML 的 `models.tran_dim`。
- 当前仓库尚未包含原始表达矩阵到该结果的提取流程。

## 7. 训练与推理的模态差异

- 当前训练 Dataset 只保留生存 CSV 与四个模态 PT 的共同样本，四个 PT 路径都必须存在。
- 推理可以把缺失模态路径设为 YAML `null`/`~`，但所有启用模态之间仍取 ID 交集。
- 模型结构维度由训练 checkpoint 决定；缺失某模态不代表可以改变该模态在 YAML 中的维度。

## 8. 最低质量检查

每次构建数据至少检查：

1. 每个结果文件的样本数、重复 ID 和空 ID。
2. 生存标签与所有启用模态的 ID 交集以及各自缺失数量。
3. 生存时间是否为数值，事件值是否只包含 0/1。
4. 临床 token 的字段、dtype 和长度是否一致。
5. 突变 Tensor 是否全部等长、是否只含 0/1、长度是否等于 gene list。
6. 所有浮点 Tensor 是否包含 NaN 或 Inf。
7. 病理与转录组 Tensor 是否等长并与 YAML 配置维度一致。

`.pt` 的更详细示例见 [pt_formats.md](pt_formats.md)。
