# PyTorch `.pt` 结果格式

M-JEPA 当前使用的 `.pt` 是通过 `torch.save` 保存的患者级 Python 字典，而不是一个无样本标识的单独 Tensor。只应使用 `torch.load` 加载可信来源的文件。

## 临床结果

建议文件名：`data/processed/clinical.pt`

```python
{
    "P001": {
        "text": "The patient ...",
        "input_ids": torch.tensor([...], dtype=torch.int64),
        "attention_mask": torch.tensor([...], dtype=torch.int64),
        "token_type_ids": torch.tensor([...], dtype=torch.int64),
    }
}
```

- 顶层键是统一的 `sample_id`。
- 每条记录必须包含上述四个键。
- 三个 token Tensor 均为一维且等长；当前默认长度为 512。
- tokenizer 没有返回 `token_type_ids` 时，转换脚本保存全零 Tensor。

## 突变结果

建议文件名：`data/processed/mutation.pt`

```python
{
    "P001": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    "P002": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
}
```

- 顶层键是统一的 `sample_id`。
- 每个值为一维 `float32` Tensor，shape 为 `[G]`。
- 所有样本向量等长，且 `G` 等于对应癌种 gene list 的非空行数。
- 第 `i` 位与 gene list 第 `i` 行严格对应，只允许 `0.0` 或 `1.0`。

## 病理与转录组结果

两者都是患者级 Tensor 字典：

```python
{"P001": torch.Tensor}
```

病理 Tensor 接受 `[D_path]` 或 `[1, D_path]`，推理 collate 会压缩首维为 1 的表示；转录组通常为 `[D_tran]`。所有样本必须等长，维度分别等于配置中的 `models.path_dim` 和 `models.tran_dim`。

## 查看实际文件

```bash
python data_processing/inspect_pt.py data/processed/clinical.pt --max-items 3
python data_processing/inspect_pt.py data/processed/mutation.pt --max-items 3
```

最低检查项包括：顶层类型、样本 ID、单样本字段、Tensor shape、dtype、NaN 和 Inf。完整跨文件检查规则见 [data_contract.md](data_contract.md)。
