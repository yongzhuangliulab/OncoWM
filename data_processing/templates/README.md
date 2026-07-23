# 可编辑数据模板

[English](README_EN.md)

本目录中的 CSV/TSV 使用合成编号，不含真实患者数据，可用 Excel、LibreOffice 或文本编辑器修改。

推荐把自己的表头改成模板中的标准字段；支持自定义列名的脚本也可以通过命令行参数映射，无需修改表头。

| 模板 | 用途 |
| --- | --- |
| `clinical_input.csv` | 临床摘要生成输入 |
| `mutation_input.csv` | `mut_基因名` 突变宽表 |
| `survival_clinical_input.csv` | 随访时间与事件 |
| `pathology_group_input.csv` | 样本癌种/病理分组 |
| `survival_output_example.csv` | 标准生存标签示例 |
| `column_mapping.csv` | 记录自己队列字段与标准字段的对应关系 |

Excel 使用注意事项：

- 患者编号、病理号和 Ensembl ID 设置为“文本”。
- 生存时间必须为数值，并保证整个队列单位一致；当前默认单位为月。
- CSV 推荐保存为 UTF-8。
- 不要把真实患者标识信息提交到 GitHub。

完整结果数据规范见 `../../docs/data_contract.md`，PT 内部结构见 `../../docs/pt_formats.md`。
