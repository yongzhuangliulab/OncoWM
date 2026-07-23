# 运行脚本

这些脚本必须从任意位置均可调用；它们会自动切换到仓库根目录。默认使用当前 Conda 环境中的 `python`，也可通过 `PYTHON_BIN=/path/to/python` 指定解释器。

```bash
bash scripts/train_two_stage.sh configs/tcga/train_BRCA_OS.yaml BRCA_OS

cp configs/inference.example.yaml configs/inference_BRCA_OS_fold_0.yaml
# 编辑复制后的推理配置，再执行：
bash scripts/run_inference.sh configs/inference_BRCA_OS_fold_0.yaml
bash scripts/evaluate_survival.sh \
  configs/inference_BRCA_OS_fold_0.yaml \
  outputs/inference/BRCA_OS_fold_0/prognosis_risk_scores.csv
```
