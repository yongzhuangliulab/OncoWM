# 训练、推理与生存评估

本仓库仅保留当前实际使用的 v5.3 调用链，公开文件名不再携带实验版本后缀：

| 当前文件 | 原服务器文件 | 职责 |
| --- | --- | --- |
| `main.py` | `main.py` | 两阶段训练入口与完成标记 |
| `src/train.py` | `train_modify_v5_3.py` | 五折训练循环 |
| `src/models/mjepa.py` | `m3_jepe_modify_v5_3.py` | M-JEPA 模型、对齐损失和预后头 |
| `src/datasets/medical.py` | `medical_dataset_v3_6.py` | 多模态训练 Dataset 和分层五折 |
| `src/inference.py` | `inference_v5_3.py` | checkpoint 推理、embedding 和风险分数 |
| `src/evaluate.py` | `evaluate_cindex.py` | C-index、时间依赖 AUC 和 iAUC |

历史训练/推理版本、批量癌种 shell、运行日志和患者级特征未复制到 GitHub 仓库。

## 1. 两阶段训练

训练必须按以下顺序执行：

1. `align`：学习多模态联合表征，每折保存 `align-best.pth.tar`。
2. `prognosis`：读取同一折的对齐 checkpoint，冻结骨干，只训练预后头，保存 `prognosis-best.pth.tar`。

TCGA 癌种可直接使用已提供的配置：

```bash
python main.py \
  --fname configs/tcga/train_BRCA_OS.yaml \
  --task both \
  --run-name BRCA_OS
```

自有队列则复制模板，修改数据路径、MedGemma 路径、模态维度和训练参数后运行：

```bash
cp configs/train.example.yaml configs/train_my_cohort.yaml
# 编辑 configs/train_my_cohort.yaml 后运行：
python main.py \
  --fname configs/train_my_cohort.yaml \
  --task both \
  --run-name my_cohort
```

等价包装脚本：

```bash
bash scripts/train_two_stage.sh configs/tcga/train_BRCA_OS.yaml BRCA_OS
```

原命令中两个用 `&&` 连接的 Python 调用现在合并为 `--task both`；只有 align 成功结束后才会进入 prognosis。

默认输出结构：

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

成功阶段会在 `outputs/train_status/` 写入 `.done`。需要重跑时删除对应标记，或在 `main.py` 命令后加 `--force`。

### 训练数据要求

当前 `src/datasets/medical.py` 的训练路径要求四种模态全部存在：

- `path_pt_path`：患者级病理 embedding；
- `text_pt_path`：临床 token；
- `geno_pt_path`：突变向量；
- `tran_pt_path`：转录组 embedding。

训练代码会与生存 CSV 取全模态样本交集。缺失任一模态的患者不会进入训练。`geno_raw_dim` 必须等于突变向量长度，其余模态维度也必须与 Tensor 和 checkpoint 一致。

prognosis 默认强制要求每折已存在 `align-best.pth.tar`，避免把“从头训练”误当成两阶段实验。只有明确设置 `allow_prognosis_from_scratch: true` 才会放宽此检查。当前 Dataset 会给完全相同的生存时间添加最多约 `0.01` 的确定性微扰，以稳定 Cox 风险集排序；正式统计方案应记录这一实现细节。

## 2. 推理

复制并修改 `configs/inference.example.yaml`，尤其是 checkpoint、输出目录、MedGemma 路径和各模态维度：

```bash
cp configs/inference.example.yaml configs/inference_BRCA_OS_fold_0.yaml
# 编辑 configs/inference_BRCA_OS_fold_0.yaml 后运行
python -m src.inference --fname configs/inference_BRCA_OS_fold_0.yaml
```

或：

```bash
bash scripts/run_inference.sh configs/inference_BRCA_OS_fold_0.yaml
```

与训练不同，推理配置允许把缺失模态写为 YAML `null`/`~`。启用的模态之间仍会取样本 ID 交集。若配置了一个非空路径但文件不存在，程序会报错，不会静默当作缺失模态。checkpoint 来自哪种维度和模型结构，推理配置就必须保持完全相同。

别人下载的 M-JEPA checkpoint 建议放在 `pretrained_models/mjepa/<CANCER>_<ENDPOINT>/fold_<N>/`；本地训练生成的 checkpoint 位于 `outputs/train/`。推理 YAML 的 `model_path` 可以指向两者之一。

prognosis 模式输出：

```text
outputs/inference/<run_name>/
├── prog_embeddings.npz
└── prognosis_risk_scores.csv
```

已有完整输出时程序默认跳过；使用 `--force` 重新生成。

## 3. C-index 评估

```bash
python -m src.evaluate \
  --scores outputs/inference/BRCA_OS_fold_0/prognosis_risk_scores.csv \
  --yaml configs/inference_BRCA_OS_fold_0.yaml
```

脚本按 `sample_id` 将风险分数与 YAML 中 `data.csv_path` 指向的生存标签合并，生成同目录下的 `evaluation_metrics.csv`。

风险方向约定：当前评估代码把较大的 `risk_score` 解释为更高风险。生存事件固定为 `0=删失`、`1=事件发生`，生存时间单位为月。

注意：当前时间依赖 AUC 使用同一队列同时构造生存分布和计算指标，因此它是队列内描述性结果。正式外部验证或论文报告应明确训练/测试参考分布，并由统计方案复核；C-index 是当前脚本最直接的主要指标。
