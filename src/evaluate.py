import os
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"读取风险评分文件: {args.scores}")
    df_scores = pd.read_csv(args.scores)
    missing_score_columns = {'sample_id', 'risk_score'}.difference(df_scores.columns)
    if missing_score_columns:
        raise ValueError(f"Risk score CSV is missing columns: {sorted(missing_score_columns)}")
    if df_scores['sample_id'].duplicated().any():
        raise ValueError("Risk score CSV contains duplicate sample IDs")
    
    logger.info(f"读取 YAML 配置文件: {args.yaml}")
    with open(args.yaml, 'r') as f:
        params = yaml.safe_load(f)
        
    time_col = 'survival_time'    
    status_col = 'survival_event' 
    
    if time_col not in df_scores.columns or status_col not in df_scores.columns:
        logger.info(f"预测结果中未找到标签列 ({time_col}, {status_col})，尝试从原始数据集加载...")
        orig_csv_path = params['data']['csv_path']
        if not os.path.exists(orig_csv_path):
            raise FileNotFoundError(f"找不到原始数据集: {orig_csv_path}")
            
        df_orig = pd.read_csv(orig_csv_path)
        
        id_col = 'sample_id' if 'sample_id' in df_scores.columns else 'PATIENT_ID'
        orig_id_col = 'sample_id' if 'sample_id' in df_orig.columns else 'PATIENT_ID'
        
        required = {orig_id_col, time_col, status_col}
        missing = required.difference(df_orig.columns)
        if missing:
            raise ValueError(f"Survival CSV is missing columns: {sorted(missing)}")
        if df_scores[id_col].duplicated().any():
            raise ValueError("Risk score CSV contains duplicate sample IDs")
        if df_orig[orig_id_col].duplicated().any():
            raise ValueError("Survival CSV contains duplicate sample IDs")
        df_scores[id_col] = df_scores[id_col].astype(str)
        df_orig[orig_id_col] = df_orig[orig_id_col].astype(str)
        df_orig = df_orig[[orig_id_col, time_col, status_col]]
        df_scores = pd.merge(
            df_scores,
            df_orig,
            left_on=id_col,
            right_on=orig_id_col,
            how='inner',
            validate='one_to_one',
        )
        logger.info(f"成功将真实标签合并至预测结果，共匹配 {len(df_scores)} 个样本。")
    
    df_scores = df_scores.dropna(subset=[time_col, status_col, 'risk_score'])
    
    if df_scores.empty:
        raise ValueError("No valid samples remain after joining labels and dropping missing values")

    time = pd.to_numeric(df_scores[time_col], errors='raise').astype(float).values
    raw_status = pd.to_numeric(df_scores[status_col], errors='raise').astype(int)
    if not set(raw_status.unique()).issubset({0, 1}):
        raise ValueError("survival_event must contain only 0 or 1")
    status = raw_status.astype(bool).values
    risk = pd.to_numeric(df_scores['risk_score'], errors='raise').astype(float).values
    if not np.isfinite(time).all() or not np.isfinite(risk).all():
        raise ValueError("survival_time and risk_score must be finite")
    if (time < 0).any():
        raise ValueError("survival_time must be non-negative")
    
    logger.info("正在计算 C-index...")
    c_index = concordance_index_censored(status, time, risk)[0]
    
    logger.info("正在尝试计算 Time-dependent AUC 和 iAUC...")
    y = np.array(
        [(s, t) for s, t in zip(status, time)],
        dtype=[('status', 'bool'), ('time', 'f8')]
    )
    
    target_times = [12, 36, 60] 
    time_labels = ['1-Year', '3-Year', '5-Year']
    
    min_time = time[status].min() if status.any() else 0
    max_time = time.max()
    
    results_list = [{'Metric': 'C-index', 'Value': c_index}]
    
    valid_times = []
    valid_labels = []
    for t, label in zip(target_times, time_labels):
        if min_time < t < max_time:
            valid_times.append(t)
            valid_labels.append(label)
        else:
            logger.warning(f"时间点 {label} ({t}) 超出随访时间范围 (Min Event: {min_time:.1f}, Max: {max_time:.1f})，跳过该时间点。")
            results_list.append({'Metric': f'AUC_{label}', 'Value': np.nan})

    if valid_times:
        try:
            auc_values, mean_auc = cumulative_dynamic_auc(y, y, risk, valid_times)
            
            for label, auc_val in zip(valid_labels, auc_values):
                results_list.append({'Metric': f'AUC_{label}', 'Value': auc_val})
            
            results_list.append({'Metric': 'iAUC', 'Value': mean_auc})
            logger.info("AUC/iAUC 计算成功。")
            
        except Exception as e:
            logger.warning(f"计算 Time-dependent AUC 和 iAUC 时失败: {e}")
            logger.warning("已跳过 iAUC 计算，将仅输出 C-index。")
            for label in valid_labels:
                results_list.append({'Metric': f'AUC_{label}', 'Value': np.nan})
            results_list.append({'Metric': 'iAUC', 'Value': np.nan})
    else:
        results_list.append({'Metric': 'iAUC', 'Value': np.nan})
        logger.warning("所有预设的时间点均不在有效范围内，无法计算 AUC。将仅输出 C-index。")

    df_results = pd.DataFrame(results_list)
    
    print("\n" + "="*40)
    print("           EVALUATION METRICS")
    print("="*40)
    print(df_results.to_string(index=False))
    print("="*40 + "\n")
    
    output_dir = os.path.dirname(args.scores)
    output_csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    df_results.to_csv(output_csv_path, index=False)
    logger.info(f"✅ 评估结果已保存至: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate C-index and Time-dependent AUC")
    parser.add_argument('--scores', type=str, required=True, help="Path to prognosis_risk_scores.csv")
    parser.add_argument('--yaml', type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    main(args)
