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
    logger.info(f"Reading risk score file: {args.scores}")
    df_scores = pd.read_csv(args.scores)
    missing_score_columns = {'sample_id', 'risk_score'}.difference(df_scores.columns)
    if missing_score_columns:
        raise ValueError(f"Risk score CSV is missing columns: {sorted(missing_score_columns)}")
    if df_scores['sample_id'].duplicated().any():
        raise ValueError("Risk score CSV contains duplicate sample IDs")
    
    logger.info(f"Reading YAML configuration: {args.yaml}")
    with open(args.yaml, 'r') as f:
        params = yaml.safe_load(f)
        
    time_col = 'survival_time'    
    status_col = 'survival_event' 
    
    if time_col not in df_scores.columns or status_col not in df_scores.columns:
        logger.info(f"Prediction output does not contain ({time_col}, {status_col}); loading labels from the source dataset.")
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
        logger.info(f"Merged ground-truth labels for {len(df_scores)} samples.")
    
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
    
    logger.info("Computing C-index.")
    c_index = concordance_index_censored(status, time, risk)[0]
    
    logger.info("Computing time-dependent AUC and iAUC.")
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
            logger.warning(f"Time point {label} ({t}) is outside the follow-up range (minimum event time: {min_time:.1f}, maximum time: {max_time:.1f}); skipping.")
            results_list.append({'Metric': f'AUC_{label}', 'Value': np.nan})

    if valid_times:
        try:
            auc_values, mean_auc = cumulative_dynamic_auc(y, y, risk, valid_times)
            
            for label, auc_val in zip(valid_labels, auc_values):
                results_list.append({'Metric': f'AUC_{label}', 'Value': auc_val})
            
            results_list.append({'Metric': 'iAUC', 'Value': mean_auc})
            logger.info("AUC and iAUC computed successfully.")
            
        except Exception as e:
            logger.warning(f"Failed to compute time-dependent AUC and iAUC: {e}")
            logger.warning("Skipping iAUC; only C-index will be reported.")
            for label in valid_labels:
                results_list.append({'Metric': f'AUC_{label}', 'Value': np.nan})
            results_list.append({'Metric': 'iAUC', 'Value': np.nan})
    else:
        results_list.append({'Metric': 'iAUC', 'Value': np.nan})
        logger.warning("No configured time point is within the valid range; only C-index will be reported.")

    df_results = pd.DataFrame(results_list)
    
    print("\n" + "="*40)
    print("           EVALUATION METRICS")
    print("="*40)
    print(df_results.to_string(index=False))
    print("="*40 + "\n")
    
    output_dir = os.path.dirname(args.scores)
    output_csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    df_results.to_csv(output_csv_path, index=False)
    logger.info(f"Evaluation results saved to: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate C-index and Time-dependent AUC")
    parser.add_argument('--scores', type=str, required=True, help="Path to prognosis_risk_scores.csv")
    parser.add_argument('--yaml', type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    main(args)
