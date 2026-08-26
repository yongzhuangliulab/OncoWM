# Copyright 2026 Geely AI Lab & Clinical AI Research
import os
import yaml
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.models.OncoWM import init_oncowm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import random

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class InferenceDataset(Dataset):
    def __init__(self, csv_path, path_pt=None, text_pt=None, geno_pt=None, tran_pt=None):
        self.full_df = pd.read_csv(csv_path)
        logger.info(f"Loaded cohort CSV with {len(self.full_df)} samples.")

        self.path_dict = self._load_optional_pt(path_pt, 'pathology')
        self.text_dict = self._load_optional_pt(text_pt, 'clinical')
        self.geno_dict = self._load_optional_pt(geno_pt, 'mutation')
        self.tran_dict = self._load_optional_pt(tran_pt, 'transcriptome')
        
        valid_id_sets = []
        if self.path_dict: 
            valid_id_sets.append(set(self.path_dict.keys()))
        if self.text_dict: 
            valid_id_sets.append(set(self.text_dict.keys()))
        if self.geno_dict: 
            valid_id_sets.append(set(self.geno_dict.keys()))
        if self.tran_dict: 
            valid_id_sets.append(set(self.tran_dict.keys()))
            
        if valid_id_sets:
            common_ids = set.intersection(*valid_id_sets)
            id_col = 'sample_id' if 'sample_id' in self.full_df.columns else ('PATIENT_ID' if 'PATIENT_ID' in self.full_df.columns else self.full_df.columns[0])
            self.full_df = self.full_df[self.full_df[id_col].astype(str).isin(common_ids)]
            self.full_df = self.full_df.reset_index(drop=True)
        else:
            raise ValueError("Inference requires at least one enabled modality PT file")

        logger.info(f"Retained {len(self.full_df)} samples after intersecting the enabled modalities.")
        if self.full_df.empty:
            raise ValueError("No shared sample_id exists between the cohort CSV and enabled modalities")

    @staticmethod
    def _load_optional_pt(path, modality):
        if path is None:
            logger.info("%s modality is disabled.", modality)
            return {}
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configured {modality} PT does not exist: {path}")
        value = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(value, dict):
            raise TypeError(f"{modality} PT must contain a sample_id dictionary: {path}")
        return value

    def __len__(self):
        return len(self.full_df)
    def __getitem__(self, idx):
        row = self.full_df.iloc[idx]
        patient_id = str(row.get('sample_id', row.get('PATIENT_ID', row.iloc[0]))) 
        path_feat = self.path_dict.get(patient_id, None)
        text_data = self.text_dict.get(patient_id, None)
        geno_raw = self.geno_dict.get(patient_id, None)
        tran_feat = self.tran_dict.get(patient_id, None)
        sample = {'patient_id': patient_id}
        if path_feat is not None: sample['path_features'] = path_feat
        if geno_raw is not None: sample['geno_raw'] = geno_raw
        if tran_feat is not None: sample['tran_features'] = tran_feat    
        if text_data is not None:
            sample['clinical_text'] = {
                'input_ids': text_data.get('input_ids', text_data),
                'attention_mask': text_data.get('attention_mask', torch.ones_like(text_data.get('input_ids', text_data))),
                'token_type_ids': text_data.get('token_type_ids', torch.zeros_like(text_data.get('input_ids', text_data)))
            }
        return sample

def collate_fn(batch):
    batch_dict = {'patient_id': [item['patient_id'] for item in batch]}
    def stack_if_present(key):
        items = []
        for item in batch:
            if key in item:
                val = item[key]
                if isinstance(val, torch.Tensor) and val.dim() > 1 and val.shape[0] == 1:
                    val = val.squeeze(0)
                items.append(val)
        return torch.stack(items) if len(items) == len(batch) else None
    batch_dict['path_features'] = stack_if_present('path_features')
    batch_dict['geno_raw'] = stack_if_present('geno_raw')
    batch_dict['tran_features'] = stack_if_present('tran_features')
    text_items = [item['clinical_text'] for item in batch if 'clinical_text' in item]
    if len(text_items) == len(batch):
        batch_dict['clinical_text'] = {
            'input_ids': torch.stack([t['input_ids'].squeeze(0) for t in text_items]),
            'attention_mask': torch.stack([t['attention_mask'].squeeze(0) for t in text_items]),
            'token_type_ids': torch.stack([t['token_type_ids'].squeeze(0) for t in text_items])
        }
    else:
        batch_dict['clinical_text'] = None
    return batch_dict


def main(args):
    with open(args.fname, 'r') as f:
        params = yaml.safe_load(f)

    set_deterministic_seed(params.get('seed', 42))
    task_mode = params.get('task_mode', 'align')  
    out_dir = os.path.join(params.get('output_path', './'), params.get('run_name', 'inference'))
    
    is_completed = False
    if task_mode == 'align':
        is_completed = os.path.exists(os.path.join(out_dir, 'joint_embeddings.npz'))
    elif task_mode == 'prognosis':
        is_completed = (os.path.exists(os.path.join(out_dir, 'prog_embeddings.npz')) and 
                        os.path.exists(os.path.join(out_dir, 'prognosis_risk_scores.csv')))
    elif task_mode == 'cluster':
        is_completed = os.path.exists(os.path.join(out_dir, 'cluster_embeddings.npz'))
        
    if is_completed and not args.force:
        logger.info(f"Output files already exist in {out_dir}; skipping {task_mode.upper()} mode.")
        return

    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(params.get('device', default_device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    use_bfloat16 = bool(params.get('use_bfloat16', True)) and device.type == 'cuda'

    logger.info(f"Initializing inference mode: {task_mode.upper()}, device: {device}")

    d_params = params['data']
    dataset = InferenceDataset(
        csv_path=d_params['csv_path'],
        path_pt=d_params.get('path_pt_path'),
        text_pt=d_params.get('text_pt_path'),
        geno_pt=d_params.get('geno_pt_path'),
        tran_pt=d_params.get('tran_pt_path')
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=d_params.get('batch_size', 16), 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=d_params.get('num_workers', 4),
        pin_memory=d_params.get('pin_mem', True)
    )

    models_cfg = params['models']

    oncowm = init_oncowm(
        text_encoder_name=models_cfg['text_encoder_name'],
        path_dim=models_cfg.get('path_dim', 768),
        geno_raw_dim=models_cfg.get('geno_raw_dim', 40543),
        geno_dim=models_cfg.get('geno_dim', 512),
        tran_dim=models_cfg.get('tran_dim', 768),
        hidden_size=models_cfg.get('hidden_size', 1024),
        dropout_rate=0.0
    )
    
    ckpt_path = params.get('model_path')
    if not (ckpt_path and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"找不到 YAML 中指定的 model_path: {ckpt_path}")
        
    logger.info(f"Loading pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    raw_state_dict = checkpoint.get(
        'oncowm', checkpoint.get('mm_jepa', checkpoint.get('state_dict', checkpoint))
    )
    state_dict = {
        k.replace('module.', '').replace('le_sigreg.', 'oncowm_sigreg.'): v
        for k, v in raw_state_dict.items()
    }
    state_dict.pop('head_joint.out_proj.weight', None)

    incompatible = oncowm.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning("Checkpoint missing keys: %s", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Checkpoint unexpected keys: %s", incompatible.unexpected_keys)

    if use_bfloat16:
        oncowm = oncowm.bfloat16()
    oncowm = oncowm.to(device)
    oncowm.eval()

    results_dict = {'sample_id': [], 'risk_scores': [], 'joint_embeddings': []}
    
    def hook_fn(_module, _inputs, output):
        joint_emb = output.cpu().float().numpy()
        results_dict['joint_embeddings'].append(joint_emb)

    hook_handle = oncowm.norm_joint.register_forward_hook(hook_fn)

    logger.info(f"Starting inference extraction, mode: {task_mode.upper()}")
    
    for sample in tqdm(dataloader, desc=f"Inference ({task_mode})"):
        clin_ids = sample['clinical_text']['input_ids'].to(device) if sample['clinical_text'] else None
        clin_mask = sample['clinical_text']['attention_mask'].to(device) if sample['clinical_text'] else None
        clin_type_ids = sample['clinical_text']['token_type_ids'].to(device) if sample['clinical_text'] else None
        
        path_feat = sample['path_features'].to(device, dtype=torch.bfloat16 if use_bfloat16 else torch.float32) if sample['path_features'] is not None else None
        geno_raw = sample['geno_raw'].to(device, dtype=torch.bfloat16 if use_bfloat16 else torch.float32) if sample['geno_raw'] is not None else None
        tran_feat = sample['tran_features'].to(device, dtype=torch.bfloat16 if use_bfloat16 else torch.float32) if sample['tran_features'] is not None else None
        results_dict['sample_id'].extend(sample['patient_id'])
        with torch.inference_mode():
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                
                forward_task = 'cluster' if task_mode == 'align' else 'prognosis'
                out = oncowm(
                    text_input_ids=clin_ids, text_attention_mask=clin_mask, text_token_type_ids=clin_type_ids,
                    path_features=path_feat, geno_raw=geno_raw, tran_features=tran_feat,
                    task=forward_task, drop_prob=0.0
                )
                if task_mode == 'prognosis':
                    risk_scores = out
                    results_dict['risk_scores'].extend(risk_scores.cpu().float().numpy().tolist())

    hook_handle.remove()

    os.makedirs(out_dir, exist_ok=True)
    all_ids = results_dict['sample_id']
    
    joint_emb_cat = np.concatenate(results_dict['joint_embeddings'], axis=0)
    
    if task_mode == 'align':
        np.savez_compressed(os.path.join(out_dir, 'joint_embeddings.npz'), sample_id=all_ids, joint_embedding=joint_emb_cat)
        logger.info(f"Alignment embeddings saved to: {out_dir}/joint_embeddings.npz")
        
    elif task_mode == 'prognosis':
        np.savez_compressed(os.path.join(out_dir, 'prog_embeddings.npz'), sample_id=all_ids, joint_embedding=joint_emb_cat)
        
        df_risk = pd.DataFrame({'sample_id': all_ids, 'risk_score': results_dict['risk_scores']})
        df_risk.to_csv(os.path.join(out_dir, 'prognosis_risk_scores.csv'), index=False)
        logger.info("Prognosis embeddings and risk scores generated.")
        logger.info(f"Prognosis embeddings saved to: {out_dir}/prog_embeddings.npz")
        logger.info(f"Risk scores saved to: {out_dir}/prognosis_risk_scores.csv")
        
    elif task_mode == 'cluster':
        np.savez_compressed(os.path.join(out_dir, 'cluster_embeddings.npz'), sample_id=all_ids, joint_embedding=joint_emb_cat)
        logger.info(f"Cluster embeddings saved to: {out_dir}/cluster_embeddings.npz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OncoWM inference from a YAML config")
    parser.add_argument('--fname', type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument('--force', action='store_true', help="Overwrite/recreate outputs that already exist")
    args = parser.parse_args()
    main(args)
