import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedKFold  

logger = logging.getLogger(__name__)

class MedicalDataset(Dataset):
    def __init__(self, dataframe, path_dict, text_dict, geno_dict, tran_dict):
        self.df = dataframe.reset_index(drop=True)
        self.path_dict = path_dict
        self.text_dict = text_dict
        self.geno_dict = geno_dict
        self.tran_dict = tran_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row.get('sample_id', row.iloc[0])) 
        
        text_data = self.text_dict[sample_id]
        text_input_ids = text_data['input_ids'].squeeze(0)
        text_attention_mask = text_data['attention_mask'].squeeze(0)
        text_token_type_ids = text_data.get('token_type_ids', torch.zeros_like(text_attention_mask)).squeeze(0)

        path_features = self._squeeze_leading_singleton(self.path_dict[sample_id])
        geno_raw = self._squeeze_leading_singleton(self.geno_dict[sample_id])
        tran_features = self._squeeze_leading_singleton(self.tran_dict[sample_id])

        survival_time = float(row.get('survival_time', 0.0))
        survival_event = float(row.get('survival_event', 0.0))

        # Break tied event times deterministically for a stable Cox gradient.
        survival_time += (idx % 1000) * 1e-5 

        return {
            'sample_id': sample_id,
            'clinical_text': {
                'input_ids': text_input_ids,
                'attention_mask': text_attention_mask,
                'token_type_ids': text_token_type_ids
            },
            'path_features': path_features,
            'geno_raw': geno_raw,
            'tran_features': tran_features,
            'survival_time': survival_time,
            'survival_event': survival_event
        }

    @staticmethod
    def _squeeze_leading_singleton(value):
        if isinstance(value, torch.Tensor) and value.dim() > 1 and value.shape[0] == 1:
            return value.squeeze(0)
        return value

class DatasetManager:
    def __init__(self, batch_size, pin_mem, num_workers, world_size, rank, 
                 csv_path, path_pt_path, text_pt_path, geno_pt_path, tran_pt_path, 
                 seed=42, num_folds=5):
        self.batch_size = batch_size
        self.pin_mem = pin_mem
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.num_folds = num_folds

        self.full_df = pd.read_csv(csv_path)
        self.path_dict, self.text_dict, self.geno_dict, self.tran_dict = self.load_offline_dicts(
            path_pt_path, text_pt_path, geno_pt_path, tran_pt_path
        )
        self._filter_valid_samples()

    def load_offline_dicts(self, path_pt, text_pt, geno_pt, tran_pt):
        if self.rank == 0: logger.info("开始加载所有模态样本级离线字典")
        paths = {
            'pathology': path_pt,
            'clinical': text_pt,
            'mutation': geno_pt,
            'transcriptome': tran_pt,
        }
        loaded = {}
        for name, path in paths.items():
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Training requires a valid {name} PT path: {path}")
            value = torch.load(path, map_location='cpu', weights_only=False)
            if not isinstance(value, dict):
                raise TypeError(f"{name} PT must contain a sample_id dictionary: {path}")
            loaded[name] = value
        path_d = loaded['pathology']
        text_d = loaded['clinical']
        geno_d = loaded['mutation']
        tran_d = loaded['transcriptome']
        return path_d, text_d, geno_d, tran_d

    def _filter_valid_samples(self):
        full_df = self.full_df
        valid_indices = []
        for idx in range(len(full_df)):
            row = full_df.iloc[idx]
            sample_id = str(row.get('sample_id', row.iloc[0]))
            if (sample_id in self.path_dict and 
                sample_id in self.text_dict and 
                sample_id in self.geno_dict and 
                sample_id in self.tran_dict):
                valid_indices.append(idx)
                
        self.full_df = full_df.loc[valid_indices].reset_index(drop=True)
        if self.rank == 0: logger.info(f"校验完成，有效样本数: {len(self.full_df)}")
        if self.full_df.empty:
            raise ValueError("No shared sample_id exists across survival labels and all four training modalities")

    def resplit_data(self, fold_idx=0):
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        indices = list(range(len(self.full_df)))
        
        events = self.full_df['survival_event'].values
        times = self.full_df['survival_time'].values
        
        time_bins = pd.qcut(times, q=4, labels=False, duplicates='drop')
        stratify_labels = [f"{int(e)}_{int(t)}" for e, t in zip(events, time_bins)]
        
        train_idx, val_idx = list(skf.split(indices, stratify_labels))[fold_idx]
        
        train_df = self.full_df.iloc[train_idx].reset_index(drop=True)
        val_df = self.full_df.iloc[val_idx].reset_index(drop=True)

        if self.rank == 0: 
            val_events = val_df['survival_event'].sum()
            logger.info(f"Fold {fold_idx}，训练样本: {len(train_df)}，验证样本: {len(val_df)}，验证事件: {int(val_events)}")

        train_dataset = MedicalDataset(train_df, self.path_dict, self.text_dict, self.geno_dict, self.tran_dict)
        val_dataset = MedicalDataset(val_df, self.path_dict, self.text_dict, self.geno_dict, self.tran_dict)

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        safe_drop_last = True if len(train_dataset) > self.batch_size else False
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, pin_memory=self.pin_mem, drop_last=safe_drop_last
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, sampler=val_sampler,
            num_workers=self.num_workers, pin_memory=self.pin_mem, drop_last=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler

    def get_full_cohort_data(self):
        if self.rank == 0: 
            logger.info(f"开始加载全量数据，共 {len(self.full_df)} 个有效多模态样本")

        full_dataset = MedicalDataset(
            self.full_df, self.path_dict, self.text_dict, self.geno_dict, self.tran_dict
        )

        full_sampler = DistributedSampler(
            full_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )

        full_loader = DataLoader(
            full_dataset, 
            batch_size=self.batch_size, 
            sampler=full_sampler,
            num_workers=self.num_workers, 
            pin_memory=self.pin_mem, 
            drop_last=False
        )
        
        return full_dataset, full_loader, full_sampler
