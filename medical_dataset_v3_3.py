import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import StratifiedKFold  # [!!! 新增 KFold !!!]

logger = logging.getLogger(__name__)

class MedicalDataset(Dataset):
    """
    (MedicalDataset 类保持完全不变，省略具体实现，直接复制你原本的代码即可)
    """
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

        path_features = self.path_dict[sample_id]
        geno_raw = self.geno_dict[sample_id]
        tran_features = self.tran_dict[sample_id]

        survival_time = float(row.get('OS.time', 0.0))
        survival_event = float(row.get('OS', 0.0))

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

class DatasetManager:
    def __init__(self, batch_size, pin_mem, num_workers, world_size, rank, 
                 csv_path, path_pt_path, text_pt_path, geno_pt_path, tran_pt_path, 
                 val_split=0.2, seed=42, num_folds=5):  # [!!! 新增 seed 和 num_folds !!!]
        self.batch_size = batch_size
        self.pin_mem = pin_mem
        self.num_workers = num_workers
        self.world_size = world_size
        self.rank = rank
        self.val_split = val_split
        self.seed = seed
        self.num_folds = num_folds

        self.full_df = pd.read_csv(csv_path)
        self.path_dict, self.text_dict, self.geno_dict, self.tran_dict = self.load_offline_dicts(
            path_pt_path, text_pt_path, geno_pt_path, tran_pt_path
        )
        self._filter_valid_samples()

    def load_offline_dicts(self, path_pt, text_pt, geno_pt, tran_pt):
        if self.rank == 0: logger.info(">>> 开始加载所有模态样本级离线字典...")
        path_d = torch.load(path_pt, map_location='cpu')
        text_d = torch.load(text_pt, map_location='cpu')
        geno_d = torch.load(geno_pt, map_location='cpu')
        tran_d = torch.load(tran_pt, map_location='cpu')
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
        if self.rank == 0: logger.info(f">>> 校验完成，有效样本数: {len(self.full_df)}")

    # [!!! 核心修改 !!!] 使用 KFold 替代单一的打乱切分
    def resplit_data(self, fold_idx=0):
        # 1. 实例化 StratifiedKFold
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        
        # 2. 获取所有的索引
        indices = list(range(len(self.full_df)))
        
        # 3. 提取分层依据：所有患者的生存状态（0=存活/删失，1=死亡）
        # 注意：这里需要填入你真实的列名，比如 'survival_event'
        events = self.full_df['survival_event'].values
        
        # 4. 根据当前的 fold_idx 拿到这一折的 训练集索引 和 验证集索引
        # split 方法现在需要传入 indices 和 分层标签 events
        train_idx, val_idx = list(skf.split(indices, events))[fold_idx]
        
        train_df = self.full_df.iloc[train_idx].reset_index(drop=True)
        val_df = self.full_df.iloc[val_idx].reset_index(drop=True)

        if self.rank == 0: 
            logger.info(f">>> Fold {fold_idx}: Train samples: {len(train_df)}, Val samples: {len(val_df)}")

        train_dataset = MedicalDataset(train_df, self.path_dict, self.text_dict, self.geno_dict, self.tran_dict)
        val_dataset = MedicalDataset(val_df, self.path_dict, self.text_dict, self.geno_dict, self.tran_dict)

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, pin_memory=self.pin_mem, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, sampler=val_sampler,
            num_workers=self.num_workers, pin_memory=self.pin_mem, drop_last=False
        )
        
        return train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler