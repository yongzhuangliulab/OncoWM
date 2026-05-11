import logging
import sys
import yaml
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA')

from src.models.m3_jepe_modify_v3_1 import init_m3jepa  
from src.datasets.medical_dataset import DatasetManager 
from src.helper import load_checkpoint, init_opt
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.optimizer import EarlyStopping

try:
    from lifelines.utils import concordance_index
except ImportError:
    concordance_index = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:1')
    
    task_mode = args.get('task_mode', 'align') 
    run_name = args.get('run_name', 'default_run') 
    use_bfloat16 = args['meta'].get('use_bfloat16', True)
    
    opt_cfg = args['optimization']
    num_epochs = opt_cfg['epochs']
    patience = opt_cfg['patience']
    base_wd = float(opt_cfg['t2i_weight_decay'])
    base_final_wd = float(opt_cfg['t2i_final_weight_decay'])
    base_warmup = opt_cfg['t2i_warmup']
    ipe_scale = opt_cfg['ipe_scale']

    task_opt = opt_cfg[task_mode]
    base_start_lr = task_opt['start_lr']
    base_lr = task_opt['lr']
    base_final_lr = task_opt['final_lr']
    
    batch_size = args['data'].get(f'{task_mode}_batch_size', 90)

    logger.info(f"=== Session: {run_name} | Task: {task_mode.upper()} | LR: {base_lr} | Batch: {batch_size} ===")
    
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    csv_path = args['data'].get('csv_path')
    path_pt_path = args['data'].get('path_pt_path')
    text_pt_path = args['data'].get('text_pt_path')
    geno_pt_path = args['data'].get('geno_pt_path')
    tran_pt_path = args['data'].get('tran_pt_path')
    val_split = args['eva']['val_spilit']
    
    text_encoder_name = args['models']['text_encoder_name']
    path_dim = args['models'].get('path_dim', 768)
    geno_raw_dim = args['models'].get('geno_raw_dim', 40543)
    geno_dim = args['models'].get('geno_dim', 512)
    tran_dim = args['models'].get('tran_dim', 643)
    
    num_experts = args['MoE']['num_experts']
    k_expert = args['MoE']['k']
    drop_rate = args['MoE']['drop']
    hidden_size = args['MoE']['hidden_size']
    max_drop_prob = args.get('modality_drop_prob', 0.35)

    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    folder = os.path.join(folder, run_name)
    os.makedirs(folder, exist_ok=True)
    
    log_file = os.path.join(folder, f'{tag}_{task_mode}.csv')
    best_path = os.path.join(folder, f'{task_mode}-best.pth.tar')
    tb_log_dir = os.path.join(args['tb']['log_dir'], run_name)
    writer = SummaryWriter(tb_log_dir)
    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%s', 'task'), ('%d', 'time (ms)'))

    mm_jepa = init_m3jepa(
        text_encoder_name=text_encoder_name, path_dim=path_dim, 
        geno_raw_dim=geno_raw_dim, geno_dim=geno_dim, 
        tran_dim=tran_dim, num_experts=num_experts, hidden_size=hidden_size, k_expert=k_expert, dropout_rate=drop_rate
    )
    
    best_metric = -float('inf')

    if task_mode == 'prognosis':
        resume_checkpoint = os.path.join(folder, 'align-best.pth.tar')
        if os.path.exists(resume_checkpoint):
            logger.info(f">>> 自动路由: 找到对齐权重，准备加载 -> {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
            state_dict = {k.replace('module.', ''): v for k, v in (checkpoint['mm_jepa'] if 'mm_jepa' in checkpoint else checkpoint).items()}
            
            # 过滤旧的预测头
            state_dict = {k: v for k, v in state_dict.items() if 'head_B_prog' not in k}
            
            mm_jepa.load_state_dict(state_dict, strict=False)
            logger.info(">>> 权重加载成功！(已自动忽略维度不匹配的旧预测头)")
            
        # [!!! 核心修改：特征冻结模式 !!!]
        logger.info(">>> 【线性探测模式】冻结所有多模态特征提取层，仅训练 1 维连续 Cox 预测头！")
        for name, param in mm_jepa.named_parameters():
            # 仅允许 1 维的 head_B_prog 更新梯度，杜绝底层特征为了拟合生存天数而发生灾难性遗忘
            if "head_B_prog" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        logger.info(">>> 对齐任务: 模型将从初始状态开始训练。")

    if use_bfloat16: mm_jepa = mm_jepa.bfloat16()
    mm_jepa = mm_jepa.to(device)

    data_manager = DatasetManager(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, 
        world_size=1, rank=0, 
        csv_path=csv_path, path_pt_path=path_pt_path, text_pt_path=text_pt_path, 
        geno_pt_path=geno_pt_path, tran_pt_path=tran_pt_path, val_split=val_split
    )
    
    _, _, train_loader, val_loader, train_sampler, _ = data_manager.resplit_data()

    ipe = len(train_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        mm_jepa=mm_jepa, wd=base_wd, final_wd=base_final_wd, start_lr=base_start_lr, ref_lr=base_lr,
        final_lr=base_final_lr, iterations_per_epoch=ipe, warmup=base_warmup, num_epochs=num_epochs,
        ipe_scale=ipe_scale, use_bfloat16=use_bfloat16
    )
    
    early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, maximize=True)

    for epoch in range(num_epochs):
        mm_jepa.train()
        if hasattr(train_sampler, 'set_epoch') and task_mode != 'prognosis': 
            train_sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        base_drop_prob = max_drop_prob * ((epoch - num_epochs * 0.2) / (num_epochs * 0.8)) if epoch >= num_epochs * 0.2 else 0.0

        for itr, sample in enumerate(train_loader):
            clinical_ids = sample['clinical_text']['input_ids'].to(device, non_blocking=True)
            clinical_mask = sample['clinical_text']['attention_mask'].to(device, non_blocking=True)
            clinical_type_ids = sample['clinical_text']['token_type_ids'].to(device, non_blocking=True)
            
            actual_max_len = int(clinical_mask.sum(dim=1).max().item())
            clinical_ids = clinical_ids[:, :actual_max_len]
            clinical_mask = clinical_mask[:, :actual_max_len]
            clinical_type_ids = clinical_type_ids[:, :actual_max_len]

            path_features = sample['path_features'].to(device, dtype=torch.bfloat16, non_blocking=True)
            geno_raw = sample['geno_raw'].to(device, dtype=torch.bfloat16, non_blocking=True)
            tran_features = sample['tran_features'].to(device, dtype=torch.bfloat16, non_blocking=True)
            
            surv_time_raw = sample['survival_time'].to(device, non_blocking=True)
            surv_event = sample['survival_event'].to(device, non_blocking=True)

            def train_step():
                optimizer.zero_grad()
                batch_risk = None
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    if task_mode == 'align':
                        loss = mm_jepa(
                            text_input_ids=clinical_ids, text_attention_mask=clinical_mask, text_token_type_ids=clinical_type_ids,
                            path_features=path_features, geno_raw=geno_raw, tran_features=tran_features,
                            task='align', surv_time=surv_time_raw, surv_event=surv_event, drop_prob=base_drop_prob
                        )
                        current_task = 'align'
                    else:
                        out = mm_jepa(
                            text_input_ids=clinical_ids, text_attention_mask=clinical_mask, text_token_type_ids=clinical_type_ids,
                            path_features=path_features, geno_raw=geno_raw, tran_features=tran_features,
                            task='prognosis', surv_time=surv_time_raw, surv_event=surv_event, drop_prob=base_drop_prob
                        )
                        loss = out[0]
                        batch_risk = out[1]
                        current_task = 'prognosis'

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                optimizer.step()
                
                risk_info = (batch_risk.min().item(), batch_risk.max().item()) if batch_risk is not None else None
                return loss.item(), scheduler.step(), wd_scheduler.step(), current_task, risk_info

            ret, etime = gpu_timer(train_step)
            loss_val, new_lr, new_wd, task_type, risk_info = ret
            
            loss_meter.update(loss_val)
            csv_logger.log(epoch + 1, itr, loss_val, task_type, etime)

            if itr % 10 == 0:
                log_str = f'[{epoch + 1}, {itr}/{ipe}] Task: {task_type:<15} | Loss: {loss_val:.4f} | LR: {new_lr:.8f}'
                if risk_info is not None:
                    log_str += f' | Risk Score Range: [{risk_info[0]:.2f}, {risk_info[1]:.2f}]'
                logger.info(log_str)

        logger.info(f'--- Epoch {epoch + 1} Internal Validation ---')
        mm_jepa.eval()
        val_loss_meter = AverageMeter()
        all_risk_scores, all_surv_times, all_surv_events = [], [], []

        with torch.no_grad():
            for val_sample in val_loader:
                v_clin_ids = val_sample['clinical_text']['input_ids'].to(device)
                v_clin_mask = val_sample['clinical_text']['attention_mask'].to(device)
                v_clin_type_ids = val_sample['clinical_text']['token_type_ids'].to(device)
                
                v_max_len = int(v_clin_mask.sum(dim=1).max().item())
                v_clin_ids = v_clin_ids[:, :v_max_len]
                v_clin_mask = v_clin_mask[:, :v_max_len]
                v_clin_type_ids = v_clin_type_ids[:, :v_max_len]
                
                v_path = val_sample['path_features'].to(device, dtype=torch.bfloat16)
                v_geno = val_sample['geno_raw'].to(device, dtype=torch.bfloat16)
                v_tran = val_sample['tran_features'].to(device, dtype=torch.bfloat16)
                
                v_time_raw = val_sample['survival_time'].to(device)
                v_event = val_sample['survival_event'].to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    out = mm_jepa(
                        text_input_ids=v_clin_ids, text_attention_mask=v_clin_mask, text_token_type_ids=v_clin_type_ids,
                        path_features=v_path, geno_raw=v_geno, tran_features=v_tran,
                        task=task_mode, surv_time=v_time_raw, surv_event=v_event, drop_prob=0.0
                    )
                
                if task_mode == 'prognosis':
                    v_loss = out[0]
                    v_scores = out[1]
                    
                    all_risk_scores.extend(v_scores.float().cpu().numpy().tolist())
                    all_surv_times.extend(v_time_raw.float().cpu().numpy().tolist())
                    all_surv_events.extend(v_event.float().cpu().numpy().tolist())
                else:
                    v_loss = out

                val_loss_meter.update(float(v_loss))

        current_metric = -val_loss_meter.avg
        metric_name = "Negative Val Loss"
        
        if task_mode == 'prognosis' and concordance_index is not None and len(all_surv_times) > 0:
            try:
                c_index = concordance_index(np.array(all_surv_times), -np.array(all_risk_scores), np.array(all_surv_events))
                current_metric = c_index
                metric_name = "C-index"
                logger.info(f"Epoch {epoch + 1} | Val C-index: {c_index:.4f} | Val Cox_Loss: {val_loss_meter.avg:.4f}")
                if writer:
                    writer.add_scalar(f'Val/{task_mode.capitalize()}_C-index', c_index, epoch)
            except Exception as e:
                logger.warning(f"C-index failed: {e}. Fallback to Loss.")
        else:
            logger.info(f"Epoch {epoch + 1} | Val Loss: {val_loss_meter.avg:.4f}")

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save({
                'mm_jepa': mm_jepa.state_dict(),
                'opt': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None, 
                'epoch': epoch + 1,
                'best_metric': best_metric,
                'metric_name': metric_name
            }, best_path)
            logger.info(f'>>> Saved new best model to {best_path} ({metric_name}: {best_metric:.4f})')
        
        early_stop(current_metric, mm_jepa)

        if early_stop.early_stop:
            logger.info(f'Early stopping triggered!')
            break

    if writer: writer.close()