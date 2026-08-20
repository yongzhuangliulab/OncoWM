import logging
import sys
import os
import numpy as np
import torch
import random  
import shutil  
import pandas as pd  
from torch.utils.tensorboard import SummaryWriter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from src.models.OncoWM import init_m3jepa
from src.datasets.medical import DatasetManager
from src.training import init_opt
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.optimizer import EarlyStopping

try:
    from lifelines.utils import concordance_index
except ImportError:
    concordance_index = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(args):
    seed = args.get('seed', 10)  
    num_folds = args.get('num_folds', 5) 
    logger.info(f"Global seed: {seed}, mode: {num_folds}-fold cross-validation")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    task_mode = args.get('task_mode', 'align') 
    run_name = args.get('run_name', 'default_run') 
    use_bfloat16 = args['meta'].get('use_bfloat16', True)
    
    opt_cfg = args['optimization']
    num_epochs = opt_cfg.get(f'{task_mode}_epochs', opt_cfg.get('epochs', 100))
    patience = opt_cfg['patience']
    base_wd = float(opt_cfg['t2i_weight_decay'])
    base_final_wd = float(opt_cfg['t2i_final_weight_decay'])
    base_warmup = opt_cfg['t2i_warmup']
    ipe_scale = opt_cfg['ipe_scale']

    task_opt = opt_cfg[task_mode]
    base_start_lr = task_opt['start_lr']
    base_lr = task_opt['lr']
    base_final_lr = task_opt['final_lr']
    
    if task_mode == 'prognosis':
        logger.info(f"Linear probing freezes the backbone and trains the prognosis head, LR: {base_lr}, WD: {base_wd}")
    
    batch_size = args['data'].get(f'{task_mode}_batch_size', 90)

    logger.info(f"Session: {run_name}, task: {task_mode.upper()}, LR: {base_lr}, batch: {batch_size}")
    
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    csv_path = args['data'].get('csv_path')
    path_pt_path = args['data'].get('path_pt_path')
    text_pt_path = args['data'].get('text_pt_path')
    geno_pt_path = args['data'].get('geno_pt_path')
    tran_pt_path = args['data'].get('tran_pt_path')
    text_encoder_name = args['models']['text_encoder_name']
    path_dim = args['models'].get('path_dim', 768)
    geno_raw_dim = args['models'].get('geno_raw_dim', 40543)
    geno_dim = args['models'].get('geno_dim', 512)
    tran_dim = args['models'].get('tran_dim', 643)
    
    drop_rate = args['models']['dropout_rate']
    hidden_size = args['models']['hidden_size']

    tag = args['logging']['write_tag']
    base_output_dir = args.get('output_path', args['logging'].get('folder', './experiment_logs'))
    folder = os.path.join(base_output_dir, run_name)
    os.makedirs(folder, exist_ok=True)

    log_file_path = os.path.join(folder, f"{run_name}_{task_mode}.log")
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)
        
    tb_log_dir_base = os.path.join(args['tb']['log_dir'], run_name)

    data_manager = DatasetManager(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, 
        world_size=1, rank=0, 
        csv_path=csv_path, path_pt_path=path_pt_path, text_pt_path=text_pt_path, 
        geno_pt_path=geno_pt_path, tran_pt_path=tran_pt_path,
        seed=seed, num_folds=num_folds
    )
    
    fold_metrics = [] 
    overall_best_metric = -float('inf')
    fold_best_epochs = []
    overall_best_fold = -1

    for fold in range(num_folds):
        current_seed = seed + fold
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        fold_dir = os.path.join(folder, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        logger.info(f"Starting training for fold {fold}/{num_folds-1}")
        
        mm_jepa = init_m3jepa(
            text_encoder_name=text_encoder_name, path_dim=path_dim, 
            geno_raw_dim=geno_raw_dim, geno_dim=geno_dim, 
            tran_dim=tran_dim, hidden_size=hidden_size, dropout_rate=drop_rate
        )
        
        log_file = os.path.join(fold_dir, f'{tag}_{task_mode}.csv')
        best_path = os.path.join(fold_dir, f'{task_mode}-best.pth.tar')
        writer = SummaryWriter(os.path.join(tb_log_dir_base, f'fold_{fold}'))
        csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%s', 'task'), ('%d', 'time (ms)'))

        if task_mode == 'prognosis':
            resume_checkpoint = os.path.join(fold_dir, 'align-best.pth.tar')
            if os.path.exists(resume_checkpoint):
                checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
                state_dict = {k.replace('module.', ''): v for k, v in (checkpoint['mm_jepa'] if 'mm_jepa' in checkpoint else checkpoint).items()}
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_")}
                mm_jepa.load_state_dict(state_dict, strict=False)
                logger.info(f"Fold {fold} loaded alignment weights and excluded prognosis-head weights.")
            else:
                if args.get('allow_prognosis_from_scratch', False):
                    logger.warning(f"Checkpoint not found at {resume_checkpoint}; prognosis training from scratch is explicitly enabled.")
                else:
                    raise FileNotFoundError(
                        f"Prognosis requires the matching alignment checkpoint: {resume_checkpoint}. "
                        "Run align first, or explicitly set allow_prognosis_from_scratch: true."
                    )
            
            for name, param in mm_jepa.named_parameters():
                if name.startswith("head_joint"):  
                    param.requires_grad = True
                else:                
                    param.requires_grad = False

        mm_jepa = mm_jepa.to(device)
        
        if use_bfloat16:
            mm_jepa = mm_jepa.bfloat16()
            for param in mm_jepa.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)

        if task_mode == 'prognosis':
            decay_params = []
            no_decay_params = []
            
            for name, param in mm_jepa.named_parameters():
                if param.requires_grad:
                    if param.ndim <= 1 or name.endswith(".bias") or "LayerNorm" in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
            
            params_to_optimize = [
                {'params': decay_params, 'lr_scale': 1.0, 'weight_decay': base_wd},
                {'params': no_decay_params, 'lr_scale': 1.0, 'weight_decay': 0.0}
            ]
        else:
            params_to_optimize = list(filter(lambda p: p.requires_grad, mm_jepa.parameters()))

        _, _, train_loader, val_loader, train_sampler, _ = data_manager.resplit_data(fold_idx=fold)
        
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        train_csv_path = os.path.join(fold_dir, 'train.csv')
        val_csv_path = os.path.join(fold_dir, 'val.csv')
        train_dataset.df.to_csv(train_csv_path, index=False)
        val_dataset.df.to_csv(val_csv_path, index=False)

        ipe = len(train_loader)

        optimizer, scheduler, wd_scheduler = init_opt(
            params_to_optimize=params_to_optimize,  
            wd=base_wd,               
            final_wd=base_final_wd,        
            start_lr=base_start_lr,        
            ref_lr=base_lr,           
            final_lr=base_final_lr,        
            iterations_per_epoch=ipe, 
            warmup=base_warmup,     
            num_epochs=num_epochs,
            ipe_scale=ipe_scale
        )
        
        early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, maximize=True)
        best_metric = -float('inf')
        metric_name = "Metric"
        best_epoch_for_fold = 1

        align_protect_epochs = 15

        for epoch in range(num_epochs):
            if task_mode == 'prognosis':
                mm_jepa.eval() 
                mm_jepa.head_joint.train()
            else:
                mm_jepa.train()
                
            if hasattr(train_sampler, 'set_epoch') and task_mode != 'prognosis': 
                train_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            
            if task_mode == 'align':
                base_drop_prob = min(0.35, 0.1 + (0.25 / align_protect_epochs) * epoch)
            elif task_mode == 'prognosis':
                base_drop_prob = 0.25
            else:
                base_drop_prob = 0.1

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
                
                surv_time_input = sample['survival_time'].to(device, non_blocking=True)
                surv_event = sample['survival_event'].to(device, non_blocking=True)

                def train_step():
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                        out = mm_jepa(
                            text_input_ids=clinical_ids, text_attention_mask=clinical_mask, text_token_type_ids=clinical_type_ids,
                            path_features=path_features, geno_raw=geno_raw, tran_features=tran_features,
                            task=task_mode, surv_time=surv_time_input, surv_event=surv_event, drop_prob=base_drop_prob
                        )
                        loss = out if task_mode == 'align' else out[0] 

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    return loss.item(), scheduler.step(), wd_scheduler.step(), task_mode

                (loss, new_lr, new_wd, task_type), etime = gpu_timer(train_step)
                loss_meter.update(loss)
                csv_logger.log(epoch + 1, itr, loss, task_type, etime)

                if itr % 10 == 0:
                    logger.info(f'Fold {fold}, epoch {epoch + 1}, iteration {itr}/{ipe}, loss: {loss:.4f}, LR: {new_lr:.8f}, drop: {base_drop_prob:.3f}')

            mm_jepa.eval()
            val_loss_meter = AverageMeter()
            all_risk_scores, all_surv_times, all_surv_events = [], [], []

            with torch.no_grad():
                eval_drop_prob = 0.35 if task_mode == 'align' else 0.0

                cpu_rng_state = torch.get_rng_state()
                if torch.cuda.is_available():
                    gpu_rng_state = torch.cuda.get_rng_state(device)
                
                torch.manual_seed(42)  
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42)

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
                    v_time_input = val_sample['survival_time'].to(device)
                    v_event = val_sample['survival_event'].to(device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                        out = mm_jepa(
                            text_input_ids=v_clin_ids, text_attention_mask=v_clin_mask, text_token_type_ids=v_clin_type_ids,
                            path_features=v_path, geno_raw=v_geno, tran_features=v_tran,
                            task=task_mode, surv_time=v_time_input, surv_event=v_event, 
                            drop_prob=eval_drop_prob 
                        )
                    
                    if task_mode == 'prognosis':
                        v_loss, v_scores = out
                        all_risk_scores.extend(v_scores.float().cpu().numpy().tolist())
                        all_surv_times.extend(v_time_input.float().cpu().numpy().tolist())
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
                    logger.info(f"Fold {fold}, epoch {epoch + 1}, validation C-index: {c_index:.4f}, validation Cox loss: {val_loss_meter.avg:.4f}")
                    if writer: writer.add_scalar(f'Val/{task_mode.capitalize()}_C-index', c_index, epoch)
                except Exception as e:
                    logger.warning(f"C-index failed: {e}. Fallback to Loss.")
            else:
                logger.info(f"Fold {fold}, epoch {epoch + 1}, validation loss: {val_loss_meter.avg:.4f}")

            is_valid_to_save = True
            if task_mode == 'align' and epoch < align_protect_epochs: 
                is_valid_to_save = False

            if current_metric > best_metric and is_valid_to_save:
                best_metric = current_metric
                best_epoch_for_fold = epoch + 1
                
                clean_state_dict = {}
                for k, v in mm_jepa.state_dict().items():
                    if k.startswith('text_model.'):
                        if 'lora' in k or 'modules_to_save' in k:
                            clean_state_dict[k] = v
                    else:
                        clean_state_dict[k] = v

                checkpoint = {
                    'mm_jepa': clean_state_dict,
                    'epoch': epoch + 1,
                    'best_metric': best_metric,
                    'metric_name': metric_name
                }

                torch.save(checkpoint, best_path)
                logger.info(f'Best model saved to: {best_path}, {metric_name}: {best_metric:.4f}')
            
            is_protected = not is_valid_to_save
            
            if not is_protected:
                early_stop(current_metric)
                if early_stop.early_stop:
                    logger.info(f'Early stopping triggered for fold {fold}.')
                    break
            else:
                logger.info(f"Fold {fold}, epoch {epoch + 1}, alignment protection period; skipping checkpointing and the early-stopping counter.")
                
        if writer: writer.close()
        
        fold_metrics.append(best_metric)
        fold_best_epochs.append(best_epoch_for_fold)
        logger.info(f"Fold {fold} completed, best {metric_name}: {best_metric:.4f}")

        if best_metric > overall_best_metric:
            overall_best_metric = best_metric
            overall_best_fold = fold
            best_overall_path = os.path.join(folder, f'{task_mode}-best-overall.pth.tar')
            shutil.copy(best_path, best_overall_path)
            logger.info(f"Updated the overall best model from fold {fold}, saved to: {best_overall_path}")

    avg_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)
    mean_epoch = np.mean(fold_best_epochs)
    std_epoch = np.std(fold_best_epochs)
    logger.info(f"Final {num_folds}-fold cross-validation report")
    logger.info(f"Task: {task_mode.upper()}, metric: {metric_name}")
    for i, m in enumerate(fold_metrics):
        logger.info(f"Fold {i}: {m:.4f}")
    logger.info(f"Average {metric_name}: {avg_metric:.4f} ± {std_metric:.4f}")
    logger.info(f"Overall best model, fold {overall_best_fold}, {metric_name}: {overall_best_metric:.4f}")

    epoch_log_path = os.path.join(folder, f"{run_name}_{task_mode}_epoch_summary.log")
    with open(epoch_log_path, 'w', encoding='utf-8') as f:
        f.write(f"========== {num_folds}-Fold Early Stopping & Best Epoch Summary ==========\n")
        f.write(f"Session: {run_name} | Task: {task_mode.upper()}\n")
        f.write(f"Metric Evaluated: {metric_name}\n")
        f.write("-" * 60 + "\n")
        for i in range(num_folds):
            f.write(f"Fold {i}: Best Epoch = {fold_best_epochs[i]:<3} | {metric_name} = {fold_metrics[i]:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Average {metric_name} : {avg_metric:.4f} ± {std_metric:.4f}\n")
        f.write(f"Average Best Epoch : {mean_epoch:.1f}\n")
        f.write("=" * 60 + "\n")

    results_dict = {
        'Fold': [f"Fold_{i}" for i in range(num_folds)] + ['Average', 'Std_Dev'],
        'Best_Epoch': fold_best_epochs + [mean_epoch, std_epoch],
        metric_name: fold_metrics + [avg_metric, std_metric]
    }
    results_df = pd.DataFrame(results_dict)
    csv_out_path = os.path.join(folder, f'{task_mode}_cross_validation_results.csv')
    results_df.to_csv(csv_out_path, index=False)

if __name__ == '__main__':
    pass


   
