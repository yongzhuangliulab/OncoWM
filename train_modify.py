import logging
import sys
import yaml
import os
import random
from datetime import datetime
import itertools

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from src.models.m3_jepa import init_m3jepa  # 更新导入名
from src.datasets.medical_dataset import DatasetManager, make_test_set  # 需要你自己建此文件
from src.helper import load_checkpoint, init_opt
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.optimizer import EarlyStopping
from src.evaluation import dual_embed, log_recall_rates, log_tensorboard_metrics
from src.utils.TaskProbabilityAdjuster import init_ProbabilityAdjuster

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main(local_rank, args):
    """Main function for Medical JEPA training."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    except KeyError:
        pass

    device = torch.device(f'cuda:{local_rank}')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 抽取参数配置 (变量保持读取你原 yaml 的字段)
    use_bfloat16 = args['meta'].get('use_bfloat16', True) # 强烈建议在医疗大模型设为 True
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder'] # 病理 Patch 路径
    val_split = args['eva']['val_spilit']
    sim_c = args['eva']['smilariity']
    num_epochs = args['optimization']['epochs']
    patience = args['optimization']['patience']
    
    # 临床到病理 (原 t2i) 的超参
    c2p_wd = float(args['optimization']['t2i_weight_decay'])
    c2p_final_wd = float(args['optimization']['t2i_final_weight_decay'])
    c2p_warmup = args['optimization']['t2i_warmup']
    c2p_start_lr = args['optimization']['t2i_start_lr']
    c2p_lr = args['optimization']['t2i_lr']
    c2p_final_lr = args['optimization']['t2i_final_lr']
    
    # 病理到临床 (原 i2t) 的超参
    p2c_wd = float(args['optimization']['i2t_weight_decay'])
    p2c_final_wd = float(args['optimization']['i2t_final_weight_decay'])
    p2c_warmup = args['optimization']['i2t_warmup']
    p2c_start_lr = args['optimization']['i2t_start_lr']
    p2c_lr = args['optimization']['i2t_lr']
    p2c_final_lr = args['optimization']['i2t_final_lr']
    
    ipe_scale = args['optimization']['ipe_scale']
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    tb_path = args['tb']['log_dir']
    port = args['tb']['port']
    imag_encoder_name = args['models']['imag_encoder_name']
    text_encoder_name = args['models']['text_encoder_name']
    num_experts = args['MoE']['num_experts']
    k_expert = args['MoE']['k']
    drop_rate = args['MoE']['drop']
    predictor_type = args['predictor']
    hidden_size = args['MoE']['hidden_size'] if predictor_type == 'moe_mlp' else args['MLP']['hidden_size']
    
    # 【注意】你的 yaml 配置文件中 proxy_tasks 必须设为 ['clinical2path', 'path2clinical']
    proxy_tasks = args['proxy_tasks'] 
    iter_task = args['iter_task']
    prob_samp = args['prob_samp']
    base_prob = args['task_prob']['base_prob']
    max_prob = args['task_prob']['max_prob']
    min_prob = args['task_prob']['min_prob']
    step = args['task_prob']['step']
    total_step = args['task_prob']['total_step']

    folder = os.path.join(folder, datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{global_rank}.csv')
    best_path = os.path.join(folder, f'{tag}-best.pth.tar')
    if global_rank == 0:
        with open(os.path.join(folder, 'params-medjepa.yaml'), 'w') as f:
            yaml.dump(args, f)
    csv_logger = CSVLogger(log_file, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), ('%d', 'time (ms)'))

    writer = None
    if global_rank == 0:
        os.makedirs(tb_path, exist_ok=True)
        writer = SummaryWriter(tb_path)
        logger.info(f"TensorBoard at {tb_path}, port {port}")

    # Initialize model
    mm_jepa = init_m3jepa(imag_encoder_name, text_encoder_name, num_experts, hidden_size, k_expert, drop_rate, predictor_type)
    
    # 【重要】确保主模型在设备上，并且半精度运行，再套 DDP
    if use_bfloat16:
        mm_jepa = mm_jepa.bfloat16()
        
    mm_jepa = mm_jepa.to(device)
    mm_jepa = DistributedDataParallel(mm_jepa, device_ids=[local_rank], find_unused_parameters=True)
    logger.info(f"Model parameters requiring grad: {sum(p.numel() for p in mm_jepa.parameters() if p.requires_grad)}")

    # Data loaders
    data_manager = DatasetManager(
        batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers, world_size=world_size, rank=global_rank,
        root_path=root_path, image_folder=image_folder, val_split=val_split, imag_encoder_name=imag_encoder_name,
        text_encoder_name=text_encoder_name
    )
    train_dataset, val_dataset, train_loader, val_loader, train_sampler, val_sampler = data_manager.resplit_data()
    test_set, test_loader, test_sampler = make_test_set(...)
    ipe = len(train_loader)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        mm_jepa=mm_jepa, wd=c2p_wd, final_wd=c2p_final_wd, start_lr=c2p_start_lr, ref_lr=c2p_lr,
        final_lr=c2p_final_lr, iterations_per_epoch=ipe, warmup=c2p_warmup, num_epochs=num_epochs,
        ipe_scale=ipe_scale, use_bfloat16=use_bfloat16
    )
    
    task_opt_params = {
        'path2clinical': {'start_lr': p2c_start_lr, 'ref_lr': p2c_lr, 'final_lr': p2c_final_lr, 'warmup': p2c_warmup, 'wd': p2c_wd, 'final_wd': p2c_final_wd},
        'clinical2path': {'start_lr': c2p_start_lr, 'ref_lr': c2p_lr, 'final_lr': c2p_final_lr, 'warmup': c2p_warmup, 'wd': c2p_wd, 'final_wd': c2p_final_wd}
    }
    
    early_stop = EarlyStopping(patience=patience, verbose=True, delta=0, maximize=True)
    prob_adjuster = init_ProbabilityAdjuster(base_prob, max_prob, min_prob, step)
    task_iterator = itertools.cycle(proxy_tasks) if iter_task else None

    def sync_task_selection(task_prob):
        weights = task_prob
        task_index = torch.tensor([0], dtype=torch.long, device=device)
        if global_rank == 0:
            current_task = random.choices(proxy_tasks, weights=weights, k=1)[0]
            task_index[0] = proxy_tasks.index(current_task)
        dist.broadcast(task_index, src=0)
        return proxy_tasks[task_index.item()]

    best_recall = 0.0
    log_freq = 10
    ks = [1, 5, 10]
    avg_p2c_recall = avg_c2p_recall = None

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        logger.info(f'Epoch {epoch + 1}')

        if prob_samp:
            p2c_prob, c2p_prob, decayed_step = prob_adjuster.update_probabilities_dec(
                recall_a=avg_p2c_recall, recall_b=avg_c2p_recall, current_epoch=epoch, total_epochs=total_step
            )
            logger.info(f'p2c_prob: {p2c_prob:.3f}, c2p_prob: {c2p_prob:.3f}, step: {decayed_step}')

        for itr, sample in enumerate(train_loader):
            # 获取病理与临床特征
            path_imgs = sample['pathology_features'].to(device, non_blocking=True)
            clinical_ids = sample['clinical_text']['input_ids'].to(device, non_blocking=True)
            clinical_mask = sample['clinical_text']['attention_mask'].to(device, non_blocking=True)

            def train_step():
                optimizer.zero_grad()
                task = sync_task_selection([p2c_prob, c2p_prob]) if prob_samp else next(task_iterator) if iter_task else proxy_tasks[0]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    # 【重要】确保 src.utils.optimizer 里支持你的新任务名称
                    # adjust_optimizer_params(optimizer, task, task_opt_params) 
                    loss, *_ = mm_jepa(clinical_ids, clinical_mask, path_imgs, task)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mm_jepa.parameters(), max_norm=1.0)
                    optimizer.step()

                return float(loss), scheduler.step(), wd_scheduler.step()

            loss, new_lr, new_wd, etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            csv_logger.log(epoch + 1, itr, loss, etime)
            if itr % log_freq == 0:
                logger.info(f'[{epoch + 1}, {itr}] Loss: {loss_meter.avg:.3f} | Mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB | Time: {time_meter.avg:.1f}ms')

        logger.info(f'Epoch {epoch + 1}, Avg Loss: {loss_meter.avg:.3f}')

        # ==== Evaluation 模块保持原本的调用风格，只需注意其内部变量逻辑 ====
        logger.info(f'Epoch {epoch + 1} validation')
        
        # ... 后续测试与验证代码与原版相同，在此省略以精简，
        # 因为 dual_embed 和 log_recall_rates 被封在 src.evaluation 里，你只要保证传参正确即可 ...
        
        # Save best model
        # ... (与原逻辑完全一致)