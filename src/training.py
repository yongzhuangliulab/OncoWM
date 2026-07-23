import math
import logging
import torch

logger = logging.getLogger(__name__)

class CosineSchedule:
    """
    支持线性 Warmup 和 Cosine 退火的参数调度器 (按 iteration 更新)。
    可用于调整 learning rate (lr) 或 weight decay (weight_decay)。
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        param_name: str, 
        base_value: float, 
        final_value: float, 
        total_iters: int, 
        warmup_iters: int = 0, 
        start_warmup_value: float = 0.0
    ):
        self.optimizer = optimizer
        self.param_name = param_name
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value
        self.current_iter = 0

    def step(self):
        """
        更新 optimizer 中对应的 param，并返回当前值。
        """
        if self.current_iter < self.warmup_iters:
            alpha = self.current_iter / self.warmup_iters
            value = self.start_warmup_value + alpha * (self.base_value - self.start_warmup_value)
        else:
            alpha = (self.current_iter - self.warmup_iters) / max(1, (self.total_iters - self.warmup_iters))
            alpha = min(max(alpha, 0.0), 1.0)
            value = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + math.cos(math.pi * alpha))
        
        for group in self.optimizer.param_groups:
            if self.param_name == 'lr':
                scale = group.get('lr_scale', 1.0)
                group[self.param_name] = value * scale
                
            elif self.param_name == 'weight_decay':
                if group.get('weight_decay', 1.0) == 0.0:
                    continue
                group[self.param_name] = value
                
            else:
                group[self.param_name] = value
            
        self.current_iter += 1
        return value


def init_opt(
    params_to_optimize,
    wd: float, 
    final_wd: float, 
    start_lr: float, 
    ref_lr: float, 
    final_lr: float, 
    iterations_per_epoch: int, 
    warmup: float, 
    num_epochs: int, 
    ipe_scale: float = 1.0
):
    """
    初始化 AdamW 优化器以及学习率和权重衰减调度器。
    """
    logger.info("Initializing Optimizer and Schedulers...")

    optimizer = torch.optim.AdamW(params_to_optimize, lr=ref_lr, weight_decay=wd)
    total_iters = int(iterations_per_epoch * num_epochs * ipe_scale)
    warmup_iters = int(iterations_per_epoch * warmup * ipe_scale)

    scheduler = CosineSchedule(
        optimizer=optimizer, 
        param_name='lr',
        base_value=ref_lr, 
        final_value=final_lr, 
        total_iters=total_iters, 
        warmup_iters=warmup_iters, 
        start_warmup_value=start_lr
    )

    wd_scheduler = CosineSchedule(
        optimizer=optimizer, 
        param_name='weight_decay',
        base_value=wd, 
        final_value=final_wd, 
        total_iters=total_iters, 
        warmup_iters=0, 
        start_warmup_value=wd
    )

    return optimizer, scheduler, wd_scheduler
