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
            # Linear warmup
            alpha = self.current_iter / self.warmup_iters
            value = self.start_warmup_value + alpha * (self.base_value - self.start_warmup_value)
        else:
            # Cosine decay
            alpha = (self.current_iter - self.warmup_iters) / max(1, (self.total_iters - self.warmup_iters))
            alpha = min(max(alpha, 0.0), 1.0) # 防止越界
            value = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + math.cos(math.pi * alpha))
        
        # ==================== 【修改点 1 开始】 ====================
        # 更新优化器参数，引入 lr_scale 支持差分学习率
        for group in self.optimizer.param_groups:
            if self.param_name == 'lr':
                # 如果更新的是 lr，乘上缩放系数
                scale = group.get('lr_scale', 1.0)
                group[self.param_name] = value * scale
                
            elif self.param_name == 'weight_decay':
                # 🚀【防背刺核心逻辑】：如果该组在初始化时 wd 被设为了 0.0（免衰减组），则跳过更新！
                if group.get('weight_decay', 1.0) == 0.0:
                    continue  # 绝对保护 Bias 和 LayerNorm
                group[self.param_name] = value
                
            else:
                group[self.param_name] = value
            
        self.current_iter += 1
        return value


def init_opt(
    params_to_optimize,  # <--- 这里改名
    wd: float, 
    final_wd: float, 
    start_lr: float, 
    ref_lr: float, 
    final_lr: float, 
    iterations_per_epoch: int, 
    warmup: float, 
    num_epochs: int, 
    ipe_scale: float = 1.0, 
    use_bfloat16: bool = True
):
    """
    初始化优化器 (AdamW)、混合精度 Scaler 和 Schedulers (LR & WD)。
    """
    logger.info("Initializing Optimizer and Schedulers...")
    
    # ==================== 【修改点 2 开始】 ====================
    # 兼容处理：判断传入的是模型，还是我们手动打包好的参数组字典
    if isinstance(params_to_optimize, torch.nn.Module):
        # 兼容原本的传参方式
        trainable_params = [p for p in params_to_optimize.parameters() if p.requires_grad]
    else:
        # 接收 Prognosis 阶段传入的 dict list（包含了 lr_scale）
        trainable_params = params_to_optimize
    # ==================== 【修改点 2 结束】 ====================
    
    optimizer = torch.optim.AdamW(trainable_params, lr=ref_lr, weight_decay=wd)
    
    # ... 下面的代码完全保持原样不需要动 ...
    scaler = torch.amp.GradScaler('cuda', enabled=use_bfloat16)

    # 3. 计算总迭代次数和 Warmup 迭代次数
    total_iters = int(iterations_per_epoch * num_epochs * ipe_scale)
    warmup_iters = int(iterations_per_epoch * warmup * ipe_scale)

    # 4. 学习率调度器 (带 Warmup)
    scheduler = CosineSchedule(
        optimizer=optimizer, 
        param_name='lr',
        base_value=ref_lr, 
        final_value=final_lr, 
        total_iters=total_iters, 
        warmup_iters=warmup_iters, 
        start_warmup_value=start_lr
    )

    # 5. 权重衰减调度器 (通常不需要 Warmup，直接做 Cosine Decay)
    wd_scheduler = CosineSchedule(
        optimizer=optimizer, 
        param_name='weight_decay',
        base_value=wd, 
        final_value=final_wd, 
        total_iters=total_iters, 
        warmup_iters=0, 
        start_warmup_value=wd
    )

    return optimizer, scaler, scheduler, wd_scheduler
