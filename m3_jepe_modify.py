# Copyright 2025 Geely AI Lab & Clinical AI Research
# 
# 修改说明：
# 1. 适配 UNI2-h (timm) 作为病理图像编码器
# 2. 适配 MedGemma-1.5 (Gemma 3) 作为临床文本编码器
# 3. 修复了嵌套多模态模型中 LoRA 无法定位文本层的问题
# 4. 修复了嵌套模型不返回 hidden_states 的问题

import sys
import logging
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import timm  # 加载 UNI2-h
from peft import get_peft_model, LoraConfig

from src.models import model_paths
# 假设你的 MMoE 和 MLP 构造函数定义在以下路径
from src.models.MMoE import MoE_predictor as create_mmoe_predictor
from src.loss_function import loss_fn_1, symmetric_contrastive_loss

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

class M3Jepa(nn.Module):
    """Pathology-Clinical Joint Embedding Predictive Architecture.
    Aligns WSI Patches (UNI2-h) with Clinical Outcomes (MedGemma).
    """
    def __init__(
        self,
        image_encoder_name: str,
        text_encoder_name: str,
        num_experts: int,
        hidden_size: int,
        k_expert: int,
        dropout_rate: float,
        predictor_type: str
    ):
        super().__init__()
        # 任务定义：临床对齐病理(c2p)，病理对齐临床(p2c)
        self.task_map = {
            'clinical2path': self._clinical2path_forward, 
            'path2clinical': self._path2clinical_forward
        }

        # 1. 加载模型（统一使用 bfloat16）
        self.image_model = self._load_image_model(image_encoder_name)
        self.text_model = self._load_text_model(text_encoder_name)

        # 2. 获取维度 (timm 使用 embed_dim, Gemma 使用 config.hidden_size)
        image_dim = self.image_model.embed_dim 
        text_dim = self.text_model.config.hidden_size
        logger.info(f"Pathology Dim (UNI2-h): {image_dim}, Clinical Dim (MedGemma): {text_dim}")

        # 3. 模态特定嵌入 (Modality Tokens)
        self.path_to_clinical_embedding = nn.Parameter(torch.empty(1, 1, image_dim))
        self.clinical_to_path_embedding = nn.Parameter(torch.empty(1, 1, text_dim))

        # 4. 预测器初始化
        if predictor_type == 'mlp':
            # 这里假设你有一个 Creat_MLP_predictor 的类定义
            from src.models.MLP import Creat_MLP_predictor
            self.predictor = Creat_MLP_predictor(
                hidden_sizes=[hidden_size, hidden_size],
                image_dim=image_dim,
                text_dim=text_dim,
                drop=dropout_rate
            )
        elif predictor_type == 'mmoe_mlp':
            self.predictor = create_mmoe_predictor(
                num_experts=num_experts,
                image_dim=image_dim,
                text_dim=text_dim,
                hidden_size=hidden_size,
                k=k_expert,
                drop=dropout_rate
            )

        # 5. 对比学习投影层
        self.cl_image_emb = nn.Linear(image_dim, hidden_size, bias=True)
        self.cl_text_emb = nn.Linear(text_dim, hidden_size, bias=True)

        # 权重设置
        self.l2_weight = 1.0
        self.cl_weight = 1.0
        self.task_weights = {'clinical2path': 0.5, 'path2clinical': 0.5}

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_uniform_(self.path_to_clinical_embedding, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.clinical_to_path_embedding, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.cl_image_emb.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.cl_text_emb.weight, nonlinearity='relu')

    def _load_image_model(self, image_encoder_name: str) -> nn.Module:
        """加载 UNI2-h (timm) 并注入 LoRA"""
        model = timm.create_model(
            "hf_hub:MahmoodLab/UNI2-h",
            pretrained=True,
            dynamic_img_size=True
        ).to(torch.bfloat16)
        
        # 应用 LoRA 到最后 3 个 block
        model = self._apply_lora_to_vm_layers(model, num_layers=3)
        
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        return model

    def _load_text_model(self, text_encoder_name: str) -> nn.Module:
        """加载 MedGemma-1.5 并使用正则注入 LoRA"""
        model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16
        )
        
        # 应用 LoRA 到文本部分的最后 3 层
        model = self._apply_lora_to_lm_layers(model, num_layers=3)
        
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        return model

    def _apply_lora_to_lm_layers(self, model: nn.Module, num_layers: int = 3) -> nn.Module:
        """使用正则表达式精准定位 Gemma 3 的文本层"""
        # 获取实际的文本解码层
        text_layers = model.model.language_model.layers
        total_layers = len(text_layers)
        target_indices = list(range(total_layers - num_layers, total_layers))
        
        indices_str = "|".join(map(str, target_indices))
        # 精准正则：锁定 language_model 路径下的特定层
        regex_target_modules = f".*language_model\\.layers\\.({indices_str})\\.self_attn\\.(q_proj|v_proj)"
        
        logger.info(f"Applying Regex LoRA to MedGemma layers: {target_indices}")
        
        lora_config = LoraConfig(
            peft_type='feature_extraction',
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='lora_only',
            init_lora_weights='pissa',
            target_modules=regex_target_modules
        )
        return get_peft_model(model, lora_config)

    def _apply_lora_to_vm_layers(self, model: nn.Module, num_layers: int = 3) -> nn.Module:
        """适配 timm ViT 结构的 LoRA 注入"""
        total_layers = len(model.blocks)
        selected_layers = model.blocks[-num_layers:]
        
        logger.info(f"Applying LoRA to {num_layers} UNI2-h blocks")
        
        lora_config = LoraConfig(
            peft_type='feature_extraction',
            r=4,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='lora_only',
            init_lora_weights='pissa',
            target_modules=["qkv"] # timm 默认命名
        )

        for layer in selected_layers:
            layer.attn = get_peft_model(layer.attn, lora_config)
        return model

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_pixel_values: torch.Tensor,
        task: str
    ):
        if task not in self.task_map:
            raise ValueError(f"Task {task} not supported.")

        # 1. 临床文本编码 (必须显式要求 output_hidden_states)
        text_outputs = self.text_model(
            input_ids=text_input_ids, 
            attention_mask=text_attention_mask,
            output_hidden_states=True
        )
        # 获取最后两层的平均值
        text_embedding = torch.mean(torch.stack(text_outputs.hidden_states[-2:]), dim=0)

        # 2. 病理图像编码 (使用 timm 专用的中间层提取方法)
        # n=2 代表提取最后两层
        img_hidden_states = self.image_model.get_intermediate_layers(image_pixel_values, n=2)
        image_embedding = torch.mean(torch.stack(img_hidden_states), dim=0)

        return self.task_map[task](text_embedding, image_embedding)

    def _clinical2path_forward(self, text_emb, img_emb):
        """临床结局 -> 病理切片"""
        # 加入 Task Token
        t2i_token = self.clinical_to_path_embedding.expand(text_emb.shape[0], text_emb.shape[1], -1)
        text_emb = text_emb + t2i_token

        pred_img, cl_text = self.predictor(text_emb, 'clinical2path')
        
        # 聚合特征 (Batch, Dim)
        pred_img = pred_img.mean(dim=1)
        target_img = img_emb.mean(dim=1)
        
        cl_text = cl_text.mean(dim=1)
        cl_img = self.cl_image_emb(img_emb).mean(dim=1)

        loss = self._compute_loss(pred_img, target_img, cl_text, cl_img)
        return self.task_weights['clinical2path'] * loss, pred_img, target_img, cl_text, cl_img

    def _path2clinical_forward(self, text_emb, img_emb):
        """病理切片 -> 临床结局"""
        i2t_token = self.path_to_clinical_embedding.expand(img_emb.shape[0], img_emb.shape[1], -1)
        img_emb = img_emb + i2t_token

        pred_text, cl_img = self.predictor(img_emb, 'path2clinical')
        
        pred_text = pred_text.mean(dim=1)
        target_text = text_emb.mean(dim=1)
        
        cl_img = cl_img.mean(dim=1)
        cl_text = self.cl_text_emb(text_emb).mean(dim=1)

        loss = self._compute_loss(pred_text, target_text, cl_img, cl_text)
        return self.task_weights['path2clinical'] * loss, pred_text, target_text, cl_img, cl_text

    def _compute_loss(self, pred, target, cl_s, cl_t):
        l2 = loss_fn_1(pred, target)
        cl = symmetric_contrastive_loss(cl_s, cl_t)
        return self.l2_weight * l2 + self.cl_weight * cl

# 保持 init_m3jepa 函数名不变，方便外部调用
def init_m3jepa(image_encoder_name, text_encoder_name, num_experts, hidden_size, k_expert, dropout_rate, predictor_type):
    return M3Jepa(image_encoder_name, text_encoder_name, num_experts, hidden_size, k_expert, dropout_rate, predictor_type)