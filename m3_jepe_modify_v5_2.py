# Copyright 2026 Geely AI Lab & Clinical AI Research
import os
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

logger = logging.getLogger(__name__)

def deepsurv_cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # 🚀【绝对防御】：强制升维到 float32，彻底杜绝 Bfloat16 在 cumsum 时发生的“大数吃小数”灾难！
    risk = risk.float()
    time = time.float()
    event = event.float()

    risk = torch.clamp(risk, min=-15.0, max=15.0) 
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)
    
    # 降序排序
    sort_idx = torch.argsort(time, descending=True)
    risk = risk[sort_idx]
    event = event[sort_idx]
    
    hazard_ratio = torch.exp(risk)
    # 此时 hazard_ratio 也是 float32，cumsum 绝对精确安全！
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + eps)
    
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * event
    num_observed_events = torch.sum(event)
    
    if num_observed_events == 0:
        return risk.sum() * 0.0
    
    neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events
    return neg_likelihood

class LeJEPASIGReg(nn.Module):
    def __init__(self, num_slices=1024, num_knots=17):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(-5, 5, num_knots, dtype=torch.float32)
        self.register_buffer("t", t)
        self.register_buffer("exp_f", torch.exp(-0.5 * t.square()))

    def forward(self, x):
        N, D = x.shape
        with torch.amp.autocast('cuda', enabled=False):
            x_f = x.float()
            A = torch.randn(D, self.num_slices, device=x.device, dtype=torch.float32)
            A = F.normalize(A, p=2, dim=0)
            
            x_t = (x_f @ A).unsqueeze(-1) * self.t.view(1, 1, -1)  
            ecf_real = torch.cos(x_t).mean(dim=0)  
            ecf_imag = torch.sin(x_t).mean(dim=0)  
            
            err = ((ecf_real - self.exp_f).square() + ecf_imag.square()) * self.exp_f
            
            if hasattr(torch, 'trapezoid'):
                T_stat = torch.trapezoid(err, self.t, dim=-1) * N
            else:
                T_stat = torch.trapz(err, self.t, dim=-1) * N
                
        return T_stat.mean().to(x.dtype)

class GenoSNNEncoder(nn.Module):
    def __init__(self, input_dim=20000, hidden_dims=[2048, 1024], output_dim=512, dropout_rate=0.05):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.raw_dropout = nn.AlphaDropout(p=0.3)
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.SELU())
            layers.append(nn.AlphaDropout(p=dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.SELU())
        self.encoder = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear') 
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.size(0) == 1 and self.training:
            x = F.batch_norm(
                x, self.bn_input.running_mean, self.bn_input.running_var, 
                self.bn_input.weight, self.bn_input.bias, 
                training=False, momentum=0.0, eps=self.bn_input.eps
            )
        else:
            x = self.bn_input(x)
            
        x = self.raw_dropout(x) 
        return self.encoder(x)

# ==========================================
# 🚀 DeepSurv SNN 预后评估头 (V5.5 容量解放版)
# 拓宽至 [128, 32]，Dropout 降至 0.1
# 释放非线性表达力，彻底击碎欠拟合！
# ==========================================
class DeepSurvHead(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list = [128, 32], dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = in_features
        
        layers.append(nn.LayerNorm(curr_dim))
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.SELU()) 
            layers.append(nn.AlphaDropout(p=dropout_rate)) 
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

        # 🚀 强制 LeCun 初始化，点燃 SELU 的自归一化引擎
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

class M3JepaQuad(nn.Module):
    def __init__(self, text_encoder_name: str, path_dim: int, geno_raw_dim: int, geno_dim: int, tran_dim: int, 
                 num_experts: int, hidden_size: int, k_expert: int, dropout_rate: float):
        super().__init__()
        self.hidden_size = hidden_size

        self.text_model = self._load_text_model(text_encoder_name)
        text_dim = self.text_model.config.text_config.hidden_size if hasattr(self.text_model.config, 'text_config') else self.text_model.config.hidden_size
        
        self.geno_encoder = GenoSNNEncoder(input_dim=geno_raw_dim, output_dim=geno_dim)
        for param in self.geno_encoder.parameters():
            param.requires_grad = False

        self.path_bn = nn.BatchNorm1d(path_dim)
        self.tran_bn = nn.BatchNorm1d(tran_dim)

        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.path_proj = nn.Linear(path_dim, hidden_size)
        self.geno_proj = nn.Linear(geno_dim, hidden_size)
        self.tran_proj = nn.Linear(tran_dim, hidden_size)

        self.text_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.path_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.geno_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.tran_type_emb = nn.Parameter(torch.empty(1, hidden_size))

        self.text_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.path_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.geno_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.tran_mask_token = nn.Parameter(torch.empty(1, hidden_size))

        self.num_fusion_tokens = 8  
        self.fusion_tokens = nn.Parameter(torch.empty(1, self.num_fusion_tokens, hidden_size))
        self.global_cls_token = nn.Parameter(torch.empty(1, 1, hidden_size)) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4, 
            dropout=dropout_rate, batch_first=True, norm_first=True 
        )
        self.fusion_layer_0 = nn.TransformerEncoder(copy.deepcopy(encoder_layer), num_layers=2, enable_nested_tensor=False)
        self.fusion_layer_1 = nn.TransformerEncoder(copy.deepcopy(encoder_layer), num_layers=1, enable_nested_tensor=False)

        self.scale_text = nn.Parameter(torch.ones(1))
        self.scale_path = nn.Parameter(torch.ones(1))
        self.scale_geno = nn.Parameter(torch.ones(1))
        self.scale_tran = nn.Parameter(torch.ones(1))
        self.norm_joint = nn.LayerNorm(hidden_size)

        self.le_sigreg = LeJEPASIGReg(num_slices=1024)
        # 🚀 替换为精简版的 DeepSurv Head，隐藏层维度为 64，抗噪 0.3
        self.head_joint = DeepSurvHead(in_features=hidden_size, hidden_dims=[128, 32], dropout_rate=0.1)
        
        self._initialize_weights()

    def _safe_bn(self, bn_layer, x, missing_mask):
        out = x.clone()
        valid_mask = ~missing_mask
        if valid_mask.any():
            valid_x = x[valid_mask]
            if valid_x.size(0) == 1 and bn_layer.training:
                out[valid_mask] = F.batch_norm(
                    valid_x, bn_layer.running_mean, bn_layer.running_var, 
                    bn_layer.weight, bn_layer.bias, 
                    training=False, momentum=0.0, eps=bn_layer.eps
                ).to(out.dtype)
            else:
                out[valid_mask] = bn_layer(valid_x).to(out.dtype)
        return out

    def _initialize_weights(self) -> None:
        for name, m in self.named_modules():
            # 屏蔽对 DeepSurvHead 的默认初始化干扰
            if isinstance(m, nn.Linear) and "geno_encoder" not in name and "fusion_layer" not in name and "text_model" not in name and "head_joint" not in name:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm) and "head_joint" not in name:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        for emb in [self.text_type_emb, self.path_type_emb, self.geno_type_emb, self.tran_type_emb,
                    self.text_mask_token, self.path_mask_token, self.geno_mask_token, self.tran_mask_token,
                    self.fusion_tokens, self.global_cls_token]:
            nn.init.normal_(emb, mean=0.0, std=0.02)


    def _load_text_model(self, text_encoder_name: str) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
                text_encoder_name, 
                dtype=torch.bfloat16, 
                attn_implementation="sdpa"
        )
        model.config.use_cache = False
        if hasattr(model, 'lm_head'): model.lm_head = nn.Identity() 
        #if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
        #model.gradient_checkpointing_enable()
        
        text_layers = model.model.language_model.layers
        target_indices = list(range(len(text_layers) - 3, len(text_layers)))
        regex_target_modules = f".*language_model\\.layers\\.({'|'.join(map(str, target_indices))})\\.self_attn\\.(q_proj|v_proj)"
        
        lora_config = LoraConfig(
            peft_type='feature_extraction', r=4, lora_alpha=16, lora_dropout=0.1,
            bias='lora_only', init_lora_weights='pissa', target_modules=regex_target_modules
        )
        model = get_peft_model(model, lora_config)
        return model

    def _modality_dropout(self, B: int, text_f, path_f, geno_f, tran_f, drop_prob: float):
        device = text_f.device
        if drop_prob > 0:
            path_drop = min(drop_prob * 1.2, 0.5)  
            other_drop = drop_prob 
            probs = torch.tensor([other_drop, path_drop, other_drop, other_drop], device=device)
            rand_tensor = torch.rand(B, 4, device=device)
            drop_mask = rand_tensor < probs.unsqueeze(0)
        else:
            drop_mask = torch.zeros(B, 4, dtype=torch.bool, device=device)

        all_dropped = drop_mask.all(dim=1)
        if all_dropped.any():
            keep_idx = torch.randint(0, 4, (all_dropped.sum(),), device=device)
            drop_mask[all_dropped, keep_idx] = False

        t_fused = torch.where(drop_mask[:, 0:1], self.text_mask_token.expand(B, -1) + self.text_type_emb, text_f)
        p_fused = torch.where(drop_mask[:, 1:2], self.path_mask_token.expand(B, -1) + self.path_type_emb, path_f)
        g_fused = torch.where(drop_mask[:, 2:3], self.geno_mask_token.expand(B, -1) + self.geno_type_emb, geno_f)
        tr_fused = torch.where(drop_mask[:, 3:4], self.tran_mask_token.expand(B, -1) + self.tran_type_emb, tran_f)
        
        return t_fused, p_fused, g_fused, tr_fused, drop_mask

    def forward(self, 
                text_input_ids: torch.Tensor = None, text_attention_mask: torch.Tensor = None, text_token_type_ids: torch.Tensor = None,
                path_features: torch.Tensor = None, geno_raw: torch.Tensor = None, 
                tran_features: torch.Tensor = None,
                task: str = 'prognosis', surv_time: torch.Tensor = None, surv_event: torch.Tensor = None, 
                drop_prob: float = 0.0):
        
        device = next(self.parameters()).device
        
        # 🚀 斩断 OOM 的绝杀：
        # 如果是 prognosis 阶段，整个骨干网络严禁构建计算图，彻底断开反向传播树，显存消耗降至冰点！
        track_backbone_grad = (task == 'align') and torch.is_grad_enabled()
        with torch.set_grad_enabled(track_backbone_grad):
            B = None
            for feat in [path_features, text_input_ids, geno_raw, tran_features]:
                if feat is not None:
                    B = feat.shape[0]; break
            if B is None: B = 1

            path_missing = (path_features.abs().sum(dim=-1) < 1e-6) if path_features is not None else torch.ones(B, dtype=torch.bool, device=device)
            tran_missing = (tran_features.abs().sum(dim=-1) < 1e-6) if tran_features is not None else torch.ones(B, dtype=torch.bool, device=device)
            geno_missing = (geno_raw.abs().sum(dim=-1) < 1e-6) if geno_raw is not None else torch.ones(B, dtype=torch.bool, device=device)
            text_missing = (text_attention_mask.sum(dim=-1) == 0) if text_attention_mask is not None else torch.ones(B, dtype=torch.bool, device=device)

            u_path = self.path_mask_token.expand(B, -1) + self.path_type_emb
            if not path_missing.all():
                p_feat = self._safe_bn(self.path_bn, path_features, path_missing)
                u_path = torch.where(path_missing.unsqueeze(1), u_path, self.path_proj(p_feat) + self.path_type_emb)

            u_tran = self.tran_mask_token.expand(B, -1) + self.tran_type_emb
            if not tran_missing.all():
                tr_feat = self._safe_bn(self.tran_bn, tran_features, tran_missing)
                u_tran = torch.where(tran_missing.unsqueeze(1), u_tran, self.tran_proj(tr_feat) + self.tran_type_emb)

            u_geno = self.geno_mask_token.expand(B, -1) + self.geno_type_emb
            if not geno_missing.all():
                self.geno_encoder.eval() 
                with torch.no_grad():
                    g_feat = self.geno_encoder(geno_raw)
                u_geno = torch.where(geno_missing.unsqueeze(1), u_geno, self.geno_proj(g_feat) + self.geno_type_emb)

            u_text = self.text_mask_token.expand(B, -1) + self.text_type_emb
            if not text_missing.all():
                text_outputs = self.text_model(
                    input_ids=text_input_ids, attention_mask=text_attention_mask, 
                    token_type_ids=text_token_type_ids, output_hidden_states=False, use_cache=False             
                )
                last_hidden_state = text_outputs[0]
                
                sequence_lengths = text_attention_mask.sum(dim=1).long() - 1
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                batch_size_txt = last_hidden_state.shape[0]
                real_text_emb = last_hidden_state[torch.arange(batch_size_txt, device=device), sequence_lengths]
                
                u_text = torch.where(text_missing.unsqueeze(1), u_text, self.text_proj(real_text_emb) + self.text_type_emb)

            if task == 'align':
                t_seq = torch.cat([
                    self.global_cls_token.expand(B, -1, -1),
                    self.fusion_tokens.expand(B, -1, -1),
                    (u_text * self.scale_text).unsqueeze(1), 
                    (u_path * self.scale_path).unsqueeze(1), 
                    (u_geno * self.scale_geno).unsqueeze(1), 
                    (u_tran * self.scale_tran).unsqueeze(1)
                ], dim=1)
                
                t_seq_l0 = self.fusion_layer_0(t_seq)
                t_fused = self.fusion_layer_1(t_seq_l0[:, :1+self.num_fusion_tokens, :])
                
                target_joint_embedding = self.norm_joint(t_fused[:, 0, :])

                if drop_prob > 0:
                    s_text, s_path, s_geno, s_tran, drop_mask = self._modality_dropout(B, u_text, u_path, u_geno, u_tran, drop_prob)
                else:
                    s_text, s_path, s_geno, s_tran = u_text, u_path, u_geno, u_tran

                s_seq = torch.cat([
                    self.global_cls_token.expand(B, -1, -1),
                    self.fusion_tokens.expand(B, -1, -1),
                    (s_text * self.scale_text).unsqueeze(1), 
                    (s_path * self.scale_path).unsqueeze(1), 
                    (s_geno * self.scale_geno).unsqueeze(1), 
                    (s_tran * self.scale_tran).unsqueeze(1)
                ], dim=1)
                
                s_seq_l0 = self.fusion_layer_0(s_seq)
                s_fused = self.fusion_layer_1(s_seq_l0[:, :1+self.num_fusion_tokens, :])
                student_joint_embedding = self.norm_joint(s_fused[:, 0, :])

                loss_pred = F.mse_loss(student_joint_embedding, target_joint_embedding)
                loss_sigreg = self.le_sigreg(student_joint_embedding)

                lambd = 0.05
                total_loss = (1.0 - lambd) * loss_pred + lambd * loss_sigreg
                return total_loss

            # --------------- 下游提取 ---------------
            if drop_prob > 0:
                s_text, s_path, s_geno, s_tran, _ = self._modality_dropout(B, u_text, u_path, u_geno, u_tran, drop_prob)
            else:
                s_text, s_path, s_geno, s_tran = u_text, u_path, u_geno, u_tran

            s_seq = torch.cat([
                self.global_cls_token.expand(B, -1, -1),
                self.fusion_tokens.expand(B, -1, -1),
                (s_text * self.scale_text).unsqueeze(1), 
                (s_path * self.scale_path).unsqueeze(1), 
                (s_geno * self.scale_geno).unsqueeze(1), 
                (s_tran * self.scale_tran).unsqueeze(1)
            ], dim=1)
            
            s_seq_l0 = self.fusion_layer_0(s_seq)
            s_fused = self.fusion_layer_1(s_seq_l0[:, :1+self.num_fusion_tokens, :])
            joint_embedding = self.norm_joint(s_fused[:, 0, :])

        # =======================================================
        # 出了 with torch.set_grad_enabled(False) 之后：
        # 以下运算重新恢复梯度跟踪能力，且只有这里需要消耗极少量的显存
        # =======================================================
        if task == 'prognosis':
            # 🚀 阻断骨干网络反向传播（绝对冻结 JEPA 流形）
            joint_embedding = joint_embedding.detach()
            
            # 🚀 让提取出的完美特征，通过 DeepSurv SNN 网络发散最后的非线性潜能
            final_risk = self.head_joint(joint_embedding).squeeze(-1)

            if surv_time is not None and surv_event is not None:
                loss_cox = deepsurv_cox_loss(final_risk, surv_time, surv_event)
                return loss_cox, final_risk
                
            return final_risk

        elif task == 'cluster':
            return joint_embedding.detach()
            
        else:
            raise ValueError(f"不支持的 Task 模式: {task}")

def init_m3jepa(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, num_experts, hidden_size, k_expert, dropout_rate):
    return M3JepaQuad(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, num_experts, hidden_size, k_expert, dropout_rate)