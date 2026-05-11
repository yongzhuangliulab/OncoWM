# Copyright 2026 Geely AI Lab & Clinical AI Research
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import sys

sys.path.append('/home/Disk30T/chengpeng/Multimodal_JEPA/OS_JEPA')
from src.models.MMoE_v2 import MoE_predictor

logger = logging.getLogger(__name__)

# ==========================================
# [!!! 大道至简 !!!] 连续 Cox 比例风险损失 (Cox-PH Loss)
# 彻底抛弃 4 分类，直接对连续生存时间进行排序优化，天然契合 C-index
# ==========================================
def cox_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    # 强制统一为 float32
    risk_scores = risk_scores.float().view(-1)
    times = times.float().view(-1)
    events = events.float().view(-1)
    
    # 按照生存时间降序排列 (时间活得越久，排在越前面)
    sort_idx = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sort_idx]
    events = events[sort_idx]
    
    # 计算 Risk Set 的对数和 (logsumexp)
    # 对于任何时间 t_i，处于风险集中的人是所有存活时间 >= t_i 的人
    risk_set_log_sum = torch.logcumsumexp(risk_scores, dim=0)
    
    # Cox 负对数偏似然函数：让发生死亡事件的患者风险得分，在当时的所有存活者中尽可能大
    losses = - (risk_scores - risk_set_log_sum) * events
    
    valid_events = events.sum()
    if valid_events > 0:
        return losses.sum() / valid_events
    else:
        return risk_scores.sum() * 0.0

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class GenoSNNEncoder(nn.Module):
    def __init__(self, input_dim=20000, hidden_dims=[2048, 1024], output_dim=512, dropout_rate=0.05):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
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
        if x.size(0) > 1:
            x = self.bn_input(x)
        return self.encoder(x)

class M3JepaQuad(nn.Module):
    def __init__(self, text_encoder_name: str, path_dim: int, geno_raw_dim: int, geno_dim: int, tran_dim: int, 
                 num_experts: int, hidden_size: int, k_expert: int, dropout_rate: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigreg = SIGReg()

        self.text_model = self._load_text_model(text_encoder_name)
        if hasattr(self.text_model.config, 'text_config'):
            text_dim = self.text_model.config.text_config.hidden_size
        else:
            text_dim = self.text_model.config.hidden_size
        
        self.geno_encoder = GenoSNNEncoder(input_dim=geno_raw_dim, output_dim=geno_dim)

        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.path_proj = nn.Linear(path_dim, hidden_size)
        self.geno_proj = nn.Linear(geno_dim, hidden_size)
        self.tran_proj = nn.Linear(tran_dim, hidden_size)

        self.norm_t = nn.LayerNorm(hidden_size)
        self.norm_p = nn.LayerNorm(hidden_size)
        self.norm_g = nn.LayerNorm(hidden_size)
        self.norm_tr = nn.LayerNorm(hidden_size)

        self.text_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.path_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.geno_type_emb = nn.Parameter(torch.empty(1, hidden_size))
        self.tran_type_emb = nn.Parameter(torch.empty(1, hidden_size))

        self.text_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.path_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.geno_mask_token = nn.Parameter(torch.empty(1, hidden_size))
        self.tran_mask_token = nn.Parameter(torch.empty(1, hidden_size))

        self.predictor = MoE_predictor(num_experts, hidden_size, k_expert, dropout_rate)
        self.head_A_text = nn.Linear(hidden_size, text_dim)
        self.head_A_path = nn.Linear(hidden_size, path_dim)
        self.head_A_geno = nn.Linear(hidden_size, geno_dim)
        self.head_A_tran = nn.Linear(hidden_size, tran_dim)
        
        # [!!! 核心精简 !!!] 从 4 维分类头，变成 1 维连续风险打分头！
        self.head_B_prog = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and "geno_encoder" not in str(m):
                # 依然保持零初始化，继承 Align 阶段最纯粹的初始特征排序
                if m == self.head_B_prog:
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        for emb in [self.text_type_emb, self.path_type_emb, self.geno_type_emb, self.tran_type_emb,
                    self.text_mask_token, self.path_mask_token, self.geno_mask_token, self.tran_mask_token]:
            nn.init.kaiming_uniform_(emb, nonlinearity='relu')

    def _load_text_model(self, text_encoder_name: str) -> nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            text_encoder_name, dtype=torch.bfloat16, attn_implementation="sdpa" 
        )
        model.config.use_cache = False
        if hasattr(model, 'lm_head'): model.lm_head = nn.Identity() 
        if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        
        text_layers = model.model.language_model.layers
        target_indices = list(range(len(text_layers) - 3, len(text_layers)))
        regex_target_modules = f".*language_model\\.layers\\.({'|'.join(map(str, target_indices))})\\.self_attn\\.(q_proj|v_proj)"
        
        lora_config = LoraConfig(
            peft_type='feature_extraction', r=4, lora_alpha=16, lora_dropout=0.1,
            bias='lora_only', init_lora_weights='pissa', target_modules=regex_target_modules
        )
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model

    def _modality_dropout(self, B: int, text_f, path_f, geno_f, tran_f, drop_prob: float):
        device = text_f.device
        drop_mask = torch.rand(B, 4, device=device) < drop_prob
        all_dropped = drop_mask.all(dim=1)
        if all_dropped.any():
            keep_idx = torch.randint(0, 4, (all_dropped.sum(),), device=device)
            drop_mask[all_dropped, keep_idx] = False

        t_fused = torch.where(drop_mask[:, 0:1], self.text_mask_token.to(text_f.dtype).expand(B, -1), text_f)
        p_fused = torch.where(drop_mask[:, 1:2], self.path_mask_token.to(path_f.dtype).expand(B, -1), path_f)
        g_fused = torch.where(drop_mask[:, 2:3], self.geno_mask_token.to(geno_f.dtype).expand(B, -1), geno_f)
        tr_fused = torch.where(drop_mask[:, 3:4], self.tran_mask_token.to(tran_f.dtype).expand(B, -1), tran_f)
        return t_fused, p_fused, g_fused, tr_fused

    def forward(self, 
                text_input_ids: torch.Tensor = None, text_attention_mask: torch.Tensor = None, text_token_type_ids: torch.Tensor = None,
                path_features: torch.Tensor = None, geno_raw: torch.Tensor = None, 
                tran_features: torch.Tensor = None,
                task: str = 'prognosis', surv_time: torch.Tensor = None, surv_event: torch.Tensor = None, 
                drop_prob: float = 0.0):
        
        B = None
        for feat in [path_features, text_input_ids, geno_raw, tran_features]:
            if feat is not None:
                B = feat.shape[0]
                break

        if text_input_ids is not None and text_attention_mask is not None:
            text_outputs = self.text_model(
                input_ids=text_input_ids, attention_mask=text_attention_mask, 
                token_type_ids=text_token_type_ids, output_hidden_states=False, use_cache=False             
            )
            last_hidden_state = text_outputs[0]
            mask_expanded = text_attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            real_text_emb = sum_embeddings / sum_mask 
            t_proj = self.text_proj(real_text_emb)
            text_proj = t_proj + self.text_type_emb.to(t_proj.dtype)
        else:
            real_text_emb = None
            text_proj = self.text_mask_token.expand(B, -1)

        if geno_raw is not None:
            real_geno_emb = self.geno_encoder(geno_raw)
            g_proj = self.geno_proj(real_geno_emb)
            geno_proj = g_proj + self.geno_type_emb.to(g_proj.dtype)
        else:
            real_geno_emb = None
            geno_proj = self.geno_mask_token.expand(B, -1)

        if path_features is not None:
            p_proj = self.path_proj(path_features)
            path_proj = p_proj + self.path_type_emb.to(p_proj.dtype)
        else:
            path_proj = self.path_mask_token.expand(B, -1)
            
        if tran_features is not None:
            tr_proj = self.tran_proj(tran_features)
            tran_proj = tr_proj + self.tran_type_emb.to(tr_proj.dtype)
        else:
            tran_proj = self.tran_mask_token.expand(B, -1)

        if self.training and drop_prob > 0:
            t_fused, p_fused, g_fused, tr_fused = self._modality_dropout(B, text_proj, path_proj, geno_proj, tran_proj, drop_prob)
        else:
            t_fused, p_fused, g_fused, tr_fused = text_proj, path_proj, geno_proj, tran_proj

        fused_input = self.norm_t(t_fused) + self.norm_p(p_fused) + self.norm_g(g_fused) + self.norm_tr(tr_fused)
        E_align, E_prog = self.predictor(fused_input)

        if task == 'align':
            pred_text = self.head_A_text(E_align)
            pred_path = self.head_A_path(E_align)
            pred_geno = self.head_A_geno(E_align)
            pred_tran = self.head_A_tran(E_align)

            inv_loss = 0.0
            kl_loss = 0.0 
            valid_modalities = 0
            
            def calc_combined_alignment(pred, target):
                mse_l = F.mse_loss(pred, target)
                T = 2.0 
                kl_l = F.kl_div(F.log_softmax(pred / T, dim=-1), F.softmax(target / T, dim=-1), reduction='batchmean') * (T * T)
                return mse_l, kl_l

            if real_text_emb is not None: 
                m_l, k_l = calc_combined_alignment(pred_text, real_text_emb)
                inv_loss += m_l; kl_loss += k_l; valid_modalities += 1
            if path_features is not None: 
                m_l, k_l = calc_combined_alignment(pred_path, path_features)
                inv_loss += m_l; kl_loss += k_l; valid_modalities += 1
            if real_geno_emb is not None: 
                m_l, k_l = calc_combined_alignment(pred_geno, real_geno_emb)
                inv_loss += m_l; kl_loss += k_l; valid_modalities += 1
            if tran_features is not None: 
                m_l, k_l = calc_combined_alignment(pred_tran, tran_features)
                inv_loss += m_l; kl_loss += k_l; valid_modalities += 1
                
            if valid_modalities > 0: 
                inv_loss = inv_loss / valid_modalities
                kl_loss = kl_loss / valid_modalities
                
            sig_loss = self.sigreg(E_align)
            return sig_loss * 0.02 + inv_loss * 0.49 + kl_loss * 0.49

        elif task == 'prognosis':
            # [!!! 极简主义 !!!]
            # 直接输出一维风险得分，不再搞 sigmoid 也不搞 cumprod，彻底切断所有报错之源。
            risk_scores = self.head_B_prog(E_prog) 
            
            if surv_time is not None and surv_event is not None:
                # 传入最真实的连续时间和事件，计算 Cox 损失
                loss = cox_loss(risk_scores, surv_time, surv_event)
                
                # 极微弱的 L2 正则，防止权重飞到天上
                l2_reg = 1e-4 * torch.mean(risk_scores ** 2)
                
                # 这里返回 risk_scores.view(-1) 让它变成一维向量，C-index 更好算
                return loss + l2_reg, risk_scores.view(-1)
                
            return risk_scores.view(-1)
            
        else:
            raise ValueError(f"不支持的 Task: {task}")

def init_m3jepa(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, num_experts, hidden_size, k_expert, dropout_rate):
    return M3JepaQuad(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, num_experts, hidden_size, k_expert, dropout_rate)