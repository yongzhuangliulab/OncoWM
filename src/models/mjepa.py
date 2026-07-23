# Copyright 2026 Geely AI Lab & Clinical AI Research
import copy
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

logger = logging.getLogger(__name__)

def deepsurv_cox_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    risk = risk.float()
    time = time.float()
    event = event.float()

    risk = torch.clamp(risk, min=-15.0, max=15.0) 
    risk = risk.view(-1)
    time = time.view(-1)
    event = event.view(-1)
    
    sort_idx = torch.argsort(time, descending=True)
    risk = risk[sort_idx]
    event = event[sort_idx]
    
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + eps)
    
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * event
    num_observed_events = torch.sum(event)
    
    if num_observed_events == 0:
        return risk.sum() * 0.0
    
    neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events
    return neg_likelihood

def pairwise_ranking_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    risk = risk.float().view(-1)
    time = time.float().view(-1)
    event = event.float().view(-1)
    time_diff = time.unsqueeze(1) - time.unsqueeze(0)  
    risk_diff = risk.unsqueeze(1) - risk.unsqueeze(0)  
    valid_pair_mask = (time_diff < 0) & (event.unsqueeze(1) == 1.0)
    if not valid_pair_mask.any(): return risk.sum() * 0.0
    loss_matrix = F.relu(margin - risk_diff)
    return loss_matrix[valid_pair_mask].mean()

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
            if hasattr(torch, 'trapezoid'): T_stat = torch.trapezoid(err, self.t, dim=-1) * N
            else: T_stat = torch.trapz(err, self.t, dim=-1) * N
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
            x = F.batch_norm(x, self.bn_input.running_mean, self.bn_input.running_var, 
                             self.bn_input.weight, self.bn_input.bias, training=False, momentum=0.0, eps=self.bn_input.eps)
        else: x = self.bn_input(x)
        x = self.raw_dropout(x) 
        return self.encoder(x)

class DeepSurvHead(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list = [128, 32], dropout_rate: float = 0.35, temperature: float = 2.0):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.num_heads = 4
        self.head_dim = in_features // self.num_heads
        
        self.q_proj = nn.Linear(in_features, in_features, bias=False)
        self.kv_proj = nn.Linear(in_features, in_features * 2, bias=False)
        self.attn_drop = nn.Dropout(p=dropout_rate)
        
        self.context_norm = nn.LayerNorm(in_features)
        
        self.film_gen = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SELU(),
            nn.Linear(in_features // 2, in_features * 2)
        )
        
        self.main_predictor = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SELU(),
            nn.AlphaDropout(p=dropout_rate),
            nn.Linear(hidden_dims[1], 1)
        )
        
        self.vip_t = nn.Linear(in_features, 1)
        self.vip_p = nn.Linear(in_features, 1)
        self.vip_g = nn.Linear(in_features, 1)
        self.vip_tr = nn.Linear(in_features, 1)
        
        self.router_v = nn.Linear(in_features, 1)
        
        self.vip_scale = nn.Parameter(torch.tensor(2.0))
        self.branch_drop_prob = 0.25

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
        nn.init.zeros_(self.main_predictor[-1].weight)
        nn.init.zeros_(self.vip_t.weight)
        nn.init.zeros_(self.vip_p.weight)
        nn.init.zeros_(self.vip_g.weight)
        nn.init.zeros_(self.vip_tr.weight)
        
        # Start FiLM as an identity modulation.
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)
        
        # Start the VIP residual gate nearly closed.
        nn.init.zeros_(self.router_v.weight)
        nn.init.constant_(self.router_v.bias, -2.0) 

    def forward(self, joint_emb, s_t, s_p, s_g, s_tr, u_t, u_p, u_g, u_tr, missing_mask):
        B, D = joint_emb.shape
        device = joint_emb.device
        
        mod_feats = torch.stack([s_t, s_p, s_g, s_tr], dim=1)
        q = self.q_proj(joint_emb).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(mod_feats).view(B, 4, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] 
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = missing_mask.unsqueeze(1).unsqueeze(2) 
        attn_scores = attn_scores.masked_fill(mask, float('-1e9'))
        
        temp = torch.clamp(self.temperature, min=0.5, max=5.0)
        valid_mask_attn = (~missing_mask).unsqueeze(1).unsqueeze(2).float() 
        
        attn_probs = torch.sigmoid(attn_scores / temp) * valid_mask_attn
        attn_probs = self.attn_drop(attn_probs)

        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, D)
        context = self.context_norm(context)
        
        cos_sim = F.cosine_similarity(joint_emb, context, dim=-1)
        ortho_loss = torch.abs(cos_sim).mean()
        
        film_params = self.film_gen(joint_emb)
        gamma = film_params[:, :D]
        beta = film_params[:, D:]
        modulated_context = context * (1.0 + gamma) + beta
        
        if self.training and self.branch_drop_prob > 0:
            drop_mask_c = (torch.rand(B, 1, device=device) > self.branch_drop_prob).float() / (1.0 - self.branch_drop_prob)
            fused_features = joint_emb + modulated_context * drop_mask_c
        else:
            fused_features = joint_emb + modulated_context
            
        main_risk = self.main_predictor(fused_features)
        
        valid_mask_vip = (~missing_mask).float() 
        r_t = self.vip_t(u_t) * valid_mask_vip[:, 0:1]
        r_p = self.vip_p(u_p) * valid_mask_vip[:, 1:2]
        r_g = self.vip_g(u_g) * valid_mask_vip[:, 2:3]
        r_tr = self.vip_tr(u_tr) * valid_mask_vip[:, 3:4]
        
        raw_vip_risk = r_t + r_p + r_g + r_tr
        
        scale_limit = torch.clamp(self.vip_scale, min=0.5, max=5.0)
        bounded_vip_risk = torch.tanh(raw_vip_risk) * scale_limit
        
        alpha_v = torch.sigmoid(self.router_v(joint_emb))
        
        if self.training and self.branch_drop_prob > 0:
            drop_mask_v = (torch.rand(B, 1, device=device) > self.branch_drop_prob).float() / (1.0 - self.branch_drop_prob)
            vip_in = alpha_v * bounded_vip_risk * drop_mask_v
        else:
            vip_in = alpha_v * bounded_vip_risk
            
        final_risk = main_risk + vip_in
        
        return final_risk, ortho_loss

class OncoJEPA(nn.Module):
    def __init__(self, text_encoder_name: str, path_dim: int, geno_raw_dim: int, geno_dim: int, tran_dim: int, 
                 hidden_size: int, dropout_rate: float):
        super().__init__()

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
        
        self.head_joint = DeepSurvHead(in_features=hidden_size, hidden_dims=[128, 32], dropout_rate=0.35, temperature=2.0)
        
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
                student_joint_embedding = self.norm_joint(s_fused[:, 0, :])

                loss_pred = F.mse_loss(student_joint_embedding, target_joint_embedding)
                loss_sigreg = self.le_sigreg(student_joint_embedding)

                lambd = 0.05
                total_loss = (1.0 - lambd) * loss_pred + lambd * loss_sigreg
                return total_loss

            if drop_prob > 0:
                s_text, s_path, s_geno, s_tran, drop_mask = self._modality_dropout(B, u_text, u_path, u_geno, u_tran, drop_prob)
            else:
                s_text, s_path, s_geno, s_tran = u_text, u_path, u_geno, u_tran
                drop_mask = torch.zeros(B, 4, dtype=torch.bool, device=device)

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

        if task == 'prognosis':
            joint_embedding = joint_embedding.detach()
            
            s_seq_l0_detached = s_seq_l0.detach()
            idx_base = 1 + self.num_fusion_tokens
            s_t  = s_seq_l0_detached[:, idx_base]      
            s_p  = s_seq_l0_detached[:, idx_base + 1]  
            s_g  = s_seq_l0_detached[:, idx_base + 2]  
            s_tr = s_seq_l0_detached[:, idx_base + 3]  
            
            u_t = u_text.detach()
            u_p = u_path.detach()
            u_g = u_geno.detach()
            u_tr = u_tran.detach()
            
            real_missing = torch.stack([text_missing, path_missing, geno_missing, tran_missing], dim=1)
            effective_mask = real_missing | drop_mask 
            
            final_risk, ortho_loss = self.head_joint(joint_embedding, s_t, s_p, s_g, s_tr, u_t, u_p, u_g, u_tr, effective_mask)
            final_risk = final_risk.squeeze(-1)

            if surv_time is not None and surv_event is not None:
                loss_cox = deepsurv_cox_loss(final_risk, surv_time, surv_event)
                loss_rank = pairwise_ranking_loss(final_risk, surv_time, surv_event, margin=0.5)
                total_loss = loss_cox + 0.05 * loss_rank + 0.1 * ortho_loss
                return total_loss, final_risk
                
            return final_risk

        elif task == 'cluster':
            return joint_embedding.detach()
            
        else:
            raise ValueError(f"不支持的 Task 模式: {task}")

def init_m3jepa(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, hidden_size, dropout_rate):
    return OncoJEPA(text_encoder_name, path_dim, geno_raw_dim, geno_dim, tran_dim, hidden_size, dropout_rate)
