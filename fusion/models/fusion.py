"""Top-level task-conditioned hyperspectral fusion network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import trunc_normal_
except ImportError:  # Compatibility with older timm releases.
    from timm.models.layers import trunc_normal_

from .experts import MoE3
from .swin import BasicLayer_up, SUNet, UpSample

class HybridPromptAdapterV2(nn.Module):
    """
    稳定版 HybridPromptAdapter（修正并改良）
    - 自动适配输入通道 in_channels
    - Query = modality token, Key/Value = task bank
    - relation 矩阵对角线主导（已初始化）
    - FiLM final linear 小初始化、零偏置，训练初期近恒等
    """
    def __init__(self,
                 in_channels,
                 n_prompts=4,
                 prompt_dim=32,
                 num_heads=4,
                 attn_scale=0.5,
                 film_scale=0.1,
                 dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.n_prompts = n_prompts
        self.prompt_dim = prompt_dim

        # learnable task bank (keys/values)
        self.task_bank = nn.Parameter(torch.randn(n_prompts, prompt_dim))

        # relation matrix init: diag-dominant + small noise (normalized per-row)
        rel = torch.eye(n_prompts) + 0.01 * torch.randn(n_prompts, n_prompts)
        rel = rel / rel.sum(dim=-1, keepdim=True)
        self.relation = nn.Parameter(rel)

        # modality -> prompt space
        self.modality_proj = nn.Linear(in_channels, prompt_dim)
        self.modality_ln = nn.LayerNorm(prompt_dim)

        # attention (query=modality, key/value=task_bank)
        self.attn = nn.MultiheadAttention(embed_dim=prompt_dim, num_heads=num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.post_attn_ln = nn.LayerNorm(prompt_dim)
        self.attn_scale = nn.Parameter(torch.tensor(attn_scale), requires_grad=True)

        # combine attended prompt with task-specific vector
        self.combine_proj = nn.Linear(prompt_dim, prompt_dim)

        # FiLM generator -> produce 2*C params
        self.film_mlp = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, 2 * in_channels)
        )
        # small init for final linear and zero bias
        nn.init.xavier_uniform_(self.film_mlp[2].weight, gain=1e-2)
        nn.init.zeros_(self.film_mlp[2].bias)

        self.film_scale = film_scale

    def forward(self, x, prompt_type):
        """
        x: [B, L, C]
        prompt_type: [B] long tensor
        returns: modulated x same shape
        """
        B, L, C = x.shape

        # 1) modality token: avg pooled context -> project
        modality = x.mean(dim=1)                       # [B, C]
        modality_proj = self.modality_proj(modality)   # [B, D]
        modality_proj = self.modality_ln(modality_proj)

        # 2) prepare query and kv for attention
        q = modality_proj.unsqueeze(1)                 # [B, 1, D]
        kv = self.task_bank.unsqueeze(0).expand(B, -1, -1)  # [B, n_prompts, D]

        # 3) compute attention: modality queries the task bank
        attn_out, _ = self.attn(q, kv, kv)             # [B, 1, D]
        attn_out = attn_out.squeeze(1)                 # [B, D]
        attn_out = self.attn_dropout(attn_out)
        attn_out = self.post_attn_ln(attn_out)

        # 4) task-specific vector via relation matrix:
        # correct multiplication: rel_w [B, n] @ task_bank [n, D] -> [B, D]
        rel_w = F.softmax(self.relation[prompt_type], dim=-1)  # [B, n_prompts]
        task_spec = rel_w @ self.task_bank                       # [B, D]

        # 5) fused prompt: combine attended modality + scaled task_spec
        fused = self.combine_proj(attn_out) + self.attn_scale * task_spec  # [B, D]
        fused = F.relu(fused)

        # 6) FiLM params (bounded)
        params = self.film_mlp(fused)          # [B, 2*C]
        params = torch.tanh(params) * self.film_scale
        scale = params[:, :C].unsqueeze(1)     # [B,1,C]
        shift = params[:, C:].unsqueeze(1)     # [B,1,C]

        # 7) apply FiLM to sequence x
        out = x * (1.0 + scale) + shift
        return out


class DecoderPromptAdapter(nn.Module):
    def __init__(self, img_size=128, patch_size=4,
                 embed_dim=96, depths=[8, 8, 8],
                 num_heads=[8, 8, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, final_upsample="Dual up-sample",
                 n_prompts=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patches_resolution = patches_resolution = [img_size // patch_size, img_size // patch_size]

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.prompt_blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        for i_layer in range(self.num_layers):
            in_ch = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            out_ch = in_ch // 2 if i_layer > 0 else in_ch
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            self.concat_back_dim.append(concat_linear)

            if i_layer == 0:
                layer_up = UpSample(input_resolution=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                    in_channels=in_ch, scale_factor=2)
                self.prompt_blocks.append(None)  # 第一层不需要 prompt
            else:
                layer_up = BasicLayer_up(dim=in_ch,
                                         input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                           patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):
                                                      sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
                self.prompt_blocks.append(HybridPromptAdapterV2(out_ch * 2, n_prompts))  # concat后通道 *2

            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(self.embed_dim)
        if self.final_upsample == "Dual up-sample":
            self.up = UpSample(input_resolution=(img_size // patch_size, img_size // patch_size),
                               in_channels=embed_dim, scale_factor=4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_up_features(self, x, x_downsample, prompt_type):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip = x_downsample[2 - inx]
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=-1)  # concat channels
                x = self.concat_back_dim[inx](x)
                # apply PromptAdapter if exists
                if self.prompt_blocks[inx] is not None:
                    x = self.prompt_blocks[inx](x, prompt_type)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        if self.final_upsample == "Dual up-sample":
            x = self.up(x)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
        return x

    def forward(self, x, x_downsample, prompt_type):
        x = self.forward_up_features(x, x_downsample, prompt_type)
        x = self.up_x4(x)
        return x
class Re_HS(nn.Module):
    def __init__(self, in_ch,  out_ch):
        super(Re_HS, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.conv(x)
        return out
# --------- Full Dual-Branch Multi-Scale MoE Fusion ---------
class HSI_Fusion_DualBranch_MoE(nn.Module):
    def __init__(self, pan_ch=64, hsi_ch=102, feat_chs=[64,128,256], n_prompts=4):
        super().__init__()
        self.pan_embedding = nn.Sequential(nn.Conv2d(1, pan_ch, 1),
                                      nn.ReLU(inplace = True))
        self.RGB_embedding = nn.Sequential(nn.Conv2d(3, pan_ch, 1),
                                      nn.ReLU(inplace = True))
        self.MSI_embedding = nn.Sequential(nn.Conv2d(4, pan_ch, 1),
                                      nn.ReLU(inplace = True))
        self.miss_embedding = nn.Sequential(nn.Conv2d(1, pan_ch, 1),
                                      nn.ReLU()
        )
        self.pan_enc = SUNet(img_size=128, patch_size=4, in_chans=64, out_chans=3,
                         embed_dim=64, depths=[8, 8, 8],
                         num_heads=[4, 4, 4],
                         window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         use_checkpoint=False, final_upsample="Dual up-sample")

        self.hsi_enc = SUNet(img_size=128, patch_size=4, in_chans=102, out_chans=3,
                        embed_dim=64, depths=[8, 8, 8],
                        num_heads=[4, 4, 4],
                        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, final_upsample="Dual up-sample")

        # 三路专家融合模块
        self.moe_blocks = nn.ModuleList([
            MoE3(in_ch, in_ch) for in_ch in feat_chs
        ])
        # 解码器
        self.decoder = DecoderPromptAdapter(img_size=128, patch_size=4,
                           embed_dim=64, depths=[8, 8, 8],
                           num_heads=[4, 4, 4],
                           window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                           norm_layer=nn.LayerNorm, use_checkpoint=False, final_upsample="Dual up-sample", n_prompts=n_prompts)

        self.re = Re_HS(in_ch=64, out_ch=102)

    def softmax_entropy(self, w: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(w, min=1e-10, max=1.0)  # ⚠️ 关键保护
        return - (w * torch.log(w)).sum(dim=-1)

    def forward(self, pan, lr_hsi, prompt_type):
        outs = []
        for t in [0, 1, 2, 3]:
            mask = (prompt_type == t)
            if mask.sum() == 0:
                continue
            aux_sub, hsi_sub = pan[mask], lr_hsi[mask]
            if t == 0:   # PAN 模式
                f = self.pan_embedding(aux_sub[:, :1, ...])
            elif t == 1:        # RGB 模式
                f = self.RGB_embedding(aux_sub[:, :3, ...])
            elif t ==2:
                f = self.MSI_embedding(aux_sub)
            else:
                f = self.miss_embedding(aux_sub[:, :1, ...])
            out1, f_pan = self.pan_enc(f)
            out2, f_hsi = self.hsi_enc(hsi_sub)
            # MoE 融合
            fused_feats = []
            for i,(p,h) in enumerate(zip(f_pan,f_hsi)):
                moe_feat = self.moe_blocks[i](p,h, prompt_type[mask])   # MoE 融合
                #moe_feat = p + h
                fused_feats.append(moe_feat)
            # 解码
            out_sub = self.decoder(out1 + out2, fused_feats, prompt_type[mask])
            out_sub = self.re(out_sub) + hsi_sub
            outs.append((mask, out_sub))

        # 拼回 batch
        out = torch.zeros_like(lr_hsi, device=lr_hsi.device)
        for mask, out_sub in outs:
            out[mask] = out_sub
        return out


# ====== 训练函数 ======
