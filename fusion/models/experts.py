"""Dual-domain and mixture-of-experts building blocks.

The class bodies in this module are migrated from the original experiment
script without changing their parameter hierarchy, preserving checkpoint
compatibility.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(ch // reduction,1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch // reduction,1), ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.equal_in_out = (in_ch == out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.act = nn.ReLU(inplace=True)
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        if use_se:
            self.se = SEBlock(out_ch)
    def forward(self, x):
        identity = x if self.equal_in_out else self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.use_se:
            out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out

class SpatialFreqBranch(nn.Module):
    """
    空间频域分支：对每个波段做 2D FFT，基于幅度学习可训练 real mask，再 IFFT 回来。
    输入: X (B, C, H, W)
    输出: X_spatial_enh (B, C, H, W) (real)
    """
    def __init__(self, in_C, out_C, mid_ch=32):
        super().__init__()
        # 处理频域幅度的网络（实值操作）
        # 我们先对 log(1+|F|) 做一个小的 conv 网络（在空间频域上）
        # 注意：频域的 HxW 大小等于输入 HxW（保持 same）
        self.freq_net = ResidualBlock(in_C, in_C, use_se= True)
        self.conv = nn.Conv2d(in_C, out_C, 1, 1)

    def forward(self, X):
        # X: (B, C, H, W)
        # 1) FFT per-band: use torch.fft.fft2 along last two dims
        # result: complex tensor (B, C, H, W, complex)
        F_comp = torch.fft.fft2(X, dim=(-2, -1))  # complex tensor (B,C,H,W)
        # 2) magnitude and phase
        mag = torch.abs(F_comp)        # (B,C,H,W) real >=0

        # 3) process magnitude in real domain: log compress -> NN -> scale in (0,1)
        mag_log = torch.log1p(mag)     # numerical stability
        # Frequency network expects channels = C, so input shape (B, C, H, W)
        scale = self.freq_net(mag_log) # (B,C,H,W) in (0,1)

        # 4) apply scaling to complex spectrum (scale real amplitude)
        # reconstruct complex spectrum: new_F = scale * F_comp
        # scale is real; multiply complex by real scalar -> complex
        new_F = F_comp * scale

        # 5) inverse fft
        x_spatial = torch.fft.ifft2(new_F, dim=(-2, -1)).real  # keep real part
        x_spatial = self.conv(x_spatial)
        return x_spatial


# class LightSpatialTransformer(nn.Module):
#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)
#         self.norm = nn.LayerNorm(dim)
#         self.attn_dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # B, HW, C
#         x = self.norm(x)
#         qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
#         attn = attn.softmax(dim=-1)
#         out = attn @ v
#         out = self.proj(out)
#         out = out.transpose(1, 2).reshape(B, C, H, W)
#         return out
#
# class FrequencyEnhance(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.enhance_conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, in_channels, 1)
#         )
#
#     def forward(self, x):
#         fft = torch.fft.fft2(x)
#         fft_shift = torch.fft.fftshift(fft)
#         mag = torch.abs(fft_shift)
#         enhanced = self.enhance_conv(mag)
#         enhanced = torch.fft.ifftshift(enhanced)
#         out = torch.fft.ifft2(enhanced).real
#         return out
#
# class SpatialExpert(nn.Module):
#     def __init__(self, in_channels, embed_dim):
#         super().__init__()
#         self.spatial_branch = LightSpatialTransformer(embed_dim)
#         self.freq_branch = FrequencyEnhance(in_channels)
#         self.fuse = nn.Conv2d(in_channels * 2, in_channels, 1)
#
#     def forward(self, x):
#         freq_feat = self.freq_branch(x)
#         spatial_feat = self.spatial_branch(x)
#         fused = torch.cat([freq_feat, spatial_feat], dim=1)
#         out = self.fuse(fused)
#         return out

class DualDomainAdaptiveSpatialExpert(nn.Module):
    """
    Dual-domain adaptive spatial expert with
    frequency-aware structural prior and task-aware modulation
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        n_prompts=4,
        prompt_dim=32,
        num_heads=4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # === (1) Spatial branch: lightweight Transformer ===
        self.spatial_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True
        )
        self.spatial_out = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # === (2) Frequency branch: magnitude + phase (stable) ===
        self.freq_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

        # === (3) Cross-domain channel-wise gate ===
        self.cross_gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # === (4) Task-adaptive modulation (FiLM-style) ===
        self.task_embed = nn.Embedding(n_prompts, prompt_dim)
        self.task_mlp = nn.Sequential(
            nn.Linear(prompt_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        # === (5) Output fusion ===
        self.fuse = nn.Conv2d(embed_dim * 2, in_channels, kernel_size=1)

    def forward(self, x, prompt_type):
        B, C, H, W = x.shape

        # ==================================================
        # (1) Spatial branch
        # ==================================================
        spatial_feat = self.spatial_proj(x)  # B, D, H, W
        x_flat = spatial_feat.flatten(2).transpose(1, 2)  # B, HW, D
        x_flat = self.spatial_norm(x_flat)

        spatial_out, _ = self.spatial_attn(x_flat, x_flat, x_flat)
        spatial_feat = spatial_out.transpose(1, 2).reshape(
            B, self.embed_dim, H, W
        )
        spatial_feat = self.spatial_out(spatial_feat)

        # ==================================================
        # (2) Frequency branch (STABLE version)
        # ==================================================
        # project before FFT (very important)
        freq_in = self.spatial_proj(x)  # reuse projection, B, D, H, W

        fft = torch.fft.fft2(freq_in, norm="ortho")
        fft = torch.fft.fftshift(fft)

        mag = torch.log1p(torch.abs(fft))  # stable
        real = torch.real(fft)
        imag = torch.imag(fft)

        # use cos/sin-like representation instead of angle
        phase_cos = real / (torch.abs(fft) + 1e-6)
        phase_sin = imag / (torch.abs(fft) + 1e-6)

        freq_input = torch.cat([mag, phase_cos, phase_sin], dim=1)  # B, 3D, H, W

        freq_feat = self.freq_conv(
            F.adaptive_avg_pool2d(freq_input, (H, W))
        )  # B, D, H, W

        freq_stat = freq_feat.mean(dim=[2, 3], keepdim=True)

        # ==================================================
        # (3) Cross-domain gate (residual form)
        # ==================================================
        gate_input = torch.cat(
            [
                spatial_feat.mean(dim=[2, 3], keepdim=True),
                freq_stat
            ],
            dim=1
        )

        gate = self.cross_gate(gate_input)  # [0,1]
        gate = 0.5 + 0.5 * gate  # avoid hard saturation

        spatial_mod = spatial_feat * gate + spatial_feat
        freq_mod = freq_feat * (1.0 - gate) + freq_feat

        # ==================================================
        # (4) Task-adaptive modulation (FiLM, centered)
        # ==================================================
        task_vec = self.task_embed(prompt_type)
        task_gate = self.task_mlp(task_vec).view(B, -1, 1, 1)

        task_gate = 0.5 + task_gate  # center at 1

        spatial_mod = spatial_mod * task_gate
        freq_mod = freq_mod * (2.0 - task_gate)

        # ==================================================
        # (5) Fusion
        # ==================================================
        fused = torch.cat([spatial_mod, freq_mod], dim=1)
        out = self.fuse(fused)

        return out

class SpectralFreqBranchMultiScale(nn.Module):
    """
    Stable multi-scale spectral frequency expert
    (Soft MoE-style spectral expert)
    """

    def __init__(
        self,
        in_C,
        out_C,
        n_prompts=4,
        prompt_dim=32,
        hidden=128,
        n_scales=3,  # low / mid / high
    ):
        super().__init__()
        self.in_C = in_C
        self.out_C = out_C
        self.n_scales = n_scales

        # ---- spectral projection before FFT (VERY IMPORTANT) ----
        self.spec_proj = nn.Conv2d(in_C, in_C, kernel_size=1)

        # ---- multi-scale spectral experts (channel-wise) ----
        self.scale_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_C, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, in_C),
                nn.Sigmoid()
            ) for _ in range(n_scales)
        ])

        # ---- task-aware modulation ----
        self.task_embed = nn.Embedding(n_prompts, prompt_dim)
        self.task_mlp = nn.Sequential(
            nn.Linear(prompt_dim, in_C),
            nn.Sigmoid()
        )

        # ---- output projection ----
        self.conv_out = nn.Conv2d(in_C, out_C, kernel_size=1)

    def forward(self, X, prompt_type):
        """
        X: [B, C, H, W]
        prompt_type: [B]
        """
        B, C, H, W = X.shape

        # =====================================================
        # (1) Spectral projection + FFT (along spectral dim)
        # =====================================================
        Xp = self.spec_proj(X)                # stabilize input
        Xp = Xp.permute(0, 2, 3, 1)           # B, H, W, C

        F_spec = torch.fft.fft(Xp, dim=-1, norm="ortho")
        mag = torch.log1p(torch.abs(F_spec))  # stable magnitude

        # =====================================================
        # (2) Global spectral statistics (NOT pixel-wise)
        # =====================================================
        # [B, H, W, C] → [B, C]
        mag_stat = mag.mean(dim=(1, 2))

        # =====================================================
        # (3) Frequency masks (true multi-scale)
        # =====================================================
        freqs = torch.fft.fftfreq(C, device=X.device).abs()  # [C]

        masks = [
            (freqs <= 0.15),                               # low
            (freqs > 0.15) & (freqs <= 0.35),              # mid
            (freqs > 0.35),                                # high
        ]

        scale_weights = []

        for i, mask in enumerate(masks):
            if mask.sum() == 0:
                scale_weights.append(
                    torch.ones_like(mag_stat)
                )
                continue

            masked_feat = mag_stat * mask.unsqueeze(0)
            w = self.scale_mlps[i](masked_feat)
            scale_weights.append(w)

        # soft mixture of experts
        scale = sum(scale_weights) / self.n_scales
        scale = scale.view(B, 1, 1, C)

        # =====================================================
        # (4) Task-aware modulation (residual, centered)
        # =====================================================
        task_vec = self.task_embed(prompt_type)    # B, P
        task_gate = self.task_mlp(task_vec)        # B, C
        task_gate = 0.5 + task_gate                # center at 1
        task_gate = task_gate.view(B, 1, 1, C)

        scale = scale * task_gate

        # =====================================================
        # (5) Spectral modulation + inverse FFT
        # =====================================================
        F_spec_mod = F_spec * scale
        X_spec = torch.fft.ifft(F_spec_mod, dim=-1, norm="ortho").real
        X_spec = X_spec.permute(0, 3, 1, 2)  # B, C, H, W

        # residual output
        out = self.conv_out(X_spec + X)

        return out

# class SpectralFreqBranch(nn.Module):
#     """
#     光谱频域分支：对每个像素做 1D FFT (over C), 在频谱上学习可训练掩码，再 IFFT。
#     输入: X (B, C, H, W)
#     输出: X_spectral_enh (B, C, H, W)
#     """
#     def __init__(self, in_C, out_C, hidden=128):
#         super().__init__()
#         self.C = in_C
#         # We will operate on spectral-frequency magnitude per-pixel (per spatial location).
#         # To keep implementation efficient, we map per-pixel spectral mag (length C) through a small MLP.
#         self.mlp = nn.Sequential(
#             nn.Linear(in_C, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, in_C),
#             nn.Sigmoid()   # scaling per spectral-frequency component in [0,1]
#         )
#         self.conv = nn.Conv2d(in_C, out_C, 1, 1)
#     def forward(self, X):
#         # X: (B,C,H,W)
#         B, C, H, W = X.shape
#         # 1) compute FFT along spectral axis (C), so permute to (B, H, W, C)
#         X_perm = X.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
#         # complex spectral transform (1D fft along last dim)
#         F_spec = torch.fft.fft(X_perm, dim=-1)  # (B,H,W,C) complex
#         mag = torch.abs(F_spec)                 # (B,H,W,C) real
#         # 2) process magnitude per-pixel: flatten spatial dims
#         mag_flat = mag.view(B * H * W, C)       # (B*H*W, C)
#         scale_flat = self.mlp(mag_flat)         # (B*H*W, C)
#         scale = scale_flat.view(B, H, W, C)     # (B,H,W,C)
#         # 3) apply scaling to complex spectrum
#         new_F_spec = F_spec * scale             # broadcasting real scale to complex
#         # 4) inverse fft along spectral axis and permute back
#         x_spec = torch.fft.ifft(new_F_spec, dim=-1).real  # (B,H,W,C)
#         x_spec = x_spec.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
#         x_spec = self.conv(x_spec)
#         return x_spec


class GateNet(nn.Module):
    """
    根据全局特征（例如 PAN/RGB 的 pooled 特征或 X 的统计）产生融合权重 alpha,beta
    返回 [alpha, beta] per-sample in [0,1] (sigmoid)
    """
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
            nn.Sigmoid()
        )

    def forward(self, feat):
        # feat: (B, in_dim)
        return self.net(feat)  # (B,2) in (0,1)


class SpatioSpectralFreqEnhancer(nn.Module):
    """
    主模块：合并空间频域与光谱频域增强，并用 gate 控制融合
    输入:
      X: (B, C, H, W) 融合特征（HS + auxiliary 已合并）
      aux: (B, A, H, W) optional，用于门控（例如 PAN 或 RGB）
    输出:
      X_out: (B, C, H, W)
    """
    def __init__(self, in_C, out_C, use_aux=True, aux_ch=1, hidden_gate=64, mid_ch=32):
        super().__init__()
        self.spatial_branch = DualDomainAdaptiveSpatialExpert(in_C, in_C)
        self.spectral_branch = SpectralFreqBranchMultiScale(in_C, in_C)
        self.use_aux = use_aux
        # gate input dim: global pooled X + pooled aux (if provided)
        gate_in = in_C  # we'll use global pooled X per-channel mean as feature
        if use_aux:
            gate_in += aux_ch
        self.gate = GateNet(in_dim=gate_in, hidden=hidden_gate)

        # final fusion projection
        self.fusion = nn.Conv2d(in_C * 3, in_C, kernel_size=1)  # combine original + spatial + spectral
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_C, out_C, kernel_size=1)

    def forward(self, X, prompt_type, aux=None):
        """
        X: (B, C, H, W)
        aux: (B, A, H, W) or None
        """
        B, C, H, W = X.shape

        # compute pooled features for gate (global stat)
        x_pool = X.mean(dim=[2, 3])        # (B, C)
        if self.use_aux and (aux is not None):
            aux_pool = aux.mean(dim=[2, 3])  # (B, A)
            gate_feat = torch.cat([x_pool, aux_pool], dim=1)
            # gate_in expected size = C? we use a compact vector: [mean(X per-sample), mean(aux per-sample)]
            # GateNet defined with in_dim = 2 (if aux_ch=1), but flexible above.
        else:
            gate_feat = x_pool  # (B,1)

        gate_values = self.gate(gate_feat)  # (B,2), values in (0,1)
        alpha = gate_values[:, 0].view(B, 1, 1, 1)
        beta  = gate_values[:, 1].view(B, 1, 1, 1)

        # branches
        x_sp = self.spatial_branch(X, prompt_type)       # (B,C,H,W)
        x_spc = self.spectral_branch(X, prompt_type)     # (B,C,H,W)

        # fuse: weighted sum + original residual, then small conv
        fused = torch.cat([X, alpha * x_sp, beta * x_spc], dim=1)  # (B, 3C, H, W)
        out = self.fusion(fused)
        out = self.act(out + X)  # residual + activation
        out = self.conv(out)
        return out


def topk_routing(logits, k):
    """
    logits: [B, N]
    return: masked weights [B, N]
    """
    topk_val, topk_idx = torch.topk(logits, k, dim=1)

    mask = torch.zeros_like(logits)
    mask.scatter_(1, topk_idx, 1.0)

    weights = F.softmax(logits, dim=1) * mask
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    return weights


# --------- MoE with 3 expert types ---------
class MoE3(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        n_space=4,
        n_spec=4,
        n_sps=4,
        n_prompts=4,
        prompt_dim=32
    ):
        super().__init__()

        # ========= experts =========
        self.space_experts = nn.ModuleList([
            DualDomainAdaptiveSpatialExpert(in_ch, out_ch)
            for _ in range(n_space)
        ])

        self.spec_experts = nn.ModuleList([
            SpectralFreqBranchMultiScale(in_ch, out_ch)
            for _ in range(n_spec)
        ])

        self.sps_experts = nn.ModuleList([
            SpatioSpectralFreqEnhancer(in_ch * 2, out_ch, use_aux=False)
            for _ in range(n_sps)
        ])

        # ========= prompt =========
        self.prompt_embed = nn.Embedding(n_prompts, prompt_dim)

        # ========= prompt-aware routing =========
        self.gate_space = nn.Linear(in_ch + prompt_dim, n_space)
        self.gate_spec  = nn.Linear(in_ch + prompt_dim, n_spec)
        self.gate_sps   = nn.Linear(in_ch * 2 + prompt_dim, n_sps)

        self.fuse = nn.Conv2d(in_ch * 3, out_ch, 3, padding=1)

    def forward(self, pan_feat, hsi_feat, prompt_type):
        """
        pan_feat, hsi_feat: [B, L, C]
        """
        B, L, C = pan_feat.shape
        H = W = int(math.sqrt(L))

        pan = pan_feat.permute(0, 2, 1).view(B, C, H, W)
        hsi = hsi_feat.permute(0, 2, 1).view(B, C, H, W)

        prompt = self.prompt_embed(prompt_type)  # [B, P]

        # ======================================================
        # 1) Prompt-routed spatial experts
        # ======================================================
        spa_stat = F.adaptive_avg_pool2d(pan, 1).view(B, C)
        gate_in_space = torch.cat([spa_stat, prompt], dim=1)
        logits = self.gate_space(gate_in_space)
        w_space = topk_routing(logits, k=2)

        out_space = sum(
            w_space[:, i].view(B, 1, 1, 1) *
            self.space_experts[i](pan, prompt_type)
            for i in range(len(self.space_experts))
        )

        # ======================================================
        # 2) Prompt-routed spectral experts
        # ======================================================
        spec_stat = F.adaptive_avg_pool2d(hsi, 1).view(B, C)
        gate_in_spec = torch.cat([spec_stat, prompt], dim=1)

        logits = self.gate_spec(gate_in_spec)
        w_spec = topk_routing(logits, k=2)

        out_spec = sum(
            w_spec[:, i].view(B, 1, 1, 1) *
            self.spec_experts[i](hsi, prompt_type)
            for i in range(len(self.spec_experts))
        )

        # ======================================================
        # 3) Prompt-routed spatio-spectral experts
        # ======================================================
        fused = torch.cat([pan, hsi], dim=1)
        sps_stat = F.adaptive_avg_pool2d(fused, 1).view(B, fused.shape[1])
        gate_in_sps = torch.cat([sps_stat, prompt], dim=1)

        logits = self.gate_sps(gate_in_sps)
        w_spec = topk_routing(logits, k=2)

        out_sps = sum(
            w_spec[:, i].view(B, 1, 1, 1) *
            self.sps_experts[i](fused, prompt_type)
            for i in range(len(self.sps_experts))
        )

        # ======================================================
        # 4) Fuse
        # ======================================================
        out = self.fuse(torch.cat([out_space, out_spec, out_sps], dim=1))
        out = (out + pan + hsi).permute(0, 2, 3, 1).view(B, L, C)

        return out
