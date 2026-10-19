"""
models/unet1d_denoiser.py

DiffuScene 风格的去噪网络（baseline，用于与 transformer_denoiser.py 做对比实验）。

────────────────────────────────────────────────────────────────
与 transformer_denoiser.py（现有的 Transformer+AdaLN 方案，来自 BuildingBlock
论文对 DiffuScene 的改造）的关键设计差异，均依据 DiffuScene 原论文
(Tang et al., CVPR 2024) Sec 3.1 的描述还原：

  1. 骨干网络：
     DiffuScene 原文明确写的是 "1D convolution (≈ 逐点共享 MLP) with skip
     connections, interleaved with attention blocks"，即 MLP-UNet + 穿插的
     self-/cross-attention，而不是纯 Transformer。论文自己的消融实验
     (Table 3, C1 vs C5) 显示：换成 DALLE·2 风格的纯 Transformer 后 FID 从
     17.21 恶化到 29.08 —— 所以这里刻意保留 UNet（编码器/解码器 + skip
     connection）结构，而不是简单堆叠 Transformer block。

  2. 时序条件注入：
     DiffuScene 原文没有提到 AdaLN（那是 DiT 论文提出的、被 BuildingBlock
     引入的改动）。这里改用更朴素的加性/FiLM 方式，在每个 ResBlock 内部把
     time embedding 投影后直接加到特征上。

  3. "位置编码"：
     DiffuScene 原文："as MLPs are shared with each object, it would be
     difficult for the denoising network to distinguish different objects
     without position information. To this end, we also use the positional
     embeddings of instance IDs and inject them into the network."
     即：只用一个和 slot 序号绑定的、与内容无关的可学习 embedding，
     不像 transformer_denoiser.py 的 SpatialEncoding 那样用 (cx, cy) 的
     内容坐标去做 MLP 编码。这里对应 InstanceIDEmbedding。

  4. 文本 / 边界条件注入：
     两者都通过 Cross-Attention，这一点与 transformer_denoiser.py 一致，
     予以保留（边界条件是你在原论文基础上的任务特定扩展，不属于
     DiffuScene 原文，但两个 baseline 都需要它，所以复用同一套接口）。

  5. 扩散过程 / 训练损失：
     不在这个文件里改动。直接复用 diffusion.py 的 GaussianDiffusion——
     它的 linear beta schedule (1e-4→0.02, T=1000)、eps 预测、以及
     compute_loss 里 "bbox/cls 分离 MSE + 按 alpha_bar 加权的 IoU 惩罚"
     的结构，本来就是对 DiffuScene 论文 Eq.(9)(10) 的忠实实现，
     两个 denoiser 可以共用同一个 GaussianDiffusion / FlowMatching 实例。

接口刻意与 TransformerDenoiser 保持一致（同样的构造参数、同样的
forward(x_t, t, text_tokens, text_mask, bnd_tokens, obj_mask) 签名），
可以直接替换使用，方便做受控对比实验。

  6. 训练稳定性（v2 补丁）：
     这个 UNet 有 num_levels*2+1 个 UNetLevel、每个 UNetLevel 又有 4 个
     子模块（ResBlock/SelfAttn/CrossAttn×2），num_levels=3 时相当于 28 层
     纯加性残差堆叠在一起。实测（见对话）在训练初期会出现 loss 先降后
     骤增、随后分类分支坍缩为全 "empty" 的发散现象。为此给每个残差分支
     加了零初始化的标量门控（ReZero, Bachlechner et al. 2020）：
     `x + gate * branch(x)`，gate 初始化为 0，训练开始时网络接近恒等
     映射，随训练自己学习每个子层要注入多少信息，避免深层残差网络
     早期激活值爆炸。
────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用现有实现的时步嵌入模块，避免重复造轮子
from .transformer_denoiser import TimeEmbedding


# ─────────────────────────────────────────────
# Instance-ID 位置嵌入（替代 SpatialEncoding）
# ─────────────────────────────────────────────

class InstanceIDEmbedding(nn.Module):
    """
    与内容无关的、按 slot 序号索引的可学习位置嵌入。

    对应 DiffuScene 原文："positional embeddings of instance IDs"。
    与 transformer_denoiser.py 的 SpatialEncoding 不同：这里完全不使用
    (cx, cy) 坐标信息，只是给每个 slot 一个身份标识，帮助共享权重的
    MLP/Attention 区分不同物体。
    """

    def __init__(self, dim: int, max_slots: int = 96):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(max_slots, dim) * 0.02)

    def forward(self, B: int, N: int, device) -> torch.Tensor:
        return self.emb[:N].unsqueeze(0).expand(B, -1, -1).to(device)


# ─────────────────────────────────────────────
# ResBlock1D：逐点共享 MLP（等价 kernel=1 的 1D 卷积）+ 时间 FiLM
# ─────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    DiffuScene UNet 的基本单元。对每个 slot 独立作用（共享权重），
    时间嵌入通过朴素加性方式注入（非 AdaLN），带残差连接。

    稳定性说明：这里额外加了一个零初始化的标量门控 res_gate（ReZero
    技巧，Bachlechner et al. 2020），让残差分支在训练初期贡献接近 0，
    网络起始状态接近恒等映射，随训练逐步"学会"要注入多少信息。
    这个 UNet 有 7 个 UNetLevel × 4 个子模块 = 28 层纯加性残差堆叠，
    没有这个保护很容易在训练初期就激活值爆炸、loss 骤增。
    """

    def __init__(self, dim_in: int, dim_out: int,
                 time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_in)
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.time_proj = nn.Linear(time_emb_dim, dim_out)
        self.norm2 = nn.LayerNorm(dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.res_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, N, dim_in)
            t_emb: (B, time_emb_dim)
        Returns:
            (B, N, dim_out)
        """
        h = F.silu(self.norm1(x))
        h = self.fc1(h)
        h = h + self.time_proj(t_emb).unsqueeze(1)   # 加性时间条件，广播到所有 slot
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.fc2(h)
        return self.skip(x) + self.res_gate * h


# ─────────────────────────────────────────────
# Attention 穿插模块（self / cross）
# ─────────────────────────────────────────────

class SelfAttnBlock1D(nn.Module):
    """物体间 self-attention，pre-norm + 残差，用于捕获全局场景上下文。"""

    def __init__(self, dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, obj_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm(x)
        key_pad = (~obj_mask.bool()) if obj_mask is not None else None
        out, _ = self.attn(h, h, h, key_padding_mask=key_pad)
        return x + self.gate * self.drop(out)


class CrossAttnBlock1D(nn.Module):
    """
    对条件序列（文本 / 边界）做 cross-attention，pre-norm + 残差。

    UNet 各层级的通道宽度会变化（dim_out），但条件序列（text_tokens /
    bnd_tokens）的维度固定为 cond_dim（= d_model），因此用 kdim/vdim
    让 MultiheadAttention 内部自动把 K/V 投影到 query 所在的维度。
    """

    def __init__(self, dim: int, cond_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True,
            kdim=cond_dim, vdim=cond_dim)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, kv, kv, key_padding_mask=key_padding_mask)
        return x + self.gate * self.drop(out)


class UNetLevel(nn.Module):
    """UNet 的一个层级：ResBlock → SelfAttn → CrossAttn(text) → CrossAttn(bnd)。"""

    def __init__(self, dim_in: int, dim_out: int, cond_dim: int, nhead: int,
                 time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.res = ResBlock1D(dim_in, dim_out, time_emb_dim, dropout)
        self.self_attn = SelfAttnBlock1D(dim_out, nhead, dropout)
        self.cross_text = CrossAttnBlock1D(dim_out, cond_dim, nhead, dropout)
        self.cross_bnd = CrossAttnBlock1D(dim_out, cond_dim, nhead, dropout)

    def forward(self, x, t_emb, text_kv, text_key_pad, bnd_kv, obj_mask=None):
        x = self.res(x, t_emb)
        x = self.self_attn(x, obj_mask)
        x = self.cross_text(x, text_kv, key_padding_mask=text_key_pad)
        x = self.cross_bnd(x, bnd_kv)
        return x


# ─────────────────────────────────────────────
# 主去噪器：MLP 编码 → UNet1D(+Attn) → MLP 解码
# ─────────────────────────────────────────────

class DiffuSceneDenoiser(nn.Module):
    """
    DiffuScene 风格的去噪网络（1D-UNet + Attention 穿插）。

    与 TransformerDenoiser 接口完全一致，可直接替换使用。

    Args:
        obj_dim, num_classes, d_model, nhead, dim_ffn, time_emb_dim, dropout:
            含义与 TransformerDenoiser 相同。
        num_levels (int): UNet 编码器/解码器的层级数（不做序列维度下采样，
            只在通道维度上收窄/放宽，对应共享 MLP + skip connection 的设计）。
        max_slots (int): instance-ID 位置嵌入的最大槽位数。
    """

    def __init__(self, obj_dim: int, num_classes: int,
                 d_model: int = 256, nhead: int = 8,
                 num_levels: int = 3, dim_ffn: int = 1024,
                 time_emb_dim: int = 256, dropout: float = 0.1,
                 max_slots: int = 96):
        super().__init__()

        self.obj_dim = obj_dim
        self.num_classes = num_classes
        self.bbox_dim = obj_dim - num_classes

        # ── 时步嵌入（复用 transformer_denoiser 的实现）──────────
        self.time_emb = TimeEmbedding(time_emb_dim, time_emb_dim)

        # ── 输入 MLP 编码：bbox / cls 分离（消融 C2 vs C5 显示更优）──
        self.bbox_in_proj = nn.Linear(self.bbox_dim, d_model // 2)
        self.cls_in_proj = nn.Linear(num_classes, d_model // 2)
        self.in_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # ── instance-ID 位置嵌入（替代坐标内容编码）───────────────
        self.instance_id_emb = InstanceIDEmbedding(d_model, max_slots)

        # ── UNet 编码器（通道逐级放宽）─────────────────────────────
        # 注意：cond_dim 固定为 d_model —— text_tokens / bnd_tokens 的维度
        # 不随 UNet 层级的通道宽度变化，CrossAttnBlock1D 内部会自动投影。
        dims = [d_model * (2 ** i) for i in range(num_levels)]   # e.g. [d, 2d, 4d]
        self.enc_levels = nn.ModuleList()
        in_dim = d_model
        for out_dim in dims:
            self.enc_levels.append(
                UNetLevel(in_dim, out_dim, d_model, nhead, time_emb_dim, dropout))
            in_dim = out_dim

        # ── Bottleneck ──────────────────────────────────────────────
        self.bottleneck = UNetLevel(in_dim, in_dim, d_model, nhead, time_emb_dim, dropout)

        # ── UNet 解码器（通道逐级收窄，带 skip connection concat）──
        self.dec_levels = nn.ModuleList()
        rev_dims = list(reversed(dims))
        cur_dim = in_dim
        for skip_dim in rev_dims:
            # 输入 = 当前特征 + 对应编码器层级的 skip（concat）
            self.dec_levels.append(
                UNetLevel(cur_dim + skip_dim, skip_dim, d_model, nhead, time_emb_dim, dropout)
            )
            cur_dim = skip_dim

        # ── 输出 MLP 解码：多头分别预测 bbox / cls 噪声 ─────────────
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.bbox_dim),
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for head in [self.bbox_head, self.cls_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x_t: torch.Tensor,
                t: torch.Tensor,
                text_tokens: torch.Tensor,
                text_mask: torch.Tensor,
                bnd_tokens: torch.Tensor,
                obj_mask: torch.Tensor = None) -> torch.Tensor:
        """
        接口与 TransformerDenoiser.forward 完全一致。

        Args:
            x_t:         (B, N, obj_dim)
            t:           (B,)
            text_tokens: (B, L, d_model)
            text_mask:   (B, L)
            bnd_tokens:  (B, P, d_model)
            obj_mask:    (B, N) 或 None

        Returns:
            eps_pred: (B, N, obj_dim)
        """
        B, N, _ = x_t.shape

        bbox_feat = x_t[..., :self.bbox_dim]
        cls_feat = x_t[..., self.bbox_dim:]

        h = torch.cat([
            self.bbox_in_proj(bbox_feat),
            self.cls_in_proj(cls_feat),
        ], dim=-1)
        h = self.in_proj(h)

        # instance-ID 位置嵌入（与坐标内容无关）
        h = h + self.instance_id_emb(B, N, x_t.device)

        t_emb = self.time_emb(t)
        text_key_pad = (text_mask == 0)

        # ── 编码器：保存每层输出用于 skip connection ────────────
        skips = []
        for level in self.enc_levels:
            h = level(h, t_emb, text_tokens, text_key_pad, bnd_tokens, obj_mask)
            skips.append(h)

        # ── Bottleneck ──────────────────────────────────────────
        h = self.bottleneck(h, t_emb, text_tokens, text_key_pad, bnd_tokens, obj_mask)

        # ── 解码器：与编码器对应层级做 skip concat ────────────────
        for level, skip in zip(self.dec_levels, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = level(h, t_emb, text_tokens, text_key_pad, bnd_tokens, obj_mask)

        eps_bbox = self.bbox_head(h)
        eps_cls = self.cls_head(h)

        return torch.cat([eps_bbox, eps_cls], dim=-1)