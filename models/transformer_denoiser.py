"""
models/transformer_denoiser.py

核心去噪网络：Transformer-based denoiser（仿 Point·E + DiffuScene）。

设计要点：
  1. 完全移除序列位置编码，用「空间编码」（建筑中心坐标 MLP）代替，
     保证输入顺序的置换不变性。
  2. 每个 Transformer 块使用 AdaLN（Adaptive Layer Norm）注入时步条件，
     类似 DiT（Peebles & Xie 2022）。
  3. 文本条件和边界条件通过 Cross-Attention 层注入，各自独立一层，
     然后相加（也可改为拼接）。
  4. 多头输出：分别预测 bbox 噪声、类别噪声（类 DiffuScene 多预测头设计）。

输入（前向推理）：
  x_t:         (B, N, obj_dim)   带噪对象特征
  t:           (B,)              当前时步
  text_tokens: (B, L, d_model)   文本条件序列
  text_mask:   (B, L)            文本有效位掩码
  bnd_tokens:  (B, P, d_model)   边界条件序列

输出：
  eps_pred: (B, N, obj_dim)   预测的噪声（与输入同形）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 时步嵌入（正弦编码 + MLP）
# ─────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    """正弦余弦时步嵌入，输出维度 dim。"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  整数时步
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )                                           # (half,)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class TimeEmbedding(nn.Module):
    """正弦嵌入 + 2 层 MLP，输出 (B, out_dim)。"""

    def __init__(self, base_dim: int, out_dim: int):
        super().__init__()
        self.sin_emb = SinusoidalTimeEmb(base_dim)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin_emb(t))


# ─────────────────────────────────────────────
# 地块尺度嵌入（全局条件，与 t_emb 相加后一起喂给 AdaLN）
# ─────────────────────────────────────────────
 
class ScaleEmbedding(nn.Module):
    """
    地块尺度嵌入：将地块真实面积编码为与 TimeEmbedding 同维度的全局条件向量。
 
    设计动机：
      - 边界 token（BoundaryEncoder 输出）用的是相对质心坐标 + 角度，
        天然是尺度不变的「形状」描述；地块被放大放小多少倍，这套特征
        完全一样。真实 GIS 场景里，地块绝对大小直接决定"该放几栋建筑、
        多大密度"，这个信息必须单独补回来。
      - 用真实面积（而不是 [0,1] 归一化时用的缩放因子）作为输入，是因为
        面积是跟预处理管线无关的物理量，直接对应容量/密度这类真正驱动
        生成结果的因素；缩放因子则依赖具体的归一化实现（按最长边/对角线/
        各向同性与否），语义不稳定，换一种归一化方式其数值含义就变了。
      - scale 和 timestep 一样，都是「无差别地调制每一层、每个建筑 slot」
        的全局标量，不像 text/boundary 那样有值得被 cross-attention
        选择性路由的内部序列结构，所以走 AdaLN 通道，而不是拼进
        bnd_tokens 当作额外 token。
      - 面积量级跨度很大（几十~几万平方米），先取 log 再做 z-score
        标准化；均值/方差用 register_buffer 存储，随 checkpoint 一起
        保存/加载，保证训练和推理时刻的归一化行为完全一致（仿照
        GaussianDiffusion 里 betas 等 buffer 的做法）。
 
    Args:
        out_dim (int):      输出维度，需与 TimeEmbedding 的 out_dim 一致，
                             以便和 t_emb 直接相加。
        hidden_mult (int):  MLP 隐层宽度相对 out_dim 的倍数。
    """
 
    def __init__(self, out_dim: int, hidden_mult: int = 4):
        super().__init__()
        # 训练集统计量（log 面积的均值/方差），默认 0/1，
        # 训练前需调用 set_stats() 用真实训练集统计量覆盖。
        self.register_buffer("log_area_mean", torch.zeros(1))
        self.register_buffer("log_area_std", torch.ones(1))
 
        # 连续标量条件，不像 timestep 那样需要周期性的正弦位置编码，
        # 直接用小 MLP 升维即可。
        self.mlp = nn.Sequential(
            nn.Linear(1, out_dim * hidden_mult),
            nn.GELU(),
            nn.Linear(out_dim * hidden_mult, out_dim),
        )
 
    def set_stats(self, mean: float, std: float):
        """训练开始前调用一次，用训练集 log(area) 的均值/方差初始化归一化参数。"""
        self.log_area_mean.fill_(mean)
        self.log_area_std.fill_(max(std, 1e-6))
 
    def forward(self, area: torch.Tensor) -> torch.Tensor:
        """
        Args:
            area: (B,)  地块真实面积（物理单位，如平方米，未做任何归一化，
                        不要传入 [0,1] 缩放因子或归一化后的坐标面积）
        Returns:
            (B, out_dim)
        """
        log_area = torch.log(area.float().clamp(min=1e-6))          # (B,)
        z = (log_area - self.log_area_mean) / self.log_area_std     # (B,)
        return self.mlp(z.unsqueeze(-1))                            # (B, out_dim)


# ─────────────────────────────────────────────
# AdaLN（Adaptive Layer Norm）
# ─────────────────────────────────────────────

class AdaLN(nn.Module):
    """
    用时步嵌入预测 scale 和 shift，调制 LayerNorm 的输出。
    仿 DiT (Peebles & Xie 2022)。
    """

    def __init__(self, d_model: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # 预测 2*d_model 个参数：[scale, shift]
        self.proj = nn.Linear(time_emb_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, N, d_model)
            t_emb: (B, time_emb_dim)
        Returns:
            (B, N, d_model)
        """
        params = self.proj(t_emb)                        # (B, 2*d_model)
        scale, shift = params.chunk(2, dim=-1)           # (B, d_model) each
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


# ─────────────────────────────────────────────
# 空间编码（代替位置编码）
# ─────────────────────────────────────────────

# v1 只使用cx cy进行空间编码代替位置编码
class SpatialEncoding(nn.Module):
    """
    用建筑中心坐标 (cx, cy) 编码空间信息，注入输入特征。
    这是 BuildingBlock 的核心创新之一：
      - 不用序列位置编码（保证置换不变）
      - 用物理坐标编码空间关系
    """

    def __init__(self, out_dim: int):
        super().__init__()
        # 输入 2 维坐标（cx, cy），输出 out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, out_dim),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (B, N, 2)  中心坐标
        Returns:
            (B, N, out_dim)
        """
        return self.mlp(xy)

# v2 把"slot 身份"和"slot 内容"解耦成两个独立的加法项——一个是 learnable per-slot embedding（与内容无关，纯粹标识"这是第几个槛位"，保证 96 个槛位即使内容全是 0 也能互相区分），另一个是你现在的 content-based spatial encoding（基于 cx, cy，仅对有效坐标有意义）。两者相加后再喂给 transformer
# class SpatialEncoding(nn.Module):
#     """
#     双通道槛位编码：
#       1. learnable slot identity embedding —— 与内容无关，保证每个槛位
#          （即使是 empty、坐标退化为 0）都有独立可学习的身份信号。
#       2. content-based spatial encoding —— 基于 (cx, cy) 的物理坐标编码，
#          仅在 real slot 上承载有意义的空间信息。
#     """

#     def __init__(self, out_dim: int, max_slots: int = 96):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(2, out_dim // 2),
#             nn.GELU(),
#             nn.Linear(out_dim // 2, out_dim),
#         )
#         # 每个槛位一个独立的 learnable embedding，初始化要小一些，
#         # 避免一开始就主导 h 的尺度
#         self.slot_id_emb = nn.Parameter(torch.randn(max_slots, out_dim) * 0.02)

#     def forward(self, xy: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             xy: (B, N, 2)  中心坐标
#         Returns:
#             (B, N, out_dim)
#         """
#         B, N, _ = xy.shape
#         content = self.mlp(xy)                         # (B, N, out_dim)
#         slot_id = self.slot_id_emb[:N].unsqueeze(0)     # (1, N, out_dim)
#         return content + slot_id.expand(B, -1, -1)

# v3 concat content和learnable embedding后再proj 解耦 避免信息混叠
# class SpatialEncoding(nn.Module):
#     def __init__(self, out_dim: int, max_slots: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(2, out_dim // 2),
#             nn.GELU(),
#             nn.Linear(out_dim // 2, out_dim),
#         )
#         self.slot_id_emb = nn.Parameter(torch.randn(max_slots, out_dim) * 0.02)
#         self.merge = nn.Linear(out_dim * 2, out_dim)

#     def forward(self, xy: torch.Tensor) -> torch.Tensor:
#         B, N, _ = xy.shape
#         content = self.mlp(xy)
#         slot_id = self.slot_id_emb[:N].unsqueeze(0).expand(B, -1, -1)
#         return self.merge(torch.cat([content, slot_id], dim=-1))


# ─────────────────────────────────────────────
# 单个 Transformer 块（Self-Attn + Cross-Attn × 2 + FFN）
# ─────────────────────────────────────────────

# 辅助函数：执行自适应归一化的缩放与平移---定义在函数外面了
# 其余方法：1.加上静态方法装饰器，此时不需要写 self 参数# @staticmethod 2. 定义在函数内部，要写self参数 
# 两种调用时都要加类名.modulate()或者self.modulate()，不能直接调用 否则会报错
def modulate(x, shift, scale):
    # scale 和 shift 的形状为 (B, 1, D)
    return x * (1 + scale) + shift

class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer 块：
      1. Self-Attention（对象间关系）+ AdaLN
      2. Cross-Attention（文本条件）+ AdaLN
      3. Cross-Attention（边界条件）+ AdaLN
      4. FFN + AdaLN
    """

    def __init__(self, d_model: int, nhead: int,
                 dim_ffn: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        # — Self-Attention —
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.adaln_sa = AdaLN(d_model, time_emb_dim)
        # self.norm_sa = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # — Cross-Attention：文本 —
        self.cross_attn_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.adaln_ca_text = AdaLN(d_model, time_emb_dim)
        # self.norm_ca_text = nn.LayerNorm(d_model)

        # — Cross-Attention：边界 —
        self.cross_attn_bnd = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.adaln_ca_bnd = AdaLN(d_model, time_emb_dim)
        # self.norm_ca_bnd = nn.LayerNorm(d_model)

        # — FFN —
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )
        self.adaln_ffn = AdaLN(d_model, time_emb_dim)
        # self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(time_emb_dim, 6 * d_model)
        # )
        # # AdaLN-zero零初始化（必须将最后一层 Linear 的权重和偏置全部初始化为 0）
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor,
                t_emb: torch.Tensor,
                text_kv: torch.Tensor,
                text_mask: torch.Tensor,
                bnd_kv: torch.Tensor,
                obj_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:          (B, N, d_model)   当前对象特征
            t_emb:      (B, time_emb_dim) 时步嵌入
            text_kv:    (B, L, d_model)   文本 key/value
            text_mask:  (B, L)            文本有效位（1=有效，0=padding）
              → MultiheadAttention 的 key_padding_mask 需要 True=忽略，
                所以内部取反
            bnd_kv:     (B, P, d_model)   边界 key/value
            obj_mask:   (B, N)            对象有效位（None=全有效）
            scale: (B,) 地块真实面积（如平方米），None 时退化为不注入尺度条件

        Returns:
            (B, N, d_model)
        """
        B, N, D = x.shape

        # ── 1. Self-Attention ────────────────────────
        x_norm = self.adaln_sa(x, t_emb)
        key_pad = (~obj_mask.bool()) if obj_mask is not None else None
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm,
                                   key_padding_mask=key_pad)
        x = x + self.drop(sa_out)

        # ── 2. Cross-Attention：文本条件 ─────────────
        x_norm = self.adaln_ca_text(x, t_emb)
        text_key_pad = (text_mask == 0)              # True = 忽略
        ca_text_out, _ = self.cross_attn_text(
            x_norm, text_kv, text_kv,
            key_padding_mask=text_key_pad)
        x = x + self.drop(ca_text_out)

        # ── 3. Cross-Attention：边界条件 ─────────────
        x_norm = self.adaln_ca_bnd(x, t_emb)
        ca_bnd_out, _ = self.cross_attn_bnd(x_norm, bnd_kv, bnd_kv)
        x = x + self.drop(ca_bnd_out)

        # ── 4. FFN ───────────────────────────────────
        x_norm = self.adaln_ffn(x, t_emb)
        x = x + self.drop(self.ffn(x_norm))

        return x

    # # HetScene的实现版本---多个条件CA时不适用AdaLN，只在SA和FFN中使用AdaLN
    # def forward(self, x: torch.Tensor,
    #             t_emb: torch.Tensor,
    #             text_kv: torch.Tensor,
    #             text_mask: torch.Tensor,
    #             bnd_kv: torch.Tensor,
    #             obj_mask: torch.Tensor = None) -> torch.Tensor:
    #     B, N, D = x.shape

    #     # ── 0. 从 t_emb 中预测 6 个调制参数 ────────────────────────
    #     # 投影并切分为 6 组参数，形状均为 (B, D)
    #     shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
    #     # 将形状 unsqueeze 匹配为 (B, 1, D)，以便与特征 (B, N, D) 进行广播计算
    #     shift_sa = shift_sa.unsqueeze(1)
    #     scale_sa = scale_sa.unsqueeze(1)
    #     gate_sa = gate_sa.unsqueeze(1)
        
    #     shift_ffn = shift_ffn.unsqueeze(1)
    #     scale_ffn = scale_ffn.unsqueeze(1)
    #     gate_ffn = gate_ffn.unsqueeze(1)

    #     # ── 1. Self-Attention (包含 AdaLN-Zero) ────────────────────
    #     x_norm = modulate(self.norm_sa(x), shift_sa, scale_sa)
    #     key_pad = (~obj_mask.bool()) if obj_mask is not None else None
    #     sa_out, _ = self.self_attn(x_norm, x_norm, x_norm,
    #                             key_padding_mask=key_pad)
    #     # 使用门控参数 gate_sa 对残差路径进行缩放
    #     x = x + gate_sa * self.drop(sa_out)

    #     # ── 2. Cross-Attention：文本条件 (普通 Pre-LN，无 AdaLN) ──────
    #     x_norm = self.norm_ca_text(x)  # 使用普通的 LayerNorm
    #     text_key_pad = (text_mask == 0)              
    #     ca_text_out, _ = self.cross_attn_text(
    #         x_norm, text_kv, text_kv,
    #         key_padding_mask=text_key_pad)
    #     x = x + self.drop(ca_text_out)  # 普通残差连接，无门控

    #     # ── 3. Cross-Attention：边界条件 (普通 Pre-LN，无 AdaLN) ──────
    #     x_norm = self.norm_ca_bnd(x)   # 使用普通的 LayerNorm
    #     ca_bnd_out, _ = self.cross_attn_bnd(x_norm, bnd_kv, bnd_kv)
    #     x = x + self.drop(ca_bnd_out)  # 普通残差连接，无门控

    #     # ── 4. FFN (包含 AdaLN-Zero) ───────────────────────────────
    #     x_norm = modulate(self.norm_ffn(x), shift_ffn, scale_ffn)
    #     ffn_out = self.ffn(x_norm)
    #     # 使用门控参数 gate_ffn 对残差路径进行缩放
    #     x = x + gate_ffn * self.drop(ffn_out)

    #     return x


# ─────────────────────────────────────────────
# 主去噪器
# ─────────────────────────────────────────────

class TransformerDenoiser(nn.Module):
    """
    街区布局扩散去噪网络。

    多预测头设计（仿 DiffuScene）：
      - bbox_head:  预测位置/尺寸/角度噪声  (B, N, 6)
      - cls_head:   预测类别噪声             (B, N, num_classes)

    Args:
        obj_dim (int):       对象特征总维度 = 6 + num_classes
        num_classes (int):   建筑类型数
        d_model (int):       内部特征维度
        nhead (int):         注意力头数
        num_layers (int):    Transformer 块数量
        dim_ffn (int):       FFN 隐层维度
        time_emb_dim (int):  时步嵌入维度
        dropout (float):     Dropout 概率
    """

    def __init__(self, obj_dim: int, num_classes: int,
                 d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_ffn: int = 1024,
                 time_emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.obj_dim = obj_dim
        self.num_classes = num_classes

        # ── 时步嵌入 ──────────────────────────────────
        self.time_emb = TimeEmbedding(time_emb_dim, time_emb_dim)

        # ── 地块尺度嵌入（全局条件，与 t_emb 相加后一起喂给每层 AdaLN）──
        self.scale_emb = ScaleEmbedding(time_emb_dim)

        # ── 输入投影（分离 bbox 和 cls，对应多头思路）──
        self.bbox_in_proj = nn.Linear(obj_dim-num_classes, d_model // 2)
        self.cls_in_proj = nn.Linear(num_classes, d_model // 2)
        # 拼接后映射到 d_model
        self.in_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 空间编码（用 cx, cy 代替位置编码）─────────
        self.spatial_enc = SpatialEncoding(d_model)

        # ── Transformer 块堆叠 ─────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ffn,
                             time_emb_dim, dropout)
            for _ in range(num_layers)
        ])

        # ── 多预测头（分别预测 bbox 噪声 和 cls 噪声）──
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, obj_dim-num_classes),           # cx,cy,w,h,sin,cos
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """零初始化输出层（参考 DiT 惯例）"""
        for head in [self.bbox_head, self.cls_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x_t: torch.Tensor,
                t: torch.Tensor,
                text_tokens: torch.Tensor,
                text_mask: torch.Tensor,
                bnd_tokens: torch.Tensor,
                obj_mask: torch.Tensor = None,
                scale: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_t:         (B, N, obj_dim)   带噪对象特征
                           前 6 维：[cx,cy,w,h,sin,cos]
                           后 num_classes 维：类别 one-hot
            t:           (B,)              整数时步 [0, T)
            text_tokens: (B, L, d_model)   文本条件（已投影到 d_model）
            text_mask:   (B, L)            文本有效位
            bnd_tokens:  (B, P, d_model)   边界条件（已投影到 d_model）
            obj_mask:    (B, N)            对象有效位（1=有效建筑，0=padding）

        Returns:
            eps_pred: (B, N, obj_dim)   预测噪声
        """
        B, N, _ = x_t.shape

        # ── 分拆 bbox / cls ───────────────────────────
        bbox_feat = x_t[..., :7]                     # (B, N, 6)
        cls_feat = x_t[..., 7:]                      # (B, N, num_classes)

        # ── 输入投影 ──────────────────────────────────
        h = torch.cat([
            self.bbox_in_proj(bbox_feat),             # (B, N, d_model//2)
            self.cls_in_proj(cls_feat),               # (B, N, d_model//2)
        ], dim=-1)                                    # (B, N, d_model)
        h = self.in_proj(h)

        # ── 空间编码（用 cx,cy 注入）─────────────────
        xy = x_t[..., :2].detach()                   # (B, N, 2) cx,cy（不传梯度）
        h = h + self.spatial_enc(xy)

        # ── 时步嵌入 ──────────────────────────────────
        t_emb = self.time_emb(t)                      # (B, time_emb_dim)

        # ── 融合地块尺度条件（仿 DiT 的 timestep + label 相加方式）──
        # scale=None 时保持旧行为不变，仅在传入真实面积时生效。
        if scale is not None:
            t_emb = t_emb + self.scale_emb(scale)      # (B, time_emb_dim)

        # ── 堆叠 Transformer 块 ───────────────────────
        for layer in self.layers:
            h = layer(h, t_emb, text_tokens, text_mask,
                      bnd_tokens, obj_mask)

        # ── 多头输出 ──────────────────────────────────
        eps_bbox = self.bbox_head(h)                 # (B, N, 6)
        eps_cls = self.cls_head(h)                   # (B, N, num_classes)

        eps_pred = torch.cat([eps_bbox, eps_cls], dim=-1)  # (B, N, obj_dim)

        return eps_pred
