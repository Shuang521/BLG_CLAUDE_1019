"""
models/boundary_encoder.py

将街区边界多边形编码为固定维度的条件向量。

输入：(B, P, 2)  P 个顶点的归一化坐标
输出：(B, S, boundary_out)  S 个 token 的序列嵌入（供 Cross-Attention 消费）
      或 (B, boundary_out)  全局池化版本

设计思路：
  1. 将多边形顶点视为无序点云（类似 PointNet）
  2. 用共享 MLP 提取每个顶点的局部特征
  3. 拼接全局 max-pool 特征（获取全局边界信息）
  4. 保留逐点序列，供 cross-attention 利用局部边界信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointNetLayer(nn.Module):
    """共享权重的 MLP，作用于每个点（等价于 1D 卷积）"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, in_dim)
        return F.gelu(self.norm(self.linear(x)))


class BoundaryEncoder(nn.Module):
    """
    街区边界多边形编码器。

    Args:
        boundary_points (int): 重采样后的边界点数 P
        hidden_dim (int):      中间特征维度
        out_dim (int):         输出嵌入维度
    """

    def __init__(self, boundary_points: int = 64,
                 hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.P = boundary_points
        self.out_dim = out_dim

        # ── 局部特征提取 MLP（逐点共享）──────────────────
        # 输入：(x, y) + 相对于质心的 (dx, dy) + 与前后点的角度 sin/cos = 6 维
        self.local_enc = nn.Sequential(
            PointNetLayer(6, hidden_dim // 2),
            PointNetLayer(hidden_dim // 2, hidden_dim),
        )

        # ── 全局特征聚合后再映射 ──────────────────────────
        # 局部特征 hidden_dim + 全局 max-pooled hidden_dim → out_dim
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

        # ── 位置感知的序列 token 输出（供 cross-attn 使用）──
        self.token_proj = nn.Linear(hidden_dim * 2, out_dim)

    def _compute_features(self, pts: torch.Tensor) -> torch.Tensor:
        """
        计算每个点的 6 维局部特征。

        Args:
            pts: (B, P, 2)  归一化坐标
        Returns:
            feat: (B, P, 6)
        """
        B, P, _ = pts.shape

        # 质心
        centroid = pts.mean(dim=1, keepdim=True)  # (B, 1, 2)
        delta = pts - centroid                     # (B, P, 2) 相对坐标

        # 前后点形成的有向角度
        prev_pts = torch.roll(pts, 1, dims=1)
        next_pts = torch.roll(pts, -1, dims=1)
        edge_vec = next_pts - prev_pts             # (B, P, 2)
        angle = torch.atan2(edge_vec[..., 1], edge_vec[..., 0])  # (B, P)

        feat = torch.cat([
            pts,                               # 绝对坐标 2
            delta,                             # 相对质心 2
            torch.sin(angle).unsqueeze(-1),    # 角度 sin 1
            torch.cos(angle).unsqueeze(-1),    # 角度 cos 1
        ], dim=-1)                             # (B, P, 6)

        return feat

    def forward(self, pts: torch.Tensor):
        """
        Args:
            pts: (B, P, 2)  边界多边形顶点，归一化到 [0,1]

        Returns:
            tokens: (B, P, out_dim)   逐点 token 序列（供 cross-attention）
            global_emb: (B, out_dim)  全局嵌入（可选）
        """
        feat = self._compute_features(pts)   # (B, P, 6)
        local_feat = self.local_enc(feat)    # (B, P, hidden_dim)

        # 全局 max-pooling
        global_feat = local_feat.max(dim=1).values  # (B, hidden_dim)
        global_feat_exp = global_feat.unsqueeze(1).expand_as(local_feat)

        # 局部 + 全局拼接
        combined = torch.cat([local_feat, global_feat_exp], dim=-1)  # (B, P, 2*hidden)

        tokens = self.token_proj(combined)          # (B, P, out_dim)
        global_emb = self.global_proj(
            torch.cat([local_feat.max(1).values,
                       local_feat.mean(1)], dim=-1)
        )  # (B, out_dim)

        return tokens, global_emb


def resample_polygon(pts: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    等弧长重采样，完全向量化，无 Python 循环。
    原版双重 for 循环在 batch_size=1024, P=64 时约 800ms；
    向量化版本约 2ms。

    Args:
        pts:      (B, P_orig, 2)
        target_n: 目标点数

    Returns:
        (B, target_n, 2)
    """
    B, P, _ = pts.shape
    device = pts.device

    # 计算每条边的长度和累积弧长
    rolled = torch.roll(pts, -1, dims=1)          # 下一个顶点
    edge_len = (rolled - pts).norm(dim=-1)         # (B, P)
    # 在前面补 0，方便 searchsorted
    cum_len = torch.cat([
        torch.zeros(B, 1, device=device),
        torch.cumsum(edge_len, dim=1)              # (B, P)
    ], dim=1)                                      # (B, P+1)
    total = cum_len[:, -1:].clamp(min=1e-6)        # (B, 1)

    # 目标采样弧长位置  (B, target_n)
    t = torch.linspace(0, 1, target_n + 1, device=device)[:-1]  # 不含终点
    t = t.unsqueeze(0) * total                     # (B, target_n)

    # 向量化搜索每个 t 落在哪条边（利用 torch.searchsorted）
    # cum_len[:, 1:] 是每条边的终止弧长
    seg_idx = torch.searchsorted(
        cum_len[:, 1:].contiguous(), t.contiguous()
    )                                              # (B, target_n)
    seg_idx = seg_idx.clamp(0, P - 1)

    # 批量取对应边的起止弧长
    seg_start = cum_len.gather(1, seg_idx)         # (B, target_n)
    seg_end   = cum_len.gather(1, seg_idx + 1)
    seg_len   = (seg_end - seg_start).clamp(min=1e-9)
    alpha = ((t - seg_start) / seg_len).clamp(0, 1)  # (B, target_n)

    # 批量取对应顶点坐标
    idx_exp  = seg_idx.unsqueeze(-1).expand(-1, -1, 2)       # (B, target_n, 2)
    idx_next = ((seg_idx + 1) % P).unsqueeze(-1).expand(-1, -1, 2)
    p0 = pts.gather(1, idx_exp)                              # (B, target_n, 2)
    p1 = pts.gather(1, idx_next)

    alpha_exp = alpha.unsqueeze(-1)                          # (B, target_n, 1)
    return p0 * (1 - alpha_exp) + p1 * alpha_exp             # (B, target_n, 2)
