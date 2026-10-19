"""
models/flow_matching.py

Flow Matching（Rectified Flow / Conditional Flow Matching）过程封装，
作为 diffusion.py 中 GaussianDiffusion（DDPM）的替代方案，用于架构对比实验。

与 DDPM 的对应关系：

  DDPM                                    Flow Matching（本实现）
  ---------------------------------       ---------------------------------
  x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε              x_t = (1-(1-σ_min)t) x0 + t ε
  t ∈ {0,...,T-1}  离散                    t ∈ [0,1]  连续
  网络预测噪声 ε                            网络预测速度场 v = ε - (1-σ_min) x0
  反向：DDPM 后验采样，固定 T 步             反向：ODE 数值积分（欧拉法），
                                            步数与训练时的连续采样解耦，可自由调节
  x̃0 = (x_t - √(1-ᾱ_t) ε_pred)/√ᾱ_t       x̃0 = x_t - t · v_pred（无需除法）

设计要点：
  - TransformerDenoiser 完全复用，不做任何改动：它只是把
    (x_t, t, text_tokens, text_mask, bnd_tokens, obj_mask)
    映射成一个与 x_t 同形的张量；这个张量被解释成"噪声"还是"速度"，
    完全是外层生成过程的语义，不影响网络结构本身。
  - 唯一需要的适配点：DDPM 里 t 是 [0, T) 的整数，Flow Matching 里 t 是
    [0, 1] 的连续值。SinusoidalTimeEmb 的频率范围是按 t ~ O(T) 设计的，
    如果直接把 [0,1] 的 t 喂进去，时步嵌入会退化成近似常数。因此在调用
    denoiser 之前，用 scale_t() 把连续 t 线性放大回 [0, T) 的尺度，
    这样 denoiser（含其 SinusoidalTimeEmb / AdaLN）不需要任何修改。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

from .diffusion import _pairwise_iou_loss, pairwise_2d_iou  # 复用同一套 IoU 惩罚实现，保证和 DDPM 版本可比较

import ot
import numpy as np

def minibatch_ot_coupling(x0: torch.Tensor, x1: torch.Tensor,
                          reg: float = 0.05, method: str = "sinkhorn") -> torch.Tensor:
    """
    在 batch 维度上对 x0（数据）和 x1（噪声）做最优传输配对。

    Args:
        x0:     (B, N, D)  数据样本
        x1:     (B, N, D)  独立采样的噪声
        reg:    Sinkhorn 正则化系数 ε，越小越接近精确 OT
        method: "sinkhorn" 或 "exact" (EMD)

    Returns:
        x1_ot:  (B, N, D)  重排后的噪声，使得 (x0[i], x1_ot[i]) 遵循 OT 计划
    """
    B = x0.shape[0]
    device = x0.device
    
    # 展平每个样本用于计算距离: (B, N*D)
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)
    
    # 代价矩阵: squared L2 distance (B, B)
    C = torch.cdist(x0_flat, x1_flat, p=2) ** 2
    C_np = C.detach().cpu().numpy()
    
    a = np.ones(B, dtype=np.float64) / B
    b = np.ones(B, dtype=np.float64) / B
    
    if method == "sinkhorn":
        P = ot.sinkhorn(a, b, C_np, reg=reg, numItermax=1000)
    else:  # exact
        P = ot.emd(a, b, C_np)
    
    P = torch.from_numpy(P).to(device, dtype=x0.dtype)  # (B, B)
    
    # P 的行和为 1/B，转换为条件概率（每行和为 1）
    P_cond = P * B
    
    # 软分配: x1_ot[i] = Σ_j P_cond[i,j] * x1[j]
    x1_ot_flat = P_cond @ x1_flat
    return x1_ot_flat.reshape_as(x1)



class FlowMatching(nn.Module):
    """
    Rectified Flow / Conditional Flow Matching 过程。

    Args:
        sigma_min (float):          OT-CFM 的最小噪声系数，取 0 即退化为最简单的
                                     线性 Rectified Flow（Liu et al. 2022 /
                                     Lipman et al. 2022, "Flow Matching for
                                     Generative Modeling"）。
        num_train_timesteps (int):  仅用于 scale_t，把连续 t 映射回和 DDPM
                                     训练时一致的量级，便于复用
                                     SinusoidalTimeEmb 而不用改动 denoiser。
                                     建议直接传 cfg.T（比如 1000），
                                     这样两个模型的时步嵌入尺度完全一致。
    """

    def __init__(self, sigma_min: float = 0.0, num_train_timesteps: int = 1000):
        super().__init__()
        self.sigma_min = sigma_min
        self.num_train_timesteps = num_train_timesteps

    # ─────────────────────────────────────────────
    # 时间尺度适配（唯一需要的"胶水代码"，denoiser 本身不用改）
    # ─────────────────────────────────────────────
    def scale_t(self, t: torch.Tensor) -> torch.Tensor:
        """把连续 t∈[0,1] 线性映射到 [0, num_train_timesteps)。"""
        return t * (self.num_train_timesteps - 1)

    # ─────────────────────────────────────────────
    # 正向过程：构造插值路径 x_t 和目标速度场
    # ─────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            x0:    (B, N, D)  干净场景
            t:     (B,)       连续时间 ∈ [0,1]
            noise: (B, N, D)  若 None 则采样标准高斯

        Returns:
            x_t:      (B, N, D)  插值路径上 t 时刻的带噪场景
            target_v: (B, N, D)  该点处的目标速度场（训练监督信号，
                                  对应 DDPM 里 q_sample 返回的 noise）
        """
        if noise is None:
            noise = torch.randn_like(x0)

        B = x0.shape[0]
        t_ = t.view(B, 1, 1)

        x_t = (1 - (1 - self.sigma_min) * t_) * x0 + t_ * noise
        target_v = noise - (1 - self.sigma_min) * x0
        return x_t, target_v

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor,
                           v: torch.Tensor) -> torch.Tensor:
        """
        由预测速度场反解 x̃_0（用于 IoU 损失构建 bbox），对应
        GaussianDiffusion.predict_x0_from_eps。

        推导：对线性插值路径，x_t = x0 + t · v_target 恒成立（与 σ_min 无关），
        所以 x0 = x_t - t · v_pred —— 不需要像 DDPM 那样再除以 √ᾱ_t，
        这是 Flow Matching 线性路径的一个自然性质。
        """
        B = x_t.shape[0]
        t_ = t.view(B, 1, 1)
        return x_t - t_ * v

    # ─────────────────────────────────────────────
    # 反向采样：ODE 数值积分（欧拉法）
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, denoiser_fn: Callable, shape: tuple,
               text_tokens: torch.Tensor, text_mask: torch.Tensor,
               bnd_tokens: torch.Tensor,
               obj_mask: Optional[torch.Tensor] = None,
               device: torch.device = None,
               num_steps: int = 50) -> torch.Tensor:
        """
        从纯高斯噪声（t=1）出发，用欧拉法把 ODE dx/dt = v_θ(x_t,t,cond)
        积分回 t=0，得到 x_0。

        num_steps 与训练时的连续时间采样是解耦的：训练时 t~U(0,1) 连续采样，
        推理时可以自由选择离散化步数（通常 20~50 步即可，远少于 DDPM 的
        T=1000 步），这是 Flow Matching 相比 DDPM 的主要工程优势之一——
        同一个训练好的模型，可以在"生成质量 vs 推理速度"之间自由取舍，
        而不需要重新训练。

        Args:
            denoiser_fn: 接受 (x_t, t_batch, text_tokens, text_mask,
                         bnd_tokens, obj_mask) → v_pred 的可调用对象。
                         t_batch 传入的是 [0,1] 连续值，denoiser_fn 内部
                         自己调用 scale_t 后再喂给 TransformerDenoiser。
            shape:       (B, N, D)
            num_steps:   ODE 积分步数（可调，越多越接近连续 ODE 的解）

        Returns:
            x_0: (B, N, D)
        """
        if device is None:
            device = text_tokens.device

        B, N, D = shape
        x_t = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        # t 从 1 递减到 dt，每步欧拉更新 x_t ← x_t - dt · v_pred
        for i in reversed(range(num_steps)):
            t_val = (i + 1) / num_steps
            t_batch = torch.full((B,), t_val, device=device)
            v_pred = denoiser_fn(x_t, t_batch, text_tokens, text_mask,
                                 bnd_tokens, obj_mask)
            x_t = x_t - dt * v_pred

        return x_t

    # ─────────────────────────────────────────────
    # 训练损失：结构与 diffusion.py 当前生效的 v6 版本一一对应，
    # 只是把"预测噪声"换成"预测速度"，把 alpha_bar 权重换成 (1-t)
    # （t→0 对应干净数据，等价于 DDPM 里 alpha_bar→1）
    # ─────────────────────────────────────────────
    # 最初的flow matching的loss
    # def compute_loss(self, x0: torch.Tensor, v_pred: torch.Tensor,
    #                  target_v: torch.Tensor, t: torch.Tensor,
    #                  x_t: torch.Tensor, obj_mask: torch.Tensor = None,
    #                  lambda_iou: float = 0.1) -> dict:
    #     # ── 速度场 MSE：几何 + 类别分开加权 ───────────────────
    #     loss_geo = F.mse_loss(v_pred[..., :7], target_v[..., :7])      # bbox + sin/cos
    #     loss_class = F.mse_loss(v_pred[..., 7:9], target_v[..., 7:9])  # real/empty bits
    #     loss_sce = loss_geo + 3.0 * loss_class

    #     # ── 反解 x0，只对模型自己判断为 real 的 slot 算 IoU ──────
    #     x0_pred = self.predict_x0_from_v(x_t, t, v_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)

    #     cls_pred = x0_pred[..., 7:9]
    #     real_prob = torch.sigmoid(cls_pred[..., 0] - cls_pred[..., 1])
    #     bbox_pred_masked = bbox_pred * real_prob.unsqueeze(-1).detach()

    #     # (1-t) 扮演 DDPM 里 alpha_bar_t 的角色：t→0（干净数据）权重更大
    #     weight = (1 - t).view(-1)
    #     loss_iou = _pairwise_iou_loss(bbox_pred_masked)
    #     loss_iou = (0.1 * weight * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou
    #     return {
    #         "loss": total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_cls": loss_class.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }

    # 与Diffuscene对齐 loss v11的变体
    def compute_loss(self, x0: torch.Tensor, v_pred: torch.Tensor,
                 target_v: torch.Tensor, t: torch.Tensor,
                 x_t: torch.Tensor, obj_mask: torch.Tensor = None,
                 lambda_iou: float = 0.1) -> dict:
        B = x0.shape[0]

        # ── 1. 分离几何损失（与 diffusion 一致，保留 batch 维度） ──
        loss_trans = ((v_pred[..., 0:2] - target_v[..., 0:2]) ** 2).mean(
            dim=list(range(1, len(v_pred.shape))))
        loss_size = ((v_pred[..., 2:5] - target_v[..., 2:5]) ** 2).mean(
            dim=list(range(1, len(v_pred.shape))))
        loss_angle = ((v_pred[..., 5:7] - target_v[..., 5:7]) ** 2).mean(
            dim=list(range(1, len(v_pred.shape))))
        loss_geo = ((v_pred[..., :7] - target_v[..., :7]) ** 2).mean(
            dim=list(range(1, len(v_pred.shape))))

        # ── 2. 分类损失 ──
        loss_class = F.mse_loss(v_pred[..., 7:9], target_v[..., 7:9])

        # ── 3. 组合基础损失（diffusion 风格，class 权重改为 1.0） ──
        # loss_geo 为 (B,)，loss_class 为标量，广播相加
        losses = loss_geo + loss_class

        # ── 4. 时间依赖加权：(1-t) 对应 diffusion 的 alpha_bar ──
        weight = (1 - t).view(-1)          # (B,)
        losses_weight = losses * weight    # (B,)

        # ── 5. IoU 损失（diffusion 核心思想：hard mask + floor 策略） ──
        x0_pred = self.predict_x0_from_v(x_t, t, v_pred)
        bbox_pred = x0_pred[..., :4].clamp(0, 1)

        cls_pred = x0_pred[..., 7:9]
        real_prob = torch.sigmoid(cls_pred[..., 0] - cls_pred[..., 1])
        valid_mask = (real_prob >= 0.5).float()   # hard mask，与 diffusion 一致

        # invalid slot 的 bbox 置零，避免参与 IoU 计算
        bbox_pred_masked = bbox_pred * valid_mask.unsqueeze(-1)

        # 使用已有的 pairwise IoU 函数
        bbox_iou = _pairwise_iou_loss(bbox_pred_masked)

        # IoU 加权：引入 floor 策略，与 diffusion 的 alpha_bar clamp 对应
        iou_weight = weight.clamp(min=0.3)
        loss_iou = (0.1 * iou_weight * bbox_iou).mean()

        # ── 6. 总损失 ──
        total_loss = losses_weight.mean() + lambda_iou * loss_iou

        return {
            'loss': total_loss,                       # 不 detach，用于反向传播
            'loss_trans': loss_trans.mean().detach(),
            'loss_size': loss_size.mean().detach(),
            'loss_angle': loss_angle.mean().detach(),
            'loss_geo': loss_geo.mean().detach(),
            'loss_cls': loss_class.detach(),
            'loss_iou': loss_iou.detach(),
            'loss_bbox_iou': bbox_iou.mean().detach(),
            'iou_weight_mean': iou_weight.mean().detach(),
        }

