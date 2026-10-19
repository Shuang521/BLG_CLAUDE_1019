"""
models/diffusion.py

DDPM 扩散过程封装：
  - 线性噪声调度
  - 正向过程（q(x_t | x_0)）
  - 反向采样（逐步去噪）
  - 训练损失（噪声预测 + IoU 惩罚）

完全参照 Ho et al. 2020 (DDPM) 和 DiffuScene 的 Eq.(3)(9)(10)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable


class GaussianDiffusion(nn.Module):
    """
    DDPM 扩散过程。

    Args:
        T (int):          总时步数
        beta_start (float): 噪声调度起点
        beta_end (float):   噪声调度终点
    """

    def __init__(self, T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.T = T

        # 线性噪声调度（DDPM 默认）---感觉这个不是很好用?
        betas = torch.linspace(beta_start, beta_end, T)           # (T,) 
        alphas = 1.0 - betas                                        # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)              # ᾱ_t  (T,)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 注册为 buffer（不参与梯度，但随 .to(device) 自动迁移）
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 正向过程系数
        self.register_buffer("sqrt_alphas_cumprod",
                             alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt())

        # 反向过程后验方差 σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var)
        # 裁掉 t=0 处的 log(0)
        self.register_buffer("log_posterior_variance",
                             posterior_var.clamp(min=1e-20).log())

    # ─────────────────────────────────────────────
    # 正向过程工具
    # ─────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor,
                 t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> tuple:
        """
        从 x_0 直接采样 x_t（DDPM Eq.3）：
            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            x0:    (B, N, D)  干净场景
            t:     (B,)       整数时步
            noise: (B, N, D)  若 None，则采样标准高斯

        Returns:
            x_t:   (B, N, D)  带噪场景
            noise: (B, N, D)  实际使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)

        B = x0.shape[0]
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        x_t = sqrt_ac * x0 + sqrt_1mac * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor,
                             t: torch.Tensor,
                             eps: torch.Tensor) -> torch.Tensor:
        """
        由预测噪声 ε_pred 反解 x̃_0（用于 IoU 损失中构建边界框）：
            x̃_0 = (x_t - √(1-ᾱ_t)*ε) / √ᾱ_t
        """
        B = x_t.shape[0]
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        return (x_t - sqrt_1mac * eps) / sqrt_ac.clamp(min=1e-6)

    # ─────────────────────────────────────────────
    # 反向采样
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor,
                      t: int,
                      eps_pred: torch.Tensor) -> torch.Tensor:
        """
        单步反向采样（DDPM Eq.6）：
            μ_t = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_pred)
            x_{t-1} ~ N(μ_t, σ_t²I)

        Args:
            x_t:      (B, N, D)  当前带噪场景
            t:        int        当前时步（标量）
            eps_pred: (B, N, D)  去噪器预测的噪声

        Returns:
            x_{t-1}: (B, N, D)
        """
        B = x_t.shape[0]
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t]

        # 均值
        mu = (1.0 / alpha_t.sqrt()) * (
            x_t - (beta_t / sqrt_1mac) * eps_pred
        )

        if t == 0:
            return mu

        # 方差（取后验方差的对数 exp，数值稳定）
        log_var = self.log_posterior_variance[t]
        noise = torch.randn_like(x_t)
        return mu + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample(self, denoiser_fn: Callable,
               shape: tuple,
               text_tokens: torch.Tensor,
               text_mask: torch.Tensor,
               bnd_tokens: torch.Tensor,
               obj_mask: Optional[torch.Tensor] = None,
               device: torch.device = None) -> torch.Tensor:
        """
        从纯高斯噪声开始，逐步去噪，得到干净场景 x_0。

        Args:
            denoiser_fn: 接受 (x_t, t_batch, text_tokens, text_mask,
                                bnd_tokens, obj_mask) → eps_pred 的可调用对象
            shape:       (B, N, D)  采样形状
            text_tokens: (B, L, d_model)
            text_mask:   (B, L)
            bnd_tokens:  (B, P, d_model)
            obj_mask:    (B, N) 或 None
            device:      目标设备

        Returns:
            x_0: (B, N, D)
        """
        if device is None:
            device = text_tokens.device

        B, N, D = shape
        x_t = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            eps_pred = denoiser_fn(x_t, t_batch,
                                   text_tokens, text_mask,
                                   bnd_tokens, obj_mask)
            x_t = self.p_sample_step(x_t, t, eps_pred)

        return x_t

    # loss v11 iou v11
    def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None,
                        lambda_iou: float = 0.1, loss_separate: bool = True,
                        iou_weight_floor: float = 0.3):
        """
        整合版 loss：结合 v6/v10 的数据结构 + 对照组实验的思想
        
        数据结构假设：
        - eps_pred[..., :4]  : 2D bbox (x, y, w, h)
        - eps_pred[..., 4:7] : angle (sin, cos, ?) - 根据你的实际维度调整
        - eps_pred[..., 7:9] : real/empty bits (2D)
        """
        B = x0.shape[0]
        
        # ── 1. 分离几何损失 ──
        # 根据你的实际数据结构调整切片
        trans_pred = eps_pred[..., 0:2]      # x, y
        size_pred = eps_pred[..., 2:5]       # w, h
        angle_pred = eps_pred[..., 5:7]      # sin, cos, ?
        bbox_pred_full = eps_pred[..., 0:7]  # 完整 bbox + angle
        
        trans_noise = noise[..., 0:2]
        size_noise = noise[..., 2:5]
        angle_noise = noise[..., 5:7]
        bbox_noise_full = noise[..., 0:7]
        
        loss_trans = ((trans_pred - trans_noise) ** 2).mean(dim=list(range(1, len(x0.shape))))
        loss_size = ((size_pred - size_noise) ** 2).mean(dim=list(range(1, len(x0.shape))))
        loss_angle = ((angle_pred - angle_noise) ** 2).mean(dim=list(range(1, len(x0.shape))))
        loss_bbox = ((bbox_pred_full - bbox_noise_full) ** 2).mean(dim=list(range(1, len(x0.shape))))
        
        # ── 2. 分类损失 ──
        loss_class = F.mse_loss(eps_pred[..., 7:9], noise[..., 7:9])
        
        # ── 3. 组合基础损失 ──
        if loss_separate:
            losses = loss_bbox + loss_class  # 将你原来的 3.0 权重改为DiffuScene对应的1
        else:
            losses = ((eps_pred - noise) ** 2).mean(dim=list(range(1, len(x0.shape))))
        
        # ── 4. 时间依赖加权 ──
        alpha_bar = self.alphas_cumprod[t].view(-1, *([1] * (len(x0.shape) - 1)))
        losses_weight = losses * alpha_bar.squeeze()
        
        # ── 5. IoU 损失（2D 轴对齐） ──
        # 预测 x0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # 提取 2Dbbox 和 objectness
        bbox_recon = x0_pred[..., :4]  # 2D bbox
        obj_recon = x0_pred[..., 7:9]  # real/empty bits
        
        # 创建 valid mask（参考对照组的思路）
        # 使用 soft mask 或 hard mask
        real_prob = torch.sigmoid(obj_recon[..., 0] - obj_recon[..., 1])
        valid_mask = (real_prob >= 0.5).float()  # hard mask，或者直接用 real_prob 作为 soft mask
        
        # 计算 2D 轴对齐 IoU
        # bbox_recon 格式假设: (x_center, y_center, width, height)
        # 转换为 corners: (x_min, y_min, x_max, y_max)
        x_center = bbox_recon[..., 0]
        y_center = bbox_recon[..., 1]
        width = bbox_recon[..., 2]
        height = bbox_recon[..., 3]
        
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        bbox_corners = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # (B, N, 4)
        
        # 计算 pairwise 2D IoU
        bbox_iou = pairwise_2d_iou(bbox_corners)  # (B, N, N)
        
        # 应用 valid mask
        bbox_iou_mask = valid_mask[:, :, None] * valid_mask[:, None, :]  # (B, N, N)
        bbox_iou_valid = bbox_iou * bbox_iou_mask
        
        # 计算有效 IoU 平均值
        bbox_iou_valid_sum = bbox_iou_valid.sum(dim=list(range(1, len(bbox_iou_valid.shape))))
        bbox_iou_mask_sum = bbox_iou_mask.sum(dim=list(range(1, len(bbox_iou_valid.shape))))
        bbox_iou_valid_avg = bbox_iou_valid_sum / (bbox_iou_mask_sum + 1e-6)
        
        # IoU 损失加权（参考对照组的 floor 策略）
        iou_weight = alpha_bar.squeeze().clamp(min=iou_weight_floor)
        loss_iou_valid_avg = (iou_weight * 0.1 * bbox_iou_valid_avg).mean()
        
        # ── 6. 总损失 ──
        total_loss = losses_weight.mean() + lambda_iou * loss_iou_valid_avg
        
        return {
            'loss': total_loss,  # 不要 detach！用于反向传播
            'loss_trans': loss_trans.mean().detach(),  # 这些可以 detach，用于日志
            'loss_size': loss_size.mean().detach(),
            'loss_angle': loss_angle.mean().detach(),
            'loss_geo': loss_bbox.mean().detach(),
            'loss_cls': loss_class.detach(),
            'loss_iou': loss_iou_valid_avg.detach(),
            'loss_bbox_iou': bbox_iou_valid_avg.mean().detach(),
            'iou_weight_mean': iou_weight.mean().detach(),
        }

    # # ─────────────────────────────────────────────
    # # v1 训练损失---仅包含预测噪声和建筑框重叠惩罚
    # # ─────────────────────────────────────────────
    # def compute_loss(self, x0: torch.Tensor,
    #                  eps_pred: torch.Tensor,
    #                  noise: torch.Tensor,
    #                  t: torch.Tensor,
    #                  x_t: torch.Tensor,
    #                  obj_mask: torch.Tensor = None, # 在这个里面并未使用 但是为了保持入参一致所以先传入
    #                  lambda_iou: float = 0.1) -> dict:
    #     """
    #     修正后的损失计算
    #     """
    #     # ── L_sce：噪声预测 MSE ───────────────────────
    #     loss_sce = F.mse_loss(eps_pred, noise)

    #     # ── L_iou：建筑框重叠惩罚 ────────────────────
    #     # 移除了 with torch.no_grad() 和 .detach()，允许梯度流回 eps_pred
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)  # (B, N, 4) cx,cy,w,h

    #     # ᾱ_t 权重
    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)  # (B,1,1)

    #     loss_iou = _pairwise_iou_loss(bbox_pred)  # (B,) 均值
    #     loss_iou = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou

    #     # 注意：这里在返回字典时再使用 .detach() 仅仅是为了日志打印，不影响反向传播
    #     return {
    #         "loss": total,
    #         "loss_sce": loss_sce.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }
    # loss v6 拆分bbox和class分别计算mse loss 以及仅对预测为真实的slot计算iou loss
    # def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None, lambda_iou=0.1):
    #     # 拆分维度
    #     loss_geo    = F.mse_loss(eps_pred[..., :7], noise[..., :7])      # bbox + sin/cos
    #     loss_class  = F.mse_loss(eps_pred[..., 7:9], noise[..., 7:9])    # real/empty bits

    #     # 给 class 维度的 MSE 单独加权，并放大其 bit scale
    #     # （这是 Analog Bits 论文的关键 trick：bit scale 越大，越像离散判别）
    #     loss_sce = loss_geo + 3.0 * loss_class   # 系数先用 3~5 之间试验

    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)  # 暂未修正 只使用二维iou检测

    #     # 关键修正：只对模型自己预测为 real 的 slot 算 IoU
    #     cls_pred = x0_pred[..., 7:9]              # 在 [-1,1] 附近
    #     real_prob = torch.sigmoid(cls_pred[..., 0] - cls_pred[..., 1])  # soft real mask
    #     bbox_pred_masked = bbox_pred * real_prob.unsqueeze(-1).detach()  # 用 soft mask 加权

    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou = _pairwise_iou_loss(bbox_pred_masked)
    #     loss_iou = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou
    #     return {
    #         "loss": total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_cls": loss_class.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }

    # loss v10
    # def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None,
    #               lambda_iou: float = 0.1, empty_weight: float = 0.1,
    #               iou_weight_floor: float = 0.3):
    #     # ── 1. 几何 loss：按 obj_mask 加权，real slot 权重高，empty slot 权重低但非零 ──
    #     geo_sq = (eps_pred[..., :7] - noise[..., :7]) ** 2  # (B, N, 7)
    #     if obj_mask is not None:
    #         w = obj_mask.float() + (1.0 - obj_mask.float()) * empty_weight  # (B, N)
    #         w_exp = w.unsqueeze(-1)  # (B, N, 1)
    #         loss_geo = (geo_sq * w_exp).sum() / (w_exp.expand_as(geo_sq).sum() + 1e-8)
    #     else:
    #         loss_geo = geo_sq.mean()

    #     # ── 2. 分类 loss：不加权，real/empty 两类都需要被充分监督 ──
    #     loss_class = F.mse_loss(eps_pred[..., 7:9], noise[..., 7:9])

    #     loss_sce = loss_geo + 3.0 * loss_class

    #     # ── 3. IoU loss：只对模型自己判断为 real 的 slot 计算 ──
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)

    #     cls_pred = x0_pred[..., 7:9]
    #     real_prob = torch.sigmoid(cls_pred[..., 0] - cls_pred[..., 1])
    #     bbox_pred_masked = bbox_pred * real_prob.unsqueeze(-1).detach()

    #     # 权重不再直接乘 alpha_bar_t（会导致 t 偏大时梯度几乎为 0），
    #     # 改为 alpha_bar_t 和 floor 值取较大者，保证任意 t 都有基础约束力度
    #     alpha_bar = self.alphas_cumprod[t].view(-1)  # (B,)
    #     iou_weight = alpha_bar.clamp(min=iou_weight_floor)

    #     loss_iou = _pairwise_iou_loss(bbox_pred_masked)
    #     loss_iou = (iou_weight * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou
    #     return {
    #         "loss": total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_cls": loss_class.detach(),
    #         "loss_iou": loss_iou.detach(),
    #         "iou_weight_mean": iou_weight.mean().detach(),  # 建议加进日志，方便确认权重量级
    #     }
    
    # 训练损失 v2
    # def compute_loss(self, x0: torch.Tensor,
    #              eps_pred: torch.Tensor,
    #              noise: torch.Tensor,
    #              t: torch.Tensor,
    #              x_t: torch.Tensor,
    #              obj_mask: torch.Tensor,        # ← 需要传入，区分有效slot
    #              lambda_iou: float = 0.1,
    #              lambda_cls: float = 0.1) -> dict:

    #     # ── 1. 预测 x0 ────────────────────────────────
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)  # (B, N, D)

    #     # ── 2. 拆分维度 ───────────────────────────────
    #     # GT
    #     bbox_gt   = x0[..., :6]          # cx,cy,w,h,sin,cos
    #     cls_gt    = x0[..., 6:]          # one-hot (B, N, 8)
    #     # Pred
    #     bbox_pred = x0_pred[..., :6]
    #     cls_pred  = x0_pred[..., 6:]     # 用于分类 loss 的 logits

    #     # ── 3. 噪声 MSE（只对 bbox 维度）──────────────
    #     noise_bbox = noise[..., :6]
    #     eps_bbox   = eps_pred[..., :6]
    #     loss_sce   = F.mse_loss(eps_bbox, noise_bbox)

    #     # ── 4. 分类 loss（交叉熵，只对有效slot）────────
    #     # obj_mask: (B, N)  1=有效建筑 0=empty slot
    #     cls_target = cls_gt.argmax(dim=-1)   # (B, N)  long

    #     # 对所有 slot 都算（包括 empty，empty 的 target 就是7）
    #     # cls_pred shape: (B, N, 8) → (B*N, 8)
    #     B, N, C = cls_pred.shape
    #     loss_cls = F.cross_entropy(
    #         cls_pred.reshape(B * N, C),
    #         cls_target.reshape(B * N).long(),
    #         label_smoothing=0.1   # 防止过度自信
    #     )

    #     # ── 5. IoU loss（只对有效slot）────────────────
    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou  = _pairwise_iou_loss(bbox_pred[..., :4].clamp(0, 1))
    #     loss_iou  = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     # ── 6. 合并 ───────────────────────────────────
    #     total = loss_sce + lambda_cls * loss_cls + lambda_iou * loss_iou

    #     return {
    #         "loss":      total,
    #         "loss_sce":  loss_sce.detach(),
    #         "loss_cls":  loss_cls.detach(),   # ← 训练时监控这个，应该从~2.08下降
    #         "loss_iou":  loss_iou.detach(),
    #     }
    
    # # v3
    # # 在您的类中更新 compute_loss 方法
    # def compute_loss(self, x0: torch.Tensor,
    #                 eps_pred: torch.Tensor,
    #                 noise: torch.Tensor,
    #                 t: torch.Tensor,
    #                 x_t: torch.Tensor,
    #                 obj_mask: torch.Tensor,        # (B, N)
    #                 lambda_noise: float = 1.0,     # 噪声预测权重
    #                 lambda_cls: float = 0.1,       # 分类权重
    #                 lambda_iou: float = 0.0,       # 重叠惩罚权重
    #                 lambda_x0_reg: float = 0.1     # 辅助 x0 回归权重（可选）
    #                 ) -> dict:

    #     B, N, _ = x0.shape

    #     # ── 1. 预测 x0 ────────────────────────────────
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)  # (B, N, D)

    #     # ── 2. 拆分维度 ───────────────────────────────
    #     # GT
    #     bbox_gt   = x0[..., :6]          # cx,cy,w,h,sin,cos
    #     cls_gt    = x0[..., 6:]          # one-hot (B, N, 8)
    #     # Pred
    #     bbox_pred = x0_pred[..., :6]
    #     cls_pred  = x0_pred[..., 6:]     # logits

    #     # ── 3. 噪声 MSE（引入 obj_mask 过滤）──────────
    #     noise_bbox = noise[..., :6]
    #     eps_bbox   = eps_pred[..., :6]
        
    #     # 计算未平均的平方误差 (B, N, 6)
    #     sce_loss_raw = F.mse_loss(eps_bbox, noise_bbox, reduction='none')
    #     # 只对有效目标计算噪声损失
    #     loss_sce = (sce_loss_raw * obj_mask.unsqueeze(-1)).sum() / (obj_mask.sum() * 6 + 1e-8)

    #     # ── 4. 辅助 x0 回归损失（仅对有效目标，细化各分量）──
    #     # 这一步能加速收敛，尤其是角度部分
    #     loss_x0_reg = torch.tensor(0.0, device=x0.device)
    #     if lambda_x0_reg > 0:
    #         # 4.1 位置和大小损失 (L1 损失通常比 MSE 对框回归更鲁棒)
    #         loss_coord = F.l1_loss(bbox_pred[..., :4], bbox_gt[..., :4], reduction='none')
    #         loss_coord = (loss_coord * obj_mask.unsqueeze(-1)).sum() / (obj_mask.sum() * 4 + 1e-8)
            
    #         # 4.2 角度损失 (对 sin, cos)
    #         loss_angle = F.mse_loss(bbox_pred[..., 4:6], bbox_gt[..., 4:6], reduction='none')
    #         loss_angle = (loss_angle * obj_mask.unsqueeze(-1)).sum() / (obj_mask.sum() * 2 + 1e-8)
            
    #         # 4.3 角度单位化约束 (sin^2 + cos^2 应该接近 1)
    #         pred_sin_cos_sq = torch.sum(bbox_pred[..., 4:6] ** 2, dim=-1) # (B, N)
    #         loss_unit = F.mse_loss(pred_sin_cos_sq, torch.ones_like(pred_sin_cos_sq), reduction='none')
    #         loss_unit = (loss_unit * obj_mask).sum() / (obj_mask.sum() + 1e-8)

    #         loss_x0_reg = loss_coord + 1.0 * loss_angle + 0.1 * loss_unit

    #     # ── 5. 分类 loss（所有 slot 都参与计算）────────
    #     cls_target = cls_gt.argmax(dim=-1)   # (B, N)
    #     C = cls_pred.shape[-1]
    #     loss_cls = F.cross_entropy(
    #         cls_pred.reshape(B * N, C),
    #         cls_target.reshape(B * N).long(),
    #         label_smoothing=0.1
    #     )

    #     # ── 6. IoU loss（只对有效 slot）────────────────
    #     # 使用安全的 .view(-1) 避免 squeeze 引入的 batch=1 时的 bug
    #     alpha_bar = self.alphas_cumprod[t].view(-1)  # (B,)
        
    #     # 限制预测框在合理范围再算 IoU 惩罚
    #     bbox_for_iou = torch.clamp(bbox_pred[..., :4], 0.0, 1.0)
    #     raw_iou_loss = _pairwise_iou_loss(bbox_for_iou, obj_mask) # (B,)
        
    #     # 扩散步数较小时，预测比较准，此时加大 IoU 惩罚权重
    #     loss_iou = (alpha_bar * raw_iou_loss).mean()

    #     # ── 7. 最终合并 ───────────────────────────────
    #     total = (lambda_noise * loss_sce + 
    #             lambda_cls * loss_cls + 
    #             lambda_iou * loss_iou + 
    #             lambda_x0_reg * loss_x0_reg)

    #     return {
    #         "loss":          total,
    #         "loss_sce":      loss_sce.detach(),
    #         "loss_x0_reg":   loss_x0_reg.detach() if lambda_x0_reg > 0 else torch.tensor(0.0),
    #         "loss_cls":      loss_cls.detach(),
    #         "loss_iou":      loss_iou.detach(),
    #     }

    # v4 20260613
    # 自己重新写了一遍loss
    # def compute_loss(
    #     self,
    #     x0: torch.Tensor,
    #     eps_pred: torch.Tensor,
    #     noise: torch.Tensor,
    #     t: torch.Tensor,
    #     x_t: torch.Tensor,
    #     obj_mask: torch.Tensor,
    #     lambda_noise: float = 1.0,
    #     lambda_cls: float = 0.1,
    #     lambda_loc: float = 0.05,
    #     lambda_size: float = 0.05,
    #     lambda_angle: float = 0.05,
    #     lambda_iou: float = 0.02,
    # ):
    #     """
    #     x0:
    #         (B,N,8)
    #         [cx,cy,w,h,sin,cos,real,empty]

    #     obj_mask:
    #         (B,N)
    #         1=real building
    #         0=pad slot
    #     """

    #     B, N, D = x0.shape

    #     eps = 1e-8

    #     # --------------------------------------------------
    #     # 1. predict x0
    #     # --------------------------------------------------
    #     x0_pred = self.predict_x0_from_eps(
    #         x_t,
    #         t,
    #         eps_pred
    #     )

    #     # --------------------------------------------------
    #     # 2. split gt/pred
    #     # --------------------------------------------------

    #     bbox_gt = x0[..., :6]
    #     cls_gt = x0[..., 6:]

    #     bbox_pred = x0_pred[..., :6]
    #     cls_pred = x0_pred[..., 6:]

    #     mask_expand = obj_mask.unsqueeze(-1)

    #     # ==================================================
    #     # 3. diffusion noise loss
    #     # ==================================================

    #     noise_gt = noise[..., :6]
    #     noise_pred = eps_pred[..., :6]

    #     noise_loss_raw = F.mse_loss(
    #         noise_pred,
    #         noise_gt,
    #         reduction="none"
    #     )

    #     loss_noise = (
    #         noise_loss_raw * mask_expand
    #     ).sum() / (obj_mask.sum() * 6 + eps)

    #     # ==================================================
    #     # 4. x0 regression
    #     # ==================================================

    #     # ---------- location ----------
    #     loss_loc = F.l1_loss(
    #         bbox_pred[..., :2],
    #         bbox_gt[..., :2],
    #         reduction="none"
    #     )

    #     loss_loc = (
    #         loss_loc * mask_expand[..., :2]
    #     ).sum() / (obj_mask.sum() * 2 + eps)

    #     # ---------- size ----------
    #     loss_size = F.l1_loss(
    #         bbox_pred[..., 2:4],
    #         bbox_gt[..., 2:4],
    #         reduction="none"
    #     )

    #     loss_size = (
    #         loss_size * mask_expand[..., :2]
    #     ).sum() / (obj_mask.sum() * 2 + eps)

    #     # ---------- angle ----------
    #     loss_angle = F.mse_loss(
    #         bbox_pred[..., 4:6],
    #         bbox_gt[..., 4:6],
    #         reduction="none"
    #     )

    #     loss_angle = (
    #         loss_angle * mask_expand[..., :2]
    #     ).sum() / (obj_mask.sum() * 2 + eps)

    #     # ---------- unit circle constraint ----------
    #     pred_norm = (
    #         bbox_pred[..., 4] ** 2
    #         + bbox_pred[..., 5] ** 2
    #     )

    #     loss_unit = F.mse_loss(
    #         pred_norm,
    #         torch.ones_like(pred_norm),
    #         reduction="none"
    #     )

    #     loss_unit = (
    #         loss_unit * obj_mask
    #     ).sum() / (obj_mask.sum() + eps)

    #     loss_angle = loss_angle + 0.1 * loss_unit

    #     # ---------- size positive constraint ----------
    #     pred_wh = bbox_pred[..., 2:4]

    #     loss_wh_positive = (
    #         F.relu(-pred_wh)
    #     )

    #     loss_wh_positive = (
    #         loss_wh_positive * mask_expand[..., :2]
    #     ).sum() / (obj_mask.sum() * 2 + eps)

    #     loss_size = loss_size + 0.1 * loss_wh_positive

    #     # ==================================================
    #     # 5. classification
    #     # ==================================================

    #     cls_target = cls_gt.argmax(dim=-1)

    #     C = cls_pred.shape[-1]

    #     loss_cls = F.cross_entropy(
    #         cls_pred.reshape(B * N, C),
    #         cls_target.reshape(B * N),
    #         label_smoothing=0.1,
    #     )

    #     # ==================================================
    #     # 6. IoU overlap loss
    #     # ==================================================

    #     alpha_bar = self.alphas_cumprod[t].view(B)

    #     bbox_for_iou = bbox_pred[..., :4]

    #     bbox_for_iou = torch.cat(
    #         [
    #             bbox_for_iou[..., :2],
    #             F.relu(bbox_for_iou[..., 2:4]),
    #         ],
    #         dim=-1,
    #     )

    #     raw_iou_loss = _pairwise_iou_loss(
    #         bbox_for_iou,
    #         obj_mask
    #     )

    #     # 后期权重大
    #     loss_iou = (
    #         alpha_bar * raw_iou_loss
    #     ).mean()

    #     # ==================================================
    #     # 7. total
    #     # ==================================================

    #     total_loss = (
    #         lambda_noise * loss_noise
    #         + lambda_cls * loss_cls
    #         + lambda_loc * loss_loc
    #         + lambda_size * loss_size
    #         + lambda_angle * loss_angle
    #         + lambda_iou * loss_iou
    #     )

    #     return {
    #         "loss": total_loss,

    #         "loss_noise": loss_noise.detach(),

    #         "loss_loc": loss_loc.detach(),

    #         "loss_size": loss_size.detach(),

    #         "loss_angle": loss_angle.detach(),

    #         "loss_cls": loss_cls.detach(),

    #         "loss_iou": loss_iou.detach(),
    #     }


    # v5
    # def compute_loss(self, x0: torch.Tensor,
    #                 eps_pred: torch.Tensor,
    #                 noise: torch.Tensor,
    #                 t: torch.Tensor,
    #                 x_t: torch.Tensor,
    #                 obj_mask: torch.Tensor = None,
    #                 lambda_iou: float = 0.1,
    #                 lambda_cls: float = 1.0) -> dict:
    #     """
    #     修正后的损失计算
    #     维度说明: x0/noise shape = (B, N, 8)
    #     各维: [cx, cy, w, h, sin, cos, is_real, is_empty]
    #     """
    #     B, N, D = eps_pred.shape  # D=8

    #     # ── 1. 几何维度 MSE（只对前6维：cx cy w h sin cos）────────────────
    #     # 不把 class one-hot 混入 MSE
    #     loss_geo = F.mse_loss(eps_pred[..., :6], noise[..., :6])

    #     # ── 2. 分类维度 Cross-Entropy（第6、7维：is_real, is_empty）──────
    #     # eps_pred 在 class 维度上是对噪声的预测，需要先 denoise 得到 x0_pred
    #     # 再对 x0_pred 的 class 维做 cross-entropy
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)  # (B, N, 8)

    #     # class logits: 取 is_real 和 is_empty 两维，shape (B*N, 2)
    #     cls_logits = x0_pred[..., 6:8].reshape(B * N, 2)

    #     # GT class labels: argmax of one-hot → 0=real, 1=empty
    #     # x0 里 [is_real, is_empty] 是 one-hot，argmax 得到整数 label
    #     cls_labels = x0[..., 6:8].argmax(dim=-1).reshape(B * N)  # (B*N,)

    #     loss_cls = F.cross_entropy(cls_logits, cls_labels)

    #     # ── 3. IoU 惩罚（只对 real slot 计算）────────────────────────────
    #     # real_mask: (B, N)，x0 里 is_real=1 的 slot
    #     real_mask = (x0[..., 6] > 0.5)  # (B, N)

    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)  # (B, N, 4)

    #     # 把 empty slot 的 bbox 置零，避免它们参与 IoU 计算
    #     bbox_pred_masked = bbox_pred * real_mask.unsqueeze(-1).float()

    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou = _pairwise_iou_loss(bbox_pred_masked, real_mask)
    #     loss_iou = (alpha_bar.squeeze(-1).squeeze(-1) * loss_iou).mean()

    #     # ── 4. 汇总 ──────────────────────────────────────────────────────
    #     total = loss_geo + lambda_cls * loss_cls + lambda_iou * loss_iou

    #     return {
    #         "loss":     total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_cls": loss_cls.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }
    # loss v7 使用obj mask去计算loss geo和iou loss 使用全部预测去计算class onehot是real还是empty slot
    # 后续可考虑将训练初期（比如前 30~50% 的训练步数）用 GT mask 计算 IoU loss，让几何分支先在干净信号下学到基本的"不重叠"先验；训练后期再切换成模型自己的 real_prob 软加权，让 IoU loss 开始真正约束生成质量。这是一种 curriculum 式的过渡---其实我更想用模型自己生成的去计算Iou Loss不然感觉很奇怪啊 就应该用它自己预测的结果去做
    # 另外 在推理阶段 也可以考虑将Iou Loss加入推理过程 但是模型参数是固定的 不过如果是一步一步生成的 到底能不能指导方向呢？
    # def get_iou_mask(self, real_prob, obj_mask, curr_step, warmup_steps):
    #     if curr_step < warmup_steps:
    #         return obj_mask.float()  # 训练初期用 GT，信号干净
    #     else:
    #         return real_prob  # 训练后期用模型自己的判断
    # def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None, lambda_iou=0.1):
    #     real_mask = obj_mask.bool()  # (B, N)  来自 GT

    #     # ── 1. 几何 loss：只对 real slot 计算 ──────────────
    #     geo_sq_err = (eps_pred[..., :6] - noise[..., :6]) ** 2  # (B, N, 6)
    #     geo_sq_err = geo_sq_err * real_mask.unsqueeze(-1).float()
    #     loss_geo = geo_sq_err.sum() / (real_mask.sum() * 6).clamp(min=1.0)

    #     # ── 2. 分类 loss：覆盖全部 slot（含 empty）──────────
    #     loss_class = F.mse_loss(eps_pred[..., 6:8], noise[..., 6:8])

    #     loss_sce = loss_geo + 3.0 * loss_class

    #     # ── 3. IoU loss：用 GT real_mask（训练阶段干净信号）──
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)

    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou = _pairwise_iou_loss_masked(bbox_pred, real_mask)
    #     loss_iou = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou

    #     return {
    #         "loss": total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_class": loss_class.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }

    # loss v8 修改使用生成的类别计算iou
    # def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None, lambda_iou=0.1):
    #     real_mask = obj_mask.bool()  # (B, N)  来自 GT

    #     # ── 1. 几何 loss：只对 real slot 计算 ──────────────
    #     geo_sq_err = (eps_pred[..., :6] - noise[..., :6]) ** 2  # (B, N, 6)
    #     geo_sq_err = geo_sq_err * real_mask.unsqueeze(-1).float()
    #     loss_geo = geo_sq_err.sum() / (real_mask.sum() * 6).clamp(min=1.0)

    #     # ── 2. 分类 loss：覆盖全部 slot（含 empty）──────────
    #     loss_class = F.mse_loss(eps_pred[..., 6:8], noise[..., 6:8])

    #     loss_sce = loss_geo + 3.0 * loss_class

    #     # ── 3. IoU loss：支队模型自己预测为real 的slot 计算Iou
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)
    #     cls_pred = x0_pred[..., 6:8]
    #     real_prob = torch.sigmoid(cls_pred[..., 0] - cls_pred[..., 1])
    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou = _pairwise_iou_loss_weighted(bbox_pred, real_prob)
    #     loss_iou = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     total = loss_sce + lambda_iou * loss_iou

    #     return {
    #         "loss": total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_class": loss_class.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }
    # loss v9
    # def compute_loss(self, x0, eps_pred, noise, t, x_t, obj_mask=None,
    #                 lambda_iou: float = 0.1, lambda_cls: float = 0.5):
    #     real_mask = obj_mask.bool()  # (B, N)  来自 GT

    #     B, N, _ = x0.shape

    #     # ── 1. 几何 loss：只对 real slot 计算（噪声空间 MSE）──────
    #     geo_sq_err = (eps_pred[..., :6] - noise[..., :6]) ** 2  # (B, N, 6)
    #     geo_sq_err = geo_sq_err * real_mask.unsqueeze(-1).float()
    #     loss_geo = geo_sq_err.sum() / (real_mask.sum() * 6).clamp(min=1.0)

    #     # ── 2. 分类 loss：在 x0 空间用 cross-entropy（覆盖全部 slot）──
    #     x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)  # (B, N, 8)

    #     # cls_logits: 直接取 x0_pred 的 class 维度作为 logits
    #     # 注意：这里不需要再做 sigmoid 差值技巧，cross_entropy 内部自带 softmax
    #     cls_logits = x0_pred[..., 6:8].reshape(B * N, 2)         # (B*N, 2)

    #     # GT label：x0 的 class one-hot（实际存的是 analog bits {-1,1} 或 {0,1}）
    #     # argmax 得到 0=real, 1=empty
    #     cls_labels = x0[..., 6:8].argmax(dim=-1).reshape(B * N)  # (B*N,)

    #     loss_cls = F.cross_entropy(cls_logits, cls_labels)

    #     # ── 3. 几何+分类 总噪声损失 ────────────────────────────────
    #     loss_sce = loss_geo  # 几何部分仍走噪声空间 MSE

    #     # ── 4. IoU loss：用模型预测为 real 的 soft 概率加权 ─────────
    #     bbox_pred = x0_pred[..., :4].clamp(0, 1)

    #     # 用 cross-entropy 的 softmax 概率作为 real_prob（比之前 sigmoid 差值更标准）
    #     real_prob = F.softmax(x0_pred[..., 6:8], dim=-1)[..., 0]   # (B, N)

    #     alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
    #     loss_iou = _pairwise_iou_loss_weighted(bbox_pred, real_prob.detach())  # detach，避免IoU影响分类
    #     loss_iou = (0.1 * alpha_bar.squeeze() * loss_iou).mean()

    #     total = loss_sce + lambda_cls * loss_cls + lambda_iou * loss_iou

    #     return {
    #         "loss":     total,
    #         "loss_geo": loss_geo.detach(),
    #         "loss_cls": loss_cls.detach(),
    #         "loss_iou": loss_iou.detach(),
    #     }
        
# iou v8 v9
# def _pairwise_iou_loss_weighted(bbox, real_prob):
#     """
#     bbox: (B, N, 4)  原始 bbox，不做概率缩放
#     real_prob: (B, N)  模型预测的 real 概率，用于加权惩罚强度
#     """
#     B, N, _ = bbox.shape
#     cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
#     x1, x2 = cx - w/2, cx + w/2
#     y1, y2 = cy - h/2, cy + h/2

#     x1_i, x2_i = x1.unsqueeze(2), x2.unsqueeze(2)
#     y1_i, y2_i = y1.unsqueeze(2), y2.unsqueeze(2)
#     x1_j, x2_j = x1.unsqueeze(1), x2.unsqueeze(1)
#     y1_j, y2_j = y1.unsqueeze(1), y2.unsqueeze(1)

#     inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(0)
#     inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(0)
#     inter = inter_w * inter_h

#     area_i = (w * h).unsqueeze(2)
#     area_j = (w * h).unsqueeze(1)
#     union = (area_i + area_j - inter).clamp(min=1e-6)
#     iou = inter / union  # (B, N, N)

#     # 用 real_prob 的成对乘积作为权重，而不是缩放 bbox
#     pair_weight = real_prob.unsqueeze(2) * real_prob.unsqueeze(1)  # (B, N, N)s

#     triu_mask = torch.triu(torch.ones(N, N, device=bbox.device), diagonal=1)
#     weighted_iou = iou * pair_weight * triu_mask.unsqueeze(0)
#     iou_sum = weighted_iou.sum(dim=[1, 2])

#     return iou_sum

# iou v7
# def _pairwise_iou_loss_masked(bbox, real_mask):
#     B, N, _ = bbox.shape
#     cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
#     x1, x2 = cx - w/2, cx + w/2
#     y1, y2 = cy - h/2, cy + h/2

#     x1_i, x2_i = x1.unsqueeze(2), x2.unsqueeze(2)
#     y1_i, y2_i = y1.unsqueeze(2), y2.unsqueeze(2)
#     x1_j, x2_j = x1.unsqueeze(1), x2.unsqueeze(1)
#     y1_j, y2_j = y1.unsqueeze(1), y2.unsqueeze(1)

#     inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(0)
#     inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(0)
#     inter = inter_w * inter_h

#     area_i = (w * h).unsqueeze(2)
#     area_j = (w * h).unsqueeze(1)
#     union = (area_i + area_j - inter).clamp(min=1e-6)
#     iou = inter / union

#     triu_mask = torch.triu(torch.ones(N, N, device=bbox.device), diagonal=1)
#     pair_mask = (real_mask.unsqueeze(2) & real_mask.unsqueeze(1)).float()
#     pair_mask = pair_mask * triu_mask.unsqueeze(0)

#     iou_sum = (iou * pair_mask).sum(dim=[1, 2])
#     return iou_sum


# v5
# def _pairwise_iou_loss(bbox: torch.Tensor,
#                        real_mask: torch.Tensor) -> torch.Tensor:
#     """
#     只对 real slot 之间计算成对 IoU 惩罚。

#     Args:
#         bbox:      (B, N, 4)  [cx, cy, w, h]，已将 empty slot 置零
#         real_mask: (B, N)     bool，True 表示该 slot 是真实建筑

#     Returns:
#         (B,)  每个样本的成对 IoU 之和
#     """
#     B, N, _ = bbox.shape

#     cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
#     x1 = cx - w / 2;  x2 = cx + w / 2
#     y1 = cy - h / 2;  y2 = cy + h / 2

#     x1_i = x1.unsqueeze(2);  x2_i = x2.unsqueeze(2)
#     y1_i = y1.unsqueeze(2);  y2_i = y2.unsqueeze(2)
#     x1_j = x1.unsqueeze(1);  x2_j = x2.unsqueeze(1)
#     y1_j = y1.unsqueeze(1);  y2_j = y2.unsqueeze(1)

#     inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(0)
#     inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(0)
#     inter   = inter_w * inter_h  # (B, N, N)

#     area_i = (w * h).unsqueeze(2)
#     area_j = (w * h).unsqueeze(1)
#     union  = (area_i + area_j - inter).clamp(min=1e-6)
#     iou    = inter / union  # (B, N, N)

#     # 只取 real-real 对的上三角
#     triu_mask = torch.triu(torch.ones(N, N, device=bbox.device, dtype=torch.bool),
#                            diagonal=1)  # (N, N)

#     # real_mask 广播：(B, N, 1) & (B, 1, N) → (B, N, N)
#     pair_mask = real_mask.unsqueeze(2) & real_mask.unsqueeze(1)  # (B, N, N)
#     pair_mask = pair_mask & triu_mask.unsqueeze(0)               # 叠加上三角

#     iou_sum = (iou * pair_mask.float()).sum(dim=[1, 2])  # (B,)
#     return iou_sum        


# ─────────────────────────────────────────────
# 工具：成对 IoU 损失（轴对齐版，快速）
# ─────────────────────────────────────────────
# # v1 v2 v6 v10 不能注释 推理的时候要用？
def _pairwise_iou_loss(bbox: torch.Tensor) -> torch.Tensor:
    """
    对轴对齐矩形框计算成对 IoU 并求和，作为惩罚项。

    Args:
        bbox: (B, N, 4)  [cx, cy, w, h]，归一化到 [0,1]

    Returns:
        (B,)  每个样本的成对 IoU 总和
    """
    B, N, _ = bbox.shape

    cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    x1 = cx - w / 2;  x2 = cx + w / 2
    y1 = cy - h / 2;  y2 = cy + h / 2  # (B, N)

    # 广播：(B, N, 1) vs (B, 1, N)
    x1_i = x1.unsqueeze(2);  x2_i = x2.unsqueeze(2)
    y1_i = y1.unsqueeze(2);  y2_i = y2.unsqueeze(2)
    x1_j = x1.unsqueeze(1);  x2_j = x2.unsqueeze(1)
    y1_j = y1.unsqueeze(1);  y2_j = y2.unsqueeze(1)

    inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(0)
    inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(0)
    inter = inter_w * inter_h                               # (B, N, N)

    area_i = (w * h).unsqueeze(2)
    area_j = (w * h).unsqueeze(1)
    union = (area_i + area_j - inter).clamp(min=1e-6)
    iou = inter / union                                     # (B, N, N)

    # 只取上三角（避免对角线自比较和重复计数）
    mask = torch.triu(torch.ones(N, N, device=bbox.device), diagonal=1)
    iou_sum = (iou * mask.unsqueeze(0)).sum(dim=[1, 2])     # (B,)

    return iou_sum

# v3
# def _pairwise_iou_loss(bbox: torch.Tensor, obj_mask: torch.Tensor) -> torch.Tensor:
#         """
#         计算有效物体之间的成对 IoU 损失。
        
#         Args:
#             bbox: (B, N, 4) [cx, cy, w, h] 归一化坐标
#             obj_mask: (B, N) 1=有效, 0=空置
#         Returns:
#             (B,) 每个样本中有效物体之间的 IoU 惩罚和
#         """
#         B, N, _ = bbox.shape

#         cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
#         x1, x2 = cx - w / 2, cx + w / 2
#         y1, y2 = cy - h / 2, cy + h / 2

#         # 广播计算两两之间的边界
#         x1_i, x2_i = x1.unsqueeze(2), x2.unsqueeze(2)
#         y1_i, y2_i = y1.unsqueeze(2), y2.unsqueeze(2)
#         x1_j, x2_j = x1.unsqueeze(1), x2.unsqueeze(1)
#         y1_j, y2_j = y1.unsqueeze(1), y2.unsqueeze(1)

#         inter_w = (torch.min(x2_i, x2_j) - torch.max(x1_i, x1_j)).clamp(min=0)
#         inter_h = (torch.min(y2_i, y2_j) - torch.max(y1_i, y1_j)).clamp(min=0)
#         inter = inter_w * inter_h

#         area_i = (w * h).unsqueeze(2)
#         area_j = (w * h).unsqueeze(1)
#         union = (area_i + area_j - inter).clamp(min=1e-6)
#         iou = inter / union  # (B, N, N)

#         # 构造掩码：只有两个 slot 都是有效目标时，才计算它们之间的 IoU 惩罚
#         # obj_mask: (B, N) -> valid_pair_mask: (B, N, N)
#         valid_pair_mask = obj_mask.unsqueeze(2) * obj_mask.unsqueeze(1)

#         # 排除对角线（自比较）并保留上三角
#         triu_mask = torch.triu(torch.ones(N, N, device=bbox.device), diagonal=1).unsqueeze(0)
#         final_mask = valid_pair_mask * triu_mask

#         iou_sum = (iou * final_mask).sum(dim=[1, 2])
#         return iou_sum

# # v4
# def _pairwise_iou_loss(
#     bbox: torch.Tensor,
#     obj_mask: torch.Tensor
# ):
#     """
#     bbox:
#         (B,N,4)
#         [cx,cy,w,h]

#     obj_mask:
#         (B,N)
#     """

#     B, N, _ = bbox.shape

#     cx = bbox[..., 0]
#     cy = bbox[..., 1]
#     w = bbox[..., 2]
#     h = bbox[..., 3]

#     x1 = cx - w / 2
#     x2 = cx + w / 2

#     y1 = cy - h / 2
#     y2 = cy + h / 2

#     x1_i = x1.unsqueeze(2)
#     x2_i = x2.unsqueeze(2)

#     y1_i = y1.unsqueeze(2)
#     y2_i = y2.unsqueeze(2)

#     x1_j = x1.unsqueeze(1)
#     x2_j = x2.unsqueeze(1)

#     y1_j = y1.unsqueeze(1)
#     y2_j = y2.unsqueeze(1)

#     inter_w = (
#         torch.min(x2_i, x2_j)
#         - torch.max(x1_i, x1_j)
#     ).clamp(min=0)

#     inter_h = (
#         torch.min(y2_i, y2_j)
#         - torch.max(y1_i, y1_j)
#     ).clamp(min=0)

#     inter = inter_w * inter_h

#     area_i = (w * h).unsqueeze(2)
#     area_j = (w * h).unsqueeze(1)

#     union = (
#         area_i + area_j - inter
#     ).clamp(min=1e-6)

#     iou = inter / union

#     valid_pair_mask = (
#         obj_mask.unsqueeze(2)
#         * obj_mask.unsqueeze(1)
#     )

#     triu_mask = torch.triu(
#         torch.ones(
#             N,
#             N,
#             device=bbox.device
#         ),
#         diagonal=1,
#     ).unsqueeze(0)

#     final_mask = valid_pair_mask * triu_mask

#     iou_sum = (
#         iou * final_mask
#     ).sum(dim=[1, 2])

#     return iou_sum



# v11
def pairwise_2d_iou(boxes):
    """
    计算 pairwise 2D IoU
    boxes: (B, N, 4) 格式为 (x_min, y_min, x_max, y_max)
    返回: (B, N, N) IoU 矩阵
    """
    B, N, _ = boxes.shape
    
    # 扩展维度以便广播
    boxes1 = boxes[:, :, None, :]  # (B, N, 1, 4)
    boxes2 = boxes[:, None, :, :]  # (B, 1, N, 4)
    
    # 计算交集
    inter_x_min = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y_min = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x_max = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y_max = torch.minimum(boxes1[..., 3], boxes2[..., 3])
    
    inter_width = (inter_x_max - inter_x_min).clamp(min=0)
    inter_height = (inter_y_max - inter_y_min).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # 计算各自面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # 计算 IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    return iou