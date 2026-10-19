"""
models/block_layout_model.py

将 BoundaryEncoder、TextEncoder、TransformerDenoiser、GaussianDiffusion
整合为一个可直接训练/推理的完整模型。

接口：
  model.forward(x0, boundary_pts, texts)  → loss_dict   (训练)
  model.sample(boundary_pts, texts, N)    → buildings   (推理)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .boundary_encoder import BoundaryEncoder, resample_polygon
from .text_encoder import TextEncoder
from .transformer_denoiser import TransformerDenoiser
from .diffusion import GaussianDiffusion


class BlockLayoutModel(nn.Module):
    """
    街区布局条件扩散生成模型。

    Args:
        cfg: DiffusionConfig 实例（来自 config.py）
        use_mock_text (bool): 测试时不加载 BERT，用随机 MLP 替代
    """

    def __init__(self, cfg, use_mock_text: bool = False):
        super().__init__()
        self.cfg = cfg

        # ── 条件编码器 ────────────────────────────────
        self.boundary_enc = BoundaryEncoder(
            boundary_points=cfg.boundary_points,
            hidden_dim=cfg.boundary_hidden,
            out_dim=cfg.d_model,      # 直接输出 d_model，无需额外投影---模型内部特征维度
        )
        self.text_enc = TextEncoder(
            model_name=cfg.text_model,
            out_dim=cfg.d_model,
            freeze=cfg.freeze_bert,
            use_mock=use_mock_text,
        )

        # ── 去噪器 ────────────────────────────────────
        self.denoiser = TransformerDenoiser(
            obj_dim=cfg.obj_dim,
            num_classes=cfg.num_classes,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ffn=cfg.dim_feedforward,
            time_emb_dim=cfg.time_emb_dim,
            dropout=cfg.dropout,
        )

        # ── 扩散过程 ──────────────────────────────────
        self.diffusion = GaussianDiffusion(
            T=cfg.T,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )

    # ─────────────────────────────────────────────
    # 编码条件（训练 & 推理共用）
    # ─────────────────────────────────────────────

    def encode_conditions(self, boundary_pts: torch.Tensor,
                          texts: List[str]):
        """
        编码边界和文本条件。

        Args:
            boundary_pts: (B, P_raw, 2)  原始边界顶点（归一化）
            texts:        List[str]      长度 B

        Returns:
            text_tokens: (B, L, d_model)
            text_mask:   (B, L)
            bnd_tokens:  (B, P, d_model)
        """
        device = boundary_pts.device

        # 重采样边界到固定点数
        pts = resample_polygon(boundary_pts, self.cfg.boundary_points)  # (B, P, 2)
        bnd_tokens, _ = self.boundary_enc(pts)                          # (B, P, d_model)

        # 文本编码
        text_tokens, _, text_mask = self.text_enc(texts, device)        # (B, L, d_model)

        return text_tokens, text_mask, bnd_tokens

    # ─────────────────────────────────────────────
    # 训练前向（计算损失）
    # ─────────────────────────────────────────────

    def forward(self, x0: torch.Tensor,
                boundary_pts: torch.Tensor,
                texts: List[str],
                obj_mask: Optional[torch.Tensor] = None) -> dict:
        """
        训练前向传播。

        Args:
            x0:           (B, N, obj_dim)  干净对象特征
                            前 6 维：[cx,cy,w,h,sinθ,cosθ]
                            前 7 维：[cx,cy,w,h,height,sin2θ,cos2θ]
                            后 num_classes 维：类别 one-hot
            boundary_pts: (B, P, 2)        边界顶点（归一化）
            texts:        List[str]         文本描述
            obj_mask:     (B, N)            有效建筑掩码（1=有效，0=padding）

        Returns:
            loss_dict: {
                "loss":     总损失
                "loss_sce": 噪声预测损失
                "loss_iou": IoU 惩罚损失
            }
        """
        B, N, D = x0.shape
        device = x0.device

        # ── 采样随机时步 ──────────────────────────────
        t = torch.randint(0, self.cfg.T, (B,), device=device)

        # ── 正向加噪 ──────────────────────────────────
        noise = torch.randn_like(x0)
        x_t, noise = self.diffusion.q_sample(x0, t, noise)

        # ── 编码条件 ──────────────────────────────────
        text_tokens, text_mask, bnd_tokens = self.encode_conditions(
            boundary_pts, texts)

        # ── 去噪预测 ──────────────────────────────────
        eps_pred = self.denoiser(
            x_t, t, text_tokens, text_mask,
            bnd_tokens, obj_mask)
            
        # ── 计算损失 ──────────────────────────────────
        return self.diffusion.compute_loss(
            x0, eps_pred, noise, t, x_t, obj_mask,
            lambda_iou=self.cfg.lambda_iou)

    # ─────────────────────────────────────────────
    # 推理采样
    # ─────────────────────────────────────────────

    def forward_with_text_cache(
            self,
            x0,
            boundary_pts,
            texts,
            pre_text_tokens,
            pre_text_mask,
            obj_mask=None):
        """
        预计算文本嵌入的快速训练路径：跳过 BERT forward，只做投影。
        pre_text_tokens: (B, L, 768)  来自 cache.pt
        pre_text_mask:   (B, L)
        """
        B, device = x0.shape[0], x0.device
        text_tokens = self.text_enc.proj(pre_text_tokens.to(device))
        text_mask   = pre_text_mask.to(device)
        pts = resample_polygon(boundary_pts, self.cfg.boundary_points)
        bnd_tokens, _ = self.boundary_enc(pts)
        t = torch.randint(0, self.cfg.T, (B,), device=device)
        noise = torch.randn_like(x0)
        x_t, noise = self.diffusion.q_sample(x0, t, noise)
        eps_pred = self.denoiser(x_t, t, text_tokens, text_mask, bnd_tokens, obj_mask = None) # 修改为none
        return self.diffusion.compute_loss(x0, eps_pred, noise, t, x_t, obj_mask,
                                           lambda_iou=self.cfg.lambda_iou) # 正常输入obj mask

    @torch.no_grad()
    def sample(self, boundary_pts: torch.Tensor,
               texts: List[str],
               n_buildings: Optional[int] = None) -> dict:
        """
        给定边界和文本，生成建筑布局。

        Args:
            boundary_pts: (B, P, 2)  边界顶点（归一化到 [0,1]）
            texts:        List[str]  文本描述（长度 B）
            n_buildings:  int        期望建筑数，默认用 cfg.max_buildings

        Returns:
            dict:
              "buildings": (B, N, obj_dim)  原始预测特征
              "bbox":      (B, N, 4)        [cx,cy,w,h] 归一化
              "angle_deg": (B, N)           朝向（度）
              "cls_id":    (B, N)           类别 ID（argmax）
              "cls_prob":  (B, N, num_classes) 类别概率（softmax）
        """
        device = boundary_pts.device
        B = boundary_pts.shape[0]
        N = n_buildings if n_buildings is not None else self.cfg.max_buildings

        # ── 编码条件 ──────────────────────────────────
        text_tokens, text_mask, bnd_tokens = self.encode_conditions(
            boundary_pts, texts)

        # ── DDPM 反向采样 ─────────────────────────────
        def denoiser_fn(x_t, t_batch, tt, tm, bt, om):
            return self.denoiser(x_t, t_batch, tt, tm, bt, om)

        shape = (B, N, self.cfg.obj_dim)
        x0_pred = self.diffusion.sample(
            denoiser_fn=lambda xt, t_batch, tt, tm, bt, om:
                self.denoiser(xt, t_batch, tt, tm, bt, om),
            shape=shape,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bnd_tokens=bnd_tokens,
            device=device,
        )

        # ── 后处理 ────────────────────────────────────
        return self._postprocess(x0_pred)
    
    
    # 通过文本缓存采样 新添加的 其他不变？
    @torch.no_grad()
    def sample_with_text_cache(self,
                                boundary_pts,
                                texts,
                                pre_text_tokens,
                                pre_text_mask):
        """
        推理时使用预计算文本嵌入（跳过 BERT），与 sample() 接口一致。
        pre_text_tokens: (B, L, 768)  来自 cache.pt（float16 或 float32）
        pre_text_mask:   (B, L)
        """
        from .boundary_encoder import resample_polygon
        device = boundary_pts.device
        B = boundary_pts.shape[0]
        N = self.cfg.max_buildings
 
        text_tokens = self.text_enc.proj(pre_text_tokens.float().to(device))
        text_mask   = pre_text_mask.float().to(device)
 
        pts = resample_polygon(boundary_pts, self.cfg.boundary_points)
        bnd_tokens, _ = self.boundary_enc(pts)
 
        shape = (B, N, self.cfg.obj_dim)
        x0_pred = self.diffusion.sample(
            denoiser_fn=lambda xt, t_batch, tt, tm, bt, om:
                self.denoiser(xt, t_batch, tt, tm, bt, om),
            shape=shape,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bnd_tokens=bnd_tokens,
            device=device,
        )
        return self._postprocess(x0_pred)

    def _postprocess(self, x0: torch.Tensor) -> dict:
        """
        将原始预测特征解析为可读结果。

        bbox 值裁到 [0,1]，角度从 sin/cos 或者sin2theta/cos2theta反算，类别取 argmax。
        """
        # bbox = x0[..., :4].clamp(0.0, 1.0)      # cx,cy,w,h
        # bbox = x0[..., :4]  # 不裁剪 只看原始数据
        # sin_t = x0[..., 4]
        # cos_t = x0[..., 5]
        bbox = x0[..., :5]
        sin_t = x0[..., 5]
        cos_t = x0[..., 6]
        angle_rad = torch.atan2(sin_t, cos_t) / 2.0  # 弧度 注意与数据预处理的s2theta保持一致
        angle_deg = angle_rad * (180.0 / 3.14159265)

        # 取出one hot编码
        cls_logit = x0[..., 7:]                  # (B, N, num_classes)
        cls_prob = torch.softmax(cls_logit, dim=-1)
        cls_id = cls_prob.argmax(dim=-1)         # (B, N)

        return {
            "buildings": x0,
            "bbox": bbox,
            "angle_deg": angle_deg,
            "cls_logit": cls_logit,
            "cls_id": cls_id,
            "cls_prob": cls_prob,
        }