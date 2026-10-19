"""
test_forward.py — 快速验证网络前向和反向是否正确运行。

不需要：
  - 真实数据集
  - BERT（用 mock 替代）
  - GPU

运行：
  cd block_layout
  python test_forward.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import CFG
from models.block_layout_model import BlockLayoutModel
from data.dataset import make_dummy_batch


def test_forward():
    print("=" * 60)
    print("BlockLayoutGen 快速前向验证")
    print("=" * 60)

    device = torch.device("cpu")
    cfg = CFG

    print(f"\n配置：")
    print(f"  max_buildings  = {cfg.max_buildings}")
    print(f"  num_classes    = {cfg.num_classes}")
    print(f"  obj_dim        = {cfg.obj_dim}")
    print(f"  d_model        = {cfg.d_model}")
    print(f"  num_layers     = {cfg.num_layers}")
    print(f"  boundary_points= {cfg.boundary_points}")

    # ── 构建模型 ──────────────────────────────────
    print("\n[1] 构建模型（mock_text=True，跳过 BERT）...")
    model = BlockLayoutModel(cfg, use_mock_text=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    可训练参数：{n_params:,}")

    # ── Dummy batch ───────────────────────────────
    print("\n[2] 生成 dummy batch（B=2）...")
    batch = make_dummy_batch(2, cfg, device)
    print(f"    boundary shape : {batch['boundary'].shape}")
    print(f"    x0 shape       : {batch['x0'].shape}")
    print(f"    obj_mask shape : {batch['obj_mask'].shape}")

    # ── 前向（训练）─────────────────────────────
    print("\n[3] 前向传播（训练模式）...")
    model.train()
    loss_dict = model(
        x0=batch["x0"],
        boundary_pts=batch["boundary"],
        texts=batch["text"],
        obj_mask=batch["obj_mask"],
    )
    print(f"    loss     = {loss_dict['loss'].item():.4f}")
    print(f"    loss_sce = {loss_dict['loss_sce'].item():.4f}")
    print(f"    loss_iou = {loss_dict['loss_iou'].item():.4f}")

    # ── 反向 ──────────────────────────────────────
    print("\n[4] 反向传播...")
    loss_dict["loss"].backward()
    # 检查梯度是否正常
    has_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    print(f"    有梯度的参数块数：{has_grad}")

    # ── 推理 ──────────────────────────────────────
    print("\n[5] 推理采样（T=1000 步，可能较慢...）")
    print("    注：生产中可用 DDIM T=100 加速")
    # 仅做 5 步快速测试
    original_T = model.diffusion.T
    # 替换为快速测试（不影响权重）
    import types

    @torch.no_grad()
    def fast_sample(self, denoiser_fn, shape, text_tokens, text_mask,
                    bnd_tokens, obj_mask=None, device=None):
        """仅跑 5 步（测试用）"""
        if device is None:
            device = text_tokens.device
        B, N, D = shape
        x_t = torch.randn(shape, device=device)
        for t in [4, 3, 2, 1, 0]:
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            eps_pred = denoiser_fn(x_t, t_batch,
                                   text_tokens, text_mask,
                                   bnd_tokens, obj_mask)
            x_t = self.p_sample_step(x_t, t, eps_pred)
        return x_t

    model.diffusion.sample = types.MethodType(fast_sample, model.diffusion)

    model.eval()
    result = model.sample(
        boundary_pts=batch["boundary"][:1],
        texts=batch["text"][:1],
        n_buildings=cfg.max_buildings,
    )
    print(f"    bbox shape     : {result['bbox'].shape}")
    print(f"    cls_id shape   : {result['cls_id'].shape}")
    print(f"    sample bbox[0,0]: cx={result['bbox'][0,0,0]:.3f}  "
          f"cy={result['bbox'][0,0,1]:.3f}  "
          f"w={result['bbox'][0,0,2]:.3f}  "
          f"h={result['bbox'][0,0,3]:.3f}")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_forward()
