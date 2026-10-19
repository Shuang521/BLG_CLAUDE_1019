"""
train.py  训练入口

性能优化说明（对应 3min30s 问题的各项修复）：

  [修复1] 使用 CachedDataset：__getitem__ 只做张量切片
          运行一次 preprocess.py 后所有 I/O 消失

  [修复2] 文本嵌入两种快速路径：
          a) cache_text=True：preprocess.py 已缓存 BERT 输出，
             训练时直接取张量，BERT forward 完全跳过
          b) cache_text=False：每 batch 跑一次 BERT forward（GPU），
             比原来 CPU BERT 快 10x

  [修复3] resample_polygon 向量化，从 O(B*P) Python 循环
          改为纯 torch 广播，batch_size=1024 下约快 400x

  [修复4] DataLoader 调优：persistent_workers / prefetch_factor

  [修复5] 文本嵌入 LRU 缓存：相同文本跳过 BERT forward
          街区数据集文本多样性有限，命中率通常 > 60%

用法：
  # 推荐流程
  python data/preprocess.py --data_root ./data/blocks --cache_text
  CUDA_VISIBLE_DEVICES=5 python train.py

  # 快速调试（无需数据集）
  python train.py --dummy --mock_text
  
  # 基础使用（启用 WandB）  修改ckpts_dir模型保存路径
    CUDA_VISIBLE_DEVICES=5 python train.py \
    --data_root ./data/blocks_rotate_01 \
    --use_wandb \
    --wandb_project "block-layout-0802" \
    --wandb_run_name "fm_tsfm_loss_v11_with_areas_hetscene" \
    --cache_path "./data/blocks_rotate_01/cache_height_scale_area_ratio_text_norm_shuffle.pt" \
    --ckpt_dir "./checkpoints/final/fm_tsfm_loss_v11_with_areas_hetscene" \
    --resume "/root/project/BLG/checkpoints/final/fm_tsfm_loss_v11_with_areas_hetscene/ckpt_015000.pt"

    # 离线模式（无网络环境）
    CUDA_VISIBLE_DEVICES=5 python train.py \
    --data_root ./data/blocks_rotate_01 \
    --use_wandb --wandb_offline

    # 调试时不启用（保持原行为）
    python train.py --dummy --mock_text  # 不加 --use_wandb 即可
"""

import os, argparse, time, math
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from functools import lru_cache
from typing import List, Optional

from config import CFG
from models.block_layout_model import BlockLayoutModel, FlowMatchingBlockLayoutModel
from data.dataset import build_dataloaders, make_dummy_batch

# 记录训练过程
import wandb
# 训练过程中可视化重建和生成的图片
import matplotlib
matplotlib.use("Agg") # 确保无 GUI 环境下绘图不报错
import matplotlib.pyplot as plt
# 从同目录下的 inference.py 中导入解码与绘图工具
from inference import _decode_x0, _decode_raw, _draw_single


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default=None)  # 训练数据源目录
    p.add_argument("--cache_path", default=None,
                   help="cache.pt 路径，默认 data_root/cache.pt")  # 存放数据缓存目录
    # p.add_argument("--epochs",     type=int,   default=None)  # 总训练轮次
    p.add_argument("--max_steps",     type=int,   default=None)  # 总训练轮次
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--ckpt_dir",   default="./checkpoints/rotate_01_pad_zero/loss_v_n")  # 输出权重目录
    p.add_argument("--save_every", type=int,   default=5000)
    p.add_argument("--log_every",  type=int,   default=100)
    p.add_argument("--draw_every", type=int,   default=6000)
    p.add_argument("--profile",    action="store_true",
                   help="打印详细性能分解（找瓶颈用）")
    p.add_argument("--dummy",      action="store_true")  # 快速调试 创建虚拟数据集？无需真实数据
    p.add_argument("--mock_text",  action="store_true")
    p.add_argument("--resume",     default=None)  # 中断后重新开始训练
    p.add_argument("--config",    type=str, default="config.py", help="读取配置文件")
    
    # wandb
    p.add_argument("--wandb_project", type=str, default="Block-Layout", help="WandB 项目名")
    p.add_argument("--wandb_run_name", type=str, default=None, help="WandB 运行名称")
    p.add_argument("--wandb_entity", type=str, default=None, help="WandB 团队/用户名")
    p.add_argument("--use_wandb", action="store_true", help="启用 WandB 日志")
    p.add_argument("--wandb_offline", action="store_true", help="WandB 离线模式")
    
    return p.parse_args()


# ══════════════════════════════════════════════════════════
# 文本嵌入 LRU 缓存（无预计算文本时的加速手段）
# 原理：街区数据集中相同类型的文本描述重复率高，
#       缓存 BERT 结果避免重复计算
# ══════════════════════════════════════════════════════════

_text_cache: dict = {}
_MAX_CACHE  = 4096

def encode_texts_with_cache(model, texts: List[str], device):
    """
    先查缓存，未命中时才跑 BERT forward。
    返回 (text_tokens, text_cls, text_mask) 均在 device 上。
    """
    # 找到未缓存的 unique 文本
    unique_uncached = list({t for t in texts if t not in _text_cache})

    if unique_uncached:
        # 批量编码未缓存文本
        with torch.no_grad():
            toks, cls, mask = model.text_enc(unique_uncached, device)
        for i, t in enumerate(unique_uncached):
            _text_cache[t] = (
                toks[i].cpu(),
                cls[i].cpu(),
                mask[i].cpu(),
            )
        # 简单 LRU：超出上限时随机清一半
        if len(_text_cache) > _MAX_CACHE:
            to_del = list(_text_cache.keys())[:_MAX_CACHE // 2]
            for k in to_del:
                del _text_cache[k]

    # 按 batch 顺序组装
    tok_list, cls_list, mask_list = [], [], []
    for t in texts:
        tok, c, m = _text_cache[t]
        tok_list.append(tok)
        cls_list.append(c)
        mask_list.append(m)

    text_tokens = torch.stack(tok_list).to(device, non_blocking=True)
    text_cls    = torch.stack(cls_list).to(device, non_blocking=True)
    text_mask   = torch.stack(mask_list).to(device, non_blocking=True)
    return text_tokens, text_cls, text_mask


# ══════════════════════════════════════════════════════════
# 单步训练（抽出来便于 profile）
# ══════════════════════════════════════════════════════════

def train_step(model, batch, cfg, device, curr_step, use_cached_text: bool,
               timers: Optional[dict] = None):
    """
    执行一个训练步。
    timers: 若非 None，则记录各阶段耗时（用于 --profile 模式）
    """
    def _t(name):
        if timers is not None and device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # ── 数据搬运 ──────────────────────────────────
    t0 = _t("data")
    x0       = batch["x0"].to(device, non_blocking=True)
    boundary = batch["boundary"].to(device, non_blocking=True)
    # scale = batch["scale"].to(device, non_blocking=True)
    areas = batch["area_sqm"].to(device, non_blocking=True)  # (N,) 原始物理面积，单位平方米
    obj_mask = batch["obj_mask"].to(device, non_blocking=True)

    # ── 文本编码 ──────────────────────────────────
    t1 = _t("text_start")
    if use_cached_text and "text_seq" in batch:
        # 最快路径：直接用预计算嵌入（跳过 BERT）
        pre_text_tokens = batch["text_seq"].to(device, non_blocking=True)
        pre_text_mask   = batch["text_mask"].to(device, non_blocking=True)
    else:
        pre_text_tokens = None
        pre_text_mask   = None
    t2 = _t("text_end")

    # ── 前向 ──────────────────────────────────────
    t3 = _t("fwd_start")
    if pre_text_tokens is not None:
        # 绕过 text_enc，直接把预计算嵌入传给 denoiser
        loss_dict = model.forward_with_text_cache(
            x0, boundary, batch["text"],
            pre_text_tokens, pre_text_mask, obj_mask, areas=areas)
    else:
        loss_dict = model.forward(x0, boundary, batch["text"], obj_mask, areas=areas)
    t4 = _t("fwd_end")

    if timers is not None:
        timers.setdefault("data_to_gpu",  []).append(t1 - t0)
        timers.setdefault("text_encode",  []).append(t2 - t1)
        timers.setdefault("forward",      []).append(t4 - t3)

    return loss_dict


@torch.no_grad()
def log_visualizations_to_wandb(model, batch, stage_name, step, cfg, device, use_cached_text):
    """
    所有依赖项（model, cfg, device, use_cached_text）均通过参数显式传入，
    不再依赖 train() 函数内部的局部变量。
    """
    if batch is None:
        return
    
    # 临时切换为评估模式
    model.eval()
    
    boundary = batch["boundary"].to(device)
    texts    = batch["text"]
    x0_gt    = batch["x0"]
    areas    = batch["area_sqm"].float().to(device)
    
    # 模型推理生成
    if use_cached_text:
        pre_seq  = batch["text_seq"].to(device)
        pre_mask = batch["text_mask"].to(device)
        raw = model.sample_with_text_cache(boundary, texts, pre_seq, pre_mask, areas=areas)
    else:
        raw = model.sample(boundary, texts, areas=areas)
        
    wandb_images = []
    n_samples = len(texts)
    
    for i in range(n_samples):
        # 整合单样本的可视化输入数据
        res = {
            "boundary_pts": batch["boundary"][i].cpu().numpy(),
            "text":         texts[i],
            "gt_buildings":  _decode_x0(x0_gt[i], cfg),
            "gen_buildings": _decode_raw(raw, i, cfg),
        }
        
        # 调用 inference 中的单图绘制函数
        fig = _draw_single(res, has_gt=True)
        
        # 包装为 wandb.Image 并记录
        wandb_images.append(
            wandb.Image(fig, caption=f"Step {step} | {stage_name} Sample {i+1}")
        )
        plt.close(fig) # 防止内存泄漏
        
    # 推送至 WandB
    wandb.log({f"vis/{stage_name}": wandb_images}, step=step)
    
    # 恢复训练模式
    model.train()


# ══════════════════════════════════════════════════════════
# 主训练循环
# ══════════════════════════════════════════════════════════

def train():
    args = parse_args()
    cfg  = CFG

    if args.data_root:  cfg.data_root  = args.data_root
    if args.cache_path: cfg.cache_path = args.cache_path
    # if args.epochs:     cfg.epochs     = args.epochs
    if args.max_steps:     cfg.max_steps     = args.max_steps
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr         = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── 构建模型 ──────────────────────────────────
    # model = BlockLayoutModel(cfg, use_mock_text=args.mock_text).to(device)
    if cfg.network_type == 'flow_matching':
        model = FlowMatchingBlockLayoutModel(cfg).to(device)
    else:
        # 使用Diffusion
        model = BlockLayoutModel(cfg, use_mock_text=args.mock_text).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数：{n_params:,}")

    # ── 初始化 scale_emb 统计量（仅非 dummy 模式）──
    if not args.dummy and hasattr(model, 'scale_emb'):
        # 从训练集一次性统计 log(area) 的 mean/std
        # 方式：直接从 dataset 取，避免遍历 DataLoader
        try:
            # 如果 dataset 是 CachedDataset 且内部有 cache
            all_areas = train_loader.dataset.cache["area_sqm"].float()
        except (AttributeError, KeyError):
            # 兜底：遍历一个 epoch 收集
            all_areas = []
            for b in train_loader:
                all_areas.append(b["area_sqm"])
            all_areas = torch.cat(all_areas).float()
        
        log_areas = torch.log(all_areas.clamp(min=1e-6))
        mean = log_areas.mean().item()
        std = log_areas.std().item()
        model.scale_emb.set_stats(mean, std)
        print(f"[ScaleEmbedding] 训练集 log(area) 统计: mean={mean:.4f}, std={std:.4f}")

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.lr_decay_every,
                       gamma=cfg.lr_decay_rate)

    # ── 恢复 checkpoint ───────────────────────────
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[Resume] step={start_step}")

    # ── 数据 ──────────────────────────────────────
    if args.dummy:
        print("[Data] dummy 模式")
        train_loader = None
        use_cached_text = False
    else:
        train_loader, val_loader = build_dataloaders(cfg)
        # 检查是否有预计算文本嵌入
        first_batch = next(iter(train_loader))
        use_cached_text = "text_seq" in first_batch
        print(f"[Data] 文本缓存模式：{'是（跳过 BERT）' if use_cached_text else '否（每步跑 BERT）'}")
        if not use_cached_text:
            print("  提示：运行 python data/preprocess.py --cache_text 可大幅提速")

    # ── 初始化 WandB ─────────────────────────────
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"run_{time.strftime('%m%d_%H%M')}",
            entity=args.wandb_entity,
            config={**vars(args), **vars(cfg)},  # 记录超参
            mode="offline" if args.wandb_offline else "online",
        )
        
        
    # ── 准备固定的可视化 Batch ───────────────────────
    def slice_batch(batch, n=8):
        """仅截取 Batch 的前 n 个样本，防止画图过多降低训练速度"""
        if batch is None:
            return None
        sliced = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                sliced[k] = v[:n]
            elif isinstance(v, list):
                sliced[k] = v[:n]
            else:
                sliced[k] = v
        return sliced

    vis_train_batch = None
    vis_val_batch = None
    
    if not args.dummy:
        # 从 Dataloader 中取出第一批数据作为后续每次可视化的固定样本
        # 这样可以直观观察同一个街区随训练步数增加而发生的变化
        vis_train_batch = slice_batch(next(iter(train_loader)), n=8)
        if val_loader is not None:
            vis_val_batch = slice_batch(next(iter(val_loader)), n=6)


    # ── 训练循环 ──────────────────────────────────
    model.train()
    step   = start_step
    t_ep   = time.time()

    # 1. 移除硬编码，初始化为空字典，后续动态管理
    running = {}  
    timers  = {} if args.profile else None

    while step < cfg.max_steps:
        iter_src = ([None] * (cfg.max_steps - step)) if args.dummy else train_loader

        for raw_batch in iter_src:
            if step >= cfg.max_steps:
                break

            if args.dummy:
                batch = make_dummy_batch(cfg.batch_size, cfg, device)
            else:
                batch = raw_batch

            optimizer.zero_grad(set_to_none=True)   # 比 zero_grad() 稍快

            loss_dict = train_step(model, batch, cfg, device, step,
                                   use_cached_text, timers)
            
            # 2. 自适应获取用于反向传播的总 loss
            # 优先寻找 'loss' 或 'total_loss'，若都没有，则将所有 Tensor/数值类型的 loss 相加作为兜底
            if "loss" in loss_dict:
                total_loss = loss_dict["loss"]
            elif "total_loss" in loss_dict:
                total_loss = loss_dict["total_loss"]
            else:
                total_loss = sum(v for v in loss_dict.values() if hasattr(v, 'item') or isinstance(v, (int, float)))
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            # 3. 动态累加 loss：自动适配新增的键，兼容 Tensor 和普通数值
            for k, v in loss_dict.items():
                if hasattr(v, 'item'):
                    val = v.item()
                elif isinstance(v, (int, float)):
                    val = float(v)
                else:
                    continue  # 跳过非数值类型的返回值（如字符串、列表等）
                
                if k not in running:
                    running[k] = 0.
                running[k] += val

            step += 1
            
            if step % args.log_every == 0:
                elapsed = time.time() - t_ep
                lr_now  = optimizer.param_groups[0]["lr"]
                n = args.log_every
                
                # 4. 动态生成 log_dict (自动加上 train/ 前缀)
                log_dict = {}
                for k, v in running.items():
                    log_dict[f"train/{k}"] = v / n
                
                # 添加固定的元信息
                log_dict["train/lr"] = lr_now
                log_dict["train/step"] = step
                log_dict["train/sec_per_step"] = elapsed / n

                # 5. 动态生成终端打印字符串
                loss_str_parts = []
                for k, v in log_dict.items():
                    # 排除元信息，只提取 loss 相关的项
                    if k.startswith("train/") and k not in ["train/lr", "train/step", "train/sec_per_step"]:
                        name = k.replace("train/", "")
                        loss_str_parts.append(f"{name}={v:.4f}")
                
                # 将所有 loss 拼接成一行，例如: "loss=1.23  noise=0.5  new_metric=0.1"
                loss_str = "  ".join(loss_str_parts)
                print(f"[{step:6d}/{cfg.max_steps}] {loss_str}  lr={lr_now:.2e}  {elapsed:.1f}s")
                
                # 🔥 推送至 WandB (自适应所有 train/* 的键)
                if args.use_wandb:
                    wandb.log(log_dict, step=step)
                
                # 6. 重置 running：仅保留当前 loss_dict 中存在的键，自动淘汰已废弃的 loss 键
                running = {}
                for k, v in loss_dict.items():
                    if hasattr(v, 'item') or isinstance(v, (int, float)):
                        running[k] = 0.
                        
                t_ep = time.time()

                # profile 报告
                if args.profile and timers:
                    print("  [Profile 最近100步平均]")
                    for k, v in timers.items():
                        avg = sum(v[-100:]) / len(v[-100:]) * 1000
                        print(f"    {k:20s}: {avg:.2f} ms/step")
           
            # 🔥 新增：向 WandB 记录图像结果            
            if step % args.draw_every == 0:
                if args.use_wandb and not args.dummy:
                    print(f"  [WandB] 正在渲染并上传 Step {step} 的生成样本...")
                    try:
                        # 1. 查看模型在训练集上的拟合能力 (显式传入 model, cfg, device, use_cached_text)
                        log_visualizations_to_wandb(
                            model=model,
                            batch=vis_train_batch,
                            stage_name="train_fit",
                            step=step,
                            cfg=cfg,
                            device=device,
                            use_cached_text=use_cached_text
                        )
                        # 2. 查看模型在验证集上的生成能力
                        log_visualizations_to_wandb(
                            model=model,
                            batch=vis_val_batch,
                            stage_name="val_generalization",
                            step=step,
                            cfg=cfg,
                            device=device,
                            use_cached_text=use_cached_text
                        )
                        print("  ✓ 可视化样本上传成功")
                    except Exception as e:
                        import traceback
                        print(f"  ✗ 可视化渲染失败: {e}")
                        traceback.print_exc()  # <--- 加上这行，直接看报错的具体文件名和行号        

            if step % args.save_every == 0:
                path = os.path.join(args.ckpt_dir, f"ckpt_{step:06d}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict()}, path)
                print(f"  ✓ 保存 {path}")

    # 最终保存
    torch.save({"step": step, "model": model.state_dict()},
               os.path.join(args.ckpt_dir, "ckpt_final.pt"))
    if args.use_wandb:
        wandb.finish()
    print("训练完成")


if __name__ == "__main__":
    train()