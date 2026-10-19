"""
inference.py — 验证集推理 + 可视化

用法---修改读取的路径 修改train和val可视化示意图：
  CUDA_VISIBLE_DEVICES=5 python inference.py \
    --ckpt checkpoints/rotate_01_pad_zero/loss_v6_mask_none/ckpt_final.pt \
    --cache_path ./data/blocks_rotate_01/cache_all_norm_cls_scale_2.pt \
    --n_samples 32 --batch_size 8 \
    --output_dir ./vis_output/rotate01_pad_zero/loss_v6_mask_none \
    --seed 629

  # 不重新推理，只重生成 HTML
  python inference.py --html_only --output_dir ./vis_output
"""

import os, sys, json, math, argparse, random, time
from typing import List, Optional, Dict

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from config import CFG
from models.block_layout_model import BlockLayoutModel
from data.dataset import CachedDataset


# ══════════════════════════════════════════════════════════
# 类别元信息
# ══════════════════════════════════════════════════════════

# CLASS_META = [
#     ("residential", "#4A90D9", "#2171B5"),
#     ("commercial",  "#E8622A", "#C0392B"),
#     ("office",      "#8E44AD", "#6C3483"),
#     ("public",      "#27AE60", "#1E8449"),
#     ("green",       "#A8D5A2", "#52A447"),
#     ("parking",     "#95A5A6", "#717D7E"),
#     ("mixed",       "#F39C12", "#D68910"),
#     ("empty",       "#ECF0F1", "#BDC3C7"),
# ]

# 两分类类别
CLASS_META = [
    ("residential", "#F39C12", "#D68910"),
    ("empty",       "#ECF0F1", "#BDC3C7"),
]
CLASS_NAMES  = [m[0] for m in CLASS_META]
CLASS_FCOLOR = [m[1] for m in CLASS_META]
CLASS_ECOLOR = [m[2] for m in CLASS_META]
EMPTY_CLS    = len(CLASS_META) - 1
RES_CLS = 0

# 定义为 Tensor 常量
# BBOX_MEAN = torch.tensor([0.5024, 0.1972, 0.1103, 0.0517])
# BBOX_STD  = torch.tensor([0.2828, 0.1439, 0.0789, 0.0481])

# 前9000个样本
# BBOX_MEAN = torch.tensor([0.5022, 0.1959, 0.1077, 0.0501])
# BBOX_STD = torch.tensor([0.2826, 0.1424, 0.0766, 0.0464])

# 所有样本
BBOX_MEAN = torch.tensor([0.5026, 0.1996, 0.0853, 0.0419, 0.2691])
BBOX_STD = torch.tensor([0.2783, 0.1451, 0.0605, 0.0339, 0.1370])

# 两个样本
# BBOX_MEAN = torch.tensor([0.8004, 0.1264, 0.1466, 0.0772, 0.7313])
# BBOX_STD = torch.tensor([0.1670, 0.0690, 0.1000, 0.0572, 0.2738])

# 5个样本
# BBOX_MEAN = torch.tensor([0.7079, 0.1198, 0.1622, 0.0987, 0.8301])
# BBOX_STD = torch.tensor([0.2492, 0.0617, 0.1004, 0.0655, 0.2579])

# 17个样本
# BBOX_MEAN = torch.tensor([0.5621, 0.1211, 0.1628, 0.1013, 0.8293])
# BBOX_STD = torch.tensor([0.2792, 0.0613, 0.0833, 0.0611, 0.2166])

# 添加height相关的标准差

# ══════════════════════════════════════════════════════════
# 推理引擎
# ══════════════════════════════════════════════════════════

class BlockLayoutInference:

    def __init__(self, ckpt_path=None, bert_model=None,
                 device="auto", mock_text=False):
        self.device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       if device == "auto" else torch.device(device))
        cfg = CFG
        if bert_model:
            cfg.text_model = bert_model
        self.cfg = cfg

        self.model = BlockLayoutModel(cfg, use_mock_text=mock_text)
        self.model.to(self.device)

        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", "?")
            print(f"[Inference] 已加载权重：{ckpt_path}  (step={step})")
        else:
            print("[Inference] 未提供 checkpoint，使用随机权重（仅调试）")

        self.model.eval()

    @torch.no_grad()
    def run_from_cache(self, cache_path, n_samples=16,
                       batch_size=8, seed=42):
        # dataset = CachedDataset(cache_path, split="val",
                                # train_ratio=self.cfg.train_split)
        dataset = CachedDataset(cache_path, split="train",
                                train_ratio=self.cfg.train_split)
        n_total = len(dataset)
        n_samples = min(n_samples, n_total)
        random.seed(seed)
        indices = random.sample(range(n_total), n_samples)
        # print(f"[Inference] 验证集共 {n_total} 样本，抽取 {n_samples} 个")
        print(f"[Inference] 训练集共 {n_total} 样本，抽取 {n_samples} 个")

        results = []
        for start in range(0, n_samples, batch_size):
            batch = [dataset[i] for i in indices[start:start+batch_size]]
            results.extend(self._run_batch(batch))
            print(f"  已完成 {min(start+batch_size, n_samples)}/{n_samples}")
        return results

    # def _run_batch(self, batch):
    #     B      = len(batch)
    #     device = self.device
    #     boundary = torch.stack([item["boundary"] for item in batch]).to(device)
    #     texts    = [item["text"] for item in batch]
    #     x0_gt    = torch.stack([item["x0"] for item in batch])

    #     has_cached = "text_seq" in batch[0]
    #     with torch.no_grad():
    #         if has_cached:
    #             pre_seq  = torch.stack([item["text_seq"]  for item in batch]).to(device)
    #             pre_mask = torch.stack([item["text_mask"] for item in batch]).to(device)
    #             raw = self.model.sample_with_text_cache(
    #                 boundary, texts, pre_seq, pre_mask)
    #         else:
    #             raw = self.model.sample(boundary, texts)

    #     # print("第一类别", raw["cls_prob"][0, :, RES_CLS].cpu().numpy())  # 打印居住类别概率
    #     # print("空类别", raw["cls_prob"][0, :, EMPTY_CLS].cpu().numpy())  # 打印空类别概率
    #     print("所有类别的概率", raw["cls_prob"][0, :, :].cpu().numpy())
        
    #     results = []
    #     for b in range(B):
    #         results.append({
    #             "boundary_pts":  batch[b]["boundary"].numpy(),
    #             "text":          texts[b],
    #             "gt_buildings":  _decode_x0(x0_gt[b], self.cfg),
    #             "gen_buildings": _decode_raw(raw, b, self.cfg),
    #         })
    #     return results
    def _run_batch(self, batch):
        B      = len(batch)
        device = self.device
        boundary = torch.stack([item["boundary"] for item in batch]).to(device)
        texts    = [item["text"] for item in batch]
        x0_gt    = torch.stack([item["x0"] for item in batch])

        has_cached = "text_seq" in batch[0]
        with torch.no_grad():
            if has_cached:
                pre_seq  = torch.stack([item["text_seq"]  for item in batch]).to(device)
                pre_mask = torch.stack([item["text_mask"] for item in batch]).to(device)
                raw = self.model.sample_with_text_cache(
                    boundary, texts, pre_seq, pre_mask)
            else:
                raw = self.model.sample(boundary, texts)

        # ══ 新增：GT vs Gen 类别分布对比 ══
        # GT 的类别统计（取第0个样本）
        gt_cls_logits = x0_gt[0, :, 7:]   # shape: [N_slots, 8]
        # print("gt_cls_logits:", gt_cls_logits) # [-3,3]或者[3,-3]
        gt_cls_ids = gt_cls_logits.argmax(dim=-1)  # [N_slots]
        print("gt_cls_ids:", gt_cls_ids)
        gt_valid_mask = gt_cls_logits.max(dim=-1).values > 0.3  # 过滤低置信度

        print("\n── GT 类别分布（样本0） ──")
        for cid in range(gt_cls_logits.shape[-1]):
            count = (gt_cls_ids == cid).sum().item()
            count_valid = ((gt_cls_ids == cid) & gt_valid_mask).sum().item()
            print(f"  类别{cid} ({CLASS_NAMES[cid]:12s}): "
                f"总{count:3d}  有效(conf>0.3): {count_valid:3d}")

        print("\n── Gen 类别分布（样本0） ──")
        gen_cls_logits = raw["cls_logit"][0] # [N_slots, 8] ?
        # print("gen_cls_logits:", gen_cls_logits)
        gen_cls_ids = raw["cls_id"][0]   # [N_slots]
        print("gen_cls_ids:", gen_cls_ids)
        gen_cls_probs = raw["cls_prob"][0]  # [N_slots, 8]
        for cid in range(gen_cls_probs.shape[-1]):
            count = (gen_cls_ids == cid).sum().item()
            avg_prob = gen_cls_probs[:, cid].mean().item()
            print(f"  类别{cid} ({CLASS_NAMES[cid]:12s}): "
                f"预测为该类: {count:3d}  平均概率: {avg_prob:.4f}")
        
        # 新添加过拟合检测  看模型对单个训练样本重建时cls_logits的方差和判别度
        # 【修改点】：加上 .cpu() 将两者统一转移到 CPU 上进行统计计算
        gt_is_real = (gt_cls_ids.cpu() == 0)         # GT 里真正是 real 的位置
        pred_is_real = (gen_cls_ids.cpu() == 0)      # 模型预测为 real 的位置（按相同映射）

        match = (pred_is_real == gt_is_real).float().mean()
        print("修正后的 Per-slot match rate:", match.item())

        # 同时输出混淆矩阵，更直观
        tp = (pred_is_real & gt_is_real).sum().item()   # 预测real，GT也是real
        tn = (~pred_is_real & ~gt_is_real).sum().item() # 预测empty，GT也是empty
        fp = (pred_is_real & ~gt_is_real).sum().item()  # 预测real，GT是empty（误报）
        fn = (~pred_is_real & gt_is_real).sum().item()  # 预测empty，GT是real（漏报）
        print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
            
        

        print("\n── Gen 原始概率统计（样本0，前5个slot） ──")
        print(raw["cls_prob"][0, :5, :].cpu().numpy())
        # ══ 新增结束 ══

        results = []
        for b in range(B):
            results.append({
                "boundary_pts":  batch[b]["boundary"].numpy(),
                "text":          texts[b],
                "gt_buildings":  _decode_x0(x0_gt[b], self.cfg),
                "gen_buildings": _decode_raw(raw, b, self.cfg),
            })
        return results


# ══════════════════════════════════════════════════════════
# 解码工具
# ══════════════════════════════════════════════════════════

# def _decode_x0(x0: torch.Tensor, cfg) -> List[Dict]:
#     out = []
    
#     # n_empty = (x0[:, :, 6:].argmax(-1) == EMPTY_CLS).float().mean()
#     # print(f"GT empty 比例: {n_empty:.3f}")
    
#     for i in range(x0.shape[0]):
#         obj = x0[i]
#         cls_logit = obj[6:]
#         if cls_logit.max().item() < 0.3:
#             continue
#         cls_id = int(cls_logit.argmax().item())
#         if cls_id == EMPTY_CLS:
#             continue
#         out.append({
#             "cx": float(obj[0]), "cy": float(obj[1]),
#             "w":  float(obj[2]), "h":  float(obj[3]),
#             "angle": math.degrees(math.atan2(float(obj[4]), float(obj[5]))),
#             "cls_id":   cls_id,
#             "cls_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
#         })
#     return out

# 进行反归一化
def _decode_x0(x0: torch.Tensor, cfg) -> List[Dict]:
    out = []
    
    # 1. 提取前5维 (cx, cy, w, h,height) 进行批量反归一化
    bboxes_norm = x0[:, :5]
    device = bboxes_norm.device
    
    # 将 mean 和 std 移动到与 x0 相同的设备 (CPU/GPU)
    bbox_mean = BBOX_MEAN.to(device)
    bbox_std = BBOX_STD.to(device)
     
    # 反归一化: x = x_norm * std + mean
    bboxes = bboxes_norm * bbox_std + bbox_mean
    
    # 2. 遍历解析
    for i in range(x0.shape[0]):
        obj = x0[i]
        bbox = bboxes[i]  # 使用反归一化后的 bbox
        
        cls_logit = obj[7:]
        if cls_logit.max().item() < 0.3:
            continue
        cls_id = int(cls_logit.argmax().item())
        if cls_id == EMPTY_CLS:
            continue
            
        out.append({
            "cx": float(bbox[0]), "cy": float(bbox[1]),
            "w":  float(bbox[2]), "h":  float(bbox[3]), "height":  float(bbox[4]),
            # 角度部分 (sin2theta, cos2theta) 未做归一化，保持原样计算,但需要除以2
            "angle": math.degrees(math.atan2(float(obj[5]), float(obj[6])) / 2.0),
            "cls_id":   cls_id,
            "cls_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
        })
    return out


# def _decode_raw(raw: Dict, b: int, cfg) -> List[Dict]:
#     out = []
#     for i in range(raw["bbox"].shape[1]):
#         cls_id = int(raw["cls_id"][b, i].item())
#         if cls_id == EMPTY_CLS:
#             continue
#         bbox  = raw["bbox"][b, i]
#         out.append({
#             "cx": float(bbox[0]), "cy": float(bbox[1]),
#             "w":  float(bbox[2]), "h":  float(bbox[3]),
#             "angle":    float(raw["angle_deg"][b, i].item()),
#             "cls_id":   cls_id,
#             "cls_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
#             "prob":     float(raw["cls_prob"][b, i, cls_id].item()),
#         })
#     return out

# 附带反标准化--归一化
def _decode_raw(raw: Dict, b: int, cfg) -> List[Dict]:
    out = []
    
    # 1. 获取当前 batch 的标准化 bbox (shape: [N, 4])
    bboxes_norm = raw["bbox"][b] 
    
    # 2. 将 mean 和 std 移动到与 bbox 相同的设备 (CPU/GPU)
    device = bboxes_norm.device
    bbox_mean = BBOX_MEAN.to(device)
    bbox_std = BBOX_STD.to(device)
    
    # 3. 批量反标准化 (利用广播机制，一次性算完所有框)
    bboxes = bboxes_norm * bbox_std + bbox_mean
    
    # 4. 遍历解析
    for i in range(bboxes.shape[0]):
        cls_id = int(raw["cls_id"][b, i].item())
        if cls_id == EMPTY_CLS:
            continue
            
        bbox = bboxes[i] # 此时已经是反标准化后的数据
        
        out.append({
            "cx": float(bbox[0]), "cy": float(bbox[1]),
            "w":  float(bbox[2]), "h":  float(bbox[3]), "height":  float(bbox[4]),
            "angle":    float(raw["angle_deg"][b, i].item()),
            "cls_id":   cls_id,
            "cls_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
            "prob":     float(raw["cls_prob"][b, i, cls_id].item()),
        })
    return out


# ══════════════════════════════════════════════════════════
# 可视化核心（全批量化，无逐个 add_patch）
# ══════════════════════════════════════════════════════════

def _rotated_corners(cx, cy, w, h, angle_deg):
    a  = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    corners = np.array([[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]], dtype=np.float32)
    rot = np.array([[ca,-sa],[sa,ca]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


# def draw_block_fast(ax, boundary_pts, buildings, title="", show_label=True):
#     """
#     批量绘制一个街区：
#     - 边界和所有建筑分别用单个 PatchCollection，只调用一次 add_collection
#     - 不用 tight_layout，调用方负责 subplots_adjust
#     """
#     ax.set_xlim(-0.02, 1.02)
#     ax.set_ylim(-0.02, 1.02)
#     # ax.set_xlim(-1.02, 1.02)
#     # ax.set_ylim(-1.02, 1.02)
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_facecolor("#F7F7F4")

#     # ── 边界 ──────────────────────────────────────
#     if len(boundary_pts) >= 3:
#         bpoly = MplPolygon(boundary_pts, closed=True)
#         pc_bnd = PatchCollection([bpoly],
#                                  facecolors=["#EEEEE8"],
#                                  edgecolors=["#222222"],
#                                  linewidths=[1.8], zorder=1)
#         ax.add_collection(pc_bnd)

#     # ── 建筑（按类别批量）────────────────────────
#     if buildings:
#         polys, fcolors, ecolors = [], [], []
#         label_info = []   # (cx, cy, label_str)

#         for bld in buildings:
#             cid = bld["cls_id"]
#             if cid == EMPTY_CLS:
#                 continue
#             corners = _rotated_corners(
#                 bld["cx"], bld["cy"], bld["w"], bld["h"], bld["angle"])
#             polys.append(MplPolygon(corners, closed=True))
#             fcolors.append(CLASS_FCOLOR[cid] if cid < len(CLASS_FCOLOR) else "#AAAAAA")
#             ecolors.append(CLASS_ECOLOR[cid] if cid < len(CLASS_ECOLOR) else "#666666")

#             if show_label and bld["w"] > 0.05 and bld["h"] > 0.05:
#                 lbl = CLASS_NAMES[cid][0].upper() if cid < len(CLASS_NAMES) else "?"
#                 label_info.append((bld["cx"], bld["cy"], lbl))

#         if polys:
#             pc = PatchCollection(polys,
#                                  facecolors=fcolors,
#                                  edgecolors=ecolors,
#                                  linewidths=0.7,
#                                  alpha=0.88,
#                                  zorder=2)
#             ax.add_collection(pc)

#         # 标签（文字量少，逐个写没关系）
#         if show_label:
#             for cx_l, cy_l, lbl in label_info:
#                 ax.text(cx_l, cy_l, lbl,
#                         ha="center", va="center",
#                         fontsize=4.5, color="white",
#                         fontweight="bold", zorder=3,
#                         path_effects=[
#                             pe.withStroke(linewidth=0.8, foreground="black")])

#     n_bld = len([b for b in buildings if b["cls_id"] != EMPTY_CLS])
#     ax.set_title(f"{title}  [{n_bld}]", fontsize=7, pad=2, fontweight="bold")


def draw_block_fast(ax, boundary_pts, buildings, title="", show_label=True, 
                    cmap_name="Blues", vmin=None, vmax=None):
    """
    批量绘制一个街区（支持建筑高度热力图渲染）：
    - 边界和所有建筑分别用单个 PatchCollection 绘制
    - 建筑颜色基于其 height 字段，通过指定的 Colormap 动态映射
    - 在子图侧边添加表示高度的 Colorbar
    """
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F7F7F4")

    # ── 1. 绘制边界 ──────────────────────────────────────
    if len(boundary_pts) >= 3:
        bpoly = MplPolygon(boundary_pts, closed=True)
        pc_bnd = PatchCollection([bpoly],
                                 facecolors=["#EEEEE8"],
                                 edgecolors=["#666666"], # 适当调浅了边界颜色
                                 linewidths=[0.8], zorder=1)
        ax.add_collection(pc_bnd)

    # ── 2. 绘制建筑（高度热力图渲染）────────────────────────
    if buildings:
        valid_buildings = [b for b in buildings if b["cls_id"] != EMPTY_CLS]
        
        if valid_buildings:
            # 动态获取高度的上下限
            heights = [bld.get("height", 0) for bld in valid_buildings]
        #     if vmin is None:
        #         vmin = min(heights) if heights else 10
        #     if vmax is None:
        #         vmax = max(heights) if heights else 70
        #     if vmin == vmax:
        #         vmin, vmax = vmin - 1, vmin + 1

            # 创建归一化和颜色映射器
            # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            norm = mcolors.Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap(cmap_name)

            polys, fcolors, ecolors = [], [], []
            label_info = []

            for bld in valid_buildings:
                corners = _rotated_corners(
                    bld["cx"], bld["cy"], bld["w"], bld["h"], bld["angle"])
                polys.append(MplPolygon(corners, closed=True))
                
                # 2.1 根据高度计算填充颜色
                h_val = bld.get("height", 0)
                # color = cmap(norm(h_val))
                color = cmap(h_val) # 不用归一化 这里的height已经是[0.1]区间内的了应该
                fcolors.append(color)
                
                # 2.2 边框颜色（如参考图所示，浅色/淡蓝色的细边框效果较好）
                ecolors.append("#B0C4DE") 

                # 若需要显示标签（可选显示类别首字母或高度数值）
                if show_label and bld["w"] > 0.05 and bld["h"] > 0.05:
                    # 默认显示高度，也可以改回类别
                    lbl = f"{int(h_val)}" 
                    label_info.append((bld["cx"], bld["cy"], lbl))

            # 统一添加到 Collection 中
            if polys:
                pc = PatchCollection(polys,
                                     facecolors=fcolors,
                                     edgecolors=ecolors,
                                     linewidths=0.5, # 细边框
                                     alpha=0.9,
                                     zorder=2)
                ax.add_collection(pc)

            # ── 3. 添加颜色条 (Colorbar) ────────────────────────
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig = ax.figure
            
            # 调整 fraction 和 pad 参数，防止颜色条挤压主图变形
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Height (m)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            # ── 4. 绘制标签文本 ──────────────────────────────────
            if show_label:
                for cx_l, cy_l, lbl in label_info:
                    ax.text(cx_l, cy_l, lbl,
                            ha="center", va="center",
                            fontsize=4.5, color="white",
                            fontweight="bold", zorder=3,
                            path_effects=[
                                pe.withStroke(linewidth=0.8, foreground="black")])

    n_bld = len([b for b in buildings if b["cls_id"] != EMPTY_CLS])
    ax.set_title(f"{title}  [{n_bld}]", fontsize=7, pad=2, fontweight="bold")



def _draw_legend(ax):
    ax.axis("off")
    ax.set_title("Legend", fontsize=7, fontweight="bold", pad=2)
    for i, (name, fc, ec) in enumerate(CLASS_META[:-1]):   # 跳过 empty
        y = 0.93 - i * 0.115
        rect = plt.Rectangle((0.03, y - 0.05), 0.18, 0.08,
                              facecolor=fc, edgecolor=ec, linewidth=0.6,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.27, y - 0.01, name.capitalize(),
                transform=ax.transAxes, fontsize=6.5, va="center")


# ── 单样本图 ──────────────────────────────────────────────

def _draw_single(res, has_gt, show_label=True):
    """
    绘制单个街区的对比图，返回 Figure。
    不调用 tight_layout，改用 subplots_adjust。
    """
    n_data_cols = 2 if has_gt else 1
    fig_w = 7.2 if has_gt else 4.2
    ratios = ([1, 1, 0.28] if has_gt else [1, 0.28])

    fig, axes = plt.subplots(1, n_data_cols + 1,
                             figsize=(fig_w, 3.6),
                             gridspec_kw={"width_ratios": ratios})
    fig.patch.set_facecolor("#FAFAFA")

    bpts = res["boundary_pts"]
    gt   = res.get("gt_buildings", [])
    gen  = res["gen_buildings"]

    if has_gt:
        draw_block_fast(axes[0], bpts, gt,  title="Ground Truth", show_label=show_label)
        draw_block_fast(axes[1], bpts, gen, title="Generated",    show_label=show_label)
        _draw_legend(axes[2])
    else:
        draw_block_fast(axes[0], bpts, gen, title="Generated",    show_label=show_label)
        _draw_legend(axes[1])

    # 文本提示（截断到固定长度，不用 wrap=True）
    prompt = res["text"]
    if len(prompt) > 100:
        prompt = prompt[:97] + "..."
    fig.text(0.5, 0.01, f'"{prompt}"',
             ha="center", fontsize=6, style="italic",
             color="#555555")

    # 用 subplots_adjust 代替 tight_layout（快 270x）
    fig.subplots_adjust(left=0.01, right=0.99,
                        top=0.93, bottom=0.08,
                        wspace=0.06)
    return fig


# ── 汇总拼图 ──────────────────────────────────────────────

def _draw_summary(results, has_gt, n_cols=4):
    n = len(results)
    n_rows = math.ceil(n / n_cols)
    data_cols = 2 if has_gt else 1
    total_cols = n_cols * data_cols + 1   # +1 legend

    cell_w = 2.4
    cell_h = 2.6
    fig_w  = total_cols * cell_w
    fig_h  = n_rows * cell_h

    ratios = ([data_cols] * n_cols + [0.4])   # legend 窄列

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#F0F0ED")
    gs  = fig.add_gridspec(n_rows, total_cols,
                           width_ratios=(([1, 1] if has_gt else [1]) * n_cols + [0.32]),
                           hspace=0.35, wspace=0.05,
                           left=0.01, right=0.99,
                           top=0.97, bottom=0.02)

    for idx, res in enumerate(results):
        row      = idx // n_cols
        base_col = (idx % n_cols) * data_cols
        bpts = res["boundary_pts"]
        gt   = res.get("gt_buildings", [])
        gen  = res["gen_buildings"]

        if has_gt:
            ax_gt  = fig.add_subplot(gs[row, base_col])
            ax_gen = fig.add_subplot(gs[row, base_col + 1])
            draw_block_fast(ax_gt,  bpts, gt,  title=f"#{idx+1} GT",  show_label=False)
            draw_block_fast(ax_gen, bpts, gen, title=f"#{idx+1} Gen", show_label=False)
        else:
            ax_gen = fig.add_subplot(gs[row, base_col])
            draw_block_fast(ax_gen, bpts, gen, title=f"#{idx+1}", show_label=False)

    # 关掉多余格子
    used = n * data_cols
    total_cells = n_rows * n_cols * data_cols
    for i in range(used, total_cells):
        row = i // (n_cols * data_cols)
        col = i %  (n_cols * data_cols)
        if row < n_rows and col < n_cols * data_cols:
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

    # 图例（右侧最后列，只在第一行画）
    ax_leg = fig.add_subplot(gs[0, total_cols - 1])
    _draw_legend(ax_leg)
    for row in range(1, n_rows):
        ax_off = fig.add_subplot(gs[row, total_cols - 1])
        ax_off.axis("off")

    return fig


# ══════════════════════════════════════════════════════════
# 入口：可视化 + HTML
# ══════════════════════════════════════════════════════════

def visualize_results(results, output_dir, n_cols=4,
                      show_gt=True, dpi=120):
    os.makedirs(output_dir, exist_ok=True)
    # img_dir = os.path.join(output_dir, "imgs")
    # os.makedirs(img_dir, exist_ok=True)
    img_dir = output_dir

    has_gt = show_gt and any(r.get("gt_buildings") for r in results)
    n = len(results) 
    print(f"[Viz] 绘制 {n} 张对比图  dpi={dpi}  has_gt={has_gt}")
    t0 = time.time()

    individual_paths = []   
    
    for idx, res in enumerate(results):
        
        # 临时诊断代码
        gt = res.get("gt_buildings", [])
        if len(gt) > 0:
            print(f"\n[DEBUG] 样本 {idx+1} 的第一个 GT 建筑原始解码值:")
            print(f"  cx: {gt[0]['cx']:.4f}, cy: {gt[0]['cy']:.4f}")
            print(f"  w:  {gt[0]['w']:.4f}, h:  {gt[0]['h']:.4f}")
            print(f"  angle: {gt[0]['angle']:.2f}, cls_id: {gt[0]['cls_id']}")
            break
        
        fig = _draw_single(res, has_gt)
        path = os.path.join(img_dir, f"sample_{idx+1:04d}.png")
        fig.savefig(path, dpi=dpi, bbox_inches=None,  # bbox_inches=None 比 tight 快很多
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        individual_paths.append(path)
        if (idx + 1) % 8 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{n}  ({elapsed:.1f}s, {elapsed/(idx+1):.2f}s/图)")

    # 汇总拼图
    print("[Viz] 绘制汇总拼图...")
    fig_sum = _draw_summary(results, has_gt, n_cols)
    summary_path = os.path.join(img_dir, "summary_grid.png")
    fig_sum.savefig(summary_path, dpi=dpi, bbox_inches=None,
                    facecolor=fig_sum.get_facecolor())
    plt.close(fig_sum)

    total = time.time() - t0
    print(f"[Viz] 绘图完成，共 {total:.1f}s，均 {total/n:.2f}s/张")

    # HTML（引用文件路径，不 base64 内嵌）
    html_path = _make_html(results, individual_paths,
                           summary_path, output_dir, has_gt)
    print(f"[Viz] HTML 报告：{html_path}")
    return html_path


# ── HTML（引用本地路径，不内嵌 base64）──────────────────

def _make_html(results, individual_paths, summary_path, output_dir, has_gt):
    """
    生成 HTML 报告。
    图片使用相对路径引用，不 base64 内嵌，避免 HTML 文件过大。
    在 output_dir 下直接打开 results.html 即可。
    """
    def _rel(p):
        return os.path.relpath(p, output_dir).replace("\\", "/")

    cards = []
    for idx, (res, img_path) in enumerate(zip(results, individual_paths)):
        gt_n  = len([b for b in res.get("gt_buildings",[]) if b["cls_id"] != EMPTY_CLS])
        gen_n = len([b for b in res["gen_buildings"]        if b["cls_id"] != EMPTY_CLS])
        text  = res["text"][:120].replace("<","&lt;").replace(">","&gt;")
        src   = _rel(img_path)

        cls_counts: Dict[str,int] = {}
        for b in res["gen_buildings"]:
            if b["cls_id"] != EMPTY_CLS:
                cls_counts[b.get("cls_name","unk")] = \
                    cls_counts.get(b.get("cls_name","unk"), 0) + 1
        tags = "".join(
            f"<span class='tag'>{k}: {v}</span>"
            for k, v in sorted(cls_counts.items()))

        meta = (f"GT: {gt_n}  |  Gen: {gen_n}" if has_gt
                else f"Gen: {gen_n} buildings")
        cards.append(f"""
  <div class="card">
    <p class="card-title">Sample {idx+1}</p>
    <img src="{src}" loading="lazy" />
    <p class="meta">{meta}</p>
    <div class="tags">{tags}</div>
    <p class="prompt">"{text}"</p>
  </div>""")

    summary_src = _rel(summary_path)
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BlockLayout 生成结果</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",system-ui,sans-serif;
     background:#F2F3F5;color:#333;padding:20px}}
h1{{font-size:1.4rem;margin-bottom:4px;color:#1a1a2e}}
.sub{{color:#666;font-size:.85rem;margin-bottom:18px}}
.summary{{background:#fff;border-radius:10px;padding:14px;
          margin-bottom:24px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
.summary h2{{font-size:1rem;margin-bottom:8px;color:#444}}
.summary img{{width:100%;border-radius:6px}}
.grid{{display:grid;
       grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
       gap:16px}}
.card{{background:#fff;border-radius:10px;padding:12px;
       box-shadow:0 2px 8px rgba(0,0,0,.07);
       transition:transform .15s,box-shadow .15s}}
.card:hover{{transform:translateY(-2px);
             box-shadow:0 6px 16px rgba(0,0,0,.12)}}
.card-title{{font-weight:600;font-size:.9rem;
             margin-bottom:6px;color:#1a1a2e}}
.card img{{width:100%;border-radius:6px;margin-bottom:6px}}
.meta{{font-size:.75rem;color:#555;margin-bottom:4px}}
.tags{{margin-bottom:5px}}
.tag{{display:inline-block;background:#EEF2FF;color:#4338CA;
      border-radius:3px;padding:1px 5px;font-size:.7rem;
      margin:1px 2px 0 0}}
.prompt{{font-size:.73rem;color:#777;font-style:italic;
         border-left:3px solid #DBEAFE;padding-left:7px;
         margin-top:5px;line-height:1.45}}
</style>
</head>
<body>
<h1>🏙 BlockLayout 生成结果验证</h1>
<p class="sub">验证集 · {len(results)} 个街区
  {"· GT 对比" if has_gt else ""}</p>

<div class="summary">
  <h2>Overview</h2>
  <img src="{summary_src}" />
</div>

<div class="grid">
{''.join(cards)}
</div>
</body></html>"""

    html_path = os.path.join(output_dir, "results.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ══════════════════════════════════════════════════════════
# 命令行
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        default=None)
    p.add_argument("--cache_path",  default="./data/blocks/cache.pt")
    p.add_argument("--bert_model",  default=None)
    p.add_argument("--n_samples",   type=int, default=16)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  default="./vis_output")
    p.add_argument("--n_cols",      type=int, default=4)
    p.add_argument("--dpi",         type=int, default=120)
    p.add_argument("--no_gt",       action="store_true")
    p.add_argument("--no_label",    action="store_true",
                   help="不在建筑上显示类别字母")
    p.add_argument("--html_only",   action="store_true",
                   help="跳过推理，仅重新生成 HTML")
    p.add_argument("--mock_text",   action="store_true")
    p.add_argument("--save_json",   action="store_true")
    # p.add_argument("--n_buildings",type=int, default=None)  # 新添加的 控制建筑数量  ？？？怎么加 怎么保持一致？？？ 现在的很多函数都不接收n_buildings函数 而且有没有过滤出来空类别？？太奇怪了 为什么 不应该有很多空类别吗？ 第七类 在数据中打印看看
    return p.parse_args()


def main():
    args = parse_args()

    if args.html_only:
        img_dir = os.path.join(args.output_dir, "imgs")
        imgs = sorted(f for f in os.listdir(img_dir)
                      if f.startswith("sample_") and f.endswith(".png"))
        fake = [{"boundary_pts": np.zeros((4,2)), "text": f"sample {i+1}",
                 "gt_buildings": [], "gen_buildings": []}
                for i in range(len(imgs))]
        summary = os.path.join(img_dir, "summary_grid.png")
        html = _make_html(fake, [os.path.join(img_dir, f) for f in imgs],
                          summary, args.output_dir, False)
        print(f"HTML 已更新：{html}")
        return

    engine = BlockLayoutInference(
        ckpt_path  = args.ckpt,
        bert_model = args.bert_model,
        mock_text  = args.mock_text,
    )

    results = engine.run_from_cache(
        cache_path = args.cache_path,
        n_samples  = args.n_samples,
        batch_size = args.batch_size,
        seed       = args.seed,
    )

    html_path = visualize_results(
        results,
        output_dir = args.output_dir,
        n_cols     = args.n_cols,
        show_gt    = not args.no_gt,
        dpi        = args.dpi,
    )

    gen_n = [len([b for b in r["gen_buildings"] if b["cls_id"] != EMPTY_CLS])
             for r in results]
    print(f"\n生成建筑数：均值={sum(gen_n)/len(gen_n):.1f}  "
          f"min={min(gen_n)}  max={max(gen_n)}")
    print(f"输出目录：{os.path.abspath(args.output_dir)}")
    print(f"HTML：file://{os.path.abspath(html_path)}")

    if args.save_json:
        jpath = os.path.join(args.output_dir, "results.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump([{
                "text":          r["text"],
                "gt_buildings":  r.get("gt_buildings", []),
                "gen_buildings": r["gen_buildings"],
            } for r in results], f, ensure_ascii=False, indent=2)
        print(f"JSON：{jpath}")


if __name__ == "__main__":
    main()