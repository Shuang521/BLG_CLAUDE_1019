#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py — BlockLayout 生成模型定量评估脚本

基于 train.py / inference.py 的数据流与模型接口，支持以下指标：

几何合法性指标:
  • Overlap Ratio (OR)       — 生成建筑间重叠面积 / 建筑总面积
  • Out-of-Block Ratio (OBR) — 生成建筑越界面积 / 建筑总面积
  • Coverage Ratio           — 建筑总面积 / 街区总面积

布局相似度指标 (vs Ground Truth):
  • Center Chamfer Distance  — 中心点双向最近邻距离均值
  • Hungarian mIoU           — 基于中心点匈牙利匹配后的平均 IoU
  • Building Count MAE/Acc   — 建筑数量绝对误差与容忍准确率
  • Class Accuracy           — 匹配建筑的类别一致率
  • Precision / Recall / F1  — 基于匈牙利匹配的建筑检测 PRF
  • Area MAPE                — 匹配建筑的面积相对误差 (%)
  • Angle MAE                — 匹配建筑的角度绝对误差 (度)
  • Angle KL Divergence      — 角度分布与 GT 的 KL 散度

图像分布指标 (栅格化后):
  • Rasterized FID           — Fréchet Inception Distance
  • Rasterized KID           — Kernel Inception Distance (多项式核 MMD)

用法:
  # 默认评估验证集，使用 FlowMatchingBlockLayoutModel（与 train.py 一致）
  CUDA_VISIBLE_DEVICES=5 python eval.py \
    --ckpt /root/project/BLG/checkpoints/rotate_01_pad_zero/rotate_01_loss_v10_height_fm_v1pos_shuffle/ckpt_050000.pt \
    --cache_path "./data/blocks_rotate_01/cache_height_scale_area_ratio_text_norm_shuffle.pt" \
    --n_samples 128 --batch_size 16 \
    --output_dir ./eval_output/ \
    --model_type block_layout
    --model_type flow_matching

  # 若需评估训练集
  python eval.py ... --split train

  # 若环境无 GPU 或无需 FID/KID
  python eval.py ... --no_fid_kid
"""

import os
import argparse
import json
import math
import time
import warnings
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import sqrtm

# ── 可选依赖：shapely（几何精确计算）─────────────────────
try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    warnings.warn(
        "[警告] shapely 未安装，几何指标将使用近似计算或返回 0。\n"
        "如需精确几何指标，请运行:  pip install shapely"
    )

# ── 项目内导入（与 train.py / inference.py 保持一致）────
from config import CFG
from models.block_layout_model import BlockLayoutModel, FlowMatchingBlockLayoutModel
from data.dataset import CachedDataset
from inference import (
    _decode_x0, _decode_raw, _rotated_corners,
    CLASS_NAMES, CLASS_FCOLOR, EMPTY_CLS, RES_CLS,
    BBOX_MEAN, BBOX_STD
)

# ── matplotlib 用于栅格化 ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ── torchvision 用于 FID/KID 特征提取 ───────────────────
try:
    from torchvision.models import inception_v3
    try:
        from torchvision.models import Inception_V3_Weights
    except ImportError:
        Inception_V3_Weights = None  # 兼容旧版 torchvision
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    warnings.warn(
        "[警告] torchvision 未安装，FID/KID 指标将被跳过。\n"
        "如需计算，请运行:  pip install torchvision"
    )


# ══════════════════════════════════════════════════════════
# 1. 几何工具函数
# ══════════════════════════════════════════════════════════

def to_polygon(bld: Dict) -> Optional[Polygon]:
    """将单个建筑字典转为 shapely Polygon（无效几何自动修复）"""
    if not HAS_SHAPELY:
        return None
    corners = _rotated_corners(bld["cx"], bld["cy"], bld["w"], bld["h"], bld["angle"])
    poly = Polygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.is_valid and poly.area > 1e-10 else None


def compute_overlap_ratio(buildings: List[Dict]) -> float:
    """
    Overlap Ratio = 所有建筑两两交集面积之和 / 所有建筑总面积
    使用 unary_union 计算：重叠面积 = 各面积之和 - 并集面积
    """
    polys = []
    for b in buildings:
        if b["cls_id"] == EMPTY_CLS:
            continue
        p = to_polygon(b)
        if p is not None:
            polys.append(p)

    if len(polys) < 2:
        return 0.0

    total_area = sum(p.area for p in polys)
    if total_area < 1e-8:
        return 0.0

    union_area = unary_union(polys).area
    overlap_area = max(0.0, total_area - union_area)
    return overlap_area / total_area


def compute_oob_ratio(buildings: List[Dict], boundary_pts: np.ndarray) -> float:
    """
    Out-of-Block Ratio = 建筑超出街区边界的总面积 / 建筑总面积
    """
    if len(boundary_pts) < 3 or not HAS_SHAPELY:
        return 0.0

    block_poly = Polygon(boundary_pts)
    if not block_poly.is_valid:
        block_poly = block_poly.buffer(0)
    if block_poly.area < 1e-8:
        return 0.0

    total_out = 0.0
    total_bld = 0.0
    for b in buildings:
        if b["cls_id"] == EMPTY_CLS:
            continue
        p = to_polygon(b)
        if p is None:
            continue
        total_out += p.difference(block_poly).area
        total_bld += p.area

    return total_out / total_bld if total_bld > 1e-8 else 0.0


def compute_coverage_ratio(buildings: List[Dict], boundary_pts: np.ndarray) -> float:
    """Coverage Ratio = 建筑总面积 / 街区总面积"""
    if len(boundary_pts) < 3 or not HAS_SHAPELY:
        return 0.0

    block_poly = Polygon(boundary_pts)
    if not block_poly.is_valid:
        block_poly = block_poly.buffer(0)
    block_area = block_poly.area
    if block_area < 1e-8:
        return 0.0

    bld_area = 0.0
    for b in buildings:
        if b["cls_id"] == EMPTY_CLS:
            continue
        p = to_polygon(b)
        if p is not None:
            bld_area += p.area
    return bld_area / block_area


def polygon_iou(poly1: Polygon, poly2: Polygon) -> float:
    """计算两个 shapely Polygon 的 IoU，自动修复无效几何"""
    if not HAS_SHAPELY:
        return 0.0
    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 1e-8 else 0.0


# ══════════════════════════════════════════════════════════
# 2. 布局相似度与匹配指标
# ══════════════════════════════════════════════════════════

def compute_layout_similarity(
    gt_buildings: List[Dict],
    gen_buildings: List[Dict],
    boundary_pts: np.ndarray,
    match_dist_thresh: float = 0.30,
) -> Dict:
    """
    计算生成布局与 GT 之间的多种相似度指标。
    返回字典包含 center_chamfer, hungarian_miou, count_mae, count_acc,
    class_acc, precision, recall, f1, area_mape, angle_mae 等。
    """
    gt = [b for b in gt_buildings if b["cls_id"] != EMPTY_CLS]
    gen = [b for b in gen_buildings if b["cls_id"] != EMPTY_CLS]
    n_gt, n_gen = len(gt), len(gen)

    # ── 数量指标 ──
    count_mae = abs(n_gt - n_gen)
    count_acc = 1.0 if count_mae <= 1 else 0.0

    # ── 中心点 Chamfer Distance ──
    gt_centers = np.array([[b["cx"], b["cy"]] for b in gt]) if n_gt else np.zeros((0, 2))
    gen_centers = np.array([[b["cx"], b["cy"]] for b in gen]) if n_gen else np.zeros((0, 2))

    if n_gt == 0 and n_gen == 0:
        center_chamfer = 0.0
        center_chamfer_norm = 1.0
    elif n_gt == 0 or n_gen == 0:
        center_chamfer = 1.0
        center_chamfer_norm = 0.0
    else:
        dist_mat = np.sqrt(((gt_centers[:, None, :] - gen_centers[None, :, :]) ** 2).sum(axis=2))
        forward = dist_mat.min(axis=1).mean()
        backward = dist_mat.min(axis=0).mean()
        center_chamfer = (forward + backward) / 2.0
        max_dist = math.sqrt(2)  # 坐标归一化在 [0,1] 附近，最大对角线距离
        center_chamfer_norm = max(0.0, 1.0 - center_chamfer / max_dist)

    # ── 匈牙利匹配（基于中心点距离） ──
    if n_gt == 0 or n_gen == 0:
        return {
            "center_chamfer": float(center_chamfer),
            "center_chamfer_norm": float(center_chamfer_norm),
            "hungarian_miou": 0.0,
            "count_mae": int(count_mae),
            "count_acc": float(count_acc),
            "class_acc": 0.0,
            "precision": 1.0 if n_gen == 0 else 0.0,
            "recall": 1.0 if n_gt == 0 else 0.0,
            "f1": 1.0 if (n_gt == 0 and n_gen == 0) else 0.0,
            "tp": 0, "fp": n_gen, "fn": n_gt,
            "area_mape": 0.0,
            "angle_mae": 0.0,
        }

    row_ind, col_ind = linear_sum_assignment(dist_mat)

    matched_ious = []
    matched_classes_gt = []
    matched_classes_gen = []
    matched_areas = []
    matched_angles = []
    matched_gt_set = set()
    matched_gen_set = set()

    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] > match_dist_thresh:
            continue  # 过远的匹配视为无效
        p_gt = to_polygon(gt[r])
        p_gen = to_polygon(gen[c])
        if p_gt is None or p_gen is None:
            continue

        iou = polygon_iou(p_gt, p_gen)
        matched_ious.append(iou)
        matched_classes_gt.append(gt[r]["cls_id"])
        matched_classes_gen.append(gen[c]["cls_id"])
        matched_gt_set.add(r)
        matched_gen_set.add(c)

        # 面积相对误差
        gt_area = p_gt.area
        gen_area = p_gen.area
        matched_areas.append(abs(gen_area - gt_area) / gt_area * 100 if gt_area > 1e-8 else 0.0)

        # 角度误差（考虑 180° 周期性）
        diff = abs(gt[r]["angle"] - gen[c]["angle"])
        diff = min(diff, 180.0 - diff)
        matched_angles.append(diff)

    tp = len(matched_ious)
    fp = n_gen - len(matched_gen_set)
    fn = n_gt - len(matched_gt_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "center_chamfer": float(center_chamfer),
        "center_chamfer_norm": float(center_chamfer_norm),
        "hungarian_miou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "count_mae": int(count_mae),
        "count_acc": float(count_acc),
        "class_acc": float(np.mean([g == h for g, h in zip(matched_classes_gt, matched_classes_gen)])) if matched_classes_gt else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "area_mape": float(np.mean(matched_areas)) if matched_areas else 0.0,
        "angle_mae": float(np.mean(matched_angles)) if matched_angles else 0.0,
    }


def compute_angle_kl(gt_buildings: List[Dict], gen_buildings: List[Dict], n_bins: int = 18) -> float:
    """
    计算 GT 与生成结果的角度分布 KL 散度。
    角度范围取 [0, 180)，因为矩形旋转 180° 后视觉上相同。
    """
    gt = [b for b in gt_buildings if b["cls_id"] != EMPTY_CLS]
    gen = [b for b in gen_buildings if b["cls_id"] != EMPTY_CLS]
    if len(gt) == 0 or len(gen) == 0:
        return 0.0

    bins = np.linspace(0, 180, n_bins + 1)
    gt_hist, _ = np.histogram([b["angle"] % 180 for b in gt], bins=bins)
    gen_hist, _ = np.histogram([b["angle"] % 180 for b in gen], bins=bins)

    gt_hist = gt_hist.astype(float) + 1e-6
    gen_hist = gen_hist.astype(float) + 1e-6
    gt_hist /= gt_hist.sum()
    gen_hist /= gen_hist.sum()

    return float(np.sum(gt_hist * np.log(gt_hist / gen_hist)))


# ══════════════════════════════════════════════════════════
# 3. 栅格化与 FID/KID
# ══════════════════════════════════════════════════════════

def rasterize_layout(
    boundary_pts: np.ndarray,
    buildings: List[Dict],
    size: int = 256,
    draw_boundary: bool = True,
) -> np.ndarray:
    """
    将布局渲染为 RGB 图像 [H, W, 3] (uint8)，用于 InceptionV3 特征提取。
    """
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 绘制街区边界
    if draw_boundary and len(boundary_pts) >= 3:
        bpoly = MplPolygon(boundary_pts, closed=True)
        pc_bnd = PatchCollection(
            [bpoly], facecolors=["#F8F8F8"], edgecolors=["#555555"],
            linewidths=[1.2], zorder=1,
        )
        ax.add_collection(pc_bnd)

    # 绘制建筑（按类别着色）
    polys, fcolors = [], []
    for b in buildings:
        if b["cls_id"] == EMPTY_CLS:
            continue
        corners = _rotated_corners(b["cx"], b["cy"], b["w"], b["h"], b["angle"])
        polys.append(MplPolygon(corners, closed=True))
        cid = b["cls_id"]
        fcolors.append(CLASS_FCOLOR[cid] if cid < len(CLASS_FCOLOR) else "#AAAAAA")

    if polys:
        pc = PatchCollection(
            polys, facecolors=fcolors, edgecolors=["#333333"],
            linewidths=0.6, alpha=0.92, zorder=2,
        )
        ax.add_collection(pc)

    fig.canvas.draw()

    # ── 修复：使用 buffer_rgba() 明确获取 RGBA，再转为 RGB ──
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape((h, w, 4))[:, :, :3]  # 丢弃 Alpha 通道，保留 RGB

    plt.close(fig)
    return img  # [H, W, 3] uint8


class InceptionFeatureExtractor:
    """基于 InceptionV3 的 2048-dim 特征提取器（用于 FID/KID）"""

    def __init__(self, device: torch.device):
        if not HAS_TORCHVISION:
            raise ImportError("torchvision 未安装")

        if Inception_V3_Weights is not None:
            weights = Inception_V3_Weights.IMAGENET1K_V1
            self.model = inception_v3(weights=weights, transform_input=False)
        else:
            self.model = inception_v3(pretrained=True, transform_input=False)

        self.model.fc = torch.nn.Identity()  # 输出 2048 维特征
        self.model = self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device

    @torch.no_grad()
    def extract(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        images: List[np.ndarray] 每个元素为 [H, W, 3] uint8
        返回: [N, 2048] numpy array
        """
        feats = []
        for i in range(0, len(images), batch_size):
            batch = []
            for img in images[i : i + batch_size]:
                batch.append(self.preprocess(img))
            batch = torch.stack(batch, dim=0).to(self.device)
            feat = self.model(batch)  # [B, 2048]
            feats.append(feat.cpu().numpy())
        return np.concatenate(feats, axis=0)


def calculate_fid(features_real: np.ndarray, features_fake: np.ndarray) -> float:
    """Fréchet Inception Distance"""
    mu1, sigma1 = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu2, sigma2 = features_fake.mean(axis=0), np.cov(features_fake, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_kid(
    features_real: np.ndarray,
    features_fake: np.ndarray,
    subset_size: int = 1000,
    num_subsets: int = 100,
    degree: int = 3,
    gamma: float = 1.0,
    coef0: float = 1.0,
) -> Tuple[float, float]:
    """
    Kernel Inception Distance（多项式核 MMD）。
    返回 (mean, std)  over num_subsets 次随机子集采样。
    """
    n = min(len(features_real), len(features_fake), subset_size)
    if n < 10:
        return 0.0, 0.0

    kid_values = []
    rng = np.random.RandomState(42)

    for _ in range(num_subsets):
        idx_r = rng.choice(len(features_real), n, replace=False)
        idx_f = rng.choice(len(features_fake), n, replace=False)
        x = features_real[idx_r]
        y = features_fake[idx_f]

        def poly_kernel(a, b):
            return (gamma * np.dot(a, b.T) + coef0) ** degree

        k_xx = poly_kernel(x, x)
        k_yy = poly_kernel(y, y)
        k_xy = poly_kernel(x, y)

        m = n - 1
        mmd2 = (k_xx.sum() - np.trace(k_xx)) / (n * m)
        mmd2 += (k_yy.sum() - np.trace(k_yy)) / (n * m)
        mmd2 -= 2 * k_xy.sum() / (n * n)
        kid_values.append(mmd2)

    return float(np.mean(kid_values)), float(np.std(kid_values))


# ══════════════════════════════════════════════════════════
# 4. 评估引擎
# ══════════════════════════════════════════════════════════

class LayoutEvaluator:
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        model_type: str = "flow_matching",
        mock_text: bool = False,
    ):
        self.device = device
        self.cfg = CFG

        # 根据训练代码选择模型类
        if model_type == "flow_matching":
            self.model = FlowMatchingBlockLayoutModel(self.cfg).to(device)
        else:
            self.model = BlockLayoutModel(self.cfg, use_mock_text=mock_text).to(device)

        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", "?")
            print(f"[Evaluator] 已加载权重：{ckpt_path}  (step={step})")
        else:
            print("[Evaluator] 警告：未提供有效 checkpoint，使用随机权重")

        self.model.eval()

        # 初始化 Inception 特征提取器
        self.feat_extractor = None
        if HAS_TORCHVISION:
            self.feat_extractor = InceptionFeatureExtractor(device)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        cache_path: str,
        split: str = "val",
        n_samples: int = 128,
        batch_size: int = 16,
        seed: int = 42,
        save_per_sample: bool = True,
    ) -> Dict:
        """
        在数据集上运行完整评估，返回汇总指标字典。
        """
        dataset = CachedDataset(cache_path, split=split, train_ratio=self.cfg.train_split)
        n_total = len(dataset)
        n_samples = min(n_samples, n_total)

        rng = np.random.RandomState(seed)
        indices = rng.choice(range(n_total), n_samples, replace=False)
        print(f"[Evaluator] {split} 集共 {n_total} 样本，评估 {n_samples} 个")

        per_sample_metrics = []
        gt_images = []
        gen_images = []

        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [dataset[i] for i in batch_indices]
            metrics, batch_gt_imgs, batch_gen_imgs = self._eval_batch(batch)

            per_sample_metrics.extend(metrics)
            gt_images.extend(batch_gt_imgs)
            gen_images.extend(batch_gen_imgs)

            done = min(start + batch_size, n_samples)
            print(f"  进度: {done}/{n_samples}")

        # ── 汇总统计 ──
        summary = self._aggregate_metrics(per_sample_metrics)

        # ── FID / KID ──
        if self.feat_extractor is not None and len(gt_images) > 0:
            print("[Evaluator] 提取 Inception 特征用于 FID/KID ...")
            t0 = time.time()
            feat_gt = self.feat_extractor.extract(gt_images)
            feat_gen = self.feat_extractor.extract(gen_images)
            print(f"  特征提取完成，耗时 {time.time() - t0:.1f}s")

            print("[Evaluator] 计算 FID ...")
            summary["fid"] = calculate_fid(feat_gt, feat_gen)

            print("[Evaluator] 计算 KID ...")
            kid_mean, kid_std = calculate_kid(feat_gt, feat_gen)
            summary["kid_mean"] = kid_mean
            summary["kid_std"] = kid_std
        else:
            summary["fid"] = None
            summary["kid_mean"] = None
            summary["kid_std"] = None

        summary["n_samples"] = n_samples
        summary["split"] = split
        if save_per_sample:
            summary["per_sample"] = per_sample_metrics
        return summary, gt_images, gen_images

    def _eval_batch(self, batch: List[Dict]) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
        B = len(batch)
        boundary = torch.stack([item["boundary"] for item in batch]).to(self.device)
        texts = [item["text"] for item in batch]
        x0_gt = torch.stack([item["x0"] for item in batch])

        # 推理生成
        has_cached = "text_seq" in batch[0]
        with torch.no_grad():
            if has_cached:
                pre_seq = torch.stack([item["text_seq"] for item in batch]).to(self.device)
                pre_mask = torch.stack([item["text_mask"] for item in batch]).to(self.device)
                raw = self.model.sample_with_text_cache(boundary, texts, pre_seq, pre_mask)
            else:
                raw = self.model.sample(boundary, texts)

        metrics = []
        gt_imgs = []
        gen_imgs = []

        for b in range(B):
            gt_buildings = _decode_x0(x0_gt[b], self.cfg)
            gen_buildings = _decode_raw(raw, b, self.cfg)
            boundary_pts = batch[b]["boundary"].numpy()

            # 几何合法性
            overlap_ratio = compute_overlap_ratio(gen_buildings)
            oob_ratio = compute_oob_ratio(gen_buildings, boundary_pts)
            coverage_gt = compute_coverage_ratio(gt_buildings, boundary_pts)
            coverage_gen = compute_coverage_ratio(gen_buildings, boundary_pts)

            # 布局相似度
            sim_metrics = compute_layout_similarity(gt_buildings, gen_buildings, boundary_pts)
            angle_kl = compute_angle_kl(gt_buildings, gen_buildings)

            m = {
                "sample_idx": b,
                "overlap_ratio": overlap_ratio,
                "oob_ratio": oob_ratio,
                "coverage_gt": coverage_gt,
                "coverage_gen": coverage_gen,
                "coverage_diff": abs(coverage_gt - coverage_gen),
                "angle_kl": angle_kl,
                **sim_metrics,
            }
            metrics.append(m)

            # 栅格化
            gt_imgs.append(rasterize_layout(boundary_pts, gt_buildings))
            gen_imgs.append(rasterize_layout(boundary_pts, gen_buildings))

        return metrics, gt_imgs, gen_imgs

    def _aggregate_metrics(self, per_sample: List[Dict]) -> Dict:
        """将每样本指标聚合成均值/标准差/最值"""
        if not per_sample:
            return {}

        numeric_keys = {k for m in per_sample for k, v in m.items() if isinstance(v, (int, float, np.number))}

        summary = {}
        for k in sorted(numeric_keys):
            vals = [m[k] for m in per_sample if k in m]
            arr = np.array(vals)
            summary[f"{k}_mean"] = float(arr.mean())
            summary[f"{k}_std"] = float(arr.std())
            summary[f"{k}_min"] = float(arr.min())
            summary[f"{k}_max"] = float(arr.max())
        return summary


# ══════════════════════════════════════════════════════════
# 5. 可视化与报告
# ══════════════════════════════════════════════════════════

def save_comparison_grid(
    gt_images: List[np.ndarray],
    gen_images: List[np.ndarray],
    output_dir: str,
    n_save: int = 16,
):
    """保存前 N 个样本的 GT vs Generated 栅格化对比图"""
    n = min(n_save, len(gt_images))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i in range(n):
        axes[i, 0].imshow(gt_images[i])
        axes[i, 0].set_title(f"#{i+1} GT", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gen_images[i])
        axes[i, 1].set_title(f"#{i+1} Generated", fontsize=8)
        axes[i, 1].axis("off")
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Eval] 对比图已保存: {path}")


def print_report(summary: Dict):
    """终端打印评估报告"""
    print("\n" + "=" * 64)
    print(" BlockLayout 生成模型 — 定量评估报告")
    print("=" * 64)

    sections = [
        ("几何合法性指标", ["overlap_ratio", "oob_ratio", "coverage_gen", "coverage_diff"]),
        ("布局相似度指标", [
            "center_chamfer", "center_chamfer_norm", "hungarian_miou",
            "count_mae", "count_acc", "class_acc"
        ]),
        ("检测质量指标", ["precision", "recall", "f1", "area_mape", "angle_mae"]),
        ("分布一致性", ["angle_kl"]),
    ]

    for title, keys in sections:
        print(f"\n【{title}】")
        for k in keys:
            mean_key = f"{k}_mean"
            if mean_key in summary:
                mean_val = summary[mean_key]
                std_val = summary.get(f"{k}_std", 0.0)
                print(f"  {k:26s}: {mean_val:8.4f}  (±{std_val:.4f})")

    if summary.get("fid") is not None:
        print(f"\n【图像分布指标 (栅格化 256×256 → InceptionV3)】")
        print(f"  {'FID':26s}: {summary['fid']:8.4f}")
        print(f"  {'KID_mean':26s}: {summary['kid_mean']:10.6f}  (±{summary.get('kid_std', 0):.6f})")

    print("\n" + "-" * 64)
    print(f" 评估样本数: {summary.get('n_samples', 'N/A')}  |  数据集划分: {summary.get('split', 'N/A')}")
    print("=" * 64 + "\n")


# ══════════════════════════════════════════════════════════
# 6. 命令行入口
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BlockLayout 生成模型定量评估")
    p.add_argument("--ckpt", required=True, help="模型 checkpoint 路径 (.pt)")
    p.add_argument("--cache_path", required=True, help="数据集 cache.pt 路径")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"], help="评估数据划分")
    p.add_argument("--n_samples", type=int, default=128, help="评估样本数")
    p.add_argument("--batch_size", type=int, default=16, help="推理 batch size")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--output_dir", default="./eval_output", help="结果输出目录")
    p.add_argument(
        "--model_type",
        type=str,
        default="flow_matching",
        choices=["flow_matching", "block_layout"],
        help="模型类型，需与训练时一致（train.py 使用 flow_matching）",
    )
    p.add_argument("--mock_text", action="store_true", help="仅在使用 block_layout 时生效")
    p.add_argument("--no_fid_kid", action="store_true", help="跳过 FID/KID 计算（节省显存/时间）")
    p.add_argument("--save_per_sample", action="store_true", help="在 JSON 中保存每样本详细指标")
    p.add_argument("--n_vis", type=int, default=16, help="保存对比图的数量")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = LayoutEvaluator(
        ckpt_path=args.ckpt,
        device=device,
        model_type=args.model_type,
        mock_text=args.mock_text,
    )

    if args.no_fid_kid:
        evaluator.feat_extractor = None

    summary, gt_imgs, gen_imgs = evaluator.evaluate_dataset(
        cache_path=args.cache_path,
        split=args.split,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        save_per_sample=args.save_per_sample,
    )

    # 保存 JSON
    json_path = os.path.join(args.output_dir, "eval_results.json")
    # 将额外字段加入 summary
    summary["cache_path"] = args.cache_path  # 或你的实际变量
    summary["ckpt"] = args.ckpt  

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存对比图
    save_comparison_grid(gt_imgs, gen_imgs, args.output_dir, n_save=args.n_vis)

    # 终端报告
    print_report(summary)
    print(f"[Eval] 详细结果已保存: {json_path}")
    print(f"[Eval] 输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()