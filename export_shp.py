#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_shp.py — 将 BlockLayout 模型输出导出为 SHP / GPKG 矢量文件

支持 2.5D 可视化：
  • 建筑 Polygon + height 属性（QGIS 2.5D Renderer / Mapbox fill-extrusion）
  • 街区边界 Polygon
  • 按 cache.pt 中的 scale 还原实际米制坐标

输出文件：
  • buildings_gen.shp   — 生成建筑（含 height, angle, cls_id, area, width, depth...）
  • boundaries_gen.shp  — 生成街区边界（含 text, area_sqm, aspect_ratio...）
  • buildings_gt.shp    — GT 建筑（若加 --export_gt）
  • boundaries_gt.shp   — GT 边界（若加 --export_gt）

用法：
  # 导出验证集生成结果（默认 .shp）
  CUDA_VISIBLE_DEVICES=4 python export_shp.py \
    --ckpt /root/project/BLG/checkpoints/rotate_01_pad_zero/rotate_01_loss_v10_height_fm_mask_none/ckpt_050000.pt \
    --cache_path "./data/blocks_final/cache_gis_norm_height_scale.pt" \
    --output_dir ./shp_output/fm \
    --split val \
    --format gpkg \
    --n_samples 10

    ./data/blocks_rotate_01/_height_scale_area_ratio_text_norm_shuffle.pt

  # 同时导出 GT 做对比 + 使用 GeoPackage 格式（推荐，中文无乱码）
  python export_shp.py ... --export_gt --format gpkg

  # 指定坐标系（若后续要与真实地图叠加）
  python export_shp.py ... --crs EPSG:32650
"""

import os
import argparse
import json
import warnings
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np

# ── 几何 / 矢量库 ──────────────────────────────────────
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ── 项目内导入（与 train.py / inference.py 保持一致）────
from config import CFG
from models.block_layout_model import BlockLayoutModel, FlowMatchingBlockLayoutModel
from data.dataset import CachedDataset
from inference import (
    _decode_raw, _decode_x0, _rotated_corners,
    CLASS_NAMES, EMPTY_CLS,
    BBOX_MEAN, BBOX_STD,
)


# ══════════════════════════════════════════════════════════
# 1. 几何工具
# ══════════════════════════════════════════════════════════

def make_building_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> Optional[Polygon]:
    """根据中心、宽高、角度创建 shapely Polygon，自动修复无效几何"""
    corners = _rotated_corners(cx, cy, w, h, angle)
    try:
        poly = Polygon(corners)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def filter_boundary_padding(boundary_pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    过滤边界点序列中的 (0,0) 填充值。
    cache.pt 中边界固定 64 点，未使用的位置通常为 (0,0)。
    """
    norms = np.linalg.norm(boundary_pts, axis=1)
    valid = norms > eps
    if valid.sum() >= 3:
        return boundary_pts[valid]
    # 若过滤后不足 3 点，说明坐标本身就在原点附近，返回原数组
    return boundary_pts


# ══════════════════════════════════════════════════════════
# 2. 导出引擎
# ══════════════════════════════════════════════════════════

class LayoutSHPExporter:
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        model_type: str = "flow_matching",
        mock_text: bool = False,
    ):
        if not HAS_GEOPANDAS:
            raise ImportError(
                "[错误] 缺少 geopandas / shapely。\n"
                "请运行:  pip install geopandas shapely"
            )

        self.device = device
        self.cfg = CFG

        # 与 train.py 保持一致：默认 flow_matching
        if model_type == "flow_matching":
            self.model = FlowMatchingBlockLayoutModel(self.cfg).to(device)
        else:
            self.model = BlockLayoutModel(self.cfg, use_mock_text=mock_text).to(device)

        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            step = ckpt.get("step", "?")
            print(f"[Exporter] 已加载权重：{ckpt_path} (step={step})")
        else:
            print("[Exporter] 警告：未提供有效 checkpoint，使用随机权重")

        self.model.eval()

    @torch.no_grad()
    def export(
        self,
        cache_path: str,
        output_dir: str,
        split: str = "val",
        n_samples: Optional[int] = None,
        batch_size: int = 16,
        seed: int = 42,
        export_gt: bool = False,
        fmt: str = "shp",
        crs: Optional[str] = None,
    ):
        """
        主导出流程。
        """
        dataset = CachedDataset(cache_path, split=split, train_ratio=self.cfg.train_split)
        n_total = len(dataset)
        if n_samples is None or n_samples > n_total:
            n_samples = n_total

        rng = np.random.RandomState(seed)
        indices = rng.choice(range(n_total), n_samples, replace=False)
        print(f"[Exporter] {split} 集共 {n_total} 条，导出 {n_samples} 条街区")

        os.makedirs(output_dir, exist_ok=True)

        gen_buildings: List[Dict] = []
        gen_boundaries: List[Dict] = []
        gt_buildings: List[Dict] = [] if export_gt else None
        gt_boundaries: List[Dict] = [] if export_gt else None

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = [dataset[i] for i in batch_idx]

            g_bld, g_bnd = self._process_batch(batch, batch_idx, export_gt=False)
            gen_buildings.extend(g_bld)
            gen_boundaries.extend(g_bnd)

            if export_gt:
                gt_bld, gt_bnd = self._process_batch(batch, batch_idx, export_gt=True)
                gt_buildings.extend(gt_bld)
                gt_boundaries.extend(gt_bnd)

            done = min(start + batch_size, n_samples)
            print(f"  进度 {done}/{n_samples}")

        # ── 驱动与后缀 ──
        suffix = ".gpkg" if fmt == "gpkg" else ".shp"
        driver = "GPKG" if fmt == "gpkg" else "ESRI Shapefile"

        # ── 保存生成结果 ──
        if gen_buildings:
            gdf = gpd.GeoDataFrame(gen_buildings, crs=crs)
            path = os.path.join(output_dir, f"buildings_gen{suffix}")
            gdf.to_file(path, driver=driver, encoding="utf-8")
            print(f"[Exporter] 生成建筑 -> {path} ({len(gdf)} 个)")

        if gen_boundaries:
            gdf = gpd.GeoDataFrame(gen_boundaries, crs=crs)
            path = os.path.join(output_dir, f"boundaries_gen{suffix}")
            gdf.to_file(path, driver=driver, encoding="utf-8")
            print(f"[Exporter] 生成边界 -> {path} ({len(gdf)} 个)")

        # ── 保存 GT（可选） ──
        if export_gt:
            if gt_buildings:
                gdf = gpd.GeoDataFrame(gt_buildings, crs=crs)
                path = os.path.join(output_dir, f"buildings_gt{suffix}")
                gdf.to_file(path, driver=driver, encoding="utf-8")
                print(f"[Exporter] GT 建筑 -> {path} ({len(gdf)} 个)")

            if gt_boundaries:
                gdf = gpd.GeoDataFrame(gt_boundaries, crs=crs)
                path = os.path.join(output_dir, f"boundaries_gt{suffix}")
                gdf.to_file(path, driver=driver, encoding="utf-8")
                print(f"[Exporter] GT 边界 -> {path} ({len(gdf)} 个)")

        # ── 元数据 JSON ──
        meta = {
            "n_samples": int(n_samples),
            "split": split,
            "n_gen_buildings": len(gen_buildings),
            "n_gen_boundaries": len(gen_boundaries),
            "export_gt": export_gt,
            "format": fmt,
            "crs": crs,
        }
        meta_path = os.path.join(output_dir, "export_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[Exporter] 元数据 -> {meta_path}")

        return meta

    def _process_batch(
        self,
        batch: List[Dict],
        global_indices: np.ndarray,
        export_gt: bool = False,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        处理一个 batch。若 export_gt=True，解码 x0 作为 GT 输出。
        """
        B = len(batch)
        device = self.device

        boundary_tensor = torch.stack([item["boundary"] for item in batch]).to(device)
        texts = [item.get("text", item.get("texts", "")) for item in batch]

        # 读取每个样本的物理尺度属性（兼容 CachedDataset 可能缺失的字段）
        scales = [float(item.get("scale", 1.0)) for item in batch]
        area_sqms = [float(item.get("area_sqm", 0.0)) for item in batch]
        aspect_ratios = [float(item.get("aspect_ratio", 0.0)) for item in batch]
        
        # ── 模型推理（仅生成模式） ──
        if not export_gt:
            has_cached = "text_seq" in batch[0]
            with torch.no_grad():
                if has_cached:
                    pre_seq = torch.stack([item["text_seq"] for item in batch]).to(device)
                    pre_mask = torch.stack([item["text_mask"] for item in batch]).to(device)
                    raw = self.model.sample_with_text_cache(
                        boundary_tensor, texts, pre_seq, pre_mask
                    )
                else:
                    raw = self.model.sample(boundary_tensor, texts)

        buildings: List[Dict] = []
        boundaries: List[Dict] = []

        for b in range(B):
            sid = int(global_indices[b])
            scale = scales[b]
            # area_sqm = area_sqms[b]
            # ar = aspect_ratios[b]
            text = texts[b]

            # ── 街区边界：过滤填充点 -> 乘 scale ──
            bnd_pts = batch[b]["boundary"].cpu().numpy()  # [64, 2]
            bnd_pts = filter_boundary_padding(bnd_pts)
            bnd_pts_real = bnd_pts * scale

            bnd_poly = None
            if len(bnd_pts_real) >= 3:
                try:
                    bnd_poly = Polygon(bnd_pts_real)
                    if not bnd_poly.is_valid:
                        bnd_poly = bnd_poly.buffer(0)
                except Exception:
                    bnd_poly = None

            if bnd_poly is not None and not bnd_poly.is_empty:
                boundaries.append({
                    "geometry": bnd_poly,
                    "smp_idx": sid,
                    "text": text,
                    "scale": scale,
                    # "area_sqm": area_sqm,
                    # "aspect_r": ar,
                })

            # ── 建筑：解码 -> 反归一化 -> 乘 scale ──
            if export_gt:
                x0 = batch[b]["x0"]  # [96, 9]
                bld_list = _decode_x0(x0, self.cfg)
                src_tag = "gt"
            else:
                bld_list = _decode_raw(raw, b, self.cfg)
                src_tag = "gen"

            for bld in bld_list:
                if bld["cls_id"] == EMPTY_CLS:
                    continue

                # _decode_raw / _decode_x0 已做反归一化（BBOX_MEAN/STD），
                # 再乘 scale 即得实际米制坐标
                cx = bld["cx"] * scale
                cy = bld["cy"] * scale
                w = bld["w"] * scale
                h = bld["h"] * scale
                height = bld.get("height", 0.0) * scale
                angle = bld["angle"]

                poly = make_building_polygon(cx, cy, w, h, angle)
                if poly is None:
                    continue

                cls_id = bld["cls_id"]
                cls_name = bld.get(
                    "cls_name",
                    CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
                )

                buildings.append({
                    "geometry": poly,
                    "smp_idx": sid,
                    "src": src_tag,
                    "cls_id": int(cls_id),
                    "cls_name": cls_name,
                    "height": float(height),   # 2.5D 核心字段
                    "angle": float(angle),
                    "width": float(w),
                    "depth": float(h),
                    "cx": float(cx),
                    "cy": float(cy),
                    "area": float(poly.area),
                    "prob": float(bld.get("prob", 1.0)),
                    "scale": scale,
                    "text": text,
                })

        return buildings, boundaries


# ══════════════════════════════════════════════════════════
# 3. 命令行入口
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BlockLayout -> SHP/GPKG 矢量导出")
    p.add_argument("--ckpt", required=True, help="模型 checkpoint 路径")
    p.add_argument("--cache_path", required=True, help="cache.pt 路径")
    p.add_argument("--output_dir", required=True, help="输出目录")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--n_samples", type=int, default=None, help="导出样本数，默认全部")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model_type",
        default="flow_matching",
        choices=["flow_matching", "block_layout"],
        help="需与训练时一致（train.py 默认 flow_matching）",
    )
    p.add_argument("--mock_text", action="store_true")
    p.add_argument(
        "--export_gt",
        action="store_true",
        help="同时导出 Ground Truth SHP，方便在 GIS 中叠置对比",
    )
    p.add_argument(
        "--format",
        default="shp",
        choices=["shp", "gpkg"],
        help="shp: 兼容最广；gpkg: 支持 UTF-8 中文与长字段名（推荐）",
    )
    p.add_argument(
        "--crs",
        default=None,
        help="坐标参考系，如 EPSG:4326；默认无 CRS（本地平面坐标，单位：米）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not HAS_GEOPANDAS:
        print("[错误] 缺少 geopandas / shapely，请运行：pip install geopandas shapely")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    exporter = LayoutSHPExporter(
        ckpt_path=args.ckpt,
        device=device,
        model_type=args.model_type,
        mock_text=args.mock_text,
    )

    exporter.export(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        split=args.split,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        export_gt=args.export_gt,
        fmt=args.format,
        crs=args.crs,
    )

    print(f"\n[Done] 所有文件已保存至：{os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()