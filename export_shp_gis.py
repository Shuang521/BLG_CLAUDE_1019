#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成功版本！！ 注意数据要去check_data.ipynb里面再转一次geojson！！
export_shp.py — 将 BlockLayout 模型输出导出为带有真实 GIS 坐标的 SHP / GPKG 矢量文件

已升级适配二次标准化缓存 (cache_gis_norm_height_scale.pt)：
  • 自动寻找并载入未归一化的原始 cache_gis.pt 提取真实物理尺度 scale (米)
  • 采用几何特征最近邻匹配（Nearest-Neighbor Matching），完美解决了打乱 Slot 顺序后 GT 建筑高度匹配错位问题
  • 自动将局部归一化 OBB 几何还原到原始 GIS 地球投影坐标
  • 支持导出 WGS84 格式（EPSG:4326），可直接导入 Kepler.gl 预览

用法：
CUDA_VISIBLE_DEVICES=5 python export_shp_gis.py \
    --ckpt /root/project/BLG/checkpoints/rotate_01_pad_zero/rotate_01_loss_v10_height_fm_mask_none/ckpt_050000.pt \
    --cache_path "./data/blocks_final/cache_gis_norm_height_scale.pt" \
    --output_dir ./shp_output/fm_gis_baseline \
    --split val \
    --format gpkg \
    --crs EPSG:32118 \
    --n_samples 100

CUDA_VISIBLE_DEVICES=5 python export_shp_gis.py \
    --ckpt /root/project/BLG/checkpoints/final/fm_tsfm_loss_v11_with_areas/ckpt_030000.pt \
    --cache_path "./data/blocks_final/cache_gis_norm_height_scale.pt" \
    --output_dir ./shp_output/test \
    --split val \
    --format gpkg \
    --crs EPSG:32118 \
    --n_samples 1

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
# 1. 几何与坐标反演工具
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
    """过滤边界点序列中的 (0,0) 填充值"""
    norms = np.linalg.norm(boundary_pts, axis=1)
    valid = norms > eps
    if valid.sum() >= 3:
        return boundary_pts[valid]
    return boundary_pts


def transform_scaled_polygon_to_real(
    poly: Polygon, 
    trans_x: float, 
    trans_y: float, 
    theta: float
) -> Optional[Polygon]:
    """
    将已进行 scale 缩放但在旋转平移前的 Polygon，反演回原始 GIS 真实坐标。
    
    反演公式：
      1. x_rot = x_scaled + trans_x
         y_rot = y_scaled + trans_y
      2. 逆向旋转变换：
         x_real = x_rot * cos(theta) - y_rot * sin(theta)
         y_real = x_rot * sin(theta) + y_rot * cos(theta)
    """
    if poly is None or poly.is_empty:
        return None
    
    coords = np.array(poly.exterior.coords)
    real_coords = []
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    for x_scaled, y_scaled in coords:
        # 平移至旋转对齐坐标系
        x_rot = x_scaled + trans_x
        y_rot = y_scaled + trans_y
        # 旋转还原
        x_real = x_rot * cos_t - y_rot * sin_t
        y_real = x_rot * sin_t + y_rot * cos_t
        real_coords.append((x_real, y_real))
        
    try:
        new_poly = Polygon(real_coords)
        if not new_poly.is_valid:
            new_poly = new_poly.buffer(0)
        return new_poly if not new_poly.is_empty else None
    except Exception:
        return None


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

        # 与 train.py 保持一致
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
        raw_cache_path: Optional[str],
        output_dir: str,
        split: str = "val",
        n_samples: Optional[int] = None,
        batch_size: int = 16,
        seed: int = 42,
        export_gt: bool = False,
        fmt: str = "shp",
        crs: Optional[str] = None,
    ):
        # 1. 载入当前的标准化 Cache (用于 Dataset 分割)
        print(f"[Exporter] 正在载入标准化 Cache 数据源: {cache_path}")
        self.norm_cache_dict = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.height_upper = float(self.norm_cache_dict.get("height_upper", 100.0))

        # 2. 自动检测并载入原始未归一化的原始 Cache 提取物理 scale [3, 5, 6, 8]
        if raw_cache_path is None:
            base_dir = os.path.dirname(cache_path)
            possible_names = ["cache_gis.pt", "cache.pt"]
            detected = None
            for name in possible_names:
                p_path = os.path.join(base_dir, name)
                if os.path.exists(p_path) and os.path.abspath(p_path) != os.path.abspath(cache_path):
                    detected = p_path
                    break
            if detected:
                raw_cache_path = detected
                print(f"[Exporter] 自动检测并载入物理坐标 Cache: {raw_cache_path}")
            else:
                raw_cache_path = cache_path
                print(f"[Exporter] [警告] 未检测到原始物理坐标 Cache，将直接使用当前 Cache（如 Scale 已被标准化，要素可能会缩水）")

        self.raw_cache_dict = torch.load(raw_cache_path, map_location="cpu", weights_only=False)

        # 3. 构造 Dataset 并构建索引映射 (将 split 内的相对索引转换为全局索引)
        dataset = CachedDataset(cache_path, split=split, train_ratio=self.cfg.train_split)
        n_total = len(dataset)
        if n_samples is None or n_samples > n_total:
            n_samples = n_total

        # 映射逻辑
        if hasattr(dataset, "indices"):
            global_indices_map = dataset.indices
        elif hasattr(dataset, "idx_map"):
            global_indices_map = dataset.idx_map
        else:
            total_cache_size = len(self.norm_cache_dict["x0"])
            n_train = int(total_cache_size * self.cfg.train_split)
            if split == "train":
                global_indices_map = list(range(n_train))
            else:
                global_indices_map = list(range(n_train, total_cache_size))
        self.global_indices_map = global_indices_map

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

        # ── 坐标参考系 (CRS) 确定 ──
        out_crs = crs if crs is not None else self.raw_cache_dict.get("crs", "")
        if not out_crs:
            out_crs = None
            print("[Exporter] 未识别到 CRS，将使用无投影本地平面米制坐标")
        else:
            print(f"[Exporter] 使用地理参考坐标系: {out_crs}")

        # ── 驱动与后缀 ──
        suffix = ".gpkg" if fmt == "gpkg" else ".shp"
        driver = "GPKG" if fmt == "gpkg" else "ESRI Shapefile"

        # ── 统一保存并生成 Kepler.gl WGS84 投影副本 ──
        def save_and_reproject(data_list, file_name_base):
            if not data_list:
                return
            gdf = gpd.GeoDataFrame(data_list, crs=out_crs)
            # 保存物理投影矢量（如 UTM）
            primary_path = os.path.join(output_dir, f"{file_name_base}{suffix}")
            gdf.to_file(primary_path, driver=driver, encoding="utf-8")
            print(f"[Exporter] 导出物理坐标数据 -> {primary_path} ({len(gdf)} 个要素)")

            # 如果存在原始投影坐标系，自动另存一份 WGS84 格式（用于 Kepler.gl 直接渲染）
            if out_crs and out_crs.upper() != "EPSG:4326":
                try:
                    wgs84_path = os.path.join(output_dir, f"{file_name_base}_wgs84{suffix}")
                    gdf_wgs84 = gdf.to_crs("EPSG:4326")
                    gdf_wgs84.to_file(wgs84_path, driver=driver, encoding="utf-8")
                    print(f"  └─ [推荐] 转换 WGS84 格式 (Kepler.gl 专用) -> {wgs84_path}")
                except Exception as e:
                    print(f"  └─ [警告] WGS84 重投影失败: {e}")

        # 保存生成结果
        save_and_reproject(gen_buildings, "buildings_gen")
        save_and_reproject(gen_boundaries, "boundaries_gen")

        # 保存 GT 结果
        if export_gt:
            save_and_reproject(gt_buildings, "buildings_gt")
            save_and_reproject(gt_boundaries, "boundaries_gt")

        # 保存元数据
        meta = {
            "n_samples": int(n_samples),
            "split": split,
            "n_gen_buildings": len(gen_buildings),
            "n_gen_boundaries": len(gen_boundaries),
            "export_gt": export_gt,
            "format": fmt,
            "crs": out_crs,
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
        B = len(batch)
        device = self.device

        boundary_tensor = torch.stack([item["boundary"] for item in batch]).to(device)
        texts = [item.get("text", item.get("texts", "")) for item in batch]
        areas = torch.stack([item["area_sqm"] for item in batch]).to(device)  # 读取面积 适配新的模型

        # 1. 直接从原始无标准化 raw_cache_dict 中提取反演所需参数与物理 scale [3, 5, 6, 8]
        trans_x_all = self.raw_cache_dict.get("trans_x", None)
        trans_y_all = self.raw_cache_dict.get("trans_y", None)
        theta_all = self.raw_cache_dict.get("theta", None)
        block_ids_all = self.raw_cache_dict.get("block_ids", None)
        source_files_all = self.raw_cache_dict.get("source_files", None)
        boundary_raw_all = self.raw_cache_dict.get("boundary_raw", None)
        heights_raw_all = self.raw_cache_dict.get("heights_raw", None)
        
        # ── 模型的推理预测 ──
        if not export_gt:
            has_cached = "text_seq" in batch[0]
            with torch.no_grad():
                if has_cached:
                    pre_seq = torch.stack([item["text_seq"] for item in batch]).to(device)
                    pre_mask = torch.stack([item["text_mask"] for item in batch]).to(device)
                    raw = self.model.sample_with_text_cache(
                        boundary_tensor, texts, pre_seq, pre_mask, areas=areas
                    )
                else:
                    raw = self.model.sample(boundary_tensor, texts, areas=areas)

        buildings: List[Dict] = []
        boundaries: List[Dict] = []

        for b in range(B):
            split_idx = int(global_indices[b])
            g_idx = int(self.global_indices_map[split_idx])  # 获取全局索引
            
            # 使用未标准化的真实物理尺度还原坐标
            scale = float(self.raw_cache_dict["scale"][g_idx])
            text = texts[b]

            tx = float(trans_x_all[g_idx]) if trans_x_all is not None else 0.0
            ty = float(trans_y_all[g_idx]) if trans_y_all is not None else 0.0
            th = float(theta_all[g_idx]) if theta_all is not None else 0.0
            block_id = block_ids_all[g_idx] if block_ids_all is not None else f"idx_{g_idx}"
            source_file = source_files_all[g_idx] if source_files_all is not None else ""

            # ── 2.1 街区边界反演 ──
            bnd_poly = None
            if boundary_raw_all is not None and g_idx < len(boundary_raw_all):
                raw_bnd_pts = boundary_raw_all[g_idx]
                if len(raw_bnd_pts) >= 3:
                    try:
                        bnd_poly = Polygon(raw_bnd_pts)
                        if not bnd_poly.is_valid:
                            bnd_poly = bnd_poly.buffer(0)
                    except Exception:
                        bnd_poly = None

            if bnd_poly is None or bnd_poly.is_empty:
                bnd_pts = batch[b]["boundary"].cpu().numpy()
                bnd_pts = filter_boundary_padding(bnd_pts)
                bnd_pts_real = bnd_pts * scale
                if len(bnd_pts_real) >= 3:
                    try:
                        bnd_poly_scaled = Polygon(bnd_pts_real)
                        bnd_poly = transform_scaled_polygon_to_real(bnd_poly_scaled, tx, ty, th)
                    except Exception:
                        bnd_poly = None

            if bnd_poly is not None and not bnd_poly.is_empty:
                boundaries.append({
                    "geometry": bnd_poly,
                    "smp_idx": split_idx,
                    "global_idx": g_idx,
                    "block_id": block_id,
                    "text": text,
                    "scale": scale,
                    "source_file": source_file,
                })

            # ── 2.2 建筑反归一化与打乱 Slot 高度匹配 ──
            if export_gt:
                x0 = batch[b]["x0"]  # [96, 9] (来自 Dataset，处于二次标准化并打乱 Slot 状态)
                bld_list = _decode_x0(x0, self.cfg)
                src_tag = "gt"
            else:
                bld_list = _decode_raw(raw, b, self.cfg)
                src_tag = "gen"

            for idx_bld, bld in enumerate(bld_list):
                if bld["cls_id"] == EMPTY_CLS:
                    continue

                # cx, cy, w, h 在解码出来后恢复在 0~1 的局部几何空间
                cx = bld["cx"] * scale
                cy = bld["cy"] * scale
                w = bld["w"] * scale
                h = bld["h"] * scale
                angle = bld["angle"]

                # ── 高度还原策略 ──
                if export_gt and heights_raw_all is not None:
                    # 采用 Nearest-Neighbor (最近邻) 特征检索解决打乱问题 [3, 5, 6, 8]
                    # 我们知道被打乱前原始未截断坐标的 cx_norm, cy_norm, w_norm, h_norm 存放在 raw_cache_dict['x0'][g_idx] 里
                    orig_x0 = self.raw_cache_dict["x0"][g_idx].cpu().numpy()  # [96, D]
                    
                    # 提取解码后的归一化特征
                    bld_geom = np.array([bld["cx"], bld["cy"], bld["w"], bld["h"]], dtype=np.float32)
                    
                    # 检索原始 x0 的前四列，寻找欧氏距离最接近的槽位
                    dists = np.sum((orig_x0[:, :4] - bld_geom) ** 2, axis=1)
                    orig_idx = int(np.argmin(dists))
                    
                    # 匹配成功，从 heights_raw 中检索原始无截断高度
                    if orig_idx < len(heights_raw_all[g_idx]):
                        height = float(heights_raw_all[g_idx][orig_idx])
                    else:
                        height = bld.get("height", 0.0) * self.height_upper
                else:
                    # 生成建筑：使用 height_norm * height_upper 恢复
                    height = bld.get("height", 0.0) * self.height_upper

                poly_scaled = make_building_polygon(cx, cy, w, h, angle)
                if poly_scaled is None:
                    continue

                # 投影到真实投影平面
                poly = transform_scaled_polygon_to_real(poly_scaled, tx, ty, th)
                if poly is None:
                    continue

                cls_id = bld["cls_id"]
                cls_name = bld.get(
                    "cls_name",
                    CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unk",
                )

                buildings.append({
                    "geometry": poly,
                    "smp_idx": split_idx,
                    "global_idx": g_idx,
                    "block_id": block_id,
                    "src": src_tag,
                    "cls_id": int(cls_id),
                    "cls_name": cls_name,
                    "height": float(height),      # 核心高度（米）
                    "angle_pred": float(angle),
                    "width": float(w),
                    "depth": float(h),
                    "cx_real": float(poly.centroid.x),
                    "cy_real": float(poly.centroid.y),
                    "area": float(poly.area),
                    "prob": float(bld.get("prob", 1.0)),
                    "text": text,
                    "source_file": source_file,
                })

        return buildings, boundaries


# ══════════════════════════════════════════════════════════
# 3. 命令行入口
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="适配二次标准化的 BlockLayout -> GIS 矢量导出")
    p.add_argument("--ckpt", required=True, help="模型 checkpoint 路径")
    p.add_argument("--cache_path", required=True, help="二次标准化 cache_gis_norm_height_scale.pt 路径")
    p.add_argument("--raw_cache_path", default=None, help="原始未标准化 cache_gis.pt 路径。不填将自动同目录检索")
    p.add_argument("--output_dir", required=True, help="输出目录")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--n_samples", type=int, default=None, help="导出样本数，默认全部")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model_type",
        default="flow_matching",
        choices=["flow_matching", "block_layout"],
    )
    p.add_argument("--mock_text", action="store_true")
    p.add_argument(
        "--export_gt",
        action="store_true",
        help="同时导出 GT，在 GIS 叠置对比",
    )
    p.add_argument(
        "--format",
        default="shp",
        choices=["shp", "gpkg"],
        help="shp: 格式通用；gpkg: 推荐（UTF-8 支持良好、字段无限制且方便 Kepler 直接载入）",
    )
    p.add_argument(
        "--crs",
        default=None,
        help="自定义坐标参考系(如 EPSG:32650)。若不填会自动从原始 cache.pt 获取",
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
        raw_cache_path=args.raw_cache_path,
        output_dir=args.output_dir,
        split=args.split,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        export_gt=args.export_gt,
        fmt=args.format,
        crs=args.crs,
    )

    print(f"\n[Done] 导出流程运行完成。文件已保存至：{os.path.abspath(args.output_dir)}")
    print("提示: 后缀包含 '_wgs84' 的矢量文件可直接载入 Kepler.gl 网站，实现真实的 3D Extrusion 大屏预览。")


if __name__ == "__main__":
    main()