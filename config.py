"""
config.py — 超参数与全局配置
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiffusionConfig:
    # ── 扩散过程 ──────────────────────────────────────
    T: int = 1000               # 总时步数
    beta_start: float = 1e-4   # 线性噪声调度起点
    beta_end: float = 0.02     # 线性噪声调度终点

    # ── 对象表示 ──────────────────────────────────────
    max_buildings: int = 96    # 每个街区最多建筑数 N
    # 每栋建筑特征维度：cx, cy, w, h, sin_theta, cos_theta + num_classes
    num_classes: int = 2      # 建筑类型数量（住宅/商业/工业/公共/绿地/停车/混合/空）8   # 
    # 注：'空'类用于 padding，索引 = num_classes-1
    bbox_dim = 7
    loc_dim = 2
    size_dim = 3
    angle_dim = 2
    shape_feature_dim = 0  # 后续考虑加入形状编码 建筑底座
    # obj_dim = 6 + num_classes，在代码中动态计算

    # ── 边界编码器 ────────────────────────────────────
    boundary_points: int = 64  # 边界多边形采样点数
    boundary_hidden: int = 256 # 边界 MLP 隐层维度
    boundary_out: int = 256    # 边界嵌入维度
    blk_extra_channel = 1 # 用于将scale作为参数拼接进去---暂时还没拼进去

    # ── 文本编码器 ────────────────────────────────────
    # text_model: str = "bert-base-uncased"  # 自动从huggingface上拉取
    text_model: str = "/root/project/BLG/bert-base-uncased"  # 修改为本地路径
    
    text_out: int = 256        # 线性投影后维度（原始 768 → 256）
    freeze_bert: bool = True   # 是否冻结 BERT 参数

    # ── Transformer 去噪器 ────────────────────────────
    d_model: int = 256         # 内部特征维度
    nhead: int = 8             # 多头注意力头数
    num_layers: int = 6        # Transformer 块数量
    dim_feedforward: int = 1024
    dropout: float = 0.0
    # 时步嵌入维度
    time_emb_dim: int = 256

    # ── 损失权重 ──────────────────────────────────────
    lambda_iou: float = 0.1    # IoU 惩罚权重（仿 DiffuScene Eq.10）

    # ── 训练 ──────────────────────────────────────────
    batch_size: int = 1024 # 1024 #4 小样本测试 测试一下绘图函数
    lr: float = 8e-4
    # epochs: int = 18000  # 先不管epoch 就是不管过了即便数据集 只看训练的step
    max_steps: int = 30000 #20000  50000
    lr_decay_rate: float = 0.9
    lr_decay_every: int = 500  # 每 N steps 衰减一次
    grad_clip: float = 1.0 # 原本是1.0

    # ── 数据 ──────────────────────────────────────────
    data_root: str = "./data/blocks"
    train_split: float = 0.95 #1.0
    num_workers: int = 0

    # ── 推理 ──────────────────────────────────────────
    inference_steps: int = 1000 # 100  # DDPM 推理步数（可用 DDIM 加速）
    use_ddim: bool = True
    ddim_steps: int = 100

    # 新添加 若使用FlowMatching
    fm_sigma_min = 1e-4
    num_train_timesteps = 1000
    fm_sample_steps = 50

    # 新添加 用于复现DiffuScene--这个也忘记在哪用了 可能是diffusion里面的参数吧 可能也还没用
    backbone: str = 'transformer' # ['transformer', 'unet1d']
    # 使用diffusion还是flow matching等的参数选择---暂时没用到 后面再改？
    network_type: str = 'flow_matching' # ['diffusion', 'flow_matching']
    num_levels: int = 3 # 复现DiffuScene时的Unet1d参数
    
    # 新添加 用于OT-CFM---模型里固定写了 这里还没有用到
    ot_reg: float = 0.05  # OT正则化参数
    ot_method: str = 'sinkhorn'  # OT方法选择 ['sinkhorn', 'emd', 'auction']



# 默认全局配置实例
CFG = DiffusionConfig()

# 动态计算对象特征维度
CFG.obj_dim = CFG.bbox_dim + CFG.num_classes  # [cx,cy,w,h,sin,cos, cls_onehot×8]
