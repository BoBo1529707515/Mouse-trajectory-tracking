#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块
功能：定义自定义数据集、准备数据、创建配置、启动训练
"""

import os
import shutil
import random
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


# 定义自定义数据集
@DATASETS.register_module()
class MouseDataset(BaseSegDataset):
    """鼠标分割数据集"""
    METAINFO = dict(
        classes=('background', 'mouse'),
        palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)


def prepare_dataset(image_dir, annotation_dir, output_dir, val_ratio=0.2, test_ratio=0.1):
    """
    准备数据集目录结构
    
    Args:
        image_dir (str): 图像目录
        annotation_dir (str): 标注目录
        output_dir (str): 输出目录
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
    
    Returns:
        str: 数据根目录的绝对路径
    """
    # 创建目录结构
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations', 'test'), exist_ok=True)

    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 检查是否已经准备好
    if len(os.listdir(os.path.join(output_dir, 'images', 'train'))) > 0:
        print("数据集看起来已经准备好了，跳过复制步骤。")
        return os.path.abspath(output_dir)

    # 随机打乱文件列表
    random.shuffle(image_files)
    total = len(image_files)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_files = image_files[:(total - val_size - test_size)]
    val_files = image_files[(total - val_size - test_size):(total - test_size)]
    test_files = image_files[(total - test_size):]

    # 复制文件函数
    def copy_files(files, split):
        for file in files:
            img_src = os.path.join(image_dir, file)
            img_dst = os.path.join(output_dir, 'images', split, file)
            ann_file = os.path.splitext(file)[0] + '.png'
            ann_src = os.path.join(annotation_dir, ann_file)
            ann_dst = os.path.join(output_dir, 'annotations', split, ann_file)
            if os.path.exists(ann_src):
                shutil.copy(img_src, img_dst)
                shutil.copy(ann_src, ann_dst)

    # 复制文件到对应目录
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"Dataset split: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return os.path.abspath(output_dir)


def create_config(data_root, config_path, model_type='unet'):
    """
    创建配置文件
    
    Args:
        data_root (str): 数据根目录
        config_path (str): 配置文件路径
        model_type (str): 模型类型，可选 'unet' 或 'segformer'
    """
    data_root = data_root.replace('\\', '/')
    
    if model_type == 'segformer':
        # SegFormer 配置
        config_text = f"""
_base_ = ['mmseg::_base_/default_runtime.py']

# 数据预处理
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512)
)

# SegFormer 模型配置
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 数据集配置
dataset_type = 'MouseDataset'
data_root = '{data_root}'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline
    )
)

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.),
            norm=dict(decay_mult=0.),
            head=dict(lr_mult=10.)
        )
    )
)

# 学习率调度
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=500, end=5000, by_epoch=False)
]

# 训练配置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# 其他设置
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
""".format(data_root=data_root)
    else:
        # U-Net 配置
        config_text = f"""
_base_ = [
    'mmseg::_base_/models/fcn_unet_s5-d16.py', 
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_40k.py'
]
model = dict(
    data_preprocessor=dict(type='SegDataPreProcessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=(512, 512)),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2),
    test_cfg=dict(mode='whole')
)
dataset_type = 'MouseDataset'
data_root = '{data_root}'
crop_size = (512, 512)
train_pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations'), dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True), dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), dict(type='RandomFlip', prob=0.5), dict(type='PackSegInputs')]
test_pipeline = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(512, 512), keep_ratio=False), dict(type='LoadAnnotations'), dict(type='PackSegInputs')]
train_dataloader = dict(batch_size=2, num_workers=0, persistent_workers=False, sampler=dict(type='InfiniteSampler', shuffle=True), dataset=dict(type=dataset_type, data_root=data_root, data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'), pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False, sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(type=dataset_type, data_root=data_root, data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'), pipeline=test_pipeline))
test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False, sampler=dict(type='DefaultSampler', shuffle=False), dataset=dict(type=dataset_type, data_root=data_root, data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'), pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
param_scheduler = [dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=3000, by_epoch=False)]
default_hooks = dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False), param_scheduler=dict(type='ParamSchedulerHook'), checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500), sampler_seed=dict(type='DistSamplerSeedHook'), visualization=dict(type='SegVisualizationHook'))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=3000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_scope = 'mmseg'
visualizer = dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
""".format(data_root=data_root)
    
    # 保存配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_text)


def train_model(image_dir, ann_dir, output_dir, config_path, model_type='unet'):
    """
    训练鼠标分割模型
    
    Args:
        image_dir (str): 图像目录
        ann_dir (str): 标注目录
        output_dir (str): 输出目录
        config_path (str): 配置文件路径
        model_type (str): 模型类型，可选 'unet' 或 'segformer'
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    print("正在准备数据集...")
    data_root = prepare_dataset(image_dir, ann_dir, 'mouse_dataset_processed')

    print(f"正在创建{model_type.upper()}配置文件...")
    create_config(data_root, config_path, model_type)

    print("加载配置...")
    cfg = Config.fromfile(config_path)

    # 创建工作目录
    if model_type == 'segformer':
        work_dir = os.path.join(os.path.dirname(output_dir), 'mouse_segmentation_segformer')
    else:
        work_dir = output_dir
    
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    cfg.work_dir = work_dir
    cfg.device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    print(f"开始训练{model_type.upper()}...")
    runner = Runner.from_cfg(cfg)
    runner.train()

    print(f"训练完成！模型保存在: {work_dir}")
    print("开始测试...")
    runner.test()
    print("所有任务完成！")