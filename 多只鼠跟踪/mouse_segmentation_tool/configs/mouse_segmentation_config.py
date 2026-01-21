# 鼠标分割配置文件
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
data_root = 'mouse_dataset_processed'
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
