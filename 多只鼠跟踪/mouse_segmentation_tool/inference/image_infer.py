#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分割推理模块
功能：对单张图像进行分割推理
"""

import cv2
import numpy as np
from mmseg.apis import init_model, inference_model
import os


def infer_image(image_path, config_path, checkpoint_path, output_path):
    """
    对单张图像进行分割推理
    
    Args:
        image_path (str): 图像路径
        config_path (str): 配置文件路径
        checkpoint_path (str): 模型权重路径
        output_path (str): 输出结果路径
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在 - {image_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在 - {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"错误：模型权重文件不存在 - {checkpoint_path}")
        return
    
    print(f"正在加载模型...")
    model = init_model(config_path, checkpoint_path, device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    
    print(f"正在推理图像: {image_path}")
    result = inference_model(model, image_path)
    
    # 获取预测结果
    pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
    
    # 读取原图
    img = cv2.imread(image_path)
    
    # 创建彩色掩码
    mask_colored = np.zeros_like(img)
    mouse_color = [0, 255, 0]  # 绿色
    mask_colored[pred_mask == 1] = mouse_color
    
    # 叠加结果
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
    
    # 保存结果
    cv2.imwrite(output_path, overlay)
    print(f"分割结果已保存到: {output_path}")
    print("推理完成！")