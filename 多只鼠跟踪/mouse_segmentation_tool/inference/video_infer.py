#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频分割推理模块
功能：对视频进行分割推理
"""

import cv2
import numpy as np
from mmseg.apis import init_model, inference_model
import os


def infer_video(video_path, config_path, checkpoint_path, output_path):
    """
    对视频进行分割推理
    
    Args:
        video_path (str): 视频路径
        config_path (str): 配置文件路径
        checkpoint_path (str): 模型权重路径
        output_path (str): 输出结果路径
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在 - {config_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"错误：模型权重文件不存在 - {checkpoint_path}")
        return
    
    print(f"正在加载模型...")
    model = init_model(config_path, checkpoint_path, device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    
    print(f"正在处理视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理视频帧
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理当前帧
        result = inference_model(model, frame)
        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
        
        # 创建彩色掩码
        mask_colored = np.zeros_like(frame)
        mouse_color = [0, 255, 0]  # 绿色
        mask_colored[pred_mask == 1] = mouse_color
        
        # 叠加结果
        overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
        
        # 写入视频
        out.write(overlay)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理进度: {frame_count}/{total_frames}")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"视频分割结果已保存到: {output_path}")
    print("视频处理完成！")