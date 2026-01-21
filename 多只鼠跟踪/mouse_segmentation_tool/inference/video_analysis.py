#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频分析模块
功能：分析视频中小鼠的交互行为，计算距离、交互次数等
"""

import cv2
import numpy as np
from mmseg.apis import init_model, inference_model
import os
import csv
import matplotlib.pyplot as plt
from scipy.spatial import distance


def analyze_video(video_path, config_path, checkpoint_path, output_video, output_csv, output_plot):
    """
    分析视频中小鼠的交互行为
    
    Args:
        video_path (str): 视频路径
        config_path (str): 配置文件路径
        checkpoint_path (str): 模型权重路径
        output_video (str): 输出视频路径
        output_csv (str): 输出CSV路径
        output_plot (str): 输出图表路径
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
    
    print(f"正在分析视频: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 分析参数
    MIN_MOUSE_SIZE = 2000  # 最小鼠标面积
    MAX_MOUSE_SIZE = 50000  # 最大鼠标面积
    INTERACTION_THRESHOLD = 50  # 交互距离阈值
    
    # 存储分析数据
    analysis_data = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interaction_count = 0
    
    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理当前帧
        result = inference_model(model, frame)
        pred_mask = result.pred_sem_seg.data.cpu().numpy()[0]
        
        # 寻找小鼠轮廓
        contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小鼠轮廓
        mice_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_MOUSE_SIZE < area < MAX_MOUSE_SIZE:
                mice_contours.append(contour)
        
        # 计算每个小鼠的质心
        mice_centroids = []
        for contour in mice_contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                mice_centroids.append((cx, cy))
        
        # 计算小鼠之间的距离
        distances = []
        if len(mice_centroids) >= 2:
            for i in range(len(mice_centroids)):
                for j in range(i + 1, len(mice_centroids)):
                    dist = distance.euclidean(mice_centroids[i], mice_centroids[j])
                    distances.append(dist)
                    
                    # 检查是否有交互
                    if dist < INTERACTION_THRESHOLD:
                        interaction_count += 1
                        # 绘制交互线
                        cv2.line(frame, mice_centroids[i], mice_centroids[j], (0, 0, 255), 2)
        
        # 绘制小鼠轮廓和质心
        for i, (contour, centroid) in enumerate(zip(mice_contours, mice_centroids)):
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"Mouse {i+1}", (centroid[0] + 10, centroid[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 计算平均距离
        avg_distance = np.mean(distances) if distances else 0
        
        # 存储分析数据
        analysis_data.append({
            'frame': frame_count,
            'time': frame_count / fps,
            'mouse_count': len(mice_centroids),
            'avg_distance': avg_distance,
            'interaction_count': interaction_count
        })
        
        # 写入视频
        out.write(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理进度: {frame_count}/{total_frames}")
    
    # 释放资源
    cap.release()
    out.release()
    
    # 保存分析数据到CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['frame', 'time', 'mouse_count', 'avg_distance', 'interaction_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_data)
    
    # 生成距离变化图表
    times = [d['time'] for d in analysis_data]
    avg_distances = [d['avg_distance'] for d in analysis_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, avg_distances, label='Average Distance')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (pixels)')
    plt.title('Mouse Distance Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_plot)
    plt.close()
    
    print(f"视频分析结果已保存到: {output_video}")
    print(f"分析数据已保存到: {output_csv}")
    print(f"距离变化图表已保存到: {output_plot}")
    print(f"总交互次数: {interaction_count}")
    print("视频分析完成！")