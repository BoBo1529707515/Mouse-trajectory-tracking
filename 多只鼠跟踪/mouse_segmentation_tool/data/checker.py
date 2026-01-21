#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注检查模块
功能：检查标注数据的质量和完整性
"""

import cv2
import numpy as np
import os


def check_annotations(ann_dir):
    """
    检查标注数据的质量和完整性
    
    Args:
        ann_dir (str): 标注文件目录
    """
    if not os.path.exists(ann_dir):
        print(f"❌ 错误：标注目录 {ann_dir} 不存在！")
        return
    
    # 获取标注文件列表
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.png')]
    
    if not ann_files:
        print(f"❌ 错误：在 {ann_dir} 中未找到 .png 标注文件！")
        return
    
    print(f"🔍 正在检查 {len(ann_files)} 个标注文件...")
    print("=" * 80)
    
    # 统计信息
    total_files = len(ann_files)
    valid_files = 0
    invalid_files = 0
    
    for i, ann_file in enumerate(ann_files[:5]):  # 只检查前5个文件作为示例
        ann_path = os.path.join(ann_dir, ann_file)
        
        try:
            # 读取标注文件
            mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"❌ {ann_file}: 无法读取文件")
                invalid_files += 1
                continue
            
            # 分析标注文件
            unique_values = np.unique(mask)
            mouse_pixels = np.sum(mask == 1)
            total_pixels = mask.size
            mouse_ratio = (mouse_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            # 输出检查结果
            print(f"📄 文件: {ann_file}")
            print(f"   尺寸: {mask.shape[1]}x{mask.shape[0]}")
            print(f"   唯一值: {unique_values}")
            print(f"   Mouse像素: {mouse_pixels}")
            print(f"   Mouse占比: {mouse_ratio:.2f}%")
            
            # 验证标注是否有效
            if 1 in unique_values:
                print(f"   ✅ 标注有效")
                valid_files += 1
            else:
                print(f"   ⚠️  警告：未检测到mouse标注")
                invalid_files += 1
                
        except Exception as e:
            print(f"❌ {ann_file}: 处理错误 - {e}")
            invalid_files += 1
        
        print("-" * 80)
    
    # 检查是否还有更多文件
    if total_files > 5:
        print(f"... 还有 {total_files - 5} 个文件未显示 ...")
    
    # 输出总体统计
    print("=" * 80)
    print("📊 检查结果统计")
    print(f"总文件数: {total_files}")
    print(f"有效文件: {valid_files}")
    print(f"无效文件: {invalid_files}")
    print(f"检查完成！")