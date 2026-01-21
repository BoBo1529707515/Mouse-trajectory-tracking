#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注转换模块
功能：将LabelMe标注的JSON文件转换为掩码图像
"""

import os
import json
import numpy as np
import cv2


def convert_annotations(json_dir, output_dir):
    """
    将LabelMe标注的JSON文件转换为掩码图像
    
    Args:
        json_dir (str): JSON标注文件目录
        output_dir (str): 输出掩码图像目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ 错误：在 {json_dir} 中未找到 .json 文件！")
        print("👉 请先使用 Labelme 标注数据。")
        return
    
    print(f"🔄 正在转换 {len(json_files)} 个标注文件...")
    
    converted = 0
    skipped = 0
    
    for filename in json_files:
        json_path = os.path.join(json_dir, filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像尺寸
            img_height = data.get('imageHeight')
            img_width = data.get('imageWidth')
            
            if img_height is None or img_width is None:
                print(f"跳过 {filename}: 缺少图像尺寸信息")
                skipped += 1
                continue
            
            # 创建空白mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # 遍历所有标注形状
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                # 检查是否是mouse标注（不区分大小写）
                if 'mouse' in label or label in ['mouse', 'mice', '小鼠', '鼠']:
                    points_array = np.array(points, dtype=np.int32)
                    
                    if shape_type == 'polygon':
                        cv2.fillPoly(mask, [points_array], 1)
                    elif shape_type == 'rectangle':
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
                    else:
                        # 其他类型也尝试用多边形填充
                        if len(points) >= 3:
                            cv2.fillPoly(mask, [points_array], 1)
            
            # 生成输出文件名（与原图同名，扩展名改为.png）
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + '.png')
            
            cv2.imwrite(output_path, mask)
            
            # 统计mask中的像素
            mouse_pixels = np.sum(mask == 1)
            total_pixels = mask.size
            
            if mouse_pixels > 0:
                print(
                    f"✓ {filename} -> {base_name}.png (mouse像素: {mouse_pixels}, 占比: {mouse_pixels * 100 / total_pixels:.2f}%)"
                )
            else:
                print(f"⚠ {filename} -> {base_name}.png (警告: 没有检测到mouse标注)")
            
            converted += 1
            
        except Exception as e:
            print(f"✗ 处理 {filename} 时出错: {e}")
            skipped += 1
    
    print(f"\n转换完成！")
    print(f"成功转换: {converted} 个文件")
    print(f"跳过: {skipped} 个文件")
    print(f"输出目录: {output_dir}")