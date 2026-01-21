#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频帧提取模块
功能：从视频中提取帧作为训练数据
"""

import cv2
import os


def extract_frames(video_paths, output_dir, start_time_sec=900, frames_per_video=20, interval=30):
    """
    从视频中提取帧作为训练数据
    
    Args:
        video_paths (list): 视频文件路径列表
        output_dir (str): 输出目录
        start_time_sec (int): 开始时间（秒）
        frames_per_video (int): 每个视频提取的帧数
        interval (int): 帧间隔
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始批量处理 {len(video_paths)} 个视频...")
    print(f"⏱️  起始时间: {start_time_sec}s | 每个视频截取: {frames_per_video} 张")

    total_saved = 0

    for video_path in video_paths:
        # 获取文件名作为图片前缀
        video_name = os.path.basename(video_path).split('.')[0]
        print(f"\n🎥 正在处理: {video_name}")

        if not os.path.exists(video_path):
            print(f"   ❌ 错误: 找不到文件，跳过！")
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 计算起始帧位置
        start_frame = int(start_time_sec * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame >= total_frames:
            print(f"   ⚠️ 警告: 视频时长不足 {start_time_sec}秒，跳过！")
            cap.release()
            continue

        # 跳转到指定位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        saved_count = 0
        current_frame = start_frame

        while saved_count < frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔指定间隔保存一次
            if (current_frame - start_frame) % interval == 0:
                # 文件名格式: 视频名_序号.jpg
                save_name = f"{video_name}_{saved_count + 1:02d}.jpg"
                cv2.imwrite(os.path.join(output_dir, save_name), frame)
                saved_count += 1
                total_saved += 1
                print(f"\r   📸 已保存: {saved_count}/{frames_per_video}", end="")

            current_frame += 1

        cap.release()

    print(f"\n\n✅ 批量处理完成！")
    print(f"📂 总共保存了 {total_saved} 张图片到 {output_dir}")
    print("👉 下一步：请打开 Labelme，导入该文件夹开始标注吧！")