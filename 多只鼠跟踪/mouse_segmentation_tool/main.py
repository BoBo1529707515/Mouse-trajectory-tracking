#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鼠标分割工具主入口脚本
支持从视频帧提取、标注转换到模型训练和推理的完整流程
"""

import argparse
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.extractor import extract_frames
from data.converter import convert_annotations
from data.checker import check_annotations
from training.trainer import train_model
from inference.image_infer import infer_image
from inference.video_infer import infer_video
from inference.video_analysis import analyze_video


def main():
    """主函数，处理命令行参数并调用相应功能"""
    parser = argparse.ArgumentParser(description='Mouse Segmentation Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 视频帧提取命令
    extract_parser = subparsers.add_parser('extract', help='Extract frames from videos')
    extract_parser.add_argument('--videos', nargs='+', help='List of video paths')
    extract_parser.add_argument('--output', default='mouse_dataset/images', help='Output directory')
    extract_parser.add_argument('--start-time', type=int, default=900, help='Start time in seconds')
    extract_parser.add_argument('--frames-per-video', type=int, default=20, help='Frames per video')
    extract_parser.add_argument('--interval', type=int, default=30, help='Frame interval')
    
    # 标注转换命令
    convert_parser = subparsers.add_parser('convert', help='Convert LabelMe annotations to masks')
    convert_parser.add_argument('--json-dir', default='mouse_dataset/images', help='Directory with JSON files')
    convert_parser.add_argument('--output-dir', default='mouse_dataset/annotations', help='Output directory for masks')
    
    # 标注检查命令
    check_parser = subparsers.add_parser('check', help='Check annotation quality')
    check_parser.add_argument('--ann-dir', default='mouse_dataset/annotations', help='Directory with annotation files')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train mouse segmentation model')
    train_parser.add_argument('--image-dir', default='mouse_dataset/images', help='Image directory')
    train_parser.add_argument('--ann-dir', default='mouse_dataset/annotations', help='Annotation directory')
    train_parser.add_argument('--output-dir', default='work_dirs/mouse_segmentation', help='Output directory for models')
    train_parser.add_argument('--config', default='configs/mouse_segmentation_config.py', help='Config file path')
    train_parser.add_argument('--model-type', default='unet', choices=['unet', 'segformer'], help='Model type')
    
    # 图像推理命令
    infer_image_parser = subparsers.add_parser('infer-image', help='Infer segmentation on single image')
    infer_image_parser.add_argument('--image', required=True, help='Image path')
    infer_image_parser.add_argument('--config', default='configs/mouse_segmentation_config.py', help='Config file path')
    infer_image_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    infer_image_parser.add_argument('--output', default='result.png', help='Output image path')
    
    # 视频推理命令
    infer_video_parser = subparsers.add_parser('infer-video', help='Infer segmentation on video')
    infer_video_parser.add_argument('--video', required=True, help='Video path')
    infer_video_parser.add_argument('--config', default='configs/mouse_segmentation_config.py', help='Config file path')
    infer_video_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    infer_video_parser.add_argument('--output', default='output_video.avi', help='Output video path')
    
    # 视频分析命令
    analyze_video_parser = subparsers.add_parser('analyze-video', help='Analyze mouse interaction in video')
    analyze_video_parser.add_argument('--video', required=True, help='Video path')
    analyze_video_parser.add_argument('--config', default='configs/mouse_segmentation_config.py', help='Config file path')
    analyze_video_parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    analyze_video_parser.add_argument('--output-video', default='analyzed_video.avi', help='Output video path')
    analyze_video_parser.add_argument('--output-csv', default='analysis_data.csv', help='Output CSV path')
    analyze_video_parser.add_argument('--output-plot', default='distance_plot.png', help='Output plot path')
    
    args = parser.parse_args()
    
    # 执行相应命令
    if args.command == 'extract':
        extract_frames(args.videos, args.output, args.start_time, args.frames_per_video, args.interval)
    elif args.command == 'convert':
        convert_annotations(args.json_dir, args.output_dir)
    elif args.command == 'check':
        check_annotations(args.ann_dir)
    elif args.command == 'train':
        train_model(args.image_dir, args.ann_dir, args.output_dir, args.config, args.model_type)
    elif args.command == 'infer-image':
        infer_image(args.image, args.config, args.checkpoint, args.output)
    elif args.command == 'infer-video':
        infer_video(args.video, args.config, args.checkpoint, args.output)
    elif args.command == 'analyze-video':
        analyze_video(args.video, args.config, args.checkpoint, args.output_video, args.output_csv, args.output_plot)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()