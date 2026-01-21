import streamlit as st
import os
import subprocess
import numpy as np
import cv2
from PIL import Image

# 设置页面标题和布局
st.set_page_config(
    page_title="鼠标分割工具",
    page_icon="🐭",
    layout="wide"
)

# 页面标题
st.title("🐭 鼠标分割工具")

# 侧边栏功能选择
option = st.sidebar.selectbox(
    "选择功能",
    ["主页", "提取视频帧", "转换标注", "检查标注", "训练模型", "图像推理", "视频分割", "视频分析"]
)

# 主页
if option == "主页":
    st.header("欢迎使用鼠标分割工具")
    st.write("这是一个基于MMSegmentation的鼠标分割任务工具，支持从视频帧提取、标注转换到模型训练和推理的完整流程。")
    
    st.subheader("功能特点")
    features = [
        "📹 视频帧提取：从视频中提取帧作为训练数据",
        "🖼️ 标注转换：将LabelMe标注转换为掩码图像",
        "✅ 标注检查：验证标注数据质量",
        "🏋️ 模型训练：训练鼠标分割模型",
        "🎯 图像推理：对单张图像进行分割",
        "📽️ 视频分割：对视频进行分割处理",
        "📊 视频分析：分析小鼠交互行为"
    ]
    for feature in features:
        st.write(feature)
    
    st.subheader("使用流程")
    st.write("1. 提取视频帧 → 2. 使用LabelMe标注 → 3. 转换标注 → 4. 检查标注 → 5. 训练模型 → 6. 推理/分析")

# 提取视频帧
elif option == "提取视频帧":
    st.header("📹 提取视频帧")
    videos = st.text_area("视频文件路径（多个路径用换行分隔）")
    output_dir = st.text_input("输出目录", "mouse_dataset/images")
    start_time = st.number_input("开始时间（秒）", min_value=0, value=900)
    frames_per_video = st.number_input("每个视频提取帧数", min_value=1, value=20)
    interval = st.number_input("帧间隔", min_value=1, value=30)
    
    if st.button("开始提取"):
        video_list = [v.strip() for v in videos.split("\n") if v.strip()]
        if video_list:
            cmd = f"python main.py extract --videos {' '.join(video_list)} --output {output_dir} --start-time {start_time} --frames-per-video {frames_per_video} --interval {interval}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            st.text(result.stdout)
            if result.stderr:
                st.error(result.stderr)
        else:
            st.error("请输入视频文件路径")

# 转换标注
elif option == "转换标注":
    st.header("🖼️ 转换标注")
    json_dir = st.text_input("JSON标注目录", "mouse_dataset/images")
    output_dir = st.text_input("输出掩码目录", "mouse_dataset/annotations")
    
    if st.button("开始转换"):
        cmd = f"python main.py convert --json-dir {json_dir} --output-dir {output_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)

# 检查标注
elif option == "检查标注":
    st.header("✅ 检查标注")
    ann_dir = st.text_input("标注目录", "mouse_dataset/annotations")
    
    if st.button("开始检查"):
        cmd = f"python main.py check --ann-dir {ann_dir}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)

# 训练模型
elif option == "训练模型":
    st.header("🏋️ 训练模型")
    image_dir = st.text_input("图像目录", "mouse_dataset/images")
    ann_dir = st.text_input("标注目录", "mouse_dataset/annotations")
    output_dir = st.text_input("输出目录", "work_dirs/mouse_segmentation")
    config = st.text_input("配置文件", "configs/mouse_segmentation_config.py")
    model_type = st.selectbox("模型类型", ["unet", "segformer"], index=0)
    
    if st.button("开始训练"):
        cmd = f"python main.py train --image-dir {image_dir} --ann-dir {ann_dir} --output-dir {output_dir} --config {config} --model-type {model_type}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.text(result.stdout)
        if result.stderr:
            st.error(result.stderr)

# 图像推理
elif option == "图像推理":
    st.header("🎯 图像推理")
    uploaded_file = st.file_uploader("上传图像", type=["jpg", "jpeg", "png"])
    checkpoint = st.file_uploader("上传模型权重", type=["pth"])
    output = st.text_input("输出图像路径", "result.png")
    
    if uploaded_file and checkpoint:
        # 保存上传的文件
        img_save_path = os.path.join("temp", uploaded_file.name)
        checkpoint_save_path = os.path.join("temp", checkpoint.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(img_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open(checkpoint_save_path, "wb") as f:
            f.write(checkpoint.getbuffer())
        
        if st.button("开始推理"):
            cmd = f"python main.py infer-image --image {img_save_path} --checkpoint {checkpoint_save_path} --output {output}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            st.text(result.stdout)
            if result.stderr:
                st.error(result.stderr)
            # 显示结果
            if os.path.exists(output):
                st.image(output, caption="分割结果")

# 视频分割
elif option == "视频分割":
    st.header("📽️ 视频分割")
    video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    checkpoint = st.file_uploader("上传模型权重", type=["pth"])
    output = st.text_input("输出视频路径", "output_video.avi")
    
    if video_file and checkpoint:
        # 保存上传的文件
        video_save_path = os.path.join("temp", video_file.name)
        checkpoint_save_path = os.path.join("temp", checkpoint.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(video_save_path, "wb") as f:
            f.write(video_file.getbuffer())
        with open(checkpoint_save_path, "wb") as f:
            f.write(checkpoint.getbuffer())
        
        if st.button("开始分割"):
            cmd = f"python main.py infer-video --video {video_save_path} --checkpoint {checkpoint_save_path} --output {output}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            st.text(result.stdout)
            if result.stderr:
                st.error(result.stderr)
            # 提供下载链接
            if os.path.exists(output):
                st.success(f"视频分割完成，结果已保存到: {output}")
                with open(output, "rb") as f:
                    st.download_button("下载结果视频", f, file_name=output)

# 视频分析
elif option == "视频分析":
    st.header("📊 视频分析")
    video_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    checkpoint = st.file_uploader("上传模型权重", type=["pth"])
    output_video = st.text_input("输出视频路径", "analyzed_video.avi")
    
    if video_file and checkpoint:
        # 保存上传的文件
        video_save_path = os.path.join("temp", video_file.name)
        checkpoint_save_path = os.path.join("temp", checkpoint.name)
        os.makedirs("temp", exist_ok=True)
        
        with open(video_save_path, "wb") as f:
            f.write(video_file.getbuffer())
        with open(checkpoint_save_path, "wb") as f:
            f.write(checkpoint.getbuffer())
        
        if st.button("开始分析"):
            cmd = f"python main.py analyze-video --video {video_save_path} --checkpoint {checkpoint_save_path} --output-video {output_video}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            st.text(result.stdout)
            if result.stderr:
                st.error(result.stderr)
            # 提供下载链接
            if os.path.exists(output_video):
                st.success(f"视频分析完成，结果已保存到: {output_video}")
                with open(output_video, "rb") as f:
                    st.download_button("下载分析结果视频", f, file_name=output_video)
