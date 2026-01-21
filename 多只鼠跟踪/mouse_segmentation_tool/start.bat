@echo off
chcp 65001 >nul
cls
echo ================= 鼠标分割工具 =================
echo 0. 安装依赖
echo 1. 提取视频帧
echo 2. 转换标注
echo 3. 检查标注
echo 4. 训练模型
echo 5. 图像推理
echo 6. 视频分割
echo 7. 视频分析
echo 8. 启动Web界面
echo 9. 退出
echo ===============================================
set /p choice=请选择要执行的功能（输入数字）: 

if "%choice%"=="0" (
    echo =============== 安装依赖 ===============
    echo 1. 创建并激活Conda环境
    echo 2. 直接安装依赖（已有环境）
    set /p install_choice=
    
    if "%install_choice%"=="1" (
        echo 正在创建Conda环境...
        conda create -n mouse_seg python=3.8 -y
        echo 正在激活Conda环境...
        conda activate mouse_seg
    )
    
    echo 步骤1: 检查CUDA版本...
    nvidia-smi
    echo 请根据上述输出中的CUDA版本选择合适的PyTorch版本
    echo 1. CUDA 12.9 / 12.8 / 12.7 / 12.6 / 12.5 / 12.4 / 12.3 / 12.2 / 12.1 / 12.0
    echo 2. CUDA 11.8 / 11.7
    echo 3. CUDA 11.6 / 11.5
    echo 4. CUDA 11.4 / 11.3 / 11.2
    echo 5. CUDA 10.2
    echo 6. CUDA 10.1 / 10.0
    echo 7. 仅CPU版本
    set /p cuda_choice=
    
    echo 步骤2: 安装PyTorch...
    if "%cuda_choice%"=="1" (
        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ) else if "%cuda_choice%"=="2" (
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
    ) else if "%cuda_choice%"=="3" (
        pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
    ) else if "%cuda_choice%"=="4" (
        pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
    ) else if "%cuda_choice%"=="5" (
        pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 --index-url https://download.pytorch.org/whl/cu102
    ) else if "%cuda_choice%"=="6" (
        pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --index-url https://download.pytorch.org/whl/cu101
    ) else (
        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
    )
    
    echo 步骤3: 安装OpenMIM...
    pip install -U openmim
    
    echo 步骤4: 安装MMCV...
    mim install mmcv-full==2.1.0
    
    echo 步骤5: 安装MMSegmentation...
    mim install mmsegmentation==1.3.0
    
    echo 步骤6: 安装其他依赖...
    pip install -r requirements.txt
    
    echo 依赖安装完成！
    pause
    goto :menu
) else if "%choice%"=="1" (
    echo 请输入视频文件路径（多个视频用空格分隔）:
    set /p videos=
    echo 请输入输出目录（默认: mouse_dataset/images）:
    set /p output=
    if "%output%"=="" set output=mouse_dataset/images
    python main.py extract --videos %videos% --output %output%
) else if "%choice%"=="2" (
    echo 请输入JSON标注文件目录（默认: mouse_dataset/images）:
    set /p json_dir=
    if "%json_dir%"=="" set json_dir=mouse_dataset/images
    echo 请输入输出掩码目录（默认: mouse_dataset/annotations）:
    set /p output_dir=
    if "%output_dir%"=="" set output_dir=mouse_dataset/annotations
    python main.py convert --json-dir %json_dir% --output-dir %output_dir%
) else if "%choice%"=="3" (
    echo 请输入标注目录（默认: mouse_dataset/annotations）:
    set /p ann_dir=
    if "%ann_dir%"=="" set ann_dir=mouse_dataset/annotations
    python main.py check --ann-dir %ann_dir%
) else if "%choice%"=="4" (
    echo 请选择模型类型:
    echo 1. U-Net (默认，速度快)
    echo 2. SegFormer (精度高，速度慢)
    set /p model_choice=
    if "%model_choice%"=="2" (
        python main.py train --model-type segformer
    ) else (
        python main.py train --model-type unet
    )
) else if "%choice%"=="5" (
    echo 请输入图像路径:
    set /p img=
    echo 请输入模型权重路径:
    set /p checkpoint=
    echo 请输入输出图像路径（默认: result.png）:
    set /p output=
    if "%output%"=="" set output=result.png
    python main.py infer-image --image %img% --checkpoint %checkpoint% --output %output%
) else if "%choice%"=="6" (
    echo 请输入视频路径:
    set /p video=
    echo 请输入模型权重路径:
    set /p checkpoint=
    echo 请输入输出视频路径（默认: output_video.avi）:
    set /p output=
    if "%output%"=="" set output=output_video.avi
    python main.py infer-video --video %video% --checkpoint %checkpoint% --output %output%
) else if "%choice%"=="7" (
    echo 请输入视频路径:
    set /p video=
    echo 请输入模型权重路径:
    set /p checkpoint=
    echo 请输入输出视频路径（默认: analyzed_video.avi）:
    set /p output_video=
    if "%output_video%"=="" set output_video=analyzed_video.avi
    python main.py analyze-video --video %video% --checkpoint %checkpoint% --output-video %output_video%
) else if "%choice%"=="8" (
    echo 正在启动Web界面...
    streamlit run app.py
) else if "%choice%"=="9" (
    exit
) else (
    echo 输入无效，请重新选择。
    pause
    goto :menu
)

:menu
pause
cls
echo ================= 鼠标分割工具 =================
echo 0. 安装依赖
echo 1. 提取视频帧
echo 2. 转换标注
echo 3. 检查标注
echo 4. 训练模型
echo 5. 图像推理
echo 6. 视频分割
echo 7. 视频分析
echo 8. 启动Web界面
echo 9. 退出
echo ===============================================
set /p choice=请选择要执行的功能（输入数字）: 
goto :EOF
