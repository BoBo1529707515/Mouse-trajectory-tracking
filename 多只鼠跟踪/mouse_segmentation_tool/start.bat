@echo off
cls
echo ================= 鼠标分割工具 =================
echo 1. 提取视频帧
echo 2. 转换标注
echo 3. 检查标注
echo 4. 训练模型
echo 5. 图像推理
echo 6. 视频分割
echo 7. 视频分析
echo 0. 退出
echo ===============================================
set /p choice=请选择要执行的功能（输入数字）: 

if "%choice%"=="1" (
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
) else if "%choice%"=="0" (
    exit
) else (
    echo 输入无效，请重新选择。
    pause
    %0
)
pause
