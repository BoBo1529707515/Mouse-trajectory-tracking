# 小鼠边缘分割工具

基于MMSegmentation的分割任务工具，支持从视频帧提取、标注转换到模型训练和推理的完整流程。

## 功能特点

- 📹 **视频帧提取**：从视频中提取帧作为训练数据
- 🖼️ **标注转换**：将LabelMe标注转换为掩码图像
- ✅ **标注检查**：验证标注数据质量
- 🏋️ **模型训练**：训练鼠标分割模型
- 🎯 **图像推理**：对单张图像进行分割
- 📽️ **视频分割**：对视频进行分割处理
- 📊 **视频分析**：分析小鼠交互行为，计算距离和交互次数

## 安装说明

### 1. 克隆仓库

```bash
git clone https://github.com/BoBo1529707515/Mouse-trajectory-tracking.git
cd Mouse-trajectory-tracking
cd "多只鼠跟踪"
cd mouse_segmentation_tool
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

```bash
# 提取视频帧
python main.py extract --videos video1.mp4 video2.mp4 --output mouse_dataset/images

# 转换标注
python main.py convert --json-dir mouse_dataset/images --output-dir mouse_dataset/annotations

# 检查标注
python main.py check --ann-dir mouse_dataset/annotations

# 训练模型
python main.py train

# 图像推理
python main.py infer-image --image test.jpg --checkpoint work_dirs/mouse_segmentation/latest.pth --output result.png

# 视频分割
python main.py infer-video --video test.mp4 --checkpoint work_dirs/mouse_segmentation/latest.pth --output output_video.avi

# 视频分析
python main.py analyze-video --video test.mp4 --checkpoint work_dirs/mouse_segmentation/latest.pth --output-video analyzed_video.avi
```

### 批处理脚本（Windows）

双击 `start.bat` 文件，按照菜单提示操作。

### Streamlit Web界面（跨平台）

```bash
streamlit run app.py
```
然后在浏览器中打开提示的地址（通常是 http://localhost:8501）。

## 项目结构

```
mouse_segmentation_tool/
├── README.md              # 项目说明
├── main.py                # 主入口脚本
├── start.bat              # Windows批处理脚本
├── app.py                 # Streamlit Web界面
├── requirements.txt       # 依赖包
├── data/
│   ├── extractor.py       # 视频帧提取
│   ├── converter.py       # 标注转换
│   └── checker.py         # 标注检查
├── training/
│   └── trainer.py         # 模型训练
├── inference/
│   ├── image_infer.py     # 图像推理
│   ├── video_infer.py     # 视频分割
│   └── video_analysis.py  # 视频分析
├── configs/
│   └── mouse_segmentation_config.py  # 配置文件
└── utils/                 # 工具函数（预留）
```

## 工作流程

1. **数据准备**：使用 `extract` 命令从视频中提取帧
2. **标注**：使用 LabelMe 工具对提取的帧进行标注
3. **转换标注**：使用 `convert` 命令将LabelMe标注转换为掩码图像
4. **检查标注**：使用 `check` 命令验证标注数据质量
5. **训练模型**：使用 `train` 命令训练分割模型
6. **推理**：使用 `infer-image` 或 `infer-video` 命令进行分割
7. **分析**：使用 `analyze-video` 命令分析小鼠交互行为

## 注意事项

- 训练模型需要GPU支持，推理视频需要足够的内存
- 标注时请使用 "mouse" 或包含 "mouse" 的标签名称
- 视频分析功能需要至少两个小鼠出现在画面中

## 示例

### 视频帧提取示例

```bash
python main.py extract --videos video1.mp4 --output mouse_dataset/images
```

### 模型训练示例

```bash
python main.py train --image-dir mouse_dataset/images --ann-dir mouse_dataset/annotations --output-dir work_dirs/mouse_segmentation
```

### 视频分析示例

```bash
python main.py analyze-video --video test.mp4 --checkpoint work_dirs/mouse_segmentation/latest.pth --output-video analyzed_video.avi --output-csv analysis_data.csv
```

## 故障排除

- **依赖安装失败**：请确保使用Python 3.8+，并尝试更新pip
- **CUDA错误**：如果没有GPU，请在代码中设置 `device='cpu'`
- **标注转换失败**：请确保LabelMe标注文件存在，且标签名称包含 "mouse"
- **模型训练失败**：请检查标注数据质量，确保每个图像都有对应的掩码文件

## 许可证

本项目基于Apache 2.0许可证开源。
