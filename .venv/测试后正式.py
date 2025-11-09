import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --- 1. 全局配置与参数 ---

# --- 输入/输出路径 ---
# 您只需要修改下面这一行来更换视频文件！
VIDEO_PATH = r"C:\Users\15297\Desktop\e116a1d3aa9a86211d99a0b826a5b2a9.mp4"

# --- 自动生成输出文件名 ---
base_filename = os.path.basename(VIDEO_PATH)
video_name_without_ext, _ = os.path.splitext(base_filename)
OUTPUT_CSV_PATH = f"{video_name_without_ext}_detailed_trajectory.csv"  # 详细数据
OUTPUT_SUMMARY_PATH = f"{video_name_without_ext}_summary_by_second.csv"  # 秒级总结
OUTPUT_PLOT_PATH = f"{video_name_without_ext}_trajectory_plot.png"  # 轨迹图
OUTPUT_VIDEO_PATH = f"{video_name_without_ext}_tracked.mp4"  # 追踪视频

# --- 功能开关 ---
SHOW_VIDEO_PREVIEW = True  # 关闭预览以最大化速度
SAVE_OUTPUT_VIDEO = True  # 保存带标记的视频以供回顾

# --- 追踪参数 ---
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([50, 255, 117])
kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)
MIN_AREA = 100

# --- 全局变量 (ROI选择) ---
roi_box = None
frame_for_selection = None


def select_roi_callback(event, x, y, flags, param):
    """鼠标回调函数，用于选择ROI"""
    global roi_box, frame_for_selection
    roi_points, selecting_roi = param
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points[0] = (x, y)
        roi_points[1] = None
        selecting_roi[0] = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting_roi[0]:
        img_copy = frame_for_selection.copy()
        cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("1. Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP and roi_points[0] is not None:
        roi_points[1] = (x, y)
        selecting_roi[0] = False
        x1, y1 = min(roi_points[0][0], roi_points[1][0]), min(
            roi_points[0][1], roi_points[1][1]
        )
        x2, y2 = max(roi_points[0][0], roi_points[1][0]), max(
            roi_points[0][1], roi_points[1][1]
        )
        roi_box = (x1, y1, x2, y2)
        cv2.rectangle(frame_for_selection, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("1. Select ROI", frame_for_selection)
        print(f"ROI区域已确定: {roi_box}。请按任意键开始追踪...")


# --- 2. 步骤一：加载视频并选择ROI ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {VIDEO_PATH}")
    exit()

# 获取视频的帧率(FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("警告: 无法获取视频帧率，将默认设置为30 FPS。时间计算可能不准确。")
    fps = 30

ret, first_frame = cap.read()
if not ret:
    print("无法读取视频第一帧")
    cap.release()
    exit()

frame_for_selection = first_frame.copy()
roi_points, selecting_roi = [None, None], [False]
cv2.namedWindow("1. Select ROI")
cv2.setMouseCallback("1. Select ROI", select_roi_callback, (roi_points, selecting_roi))
print("请在 '1. Select ROI' 窗口中用鼠标拖拽来选择一个区域，然后按任意键。")
cv2.imshow("1. Select ROI", frame_for_selection)
cv2.waitKey(0)
cv2.destroyAllWindows()
if roi_box is None:
    print("未选择ROI区域，程序退出。")
    cap.release()
    exit()

# --- 3. 步骤二：初始化追踪和输出 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
raw_trajectory_data = []
video_writer = None

if SAVE_OUTPUT_VIDEO:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height)
    )

# --- 4. 步骤三：循环处理视频帧 (数据采集阶段) ---
print(f"\n阶段 1/3: 正在从视频 '{base_filename}' (FPS: {fps:.2f}) 中采集坐标...")
for frame_num in tqdm(range(total_frames), desc="追踪进度"):
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1, x2, y2 = roi_box
    roi_frame = frame[y1:y2, x1:x2]

    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)
    contours, _ = cv2.findContours(
        mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    time_sec = frame_num / fps
    abs_x, abs_y, radius = np.nan, np.nan, np.nan  # 默认为NaN
    found_object = False

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > MIN_AREA:
            ((x_rel, y_rel), radius) = cv2.minEnclosingCircle(largest_contour)
            abs_x, abs_y = int(x_rel + x1), int(y_rel + y1)
            found_object = True

    raw_trajectory_data.append([frame_num + 1, time_sec, abs_x, abs_y])

    # 绘制和保存视频帧
    if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if found_object:
            cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (abs_x, abs_y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"T: {time_sec:.2f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        if SAVE_OUTPUT_VIDEO:
            video_writer.write(frame)
        if SHOW_VIDEO_PREVIEW:
            cv2.imshow("Tracking Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# --- 5. 步骤四：数据后处理与分析 ---
print("\n阶段 2/3: 正在分析轨迹数据 (计算距离和速度)...")
detailed_data = []
header = [
    "frame",
    "time_sec",
    "x",
    "y",
    "distance_pixels",
    "cumulative_distance_pixels",
]
detailed_data.append(header)

total_distance = 0.0
for i in tqdm(range(len(raw_trajectory_data)), desc="分析进度"):
    frame, time_sec, x, y = raw_trajectory_data[i]
    distance_this_frame = 0.0

    if i > 0:
        prev_x, prev_y = raw_trajectory_data[i - 1][2], raw_trajectory_data[i - 1][3]
        # 只有当当前帧和上一帧都有有效坐标时，才计算距离
        if not (np.isnan(x) or np.isnan(y) or np.isnan(prev_x) or np.isnan(prev_y)):
            distance_this_frame = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            total_distance += distance_this_frame

    detailed_data.append([frame, time_sec, x, y, distance_this_frame, total_distance])

# --- 6. 步骤五：生成输出文件 ---
print("\n阶段 3/3: 正在生成报告和图像...")

# 6.1 保存详细的CSV文件
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(detailed_data)
print(f"  - 详细轨迹数据已保存到: {OUTPUT_CSV_PATH}")

# 6.2 生成并保存轨迹图
x_coords = [row[2] for row in detailed_data[1:]]
y_coords = [row[3] for row in detailed_data[1:]]
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, color="cornflowerblue", alpha=0.8)
plt.title(f"Mouse Trajectory: {base_filename}")
plt.xlabel("X-Position (pixels)")
plt.ylabel("Y-Position (pixels)")
# 反转Y轴以匹配视频坐标系 (0,0在左上角)
plt.gca().invert_yaxis()
# 设置坐标轴范围与ROI一致，并保持比例
plt.xlim(roi_box[0], roi_box[2])
plt.ylim(roi_box[3], roi_box[1])  # 注意ylim是(bottom, top)
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")  # 保证X和Y轴比例相同
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()
print(f"  - 运动轨迹图已保存到: {OUTPUT_PLOT_PATH}")

# 6.3 生成并保存秒级总结报告
summary_report = [["Time_End (s)", "Cumulative_Distance_at_Second_End (pixels)"]]
target_second = 1
for row in detailed_data[1:]:
    time_sec = row[1]
    cum_dist = row[5]
    if time_sec >= target_second:
        summary_report.append([target_second, cum_dist])
        target_second += 1
# 确保即使视频很短，也能处理
while target_second <= detailed_data[-1][1]:
    summary_report.append([target_second, detailed_data[-1][5]])
    target_second += 1

with open(OUTPUT_SUMMARY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(summary_report)
print(f"  - 秒级运动总结已保存到: {OUTPUT_SUMMARY_PATH}")

# --- 7. 释放资源 ---
cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print("\n所有处理完成！")
