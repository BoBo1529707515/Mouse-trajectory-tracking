import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 全局配置与参数 ----

# --- 输入路径 ---
# 您只需要修改下面这一行来更换视频文件！
VIDEO_PATH = r"C:\Users\15297\Desktop\e116a1d3aa9a86211d99a0b826a5b2a9.mp4"

# --- 自动生成输出文件夹和路径 ---
base_filename = os.path.basename(VIDEO_PATH)
video_name_without_ext, _ = os.path.splitext(base_filename)
# 创建与视频同名的输出文件夹
OUTPUT_FOLDER = video_name_without_ext
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 所有输出文件都将保存在这个新文件夹中
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "detailed_trajectory.csv")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_FOLDER, "trajectory_plot.png")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "tracked_video.mp4")

# --- 功能开关 ---
SHOW_VIDEO_PREVIEW = False
SAVE_OUTPUT_VIDEO = True

# --- 追踪参数 ---
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([50, 255, 117])
kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)
MIN_AREA = 100

# --- 全局变量 (用于多ROI选择) ---
roi_boxes = []
current_roi_points = [None, None]
is_drawing = False
frame_for_selection = None
display_frame = None


def multi_roi_callback(event, x, y, flags, param):
    """鼠标回调函数，用于选择多个ROI"""
    global current_roi_points, is_drawing, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi_points = [(x, y), None]
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        # 实时显示正在绘制的框
        temp_frame = display_frame.copy()
        cv2.rectangle(temp_frame, current_roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("1. Select ROIs", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        current_roi_points[1] = (x, y)
        x1 = min(current_roi_points[0][0], current_roi_points[1][0])
        y1 = min(current_roi_points[0][1], current_roi_points[1][1])
        x2 = max(current_roi_points[0][0], current_roi_points[1][0])
        y2 = max(current_roi_points[0][1], current_roi_points[1][1])

        # 添加新的ROI到列表中
        roi_boxes.append((x1, y1, x2, y2))

        # 在显示帧上永久画出这个已确定的ROI
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            display_frame,
            f"ROI {len(roi_boxes)}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.imshow("1. Select ROIs", display_frame)


# --- 2. 步骤一：加载视频并选择多个ROI ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
    print("警告: 无法获取帧率, 默认设置为30 FPS.")

ret, first_frame = cap.read()
if not ret:
    print("无法读取视频第一帧")
    cap.release()
    exit()

frame_for_selection = first_frame.copy()
display_frame = first_frame.copy()  # 用于永久绘制已选ROI的帧

cv2.namedWindow("1. Select ROIs")
cv2.setMouseCallback("1. Select ROIs", multi_roi_callback)
print("--- ROI选择说明 ---")
print("1. 在窗口中用鼠标拖拽来选择一个区域。")
print("2. 您可以重复此操作以选择多个区域。")
print("3. 选择完毕后，按键盘上的 'c' 或 'Enter' 键开始追踪。")

while True:
    cv2.imshow("1. Select ROIs", display_frame)
    key = cv2.waitKey(1) & 0xFF
    # 按 'c' 或 Enter 键确认
    if key == ord("c") or key == 13:
        break
    # 按 'r' 键重置选择
    if key == ord("r"):
        roi_boxes = []
        display_frame = frame_for_selection.copy()
        print("ROI选择已重置，请重新选择。")

cv2.destroyAllWindows()
if not roi_boxes:
    print("未选择任何ROI区域，程序退出。")
    cap.release()
    exit()
print(f"已选择 {len(roi_boxes)} 个ROI区域，开始处理...")

# --- 3. 步骤二：初始化追踪和输出 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
raw_trajectory_data = []  # 格式: [frame, time_sec, roi_id, x, y]
video_writer = None

if SAVE_OUTPUT_VIDEO:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height)
    )

# --- 4. 步骤三：循环处理视频帧 (数据采集) ---
print(f"\n阶段 1/3: 正在从视频 '{base_filename}' (FPS: {fps:.2f}) 中采集坐标...")
for frame_num in tqdm(range(total_frames), desc="追踪进度"):
    ret, frame = cap.read()
    if not ret:
        break

    time_sec = frame_num / fps

    # 对每个ROI进行处理
    for roi_id, roi_box in enumerate(roi_boxes):
        x1, y1, x2, y2 = roi_box
        roi_frame = frame[y1:y2, x1:x2]

        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(
            mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        abs_x, abs_y, radius = np.nan, np.nan, np.nan
        found_object = False

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_AREA:
                ((x_rel, y_rel), radius) = cv2.minEnclosingCircle(largest_contour)
                abs_x, abs_y = int(x_rel + x1), int(y_rel + y1)
                found_object = True

        raw_trajectory_data.append([frame_num + 1, time_sec, roi_id + 1, abs_x, abs_y])

        # 在视频帧上绘制标记
        if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ROI {roi_id + 1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            if found_object:
                cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)
                cv2.circle(frame, (abs_x, abs_y), 5, (0, 0, 255), -1)

    if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
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
print("\n阶段 2/3: 正在分析轨迹数据...")
detailed_data = []
header = [
    "frame",
    "time_sec",
    "roi_id",
    "x",
    "y",
    "distance_pixels",
    "cumulative_distance_pixels",
]
detailed_data.append(header)

# 按ROI分组计算累计距离
total_distances = {roi_id + 1: 0.0 for roi_id in range(len(roi_boxes))}
last_coords = {roi_id + 1: (np.nan, np.nan) for roi_id in range(len(roi_boxes))}

for frame, time_sec, roi_id, x, y in tqdm(raw_trajectory_data, desc="分析进度"):
    distance_this_frame = 0.0
    prev_x, prev_y = last_coords[roi_id]

    if not (np.isnan(x) or np.isnan(y) or np.isnan(prev_x) or np.isnan(prev_y)):
        distance_this_frame = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        total_distances[roi_id] += distance_this_frame

    detailed_data.append(
        [frame, time_sec, roi_id, x, y, distance_this_frame, total_distances[roi_id]]
    )
    last_coords[roi_id] = (x, y)

# --- 6. 步骤五：生成输出文件 ---
print("\n阶段 3/3: 正在生成报告和图像...")

# 6.1 保存详细的CSV文件
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(detailed_data)
print(f"  - 详细轨迹数据已保存到: {OUTPUT_CSV_PATH}")

# 6.2 生成并保存轨迹图 (包含所有ROI)
plt.figure(figsize=(10, 10))
ax = plt.gca()
colors = plt.cm.get_cmap("tab10", len(roi_boxes))  # 为每个ROI生成不同颜色

for roi_id_to_plot in range(1, len(roi_boxes) + 1):
    roi_data = [row for row in detailed_data[1:] if row[2] == roi_id_to_plot]
    x_coords = [row[3] for row in roi_data]
    y_coords = [row[4] for row in roi_data]

    # 绘制轨迹
    ax.plot(
        x_coords,
        y_coords,
        color=colors(roi_id_to_plot - 1),
        label=f"ROI {roi_id_to_plot}",
        alpha=0.8,
    )

    # 绘制ROI框
    x1, y1, x2, y2 = roi_boxes[roi_id_to_plot - 1]
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1.5,
        edgecolor=colors(roi_id_to_plot - 1),
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

ax.set_title(f"Mouse Trajectories: {base_filename}")
ax.set_xlabel("X-Position (pixels)")
ax.set_ylabel("Y-Position (pixels)")
ax.invert_yaxis()
ax.grid(True, linestyle="--", alpha=0.5)
ax.axis("equal")
ax.legend()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()
print(f"  - 组合轨迹图已保存到: {OUTPUT_PLOT_PATH}")

# 6.3 为每个ROI生成秒级总结报告
for roi_id_to_summarize in range(1, len(roi_boxes) + 1):
    roi_data = [row for row in detailed_data[1:] if row[2] == roi_id_to_summarize]
    summary_report = [["Time_End (s)", "Cumulative_Distance_at_Second_End (pixels)"]]
    target_second = 1

    if not roi_data:
        continue

    last_dist = 0
    for row in roi_data:
        time_sec, cum_dist = row[1], row[5]
        if time_sec >= target_second:
            summary_report.append([target_second, cum_dist])
            target_second += 1
        last_dist = cum_dist

    # 确保最后的时间点也被记录
    if not summary_report or summary_report[-1][0] < int(roi_data[-1][1]):
        summary_report.append([int(roi_data[-1][1]), last_dist])

    summary_filename = os.path.join(
        OUTPUT_FOLDER, f"summary_roi_{roi_id_to_summarize}.csv"
    )
    with open(summary_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_report)
    print(f"  - ROI {roi_id_to_summarize} 的秒级总结已保存到: {summary_filename}")

# --- 7. 释放资源 ---
cap.release()
if video_writer:
    video_writer.release()
print(f"\n所有处理完成！请查看文件夹 '{OUTPUT_FOLDER}' 获取结果。")
