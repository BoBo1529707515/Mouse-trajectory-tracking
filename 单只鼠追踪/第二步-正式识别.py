import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import math

# --- 1. 全局配置与参数 ---
VIDEO_PATH = r'C:\Users\15297\Desktop\b1d914e513f5dc3f0fafd2824ea55ac3.mp4'
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([179, 153, 193])
MIN_AREA = 100

# --- 自动生成输出文件夹和路径 ---
base_filename = os.path.basename(VIDEO_PATH)
video_name_without_ext, _ = os.path.splitext(base_filename)
OUTPUT_FOLDER = video_name_without_ext
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "detailed_trajectory_pixels.csv")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_FOLDER, "trajectory_subplots.png")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "tracked_video.mp4")

# --- 其他功能开关 ----
SHOW_VIDEO_PREVIEW = True
SAVE_OUTPUT_VIDEO = True

# --- 全局变量 ---
roi_polygons = []; current_polygon_points = []
frame_for_selection = None; display_frame = None
start_frame_index = 0

# --- 回调函数 ---
def irregular_roi_callback(event, x, y, flags, param):
    global current_polygon_points, display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon_points.append((x, y)); cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        if len(current_polygon_points) > 1: cv2.line(display_frame, current_polygon_points[-2], current_polygon_points[-1], (0, 255, 255), 2)
        cv2.imshow("Step 1: Select Irregular ROIs", display_frame)

def start_frame_callback(pos):
    global frame_for_selection
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        frame_for_selection = frame.copy()
        cv2.imshow("Step 0: Select Start Frame", frame_for_selection)

# --- 2. 视频加载与交互式选择 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): print(f"错误：无法打开视频文件 {VIDEO_PATH}"); exit()
fps = cap.get(cv2.CAP_PROP_FPS); fps = 30 if fps == 0 else fps
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("--- 步骤 0: 选择分析的起始帧 ---")
print("1. 拖动滑块，找到您希望开始分析的画面。"); print("2. 确认后，按 'c' 或 'Enter' 键。")
cv2.namedWindow("Step 0: Select Start Frame")
cv2.createTrackbar("Frame", "Step 0: Select Start Frame", 0, total_frames - 1, start_frame_callback)
start_frame_callback(0)
while True:
    # 确保焦点在窗口上才能接收按键
    cv2.imshow("Step 0: Select Start Frame", frame_for_selection)
    key = cv2.waitKey(20) & 0xFF
    if key in [ord('c'), 13]:
        start_frame_index = cv2.getTrackbarPos("Frame", "Step 0: Select Start Frame"); break
cv2.destroyAllWindows()
print(f"分析将从第 {start_frame_index} 帧开始。")
display_frame = frame_for_selection.copy()
cv2.namedWindow("Step 1: Select Irregular ROIs")
cv2.setMouseCallback("Step 1: Select Irregular ROIs", irregular_roi_callback)

# *** 优化用户指引 START ***
print("\n--- 步骤 1: ROI选择说明 ---")
print("1. 在弹出的 'Step 1' 窗口中用鼠标左键点击，定义多边形的各个顶点。")
print("2. 完成一个区域后，按 'n' 键保存并开始下一个。")
print("3. 如果画错了，按 'r' 键清空所有已选区域。")
print("\n*** 重要提示 ***")
print("4. 所有区域选择完毕后，请先【用鼠标点击一下 'Step 1' 窗口】，确保其为活动窗口。")
print("5. 然后，再按 'c' 或 'Enter' 键，即可开始追踪。")
# *** 优化用户指引 END ***

while True:
    cv2.imshow("Step 1: Select Irregular ROIs", display_frame)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('n'):
        if len(current_polygon_points) > 2:
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32)); cv2.polylines(display_frame, [roi_polygons[-1]], isClosed=True, color=(255, 0, 0), thickness=2)
            label_pos = tuple(np.mean(roi_polygons[-1], axis=0, dtype=np.int32)); cv2.putText(display_frame, f"ROI {len(roi_polygons)}", (label_pos[0], label_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            current_polygon_points = []; print(f"  > ROI {len(roi_polygons)} 已保存。")
    elif key == ord('r'):
        roi_polygons = []; current_polygon_points = []; display_frame = frame_for_selection.copy(); print("  > 所有ROI选择已重置。")
    elif key in [ord('c'), 13]:
        if len(current_polygon_points) > 2:
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32))
        break
cv2.destroyAllWindows()
if not roi_polygons: print("未选择任何ROI区域，程序退出。"); cap.release(); exit()
print(f"\n已选择 {len(roi_polygons)} 个不规则ROI区域，开始处理...")

# --- 后续代码完全不变，因此省略以保持简洁 ---
# ... (主循环、数据处理、绘图等代码) ...
# 请将这部分代码粘贴到您现有代码的开头，替换掉从开始到 `cv2.destroyAllWindows()` 的部分即可。
# 后面的主循环、数据处理、绘图代码都是正确的，无需改动。
# 为了完整性，我还是将完整代码附在下面。

# --- 3. 初始化和主循环 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
frames_to_process = total_frames - start_frame_index
raw_trajectory_data = []
video_writer = None
if SAVE_OUTPUT_VIDEO:
    h, w = frame_for_selection.shape[:2]; fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
kernel_open = np.ones((5, 5), np.uint8); kernel_dilate = np.ones((10, 10), np.uint8)

print(f"\n阶段 1/2: 正在从视频 '{base_filename}' 中采集坐标...")
for frame_offset in tqdm(range(frames_to_process), desc="追踪进度"):
    frame_num = start_frame_index + frame_offset
    ret, frame = cap.read()
    if not ret: break
    time_sec_absolute = frame_num / fps
    time_sec_relative = frame_offset / fps
    full_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_color_mask = cv2.inRange(full_hsv_frame, lower_bound, upper_bound)
    for roi_id, polygon_points in enumerate(roi_polygons, 1):
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon_points], 255)
        final_mask = cv2.bitwise_and(full_color_mask, full_color_mask, mask=roi_mask)
        mask_processed = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        abs_x, abs_y, radius, found_object = np.nan, np.nan, np.nan, False
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_AREA:
                ((x_raw, y_raw), radius) = cv2.minEnclosingCircle(largest_contour)
                abs_x, abs_y = int(x_raw), int(y_raw); found_object = True
        raw_trajectory_data.append([frame_num, time_sec_absolute, time_sec_relative, roi_id, abs_x, abs_y])
        if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
            cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
            if found_object: cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)
    if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
        cv2.putText(frame, f"Analysis Time: {time_sec_relative:.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if SAVE_OUTPUT_VIDEO: video_writer.write(frame)
        if SHOW_VIDEO_PREVIEW:
            cv2.imshow('Tracking Preview', frame);
            if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- 4. 数据后处理与输出 ---
print("\n阶段 2/2: 正在分析数据并生成报告...")
detailed_data = []
header = ['frame_number', 'video_real_time_sec', 'analysis_relative_time_sec', 'roi_id', 'x_pixel', 'y_pixel', 'distance_pixels', 'cumulative_distance_pixels']
detailed_data.append(header)
total_distances = {i + 1: 0.0 for i in range(len(roi_polygons))}; last_coords = {i + 1: (np.nan, np.nan) for i in range(len(roi_polygons))}
for frame, time_abs, time_rel, roi_id, x_pixel, y_pixel in tqdm(raw_trajectory_data, desc="分析进度"):
    distance_this_frame = 0.0; prev_x, prev_y = last_coords[roi_id]
    if not (np.isnan(x_pixel) or np.isnan(y_pixel) or np.isnan(prev_x) or np.isnan(prev_y)):
        distance_this_frame = np.sqrt((x_pixel - prev_x)**2 + (y_pixel - prev_y)**2); total_distances[roi_id] += distance_this_frame
    detailed_data.append([frame, time_abs, time_rel, roi_id, x_pixel, y_pixel, distance_this_frame, total_distances[roi_id]])
    last_coords[roi_id] = (x_pixel, y_pixel)
with open(OUTPUT_CSV_PATH, 'w', newline='') as f: writer = csv.writer(f); writer.writerows(detailed_data)
print(f"  - 详细轨迹数据已保存到: {OUTPUT_CSV_PATH}")
num_rois = len(roi_polygons)
if num_rois > 0:
    cols = int(math.ceil(math.sqrt(num_rois))); rows = int(math.ceil(num_rois / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False); axes = axes.flatten()
    for i, ax in enumerate(axes):
        roi_id = i + 1
        if roi_id > num_rois: ax.axis('off'); continue
        roi_data = [row for row in detailed_data[1:] if row[3] == roi_id]
        x_coords = [row[4] for row in roi_data]; y_coords = [row[5] for row in roi_data]
        ax.plot(x_coords, y_coords, color='blue', alpha=0.8)
        ax.set_title(f'ROI {roi_id} Trajectory'); ax.set_xlabel('X-Position (pixels)'); ax.set_ylabel('Y-Position (pixels)')
        roi_polygon = roi_polygons[i]; x_min, y_min = np.min(roi_polygon, axis=0); x_max, y_max = np.max(roi_polygon, axis=0)
        margin = 20; ax.set_xlim(x_min - margin, x_max + margin); ax.set_ylim(y_min - margin, y_max + margin)
        ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(f'Trajectories for {base_filename}', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(OUTPUT_PLOT_PATH, dpi=300); plt.close()
    print(f"  - 独立轨迹子图已保存到: {OUTPUT_PLOT_PATH}")
for roi_id_to_summarize in range(1, len(roi_polygons) + 1):
    roi_data = [row for row in detailed_data[1:] if row[3] == roi_id_to_summarize]
    summary_report = [['Time_End (s)', 'Cumulative_Distance_at_Second_End (pixels)']]; target_second = 1; last_dist = 0
    if not roi_data: continue
    for row in roi_data:
        time_sec, cum_dist = row[2], row[7]
        if time_sec >= target_second: summary_report.append([target_second, cum_dist]); target_second += 1
        last_dist = cum_dist
    if not summary_report or (roi_data and roi_data[-1][2] > 0 and summary_report[-1][0] < int(roi_data[-1][2])):
        summary_report.append([int(roi_data[-1][2]), last_dist])
    summary_filename = os.path.join(OUTPUT_FOLDER, f"summary_roi_{roi_id_to_summarize}_pixels.csv")
    with open(summary_filename, 'w', newline='') as f: summary_writer = csv.writer(f); summary_writer.writerows(summary_report)
    print(f"  - ROI {roi_id_to_summarize} 的秒级总结已保存到: {summary_filename}")

# --- 5. 释放资源 ---
cap.release()
if video_writer: video_writer.release()
print(f"\n所有处理完成！请查看文件夹 '{OUTPUT_FOLDER}' 获取结果。")
