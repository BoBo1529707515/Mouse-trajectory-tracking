import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 全局配置与参数 ---

# --- 输入路径 ---
# 您只需要修改下面这一行来更换视频文件！
VIDEO_PATH = r"C:\Users\15297\Desktop\e116a1d3aa9a86211d99a0b826a5b2a9.mp4"

# --- 自动生成输出文件夹和路径 ---
base_filename = os.path.basename(VIDEO_PATH)
video_name_without_ext, _ = os.path.splitext(base_filename)
OUTPUT_FOLDER = video_name_without_ext
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# --- 全局变量 (用于不规则ROI选择) ---
roi_polygons = []  # 存储所有已确定的多边形ROI
current_polygon_points = []  # 存储当前正在绘制的多边形的点
frame_for_selection = None
display_frame = None


def irregular_roi_callback(event, x, y, flags, param):
    """鼠标回调函数，通过点击描点来定义不规则ROI"""
    global current_polygon_points, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # 添加新点
        current_polygon_points.append((x, y))
        # 在显示帧上画出点和线
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)  # 画出当前点
        if len(current_polygon_points) > 1:
            # 连接到上一个点
            cv2.line(
                display_frame,
                current_polygon_points[-2],
                current_polygon_points[-1],
                (0, 255, 255),
                2,
            )
        cv2.imshow("1. Select Irregular ROIs", display_frame)


# --- 2. 步骤一：加载视频并选择不规则ROI ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if fps == 0 else fps
ret, first_frame = cap.read()
if not ret:
    print("无法读取视频第一帧")
    cap.release()
    exit()

frame_for_selection = first_frame.copy()
display_frame = first_frame.copy()

cv2.namedWindow("1. Select Irregular ROIs")
cv2.setMouseCallback("1. Select Irregular ROIs", irregular_roi_callback)

print("--- 不规则ROI选择说明 ---")
print("1. 在窗口中用鼠标左键点击，定义多边形的各个顶点。")
print("2. 完成一个区域的绘制后，按 'n' 键保存该区域并开始下一个。")
print("3. 如果画错，按 'r' 键清空所有选择，重新开始。")
print("4. 所有区域选择完毕后，按 'c' 或 'Enter' 键开始追踪。")

while True:
    cv2.imshow("1. Select Irregular ROIs", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # 按 'n' 键完成当前多边形，并准备画下一个
    if key == ord("n"):
        if len(current_polygon_points) > 2:  # 一个多边形至少需要3个点
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32))
            # 在显示帧上将这个多边形固化为蓝色
            cv2.polylines(
                display_frame,
                [roi_polygons[-1]],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            # 添加标签
            label_pos = tuple(np.mean(roi_polygons[-1], axis=0, dtype=np.int32))
            cv2.putText(
                display_frame,
                f"ROI {len(roi_polygons)}",
                (label_pos[0], label_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            current_polygon_points = []  # 清空，准备下一个
            print(f"ROI {len(roi_polygons)} 已保存。请绘制下一个或按 'c' 确认。")
        else:
            print("绘制错误：一个区域至少需要3个点。")

    # 按 'r' 键重置所有选择
    elif key == ord("r"):
        roi_polygons = []
        current_polygon_points = []
        display_frame = frame_for_selection.copy()
        print("所有ROI选择已重置，请重新开始。")

    # 按 'c' 或 Enter 键确认所有选择
    elif key == ord("c") or key == 13:
        # 如果用户在按c之前没有按n，自动保存最后一个正在绘制的多边形
        if len(current_polygon_points) > 2:
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32))
        break

cv2.destroyAllWindows()
if not roi_polygons:
    print("未选择任何ROI区域，程序退出。")
    cap.release()
    exit()
print(f"已选择 {len(roi_polygons)} 个不规则ROI区域，开始处理...")

# --- 3. 初始化追踪与输出 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
raw_trajectory_data = []  # [frame, time_sec, roi_id, x, y]
video_writer = None

if SAVE_OUTPUT_VIDEO:
    h, w = frame_for_selection.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

# --- 4. 循环处理视频帧 (核心追踪逻辑) ---
print(f"\n阶段 1/3: 正在从视频 '{base_filename}' (FPS: {fps:.2f}) 中采集坐标...")
for frame_num in tqdm(range(total_frames), desc="追踪进度"):
    ret, frame = cap.read()
    if not ret:
        break
    time_sec = frame_num / fps

    # 对每个多边形ROI进行处理
    for roi_id, polygon_points in enumerate(roi_polygons, 1):
        # 1. 创建一个黑色的蒙版
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # 2. 在蒙版上将多边形区域填充为白色
        cv2.fillPoly(mask, [polygon_points], 255)
        # 3. 使用蒙版从原图中“抠出”ROI区域
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 在抠出的ROI上进行颜色识别和轮廓查找
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        # 形态学操作
        mask_processed = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
        mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)

        contours, _ = cv2.findContours(
            mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        abs_x, abs_y, radius = np.nan, np.nan, np.nan
        found_object = False

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_AREA:
                # 注意：因为是在全尺寸蒙版上找轮廓，坐标已经是绝对坐标了！
                ((abs_x, abs_y), radius) = cv2.minEnclosingCircle(largest_contour)
                abs_x, abs_y = int(abs_x), int(abs_y)
                found_object = True

        raw_trajectory_data.append([frame_num + 1, time_sec, roi_id, abs_x, abs_y])

        # 在视频帧上绘制标记
        if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
            cv2.polylines(
                frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2
            )
            cv2.putText(
                frame,
                f"ROI {roi_id}",
                tuple(polygon_points[0] - np.array([0, 10])),
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

# --- 5. 数据后处理与分析 (与之前版本基本相同) ---
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

total_distances = {roi_id + 1: 0.0 for roi_id in range(len(roi_polygons))}
last_coords = {roi_id + 1: (np.nan, np.nan) for roi_id in range(len(roi_polygons))}

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

# --- 6. 生成输出文件 ---
print("\n阶段 3/3: 正在生成报告和图像...")

# 6.1 保存详细CSV
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(detailed_data)
print(f"  - 详细轨迹数据已保存到: {OUTPUT_CSV_PATH}")

# 6.2 生成并保存轨迹图 (现在绘制多边形ROI)
plt.figure(figsize=(10, 10))
ax = plt.gca()
colors = plt.cm.get_cmap("tab10", len(roi_polygons))

for roi_id, polygon_points in enumerate(roi_polygons, 1):
    roi_data = [row for row in detailed_data[1:] if row[2] == roi_id]
    x_coords = [row[3] for row in roi_data]
    y_coords = [row[4] for row in roi_data]

    ax.plot(
        x_coords, y_coords, color=colors(roi_id - 1), label=f"ROI {roi_id}", alpha=0.8
    )

    # 绘制多边形ROI边界
    poly_patch = patches.Polygon(
        polygon_points,
        closed=True,
        linewidth=1.5,
        edgecolor=colors(roi_id - 1),
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(poly_patch)

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

# 6.3 为每个ROI生成秒级总结报告 (与之前版本相同)
# ... (这部分代码无需修改，直接复用)
for roi_id_to_summarize in range(1, len(roi_polygons) + 1):
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
    if not summary_report or summary_report[-1][0] < int(roi_data[-1][1]):
        summary_report.append([int(roi_data[-1][1]), last_dist])
    summary_filename = os.path.join(
        OUTPUT_FOLDER, f"summary_roi_{roi_id_to_summarize}.csv"
    )
    with open(summary_filename, "w", newline="") as f:
        writer.writerows(summary_report)
    print(f"  - ROI {roi_id_to_summarize} 的秒级总结已保存到: {summary_filename}")


# --- 7. 释放资源 ---
cap.release()
if video_writer:
    video_writer.release()
print(f"\n所有处理完成！请查看文件夹 '{OUTPUT_FOLDER}' 获取结果。")
