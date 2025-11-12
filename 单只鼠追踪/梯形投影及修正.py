import cv2
import numpy as np
import csv
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# --- 1. 全局配置与参数 ---
VIDEO_PATH = r"C:\Users\15297\Desktop\b1d914e513f5dc3f0fafd2824ea55ac3.mp4"

ENABLE_PERSPECTIVE_CORRECTION = True
REAL_WIDTH_CM = 40.0
REAL_HEIGHT_CM = 40.0

# --- 追踪参数 (请使用调试器获得最佳值) ---
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([179, 255, 60])
MIN_AREA = 100

# --- 自动生成输出文件夹和路径 ---
base_filename = os.path.basename(VIDEO_PATH)
video_name_without_ext, _ = os.path.splitext(base_filename)
OUTPUT_FOLDER = video_name_without_ext
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_CSV_PATH = os.path.join(OUTPUT_FOLDER, "detailed_trajectory.csv")
OUTPUT_PLOT_PATH = os.path.join(OUTPUT_FOLDER, "trajectory_plot.png")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "tracked_video.mp4")

# --- 其他功能开关 ---
SHOW_VIDEO_PREVIEW = True
SAVE_OUTPUT_VIDEO = True

# --- 全局变量 ---
perspective_points = []
current_polygon_points = []
roi_polygons = []
display_frame = None
start_frame_confirmed = False  # ### NEW ###: 用于按钮确认的全局标志
button_rect = None  # ### NEW ###: 按钮区域


# --- 回调函数 ---
def perspective_callback(event, x, y, flags, param):
    # (此函数无需修改)
    global display_frame, perspective_points
    if event == cv2.EVENT_LBUTTONDOWN and len(perspective_points) < 4:
        perspective_points.append([x, y])
        cv2.circle(display_frame, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            display_frame,
            str(len(perspective_points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("0. Calibrate Perspective", display_frame)


def irregular_roi_callback(event, x, y, flags, param):
    # (此函数无需修改)
    global current_polygon_points, display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon_points.append((x, y))
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        if len(current_polygon_points) > 1:
            cv2.line(
                display_frame,
                current_polygon_points[-2],
                current_polygon_points[-1],
                (0, 255, 255),
                2,
            )
        cv2.imshow("1. Select Irregular ROIs", display_frame)


# ### NEW ###: 帧选择器的鼠标回调，用于处理按钮点击
def frame_selector_callback(event, x, y, flags, param):
    global start_frame_confirmed, button_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        if (
            button_rect
            and button_rect[0] <= x <= button_rect[0] + button_rect[2]
            and button_rect[1] <= y <= button_rect[1] + button_rect[3]
        ):
            start_frame_confirmed = True


# --- 2. 视频加载与交互式设置 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30 if fps == 0 else fps
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# --- 步骤 -1: 交互式选择开始帧 (带确认按钮) ---
def on_trackbar(val):
    pass


SELECTOR_WINDOW_WIDTH = 1200
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_height / original_width
selector_window_height = int(SELECTOR_WINDOW_WIDTH * aspect_ratio)

frame_selector_window = "Select Start Frame"
cv2.namedWindow(frame_selector_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(frame_selector_window, SELECTOR_WINDOW_WIDTH, selector_window_height)
cv2.createTrackbar("Frame", frame_selector_window, 0, total_frames - 1, on_trackbar)
cv2.setMouseCallback(
    frame_selector_window, frame_selector_callback
)  # ### NEW ###: 绑定新的鼠标回调

print("--- 步骤 -1: 选择开始帧 ---")
print("1. 拖动滑动条选择开始帧。")
print("2. 点击右下角的 [Confirm] 按钮或按 'c'/'Enter' 键确认。")

current_frame_pos = -1
START_FRAME = 0
while not start_frame_confirmed:  # ### MODIFIED ###: 循环直到按钮被点击
    trackbar_pos = cv2.getTrackbarPos("Frame", frame_selector_window)
    if trackbar_pos != current_frame_pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos)
        ret, frame = cap.read()
        if not ret:
            break

        # ### NEW ###: 绘制按钮和信息
        # 定义按钮位置和大小
        button_w, button_h = 150, 50
        button_x = frame.shape[1] - button_w - 20
        button_y = frame.shape[0] - button_h - 20
        button_rect = (button_x, button_y, button_w, button_h)

        # 绘制按钮背景
        cv2.rectangle(
            frame,
            (button_x, button_y),
            (button_x + button_w, button_y + button_h),
            (0, 200, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (button_x, button_y),
            (button_x + button_w, button_y + button_h),
            (255, 255, 255),
            2,
        )

        # 绘制按钮文字
        text = "Confirm"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = button_x + (button_w - text_size[0]) // 2
        text_y = button_y + (button_h + text_size[1]) // 2
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 绘制帧信息
        info_text = f"Frame: {trackbar_pos} / {total_frames - 1}"
        cv2.putText(
            frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )

        cv2.imshow(frame_selector_window, frame)
        current_frame_pos = trackbar_pos

    key = cv2.waitKey(20) & 0xFF
    # ### MODIFIED ###: 检查按键或按钮标志
    if key in [ord("c"), 13]:
        start_frame_confirmed = True  # 按键也触发确认

    if start_frame_confirmed:
        START_FRAME = current_frame_pos
        break

    if cv2.getWindowProperty(frame_selector_window, cv2.WND_PROP_VISIBLE) < 1:
        print("帧选择窗口已关闭，程序退出。")
        exit()

cv2.destroyAllWindows()

print(f"\n已选择第 {START_FRAME} 帧作为起点。")

# (后续代码无需修改，为了完整性全部粘贴)

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
ret, setup_frame = cap.read()
if not ret:
    print(f"无法读取指定的开始帧 ({START_FRAME})")
    cap.release()
    exit()

# --- 3. 透视校准与ROI选择 ---
if ENABLE_PERSPECTIVE_CORRECTION:
    display_frame = setup_frame.copy()
    perspective_points = []
    cv2.namedWindow("0. Calibrate Perspective")
    cv2.setMouseCallback("0. Calibrate Perspective", perspective_callback)
    print("--- 步骤 0: 透视校准 ---")
    print(
        "请严格按照顺序点击场地的四个角点：\n1.左上角 -> 2.右上角 -> 3.右下角 -> 4.左下角"
    )
    print("完成后，按 'c' 或 'Enter' 键确认。")
    while True:
        cv2.imshow("0. Calibrate Perspective", display_frame)
        if cv2.waitKey(20) & 0xFF in [ord("c"), 13] or len(perspective_points) >= 4:
            break
    cv2.destroyAllWindows()
    if len(perspective_points) == 4:
        src_pts = np.float32(perspective_points)
        output_width_pixels = int(REAL_WIDTH_CM * 10)
        output_height_pixels = int(REAL_HEIGHT_CM * 10)
        dst_pts = np.float32(
            [
                [0, 0],
                [output_width_pixels, 0],
                [output_width_pixels, output_height_pixels],
                [0, output_height_pixels],
            ]
        )
        perspective_transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("透视校准完成！")
    else:
        print("未完成四点选择，将禁用透视校正。")
        ENABLE_PERSPECTIVE_CORRECTION = False
        perspective_transform_matrix = None

frame_for_selection = setup_frame.copy()
display_frame = setup_frame.copy()
roi_polygons = []
current_polygon_points = []
cv2.namedWindow("1. Select Irregular ROIs")
cv2.setMouseCallback("1. Select Irregular ROIs", irregular_roi_callback)
print("\n--- 步骤 1: ROI选择说明 ---")
print("1. 在窗口中用鼠标左键点击，定义多边形的各个顶点。")
print("2. 完成一个区域后，按 'n' 键保存并开始下一个。")
print("3. 所有区域选择完毕后，按 'c' 或 'Enter' 键开始追踪。")
while True:
    cv2.imshow("1. Select Irregular ROIs", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        if len(current_polygon_points) > 2:
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32))
            cv2.polylines(
                display_frame,
                [roi_polygons[-1]],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
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
            current_polygon_points = []
            print(f"ROI {len(roi_polygons)} 已保存。")
    elif key == ord("r"):
        roi_polygons = []
        current_polygon_points = []
        display_frame = frame_for_selection.copy()
        print("所有ROI选择已重置。")
    elif key in [ord("c"), 13]:
        if len(current_polygon_points) > 2:
            roi_polygons.append(np.array(current_polygon_points, dtype=np.int32))
        break
cv2.destroyAllWindows()
if not roi_polygons:
    print("未选择任何ROI区域，程序退出。")
    cap.release()
    exit()
print(f"已选择 {len(roi_polygons)} 个不规则ROI区域，开始处理...")

# --- 4. 初始化和主循环 ---
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
raw_trajectory_data = []
video_writer = None
if SAVE_OUTPUT_VIDEO:
    h, w = setup_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)
print(f"\n阶段 1/3: 正在从视频 '{base_filename}' 中采集和校正坐标...")

for frame_num in tqdm(range(START_FRAME, total_frames), desc="追踪进度"):
    ret, frame = cap.read()
    if not ret:
        break
    time_sec = (frame_num - START_FRAME) / fps

    full_hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    full_color_mask = cv2.inRange(full_hsv_frame, lower_bound, upper_bound)

    for roi_id, polygon_points in enumerate(roi_polygons, 1):
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon_points], 255)
        final_mask = cv2.bitwise_and(full_color_mask, full_color_mask, mask=roi_mask)
        mask_processed = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(
            mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        abs_x, abs_y, radius, x_corr, y_corr, found_object = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
        )
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > MIN_AREA:
                ((x_raw, y_raw), radius) = cv2.minEnclosingCircle(largest_contour)
                abs_x, abs_y = int(x_raw), int(y_raw)
                found_object = True
                if (
                    ENABLE_PERSPECTIVE_CORRECTION
                    and perspective_transform_matrix is not None
                ):
                    point_to_transform = np.float32([[[abs_x, abs_y]]])
                    corrected_point = cv2.perspectiveTransform(
                        point_to_transform, perspective_transform_matrix
                    )
                    x_corr = corrected_point[0][0][0] / 10.0
                    y_corr = corrected_point[0][0][1] / 10.0

        raw_trajectory_data.append(
            [frame_num, time_sec, roi_id, abs_x, abs_y, x_corr, y_corr]
        )

        if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
            cv2.polylines(
                frame, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2
            )
            if found_object:
                cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)

    if SAVE_OUTPUT_VIDEO or SHOW_VIDEO_PREVIEW:
        cv2.putText(
            frame,
            f"F:{frame_num} T:{time_sec:.2f}s",
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

# --- 5, 6, 7. 数据处理、输出与释放 ---
print("\n阶段 2/3: 正在分析轨迹数据...")
detailed_data = []
unit = "cm" if ENABLE_PERSPECTIVE_CORRECTION else "pixels"
header_base = ["frame", "time_sec", "roi_id", "x_pixel", "y_pixel"]
if ENABLE_PERSPECTIVE_CORRECTION:
    header = header_base + [
        "x_cm",
        "y_cm",
        f"distance_{unit}",
        f"cumulative_distance_{unit}",
    ]
else:
    header = header_base + [f"distance_{unit}", f"cumulative_distance_{unit}"]
detailed_data.append(header)

total_distances = {i + 1: 0.0 for i in range(len(roi_polygons))}
last_coords = {i + 1: (np.nan, np.nan) for i in range(len(roi_polygons))}
for frame, time_sec, roi_id, x_pixel, y_pixel, x_corr, y_corr in tqdm(
    raw_trajectory_data, desc="分析进度"
):
    distance_this_frame = 0.0
    current_x, current_y = (
        (x_corr, y_corr) if ENABLE_PERSPECTIVE_CORRECTION else (x_pixel, y_pixel)
    )
    prev_x, prev_y = last_coords[roi_id]
    if not (
        np.isnan(current_x)
        or np.isnan(current_y)
        or np.isnan(prev_x)
        or np.isnan(prev_y)
    ):
        distance_this_frame = np.sqrt(
            (current_x - prev_x) ** 2 + (current_y - prev_y) ** 2
        )
        total_distances[roi_id] += distance_this_frame

    row_data = [frame, time_sec, roi_id, x_pixel, y_pixel]
    if ENABLE_PERSPECTIVE_CORRECTION:
        row_data.extend([x_corr, y_corr])
    row_data.extend([distance_this_frame, total_distances[roi_id]])
    detailed_data.append(row_data)
    last_coords[roi_id] = (current_x, current_y)

print("\n阶段 3/3: 正在生成报告和图像...")
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(detailed_data)
print(f"  - 详细轨迹数据已保存到: {OUTPUT_CSV_PATH}")

plt.figure(figsize=(10, 10))
ax = plt.gca()
cmap = plt.get_cmap("tab10")
for roi_id in range(1, len(roi_polygons) + 1):
    roi_data = [row for row in detailed_data[1:] if row[2] == roi_id]
    if not roi_data:
        continue

    x_col_idx = 5 if ENABLE_PERSPECTIVE_CORRECTION else 3
    y_col_idx = 6 if ENABLE_PERSPECTIVE_CORRECTION else 4
    x_coords = [r[x_col_idx] for r in roi_data]
    y_coords = [r[y_col_idx] for r in roi_data]
    ax.plot(
        x_coords, y_coords, color=cmap(roi_id - 1), label=f"ROI {roi_id}", alpha=0.8
    )

if ENABLE_PERSPECTIVE_CORRECTION:
    ax.set_title(f"Corrected Trajectories: {base_filename}")
    ax.set_xlabel(f"X-Position ({unit})")
    ax.set_ylabel(f"Y-Position ({unit})")
    ax.set_xlim(0, REAL_WIDTH_CM)
    ax.set_ylim(0, REAL_HEIGHT_CM)
    ax.set_aspect("equal", adjustable="box")
else:
    h, w = setup_frame.shape[:2]
    ax.set_title(f"Pixel Trajectories: {base_filename}")
    ax.set_xlabel(f"X-Position ({unit})")
    ax.set_ylabel(f"Y-Position ({unit})")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
plt.close()
print(f"  - 轨迹图已保存到: {OUTPUT_PLOT_PATH}")

for roi_id_to_summarize in range(1, len(roi_polygons) + 1):
    roi_data = [row for row in detailed_data[1:] if row[2] == roi_id_to_summarize]
    if not roi_data:
        continue

    summary_report = [[f"Time_End (s)", f"Cumulative_Distance_at_Second_End ({unit})"]]
    target_second = 1
    last_dist = 0
    dist_col_index = -1
    for row in roi_data:
        time_sec, cum_dist = row[1], row[dist_col_index]
        if time_sec >= target_second:
            while target_second < time_sec:
                summary_report.append([target_second, last_dist])
                target_second += 1
            summary_report.append([target_second, cum_dist])
            target_second += 1
        last_dist = cum_dist

    if roi_data:
        final_time = int(roi_data[-1][1])
        # Fill any remaining seconds up to the final time
        while target_second <= final_time:
            summary_report.append([target_second, last_dist])
            target_second += 1

    summary_filename = os.path.join(
        OUTPUT_FOLDER, f"summary_roi_{roi_id_to_summarize}.csv"
    )
    with open(summary_filename, "w", newline="") as f:
        summary_writer = csv.writer(f)
        summary_writer.writerows(summary_report)
    print(f"  - ROI {roi_id_to_summarize} 的秒级总结已保存到: {summary_filename}")

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print(f"\n所有处理完成！请查看文件夹 '{OUTPUT_FOLDER}' 获取结果。")
