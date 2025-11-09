import cv2
import numpy as np
import csv
from tqdm import tqdm
import os  # 导入os模块来处理文件路径

# --- 1. 全局配置与参数 ---

# --- 输入/输出路径 ---
# 您只需要修改下面这一行来更换视频文件！
VIDEO_PATH = r"C:\Users\15297\Desktop\e116a1d3aa9a86211d99a0b826a5b2a9.mp4"

# --- 自动生成输出文件名 (新功能) ---
# 1. 获取视频文件的基本名称 (例如: 'video.mp4')
base_filename = os.path.basename(VIDEO_PATH)
# 2. 将基本名称和扩展名分开 (例如: 'video' 和 '.mp4')
video_name_without_ext, _ = os.path.splitext(base_filename)
# 3. 使用视频主名构建新的输出文件名
OUTPUT_CSV_PATH = f"{video_name_without_ext}_trajectory.csv"
OUTPUT_VIDEO_PATH = f"{video_name_without_ext}_tracked.mp4"
# ------------------------------------qqq

# --- 功能开关 ---
# 关闭预览可以极大提速！
SHOW_VIDEO_PREVIEW = False  # 是否显示实时追踪窗口？ (True/False)
SAVE_OUTPUT_VIDEO = True  # 是否保存带标记的视频文件？ (True/False)

# --- 追踪参数 ---
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([50, 255, 117])
kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)
MIN_AREA = 100

# --- 全局变量 (用于ROI选择) ---
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
trajectory_data = []
video_writer = None

if SAVE_OUTPUT_VIDEO:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height)
    )
    print(f"将保存带标记的视频到: {OUTPUT_VIDEO_PATH}")

# --- 4. 步骤三：循环处理视频帧并生成数据 ---
print(f"\n开始处理视频: {base_filename}")
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

    found_object = False
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > MIN_AREA:
            ((x_rel, y_rel), radius) = cv2.minEnclosingCircle(largest_contour)
            abs_x, abs_y = int(x_rel + x1), int(y_rel + y1)
            trajectory_data.append([frame_num + 1, abs_x, abs_y])
            found_object = True

    if not found_object:
        trajectory_data.append([frame_num + 1, np.nan, np.nan])

    if SHOW_VIDEO_PREVIEW or SAVE_OUTPUT_VIDEO:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if found_object:
            cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (abs_x, abs_y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Frame: {frame_num + 1}",
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

# --- 5. 步骤四：保存数据并释放资源 ---
print("\n处理完成！正在保存数据...")
with open(OUTPUT_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x", "y"])
    writer.writerows(trajectory_data)

print(f"轨迹数据已成功保存到: {OUTPUT_CSV_PATH}")

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
