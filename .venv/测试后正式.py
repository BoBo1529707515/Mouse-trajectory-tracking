import cv2
import numpy as np

# --- 1. 定义常量和全局变量 ---

# 视频文件路径
VIDEO_PATH = r'C:\Users\15297\Desktop\e116a1d3aa9a86211d99a0b826a5b2a9.mp4'

# 您通过调试找到的最佳HSV阈值
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([50, 255, 117])

# 形态学操作的内核
kernel_open = np.ones((5, 5), np.uint8)
kernel_dilate = np.ones((10, 10), np.uint8)

# 用于ROI选择的全局变量
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
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi[0]:
            img_copy = frame_for_selection.copy()
            cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("1. Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        if roi_points[0] is None: return
        roi_points[1] = (x, y)
        selecting_roi[0] = False

        x1 = min(roi_points[0][0], roi_points[1][0])
        y1 = min(roi_points[0][1], roi_points[1][1])
        x2 = max(roi_points[0][0], roi_points[1][0])
        y2 = max(roi_points[0][1], roi_points[1][1])
        roi_box = (x1, y1, x2, y2)

        cv2.rectangle(frame_for_selection, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("1. Select ROI", frame_for_selection)
        print(f"ROI区域已确定: {roi_box}。请按任意键开始追踪...")

# --- 2. 步骤一：选择ROI ---
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
roi_points = [None, None]
selecting_roi = [False]

cv2.namedWindow("1. Select ROI")
cv2.setMouseCallback("1. Select ROI", select_roi_callback, (roi_points, selecting_roi))
print("请在 '1. Select ROI' 窗口中用鼠标拖拽来选择一个区域，然后按任意键。")
cv2.imshow("1. Select ROI", frame_for_selection)
cv2.waitKey(0)
cv2.destroyWindow("1. Select ROI")

if roi_box is None:
    print("未选择ROI区域，程序退出。")
    cap.release()
    exit()

# --- 3. 步骤二：循环追踪 ---
# 将视频重新定位到开头
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("视频处理完成。")
        break

    # 从ROI框坐标中提取 x1, y1, x2, y2
    x1, y1, x2, y2 = roi_box

    # 关键：只对ROI区域进行处理，这样更快且更准确
    roi_frame = frame[y1:y2, x1:x2]

    # a. 转换到HSV并应用阈值
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)

    # b. 形态学后处理
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_processed = cv2.dilate(mask_processed, kernel_dilate, iterations=1)

    # c. 寻找轮廓
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始的、完整的帧上绘制ROI框，方便观察
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 蓝色ROI框

    # d. 找到最大的轮廓并绘制
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # 只有当轮廓面积足够大时才认为是目标
        if cv2.contourArea(largest_contour) > 100:
            ((x_rel, y_rel), radius) = cv2.minEnclosingCircle(largest_contour)

            # 关键：将ROI内的相对坐标转换回整个画面的绝对坐标
            abs_x = int(x_rel + x1)
            abs_y = int(y_rel + y1)

            # 在原始帧上绘制追踪结果
            cv2.circle(frame, (abs_x, abs_y), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (abs_x, abs_y), 5, (0, 0, 255), -1)

            # 在左上角显示坐标
            cv2.putText(frame, f"Pos: ({abs_x}, {abs_y})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # --- 4. 显示结果 ---
    cv2.imshow('Tracking Result', frame)
    # 也可以显示处理后的掩码，方便调试
    # cv2.imshow('Processed Mask in ROI', mask_processed)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# --- 5. 释放资源 ---
cap.release()
cv2.destroyAllWindows()
