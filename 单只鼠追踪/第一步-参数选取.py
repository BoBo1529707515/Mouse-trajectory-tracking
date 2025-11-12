import cv2
import numpy as np

# --- 全局变量 ---
roi_box = None
frame_clone = None
hsv_roi = None
original_roi = None


def select_roi_callback(event, x, y, flags, param):
    """鼠标回调函数，用于选择ROI (已加固)"""
    global roi_box, frame_clone

    roi_points, selecting_roi = param

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points[0] = (x, y)
        # 确保第二个点在新的拖拽开始时被重置
        roi_points[1] = None
        selecting_roi[0] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi[0]:
            img_copy = frame_clone.copy()
            cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("1. Select ROI", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        # --- 关键的加固和修正部分 ---

        # 1. 安全检查：如果起始点不存在，则直接返回，防止崩溃。
        if roi_points[0] is None:
            selecting_roi[0] = False
            return

        # 2. 正确赋值：将结束点坐标赋值给列表的第二个元素。
        roi_points[1] = (x, y)
        selecting_roi[0] = False

        # -----------------------------

        # 现在我们可以安全地计算矩形框
        x1 = min(roi_points[0][0], roi_points[1][0])
        y1 = min(roi_points[0][1], roi_points[1][1])
        x2 = max(roi_points[0][0], roi_points[1][0])
        y2 = max(roi_points[0][1], roi_points[1][1])
        roi_box = (x1, y1, x2, y2)

        cv2.rectangle(frame_clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("1. Select ROI", frame_clone)
        print(f"ROI区域已确定。请按任意键进入下一步调参...")


def on_trackbar_change(val):
    """滑动条回调函数"""
    if hsv_roi is None:
        return
    h_min = cv2.getTrackbarPos("H_min", "2. Adjust Thresholds")
    s_min = cv2.getTrackbarPos("S_min", "2. Adjust Thresholds")
    v_min = cv2.getTrackbarPos("V_min", "2. Adjust Thresholds")
    h_max = cv2.getTrackbarPos("H_max", "2. Adjust Thresholds")
    s_max = cv2.getTrackbarPos("S_max", "2. Adjust Thresholds")
    v_max = cv2.getTrackbarPos("V_max", "2. Adjust Thresholds")
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    result = cv2.bitwise_and(original_roi, original_roi, mask=mask)
    cv2.imshow("Mask (ROI only)", mask)
    cv2.imshow("Result (ROI only)", result)


# --- 主程序流程 ---

# 1. 读取视频
VIDEO_PATH = r"C:\Users\15297\Desktop\WeChat_20251109195318.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit()
cap.release()
frame_clone = frame.copy()

# 2. 步骤一：选择ROI
roi_points = [None, None]
selecting_roi = [False]
cv2.namedWindow("1. Select ROI")
cv2.setMouseCallback("1. Select ROI", select_roi_callback, (roi_points, selecting_roi))
print("请在 '1. Select ROI' 窗口中用鼠标拖拽来选择一个区域，然后按任意键。")
cv2.imshow("1. Select ROI", frame)
cv2.waitKey(0)
cv2.destroyWindow("1. Select ROI")

if roi_box is None:
    print("未选择ROI区域，程序退出。")
    exit()

# 3. 步骤二：调节参数
x1, y1, x2, y2 = roi_box
original_roi = frame[y1:y2, x1:x2]
hsv_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2HSV)
cv2.namedWindow("2. Adjust Thresholds")
cv2.resizeWindow("2. Adjust Thresholds", 640, 240)
cv2.createTrackbar("H_min", "2. Adjust Thresholds", 0, 179, on_trackbar_change)
cv2.createTrackbar("S_min", "2. Adjust Thresholds", 0, 255, on_trackbar_change)
cv2.createTrackbar("V_min", "2. Adjust Thresholds", 150, 255, on_trackbar_change)
cv2.createTrackbar("H_max", "2. Adjust Thresholds", 179, 179, on_trackbar_change)
cv2.createTrackbar("S_max", "2. Adjust Thresholds", 50, 255, on_trackbar_change)
cv2.createTrackbar("V_max", "2. Adjust Thresholds", 255, 255, on_trackbar_change)
cv2.imshow("Original ROI", original_roi)
on_trackbar_change(0)
print("\n现在请拖动 '2. Adjust Thresholds' 窗口中的滑动条来寻找最佳阈值。")
print("完成后，记下参数，然后按 'q' 键退出。")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        h_min, s_min, v_min = (
            cv2.getTrackbarPos("H_min", "2. Adjust Thresholds"),
            cv2.getTrackbarPos("S_min", "2. Adjust Thresholds"),
            cv2.getTrackbarPos("V_min", "2. Adjust Thresholds"),
        )
        h_max, s_max, v_max = (
            cv2.getTrackbarPos("H_max", "2. Adjust Thresholds"),
            cv2.getTrackbarPos("S_max", "2. Adjust Thresholds"),
            cv2.getTrackbarPos("V_max", "2. Adjust Thresholds"),
        )
        print("\n--- 最终确定的阈值 ---")
        print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
        break
cv2.destroyAllWindows()
