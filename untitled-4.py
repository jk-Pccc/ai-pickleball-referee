# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import os

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 1. 加载模型
model_path = r"Y:/deeplearning/ultralytics-8.3.163/runs/detect/train85_pk_airborne3/weights/best.pt"
if not os.path.exists(model_path):
    # 尝试使用相对路径
    model_path = r"runs/detect/train85_pk_airborne3/weights/best.pt"
print(f"加载模型: {model_path}")
model = YOLO(model_path)

# 2. 视频路径与轨迹存储（用于绘制跟踪轨迹）
video_path = r"Y:/deeplearning/ultralytics-8.3.163/TESTVIDEO/pkvideo3.mp4"
if not os.path.exists(video_path):
    # 尝试使用相对路径
    video_path = r"TESTVIDEO/pkvideo3.mp4"
print(f"使用视频: {video_path}")

# 轨迹存储，最多保存50个点
track_history = deque(maxlen=50)

# 3. 设置检测参数
# 修复imgsz参数：使用单个整数或两个元素的列表[h, w]
imgsz = 720  # 或者使用 [720, 1280] 这样的格式
conf_threshold = 0.25  # 置信度阈值
iou_threshold = 0.45  # IoU阈值
classes = [0]  # 假设匹克球是类别0

# 4. 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

# 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频信息: {width}x{height} @ {fps}fps")

# 5. 处理视频帧
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频处理完成或无法读取帧")
        break
    
    frame_count += 1
    
    # 运行跟踪（修复imgsz参数）
    results = model.track(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=classes,
        imgsz=imgsz,  # 修复：使用正确格式的imgsz参数
        persist=True,  # 启用持久化跟踪
        show=False,  # 不显示默认窗口
        tracker="botsort.yaml"  # 使用botsort跟踪器
    )
    
    # 处理结果
    annotated_frame = results[0].plot()
    
    # 获取检测框和跟踪ID（如果有）
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        # 假设我们跟踪第一个检测到的对象
        box = boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # 添加到轨迹历史
        track_history.append((center_x, center_y))
        
        # 绘制轨迹
        for i in range(1, len(track_history)):
            if track_history[i-1] is None or track_history[i] is None:
                continue
            # 使用渐变色绘制轨迹
            thickness = int(np.clip(len(track_history) - i, 1, 3))
            cv2.line(annotated_frame, track_history[i-1], track_history[i], 
                    (0, 255 - i * 5, i * 5), thickness)
    
    # 显示帧率和帧数
    fps_info = f"FPS: {fps:.2f}"
    frame_info = f"Frame: {frame_count}"
    cv2.putText(annotated_frame, fps_info, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, frame_info, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow("匹克球检测与跟踪", annotated_frame)
    
    # 按'q'退出，按'space'暂停
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("用户退出")
        break
    elif key == ord(' '):
        print("视频暂停")
        cv2.waitKey(0)  # 等待任意键继续

# 清理
cap.release()
cv2.destroyAllWindows()
print("处理完成")
