from ultralytics import YOLO
import cv2
import numpy as np

if __name__ == "__main__":
    # 1. 加载训练后的模型
    model = YOLO(r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train86_pk_far_small_30ep_backup\weights\best.pt")

    # 2. 视频路径与轨迹存储（用于绘制跟踪轨迹）
    video_path = r"Y:\deeplearning\ultralytics-8.3.163\TESTVIDEO\pkvideo3.mp4"
    output_path = r"runs/track/pkvideo_far_tracked.mp4"  # 保存带跟踪的视频
    trajectories = {}  # 存储每个目标的轨迹，key为ID，value为坐标列表

    # 3. 打开视频并初始化视频写入器
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 4. 逐帧检测+跟踪（解决断断续续）
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 5. 多尺度推理+跟踪（针对远处球优化）
        results = model.track(
            source=frame,
            imgsz=[640, 720, 800],  # 多尺度推理，覆盖不同距离的球
            conf=0.2,  # 进一步降低置信度阈值，捕捉远处低置信度球
            iou=0.5,   # 降低IOU阈值，帮助跟踪器关联远处球的检测框
            max_det=200,
            augment=False,  # 跟踪时关闭增强，保证速度与稳定性
            agnostic_nms=True,  # 跨类别NMS，优化小目标检测
            vid_stride=1,
            tracker="bytetrack.yaml",  # 用ByteTrack跟踪，补全轨迹
            persist=True,  # 目标短暂消失后仍保留ID，解决“断断续续”
        )

        # 6. 绘制检测框+轨迹（直观展示跟踪效果）
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                # 为不同ID分配颜色，区分目标
                color = (id * 60 % 255, id * 120 % 255, id * 180 % 255)
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 绘制ID+置信度（标注远处球的低置信度）
                label = f"ID:{id} Conf:{conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 更新轨迹，补全远处球的运动路径
                center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
                if id not in trajectories:
                    trajectories[id] = []
                trajectories[id].append((center_x, center_y))
                # 绘制轨迹（保留最近50个点，避免轨迹过长）
                if len(trajectories[id]) > 50:
                    trajectories[id].pop(0)
                for i in range(1, len(trajectories[id])):
                    cv2.line(frame, trajectories[id][i-1], trajectories[id][i], color, 2)

        # 7. 显示并保存帧
        cv2.imshow("Pickleball Tracking (Far & Small)", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"跟踪后的视频已保存至：{output_path}")