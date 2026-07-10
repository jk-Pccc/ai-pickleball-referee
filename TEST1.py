from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
import tkinter as tk

class PickleBallScorer:
    def __init__(self, model_path, source=0, save_video=False):
        # 模型和视频参数
        self.model = YOLO(model_path)
        self.source = source
        self.save_video = save_video

        # 计分板参数
        self.score_a = 0
        self.score_b = 0
        self.current_server = "A"
        self.rally_count = 0

        # 球跟踪基础参数
        self.ball_positions = deque(maxlen=30)
        self.ball_detected = False
        self.last_detection_time = 0
        self.detection_timeout = 0.8

        # 球场绘制参数
        self.court_corners = []
        self.court_transform = None
        self.draw_complete = False

        # 球状态参数
        self.ball_last_side = None
        self.ball_crossed_net = False

        # 可视化参数
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 255, 255)
        self.thickness = 2

        # 优化参数
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

        self.smooth_positions = deque(maxlen=7)
        self.max_jump = 40
        self.min_ball_size = 8
        self.max_ball_size = 40
        self.min_aspect_ratio = 0.7
        self.max_aspect_ratio = 1.3

        self.track_health = 0
        self.max_bad_health = 3
        self.good_health_threshold = 2

        # 显示自适应参数
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.scale_ratio = 1.0
        self.frame_display_width = 0
        self.frame_display_height = 0

    def get_screen_resolution(self):
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height

    def calculate_scale_ratio(self, frame_width, frame_height):
        max_display_width = self.screen_width * 0.8
        max_display_height = self.screen_height * 0.8
        width_ratio = max_display_width / frame_width
        height_ratio = max_display_height / frame_height
        self.scale_ratio = min(width_ratio, height_ratio)
        self.frame_display_width = int(frame_width * self.scale_ratio)
        self.frame_display_height = int(frame_height * self.scale_ratio)

    def map_display_to_original(self, x, y):
        original_x = int(x / self.scale_ratio)
        original_y = int(y / self.scale_ratio)
        return (original_x, original_y)

    def start_scoring(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("❌ 无法打开视频源")
            return

        self.draw_court_corners(cap)

        if self.draw_complete:
            print("\n✅ 绘制完成，开始自动计分...")

            out = None
            if self.save_video:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter('pickleball_scoring.avi', fourcc, fps, (original_width, original_height))

            predict_params = {
                'conf': 0.4,
                'iou': 0.4,
                'max_det': 3,
                'augment': True,
                'agnostic_nms': True
            }

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\n📹 视频处理完成")
                    break

                # 关键修复：添加模型预测步骤
                results = self.model.predict(frame,** predict_params)

                self.process_detections(results, frame)
                self.update_game_state(frame)
                self.draw_info(frame)

                display_frame = cv2.resize(frame, (self.frame_display_width, self.frame_display_height))
                cv2.imshow('Pickleball Scorer', display_frame)

                if self.save_video and out is not None:
                    out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n👋 用户退出程序")
                    break

            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

    def draw_court_corners(self, cap):
        print("\n🎨 进入球场绘制模式：")
        print("1. 请按【顺时针/逆时针】顺序点击球场的4个角点")
        print("2. 按键操作：'r'=撤销最后一个点，'c'=完成绘制，'q'=退出")

        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧，绘制模式退出")
            return

        original_height, original_width = frame.shape[:2]
        self.calculate_scale_ratio(original_width, original_height)
        draw_frame = cv2.resize(frame, (self.frame_display_width, self.frame_display_height))

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.court_corners) < 4:
                    original_xy = self.map_display_to_original(x, y)
                    self.court_corners.append(original_xy)
                    print(f"✅ 已选择角点 {len(self.court_corners)}: {original_xy}（原始坐标）")

        cv2.namedWindow('Draw Court Corners', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Draw Court Corners', mouse_callback)
        cv2.resizeWindow('Draw Court Corners', self.frame_display_width, self.frame_display_height)

        while True:
            display_frame = draw_frame.copy()

            if len(self.court_corners) > 0:
                for i, (orig_x, orig_y) in enumerate(self.court_corners):
                    disp_x = int(orig_x * self.scale_ratio)
                    disp_y = int(orig_y * self.scale_ratio)
                    cv2.circle(display_frame, (disp_x, disp_y), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"{i+1}", (disp_x+10, disp_y), self.font, 0.8, (0, 255, 0), 2)

                if len(self.court_corners) > 1:
                    disp_pts = np.array([[int(x * self.scale_ratio), int(y * self.scale_ratio)] for x, y in self.court_corners], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [disp_pts], isClosed=False, color=(255, 0, 0), thickness=2)

            info_text = f"已选择 {len(self.court_corners)}/4 个角点 | 缩放比例: {self.scale_ratio:.2f}"
            cv2.putText(display_frame, info_text, (20, 30), self.font, 0.8, (255, 255, 0), 2)
            help_text = "点击选点 | 'r'=撤销 | 'c'=完成 | 'q'=退出"
            cv2.putText(display_frame, help_text, (20, self.frame_display_height-20), self.font, 0.7, (255, 255, 0), 2)

            cv2.imshow('Draw Court Corners', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if self.court_corners:
                    self.court_corners.pop()
                    print("🔄 已撤销最后一个角点")
            elif key == ord('c'):
                if len(self.court_corners) == 4:
                    self.create_court_transform()
                    self.draw_complete = True
                    print("\n✅ 4个角点绘制完成，生成透视变换矩阵")
                else:
                    print(f"❌ 绘制未完成：请选择4个角点（当前{len(self.court_corners)}个）")
                break
            elif key == ord('q'):
                print("👋 退出绘制模式")
                break

        cv2.destroyWindow('Draw Court Corners')

    def create_court_transform(self):
        src_pts = np.array(self.court_corners, dtype=np.float32)
        court_width, court_height = 400, 300
        dst_pts = np.array([[0, 0], [court_width, 0], [court_width, court_height], [0, court_height]], dtype=np.float32)
        self.court_transform = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def transform_point(self, point):
        if self.court_transform is None:
            return None
        pt = np.array([[point]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, self.court_transform)[0][0]
        return transformed_pt

    def process_detections(self, results, frame):
        current_time = time.time()
        height, width = frame.shape[:2]
        detected_balls = []

        for result in results:
            for box in result.boxes:
                if box.cls == 0:
                    try:
                        bbox = box.xyxy[0].cpu().numpy().tolist()[:4]
                        x1, y1, x2, y2 = map(int, bbox)
                    except (ValueError, IndexError):
                        continue

                    conf = float(box.conf)
                    ball_width = x2 - x1
                    ball_height = y2 - y1
                    aspect_ratio = ball_width / ball_height if ball_height > 0 else 0

                    if (self.min_ball_size < ball_width < self.max_ball_size and
                        self.min_ball_size < ball_height < self.max_ball_size and
                        self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detected_balls.append({
                            'pos': (center_x, center_y),
                            'conf': conf,
                            'bbox': (x1, y1, x2, y2)
                        })

        best_ball = None
        if detected_balls:
            if len(self.ball_positions) > 1:
                last_pos = self.ball_positions[-1]
                predicted_kalman_pos = self.kalman.predict()[:2].flatten()

                min_score = float('inf')
                for ball in detected_balls:
                    dist_to_history = np.linalg.norm(np.array(ball['pos']) - np.array(last_pos))
                    dist_to_prediction = np.linalg.norm(np.array(ball['pos']) - predicted_kalman_pos)
                    score = (dist_to_history * 0.6) + (dist_to_prediction * 0.3) - (ball['conf'] * 10)

                    if score < min_score and dist_to_history < self.max_jump:
                        min_score = score
                        best_ball = ball
            else:
                best_ball = max(detected_balls, key=lambda b: b['conf'])

        if best_ball:
            ball_center = best_ball['pos']
            x1, y1, x2, y2 = best_ball['bbox']

            transformed_pos = self.transform_point(ball_center)
            if transformed_pos is not None and 0 <= transformed_pos[0] <= 400 and 0 <= transformed_pos[1] <= 300:
                self.last_detection_time = current_time
                self.ball_detected = True
                self.track_health = min(self.track_health + 1, self.good_health_threshold)

                self.smooth_positions.append(ball_center)
                smooth_center = tuple(np.mean(self.smooth_positions, axis=0).astype(int))

                measurement = np.array(smooth_center, dtype=np.float32).reshape(-1, 1)
                self.kalman.correct(measurement)
                corrected_pos = self.kalman.statePost[:2].flatten().astype(int)

                self.ball_positions.append(corrected_pos)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, corrected_pos, 5, (0, 0, 255), -1)
            else:
                self.ball_detected = False
                self.track_health = max(self.track_health - 1, -self.max_bad_health)
        else:
            self.ball_detected = False
            self.track_health = max(self.track_health - 1, -self.max_bad_health)

            if current_time - self.last_detection_time < self.detection_timeout and len(self.ball_positions) > 0:
                prediction = self.kalman.predict()
                predicted_pos = (int(prediction[0]), int(prediction[1]))

                if 0 <= predicted_pos[0] < width and 0 <= predicted_pos[1] < height:
                    self.ball_positions.append(predicted_pos)
                    cv2.circle(frame, predicted_pos, 5, (255, 255, 0), 1, cv2.LINE_AA)

        if self.track_health <= -self.max_bad_health:
            print("🔄 轨迹质量差，已重置轨迹")
            self.ball_positions.clear()
            self.smooth_positions.clear()
            self.ball_last_side = None
            self.track_health = 0
            self.last_detection_time = 0

    def update_game_state(self, frame):
        if not self.ball_detected or len(self.ball_positions) < 2:
            return

        current_pos = self.ball_positions[-1]
        transformed_pos = self.transform_point(current_pos)

        if transformed_pos is not None:
            net_x_transformed = 200
            current_side = "left" if transformed_pos[0] < net_x_transformed else "right"

            if self.ball_last_side is not None and self.ball_last_side != current_side:
                self.ball_crossed_net = True
                self.rally_count += 1
                print(f"🎾 球过网！当前回合数：{self.rally_count}")

            self.ball_last_side = current_side

    def draw_info(self, frame):
        height, width = frame.shape[:2]

        cv2.putText(frame, f"Player A: {self.score_a}", (50, 50), self.font, self.font_scale, self.font_color, self.thickness)
        cv2.putText(frame, f"Player B: {self.score_b}", (width - 200, 50), self.font, self.font_scale, self.font_color, self.thickness)
        cv2.putText(frame, f"Server: {self.current_server}", (50, 100), self.font, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Rally: {self.rally_count}", (width - 200, 100), self.font, 0.8, (0, 255, 255), 2)

        health_color = (0, 255, 0) if self.track_health >= 0 else (0, 0, 255)
        text_x = max(50, width - 220)
        cv2.putText(frame, f"Track Health: {self.track_health}", (text_x, 150), self.font, 0.7, health_color, 2)

        if len(self.court_corners) == 4:
            pts = np.array(self.court_corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        if len(self.ball_positions) > 1:
            recent_pos = list(self.ball_positions)[-15:]
            for i in range(1, len(recent_pos)):
                alpha = i / len(recent_pos)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.line(frame, recent_pos[i-1], recent_pos[i], color, 2, cv2.LINE_AA)

if __name__ == "__main__":
    MODEL_PATH = r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train85_pk_airborne3\weights\best.pt"
    VIDEO_PATH = r"Y:\deeplearning\ultralytics-8.3.163\TESTVIDEO\pkvideo3.mp4"
    SAVE_VIDEO = False

    scorer = PickleBallScorer(
        model_path=MODEL_PATH,
        source=VIDEO_PATH,
        save_video=SAVE_VIDEO
    )
    scorer.start_scoring()