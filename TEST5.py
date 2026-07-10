import os
from collections import deque

import cv2
import numpy as np

from ultralytics import YOLO


class PickleballScorer:
    def __init__(self):
        self.model_path = (
            r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train86_pk_far_small_30ep_backup\weights\best.pt"
        )
        self.video_path = r"Y:\deeplearning\ultralytics-8.3.163\TESTVIDEO\pkvideo4.mp4"

        # 初始化模型
        self.model = YOLO(self.model_path, task="detect")

        # 计分变量
        self.score_A = 0
        self.score_B = 0

        # 场地边界和状态变量
        self.court_coords = None
        self.midline_x = None

        # 球的状态跟踪（优化后的轨迹相关变量）
        self.last_ball_position = None
        self.current_ball_position = None
        self.ball_in_left = False
        self.ball_in_right = False
        self.last_hit_side = None

        # 优化：使用双端队列存储轨迹，限制长度
        self.ball_trajectory = deque(maxlen=30)  # 限制轨迹长度
        self.trajectory_color = (255, 215, 0)  # 轨迹颜色
        self.trajectory_thickness = 2

        # 新增：轨迹平滑和过滤参数
        self.position_buffer = deque(maxlen=5)  # 位置缓冲区用于平滑
        self.min_confidence = 0.25  # 提高置信度阈值
        self.max_jump_distance = 100  # 最大允许的帧间跳跃距离（像素）

        # 新增：状态标志
        self.ball_lost_frames = 0
        self.max_lost_frames = 5  # 允许球丢失的最大帧数

        # 新增：得分区域定义
        self.scoring_zone_y = 0.6  # 得分区域占画面高度的比例

        # 显示设置
        self.display_width = 1280
        self.display_height = 720

    def select_court_boundaries(self, frame):
        """手动选择球场边界."""
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    if len(points) > 1:
                        cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
                    cv2.imshow("Select Court Boundaries", frame)

                if len(points) == 4:
                    cv2.line(frame, points[3], points[0], (0, 255, 0), 2)
                    cv2.imshow("Select Court Boundaries", frame)

        clone = frame.copy()
        cv2.imshow("Select Court Boundaries", clone)
        cv2.setMouseCallback("Select Court Boundaries", mouse_callback)

        print("请按顺序点击球场的四个角点（顺时针或逆时针）")
        print("选择完成后，按 'c' 确认，按 'r' 重新选择")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and len(points) == 4:
                break
            elif key == ord("r"):
                points = []
                frame = clone.copy()
                cv2.imshow("Select Court Boundaries", frame)

        cv2.destroyWindow("Select Court Boundaries")
        return points

    def calculate_midline(self, court_coords):
        """计算球场中线."""
        left_points = [p for p in court_coords if p[0] < self.display_width // 2]
        right_points = [p for p in court_coords if p[0] >= self.display_width // 2]

        if left_points and right_points:
            avg_left_x = np.mean([p[0] for p in left_points])
            avg_right_x = np.mean([p[0] for p in right_points])
            midline_x = (avg_left_x + avg_right_x) / 2
        else:
            midline_x = self.display_width // 2

        return midline_x

    def is_point_in_polygon(self, point, polygon):
        """判断点是否在多边形内."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def smooth_position(self, new_position):
        """使用移动平均平滑球的位置."""
        if new_position is None:
            return None

        self.position_buffer.append(new_position)

        # 如果缓冲区数据太少，返回原始位置
        if len(self.position_buffer) < 3:
            return new_position

        # 计算移动平均
        avg_x = np.mean([p[0] for p in self.position_buffer])
        avg_y = np.mean([p[1] for p in self.position_buffer])

        return (int(avg_x), int(avg_y))

    def is_valid_movement(self, prev_pos, curr_pos):
        """检查移动是否合理."""
        if prev_pos is None or curr_pos is None:
            return True

        distance = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
        return distance <= self.max_jump_distance

    def process_ball_detection(self, results, frame):
        """处理球检测结果 - 优化版本."""
        best_ball = None
        best_confidence = 0

        # 寻找置信度最高的球检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i, ball_box in enumerate(results[0].boxes):
                confidence = ball_box.conf[0].cpu().numpy()

                # 选择置信度最高的检测结果
                if confidence > best_confidence and confidence >= self.min_confidence:
                    best_confidence = confidence
                    best_ball = ball_box

        if best_ball is not None:
            ball_xyxy = best_ball.xyxy[0].cpu().numpy()

            # 计算球中心点
            x_center = int((ball_xyxy[0] + ball_xyxy[2]) / 2)
            y_center = int((ball_xyxy[1] + ball_xyxy[3]) / 2)

            raw_position = (x_center, y_center)

            # 位置平滑处理
            smoothed_position = self.smooth_position(raw_position)

            # 检查移动是否合理
            if not self.is_valid_movement(self.last_ball_position, smoothed_position):
                print("检测到异常移动，忽略当前帧检测")
                self.ball_lost_frames += 1
                if self.ball_lost_frames > self.max_lost_frames:
                    self.current_ball_position = None
                return

            self.current_ball_position = smoothed_position
            self.ball_lost_frames = 0  # 重置丢失计数器

            # 绘制球检测框
            cv2.rectangle(
                frame, (int(ball_xyxy[0]), int(ball_xyxy[1])), (int(ball_xyxy[2]), int(ball_xyxy[3])), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Ball: {best_confidence:.2f}",
                (int(ball_xyxy[0]), int(ball_xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # 绘制球当前位置标记
            cv2.circle(frame, smoothed_position, 5, (0, 0, 255), -1)

            # 判断球在哪半场
            in_left = smoothed_position[0] < self.midline_x and self.is_point_in_polygon(
                smoothed_position, self.court_coords
            )
            in_right = smoothed_position[0] >= self.midline_x and self.is_point_in_polygon(
                smoothed_position, self.court_coords
            )
            self.ball_in_left = in_left
            self.ball_in_right = in_right

            # 计分逻辑
            if self.last_ball_position is not None:
                last_x, last_y = self.last_ball_position
                curr_x, curr_y = smoothed_position

                # 检测球是否从一方飞到另一方
                if (
                    last_x < self.midline_x
                    and curr_x >= self.midline_x
                    and self.is_point_in_polygon((last_x, last_y), self.court_coords)
                    and self.is_point_in_polygon((curr_x, curr_y), self.court_coords)
                ):
                    self.last_hit_side = "A"
                    print("A方击球到B方场地")

                elif (
                    last_x >= self.midline_x
                    and curr_x < self.midline_x
                    and self.is_point_in_polygon((last_x, last_y), self.court_coords)
                    and self.is_point_in_polygon((curr_x, curr_y), self.court_coords)
                ):
                    self.last_hit_side = "B"
                    print("B方击球到A方场地")

            # 更新轨迹
            self.update_trajectory(smoothed_position)

            # 计分判断
            if (
                self.last_hit_side == "A"
                and self.ball_in_right
                and smoothed_position[1] > self.display_height * self.scoring_zone_y
            ):
                self.score_A += 1
                print(f"A方得分！当前比分: A {self.score_A} - B {self.score_B}")
                self.last_hit_side = None
                self.reset_trajectory()

            elif (
                self.last_hit_side == "B"
                and self.ball_in_left
                and smoothed_position[1] > self.display_height * self.scoring_zone_y
            ):
                self.score_B += 1
                print(f"B方得分！当前比分: A {self.score_A} - B {self.score_B}")
                self.last_hit_side = None
                self.reset_trajectory()

        else:
            # 当前帧没有检测到球
            self.ball_lost_frames += 1
            if self.ball_lost_frames > self.max_lost_frames:
                self.current_ball_position = None
                # 球丢失时间过长，清空位置缓冲区
                self.position_buffer.clear()

        self.last_ball_position = self.current_ball_position

    def update_trajectory(self, current_pos):
        """更新球轨迹 - 优化版本."""
        if not current_pos:
            return

        # 去重逻辑：避免同一位置重复添加
        if self.ball_trajectory:
            last_pos = self.ball_trajectory[-1]
            distance = np.sqrt((current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2)
            # 只有当移动距离大于阈值时才添加新点
            if distance < 3:  # 3像素阈值
                return

        self.ball_trajectory.append(current_pos)

    def reset_trajectory(self):
        """重置轨迹."""
        self.ball_trajectory.clear()
        self.position_buffer.clear()
        self.ball_lost_frames = 0

    def draw_court_and_info(self, frame):
        """绘制球场、计分信息和轨迹."""
        # 绘制球场边界
        if self.court_coords is not None:
            court_points = np.array(self.court_coords, np.int32)
            cv2.polylines(frame, [court_points], True, (255, 0, 0), 2)

            # 绘制中线
            if self.midline_x is not None:
                cv2.line(frame, (int(self.midline_x), 0), (int(self.midline_x), self.display_height), (255, 0, 0), 2)

            # 添加场地标签
            cv2.putText(frame, "A", (int(self.midline_x) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "B", (int(self.midline_x) + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 绘制得分区域线
            scoring_line_y = int(self.display_height * self.scoring_zone_y)
            cv2.line(frame, (0, scoring_line_y), (self.display_width, scoring_line_y), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Scoring Line", (10, scoring_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 显示比分
        cv2.putText(
            frame, f"Score: A {self.score_A} - B {self.score_B}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        # 显示球状态
        status_text = "Ball: "
        if self.ball_in_left:
            status_text += "Left Court (A)"
        elif self.ball_in_right:
            status_text += "Right Court (B)"
        else:
            status_text += "Not in court"

        # 添加轨迹点数量信息
        status_text += f" | Trajectory: {len(self.ball_trajectory)} points"

        cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 绘制球轨迹
        if len(self.ball_trajectory) >= 2:
            # 使用渐变色表示轨迹时间顺序
            for i in range(1, len(self.ball_trajectory)):
                prev_pos = self.ball_trajectory[i - 1]
                curr_pos = self.ball_trajectory[i]

                # 根据轨迹点的新旧程度调整颜色（越新越亮）
                color_intensity = int(255 * i / len(self.ball_trajectory))
                color = (color_intensity, 215, 255 - color_intensity)

                cv2.line(frame, prev_pos, curr_pos, color, self.trajectory_thickness)

            # 绘制轨迹起点和终点标记
            if len(self.ball_trajectory) > 0:
                start_pos = self.ball_trajectory[0]
                end_pos = self.ball_trajectory[-1]

                cv2.circle(frame, start_pos, 4, (0, 255, 0), -1)  # 绿色起点
                cv2.circle(frame, end_pos, 4, (0, 0, 255), -1)  # 红色终点

                cv2.putText(
                    frame, "Start", (start_pos[0] + 5, start_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
                cv2.putText(
                    frame, "End", (end_pos[0] + 5, end_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                )

    def run(self):
        """运行主程序."""
        if not os.path.exists(self.video_path):
            print(f"视频文件不存在: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return

        # 读取第一帧用于选择球场边界
        ret, first_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
        first_frame = cv2.resize(first_frame, (self.display_width, self.display_height))

        # 选择球场边界
        self.court_coords = self.select_court_boundaries(first_frame)
        if len(self.court_coords) != 4:
            print("必须选择4个点来定义球场边界")
            return
        self.midline_x = self.calculate_midline(self.court_coords)

        print("开始处理视频...")
        print("按 'q' 退出，按 'r' 重置比分，按 't' 重置轨迹")
        print("按 '+' 增加得分区域高度，按 '-' 降低得分区域高度")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (self.display_width, self.display_height))

            # 使用YOLO模型进行检测和跟踪
            results = self.model.track(
                source=frame,
                show=False,
                conf=0.1,  # 稍微降低置信度阈值以捕捉更多可能
                iou=0.4,
                max_det=10,  # 减少最大检测数量
                imgsz=720,
                tracker="botsort.yaml",
                persist=True,
                verbose=False,  # 减少输出
            )

            # 处理球检测
            self.process_ball_detection(results, frame)

            # 绘制球场、信息、轨迹
            self.draw_court_and_info(frame)

            # 显示帧计数
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, self.display_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # 显示结果
            cv2.imshow("Pickleball Scorer (Optimized)", frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.score_A = 0
                self.score_B = 0
                self.last_hit_side = None
                self.last_ball_position = None
                self.reset_trajectory()
                print("比分和轨迹已重置")
            elif key == ord("t"):
                self.reset_trajectory()
                print("轨迹已重置")
            elif key == ord("+"):
                self.scoring_zone_y = min(0.9, self.scoring_zone_y + 0.05)
                print(f"得分区域调整到: {self.scoring_zone_y:.2f}")
            elif key == ord("-"):
                self.scoring_zone_y = max(0.3, self.scoring_zone_y - 0.05)
                print(f"得分区域调整到: {self.scoring_zone_y:.2f}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"最终比分: A {self.score_A} - B {self.score_B}")


def main():
    scorer = PickleballScorer()
    scorer.run()


if __name__ == "__main__":
    main()
