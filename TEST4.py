import os

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

        # 球的状态跟踪（新增轨迹相关变量）
        self.last_ball_position = None  # 上一次球的位置 (x, y)
        self.current_ball_position = None  # 当前球的位置
        self.ball_in_left = False
        self.ball_in_right = False
        self.last_hit_side = None
        # 新增：轨迹存储（保留最近N帧的球位置，避免轨迹过长）
        self.ball_trajectory = []  # 格式：[(x1,y1), (x2,y2), ..., (xn,yn)]
        self.trajectory_length = 30  # 轨迹最大长度（可调整，建议20-40帧）
        self.trajectory_color = (255, 215, 0)  # 轨迹颜色（金色，醒目且不遮挡）
        self.trajectory_thickness = 2  # 轨迹线段粗细

        # 显示设置
        self.display_width = 1280
        self.display_height = 720

    def select_court_boundaries(self, frame):
        """手动选择球场边界（不变）."""
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
        """计算球场中线（不变）."""
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
        """判断点是否在多边形内（不变）."""
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

    def process_ball_detection(self, results, frame):
        """处理球检测结果+新增轨迹更新逻辑."""
        # 初始化：默认当前帧无球，轨迹暂不更新
        has_ball = False

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # 获取检测到的第一个球（假设只有一个球）
            ball_box = results[0].boxes[0]
            ball_xyxy = ball_box.xyxy[0].cpu().numpy()
            confidence = ball_box.conf[0].cpu().numpy()

            # 过滤低置信度球（避免误检影响轨迹）
            if confidence < 0.15:
                self.current_ball_position = None
                return

            # 计算球中心点
            x_center = int((ball_xyxy[0] + ball_xyxy[2]) / 2)
            y_center = int((ball_xyxy[1] + ball_xyxy[3]) / 2)

            self.current_ball_position = (x_center, y_center)
            has_ball = True  # 标记当前帧有球

            # 绘制球检测框（不变）
            cv2.rectangle(
                frame, (int(ball_xyxy[0]), int(ball_xyxy[1])), (int(ball_xyxy[2]), int(ball_xyxy[3])), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Ball: {confidence:.2f}",
                (int(ball_xyxy[0]), int(ball_xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # 绘制球当前位置标记（红色实心点）
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)

            # 判断球在哪半场（不变）
            in_left = x_center < self.midline_x and self.is_point_in_polygon((x_center, y_center), self.court_coords)
            in_right = x_center >= self.midline_x and self.is_point_in_polygon((x_center, y_center), self.court_coords)
            self.ball_in_left = in_left
            self.ball_in_right = in_right

            # 计分逻辑（不变）
            if self.last_ball_position is not None:
                last_x, last_y = self.last_ball_position
                # 检测球是否从一方飞到另一方
                if (
                    last_x < self.midline_x
                    and x_center >= self.midline_x
                    and self.is_point_in_polygon((last_x, last_y), self.court_coords)
                    and self.is_point_in_polygon((x_center, y_center), self.court_coords)
                ):
                    self.last_hit_side = "A"
                    print("A方击球到B方场地")

                elif (
                    last_x >= self.midline_x
                    and x_center < self.midline_x
                    and self.is_point_in_polygon((last_x, last_y), self.court_coords)
                    and self.is_point_in_polygon((x_center, y_center), self.court_coords)
                ):
                    self.last_hit_side = "B"
                    print("B方击球到A方场地")

            # 新增：更新轨迹（仅当当前帧有球时）
            self.update_trajectory((x_center, y_center))

            # 计分判断（不变）
            if self.last_hit_side == "A" and self.ball_in_right and y_center > self.display_height * 0.6:
                self.score_A += 1
                print(f"A方得分！当前比分: A {self.score_A} - B {self.score_B}")
                self.last_hit_side = None
                # 得分后重置轨迹（避免跨回合轨迹干扰）
                self.reset_trajectory()

            elif self.last_hit_side == "B" and self.ball_in_left and y_center > self.display_height * 0.6:
                self.score_B += 1
                print(f"B方得分！当前比分: A {self.score_A} - B {self.score_B}")
                self.last_hit_side = None
                # 得分后重置轨迹
                self.reset_trajectory()

        # 若当前帧无球，轨迹暂存（避免短暂丢失导致轨迹断裂）
        if not has_ball:
            self.current_ball_position = None

        self.last_ball_position = self.current_ball_position if has_ball else self.last_ball_position

    def update_trajectory(self, current_pos):
        """新增：更新球轨迹（限制长度+去重）."""
        # 去重：避免同一位置重复添加（如球暂时静止）
        if (
            self.ball_trajectory
            and abs(current_pos[0] - self.ball_trajectory[-1][0]) < 3
            and abs(current_pos[1] - self.ball_trajectory[-1][1]) < 3
        ):
            return
        # 添加当前位置到轨迹
        self.ball_trajectory.append(current_pos)
        # 限制轨迹长度：超过最大长度则删除最早的位置
        if len(self.ball_trajectory) > self.trajectory_length:
            self.ball_trajectory.pop(0)

    def reset_trajectory(self):
        """新增：重置轨迹（得分/回合结束时调用）."""
        self.ball_trajectory.clear()

    def draw_court_and_info(self, frame):
        """绘制球场、计分信息+新增轨迹绘制."""
        # 1. 绘制球场边界（不变）
        if self.court_coords is not None:
            court_points = np.array(self.court_coords, np.int32)
            cv2.polylines(frame, [court_points], True, (255, 0, 0), 2)

            # 绘制中线（不变）
            if self.midline_x is not None:
                cv2.line(frame, (int(self.midline_x), 0), (int(self.midline_x), self.display_height), (255, 0, 0), 2)

            # 添加场地标签（不变）
            cv2.putText(frame, "A", (int(self.midline_x) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "B", (int(self.midline_x) + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 2. 显示比分（不变）
        cv2.putText(
            frame, f"Score: A {self.score_A} - B {self.score_B}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        # 3. 显示球状态（不变）
        status_text = "Ball: "
        if self.ball_in_left:
            status_text += "Left Court (A)"
        elif self.ball_in_right:
            status_text += "Right Court (B)"
        else:
            status_text += "Not in court"
        cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. 新增：绘制球轨迹（金色线段）
        if len(self.ball_trajectory) >= 2:  # 至少2个点才绘制线段
            for i in range(1, len(self.ball_trajectory)):
                # 获取相邻两个轨迹点
                prev_pos = self.ball_trajectory[i - 1]
                curr_pos = self.ball_trajectory[i]
                # 绘制轨迹线段（金色，粗细2）
                cv2.line(frame, prev_pos, curr_pos, self.trajectory_color, self.trajectory_thickness)
            # 绘制轨迹起点标记（浅绿色，区分起点）
            cv2.circle(frame, self.ball_trajectory[0], 4, (0, 255, 127), -1)
            cv2.putText(
                frame,
                "Start",
                (self.ball_trajectory[0][0] + 5, self.ball_trajectory[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 127),
                1,
            )

    def run(self):
        """运行主程序（新增轨迹重置快捷键）."""
        if not os.path.exists(self.video_path):
            print(f"视频文件不存在: {self.video_path}")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return

        # 读取第一帧用于选择球场边界（不变）
        ret, first_frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
        first_frame = cv2.resize(first_frame, (self.display_width, self.display_height))

        # 选择球场边界（不变）
        self.court_coords = self.select_court_boundaries(first_frame)
        if len(self.court_coords) != 4:
            print("必须选择4个点来定义球场边界")
            return
        self.midline_x = self.calculate_midline(self.court_coords)

        print("开始处理视频...")
        print("按 'q' 退出，按 'r' 重置比分，按 't' 重置轨迹")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.display_width, self.display_height))

            # 使用YOLO模型进行检测和跟踪（不变）
            results = self.model.track(
                source=frame,
                show=False,
                show_conf=True,
                show_labels=True,
                conf=0.05,
                iou=0.4,
                max_det=20,
                imgsz=720,
                batch=1,
                cache=False,
                workers=0,
                val=False,
                augment=True,
                agnostic_nms=True,
                vid_stride=1,
                tracker="botsort.yaml",
                persist=True,
                classes=None,
            )

            # 处理球检测（含轨迹更新）
            self.process_ball_detection(results, frame)

            # 绘制球场、信息、轨迹
            self.draw_court_and_info(frame)

            # 显示结果（不变）
            cv2.imshow("Pickleball Scorer (With Trajectory)", frame)

            # 键盘控制（新增 't' 重置轨迹）
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # 重置比分+轨迹
                self.score_A = 0
                self.score_B = 0
                self.last_hit_side = None
                self.last_ball_position = None
                self.reset_trajectory()
                print("比分和轨迹已重置")
            elif key == ord("t"):
                # 单独重置轨迹
                self.reset_trajectory()
                print("轨迹已重置")

        cap.release()
        cv2.destroyAllWindows()
        print(f"最终比分: A {self.score_A} - B {self.score_B}")


def main():
    scorer = PickleballScorer()
    scorer.run()


if __name__ == "__main__":
    main()
