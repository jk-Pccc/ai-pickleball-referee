# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont


def cv2_add_chinese_text(img, text, position, text_color=(255, 255, 255), text_size=20):
    """
    在OpenCV图像上绘制中文字符
    使用PIL库来确保中文字体正确显示
    """
    try:
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 尝试加载中文字体
        # Windows系统默认字体路径
        font_paths = [
            "C:\\Windows\\Fonts\\simhei.ttf",  # 黑体
            "C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
            "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
            "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",  # Linux常见字体
            "/Library/Fonts/SimHei.ttf"  # macOS常见字体
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, text_size, encoding="utf-8")
                break
            except:
                continue

        # 如果找不到字体，使用默认字体
        if font is None:
            font = ImageFont.load_default()

        # 绘制文本
        draw.text(position, text, font=font, fill=text_color)

        # 将PIL图像转换回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 如果出错，使用OpenCV默认方法作为备选
        print(f"中文显示出错: {e}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, position, font, text_size / 20, text_color, 1, cv2.LINE_AA)
        return img


def cv2_add_chinese_text_to_info_overlay(overlay, text, position, text_color=(255, 255, 255), text_size=20):
    """
    为RGBA格式的info_overlay添加中文字符
    """
    try:
        # 将RGBA转换为RGB用于PIL处理
        overlay_rgb = overlay[:, :, :3].copy()
        # 处理文本
        overlay_with_text = cv2_add_chinese_text(overlay_rgb, text, position, text_color, text_size)
        # 将结果放回原overlay
        overlay[:, :, :3] = overlay_with_text
        return overlay
    except:
        # 备选方案：将文本添加到RGB通道
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay[:, :, :3], text, position, font, text_size / 20, text_color, 1, cv2.LINE_AA)
        return overlay


class PickleBallScorer:
    def __init__(self, model_path, source=0, save_video=False):
        # 加载模型
        self.model = YOLO(model_path)
        # 视频源（0表示摄像头，也可以是视频文件路径）
        self.source = source
        # 视频保存选项
        self.save_video = save_video

        # 计分板
        self.score_a = 0
        self.score_b = 0
        self.current_server = "A"  # 初始发球方
        self.rally_count = 0  # 回合计数

        # 球跟踪参数
        self.ball_positions = deque(maxlen=50)  # 增加存储帧数
        self.ball_detected = False
        self.last_detection_time = 0
        self.detection_timeout = 0.5  # 0.5秒内没有检测到球仍认为球存在
        self.conf_threshold = 0.05  # 极低的初始置信度阈值

        # 球场边框设置
        self.court_setup_mode = False
        self.court_points = []
        self.court_defined = False
        self.court_polygon = None
        self.left_region = None
        self.right_region = None
        self.net_line = None

        # 调试模式
        self.debug_mode = False

        # 预测参数
        self.velocity = [0, 0]
        self.acceleration = [0, 0.2]  # 添加重力加速度
        self.last_valid_pos = None

        # 可视化参数
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.thickness = 2
        self.bg_color = (0, 0, 0, 128)  # 半透明背景

    def start_scoring(self):
        # 打开视频流
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("无法打开视频源")
            return

        # 设置视频保存
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter('pickleball_scoring.avi', fourcc, fps, (width, height))

        # 窗口设置
        cv2.namedWindow('Pickleball Scorer', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Pickleball Scorer', self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 创建透明背景图层用于叠加
            overlay = frame.copy()
            info_overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

            # 进行预测
            results = self.model.predict(frame,
                                         conf=self.conf_threshold,  # 使用动态置信度阈值
                                         iou=0.4,  # 降低IOU阈值以捕获更多潜在目标
                                         max_det=20,  # 增加最大检测数量
                                         augment=True,  # 启用预测增强
                                         agnostic_nms=True)  # 跨类别NMS

            # 处理检测结果
            self.process_detections(results, frame, overlay)

            # 如果在球场设置模式，显示设置提示
            if self.court_setup_mode:
                self.draw_court_setup_guide(overlay)

            # 更新球状态和计分
            if self.court_defined:
                self.update_game_state()

            # 绘制信息到画面
            self.draw_info(overlay, info_overlay)

            # 合并图层
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 从info_overlay提取透明度通道
            alpha_channel = info_overlay[:, :, 3] / 255.0
            for c in range(0, 3):
                frame[:, :, c] = (1 - alpha_channel) * frame[:, :, c] + alpha_channel * info_overlay[:, :, c]

            # 显示画面
            cv2.imshow('Pickleball Scorer', frame)

            # 保存视频
            if self.save_video:
                out.write(frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # 重置分数
                self.score_a = 0
                self.score_b = 0
                self.rally_count = 0
                self.ball_positions.clear()
            elif key == ord('+') and self.conf_threshold < 0.9:  # 增加置信度阈值
                self.conf_threshold += 0.05
                print(f"置信度阈值调整为: {self.conf_threshold:.2f}")
            elif key == ord('-') and self.conf_threshold > 0.05:  # 降低置信度阈值
                self.conf_threshold -= 0.05
                print(f"置信度阈值调整为: {self.conf_threshold:.2f}")
            elif key == ord('d'):  # 切换调试模式
                self.debug_mode = not self.debug_mode
                print(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
            elif key == ord('c'):  # 进入/退出球场设置模式
                self.court_setup_mode = not self.court_setup_mode
                if self.court_setup_mode:
                    self.court_points = []
                    self.court_defined = False
                    print("进入球场设置模式。请按顺序点击球场的左上角、右上角、右下角和左下角，然后按Enter完成")
                else:
                    print("退出球场设置模式")
            elif key == 13 and self.court_setup_mode and len(self.court_points) == 4:  # Enter键确认球场设置
                self.define_court_regions()
                self.court_setup_mode = False
                print("球场设置完成")

        # 释放资源
        cap.release()
        if self.save_video:
            out.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        # 在球场设置模式下处理鼠标点击
        if self.court_setup_mode and event == cv2.EVENT_LBUTTONDOWN:
            if len(self.court_points) < 4:
                self.court_points.append((x, y))
                print(f"已添加点 {len(self.court_points)}: ({x}, {y})")
                if len(self.court_points) == 4:
                    print("已收集全部4个点，请按Enter键完成设置")

    def define_court_regions(self):
        if len(self.court_points) != 4:
            return

        # 创建球场多边形
        self.court_polygon = np.array(self.court_points, np.int32)
        self.court_polygon = self.court_polygon.reshape((-1, 1, 2))

        # 计算球网位置（左右半场的分割线）
        # 简化实现：找到球场的左右边界的中心点作为球网位置
        leftmost = min(p[0] for p in self.court_points)
        rightmost = max(p[0] for p in self.court_points)
        net_x = (leftmost + rightmost) // 2

        # 找到球场的上下边界
        topmost = min(p[1] for p in self.court_points)
        bottommost = max(p[1] for p in self.court_points)

        self.net_line = [(net_x, topmost), (net_x, bottommost)]

        # 定义左右半场区域
        self.left_region = [(leftmost, topmost), (net_x, topmost), (net_x, bottommost), (leftmost, bottommost)]
        self.right_region = [(net_x, topmost), (rightmost, topmost), (rightmost, bottommost), (net_x, bottommost)]

        self.court_defined = True

    def draw_court_setup_guide(self, frame):
        # 绘制已添加的点
        for i, point in enumerate(self.court_points):
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
            # 使用支持中文的函数绘制文本
            frame = cv2_add_chinese_text(frame, f"点{i + 1}", (point[0] + 10, point[1] - 10),
                                         (0, 255, 0), 15)

        # 连接点形成多边形
        if len(self.court_points) > 1:
            for i in range(1, len(self.court_points)):
                cv2.line(frame, self.court_points[i - 1], self.court_points[i], (0, 255, 0), 2)
            # 连接最后一个点和第一个点形成闭环（如果已添加4个点）
            if len(self.court_points) == 4:
                cv2.line(frame, self.court_points[3], self.court_points[0], (0, 255, 0), 2)

        # 显示提示信息
        messages = [
            "球场设置模式",
            f"已添加点: {len(self.court_points)}/4",
            "请按顺序点击: 左上角 -> 右上角 -> 右下角 -> 左下角",
            "完成后按Enter键，按'c'键取消"
        ]

        y_pos = 30
        for msg in messages:
            # 绘制带背景的文本
            # 估算文本大小
            (text_width, text_height), _ = cv2.getTextSize(msg, self.font, 0.7, 2)
            cv2.rectangle(frame, (10, y_pos - 20), (10 + text_width, y_pos + 5), (0, 0, 0), -1)
            # 使用支持中文的函数绘制文本
            frame = cv2_add_chinese_text(frame, msg, (10, y_pos), (0, 255, 255), 20)
            y_pos += 30

    def process_detections(self, results, frame, overlay):
        current_time = time.time()
        height, width = frame.shape[:2]

        # 检查是否检测到球
        ball_detected_this_frame = False
        ball_center = None
        best_conf = 0
        best_box = None

        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 检查是否是匹克球（假设类别0是匹克球）
                if box.cls == 0:
                    conf = float(box.conf)
                    # 保存置信度最高的球
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
                        ball_detected_this_frame = True
                        self.last_detection_time = current_time

        # 如果找到最佳球，处理它
        if best_box is not None:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            # 计算球的中心
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            ball_center = (center_x, center_y)

            # 验证球是否在球场区域内
            if self.court_defined:
                if not cv2.pointPolygonTest(self.court_polygon, (center_x, center_y), False) >= 0:
                    # 球不在球场区域内，可能是误检
                    if self.debug_mode:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色表示球场外
                    ball_detected_this_frame = False
                    ball_center = None

            if ball_detected_this_frame:
                # 记录有效位置
                self.last_valid_pos = ball_center
                # 更新速度和加速度
                self.update_velocity_and_acceleration(ball_center)
                # 记录球的位置
                self.ball_positions.append(ball_center)

                # 绘制边界框和中心点
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(overlay, ball_center, 5, (0, 0, 255), -1)

                # 在调试模式下显示置信度
                if self.debug_mode:
                    overlay = cv2_add_chinese_text(overlay, f"置信度: {best_conf:.2f}",
                                                   (x1, y1 - 10), (0, 255, 255), 15)

        # 处理漏检情况 - 使用物理模型预测位置
        if not ball_detected_this_frame and self.last_valid_pos is not None:
            # 如果最近检测到过球，使用历史位置和物理模型预测当前位置
            if (current_time - self.last_detection_time < self.detection_timeout):
                # 使用物理模型预测
                predicted_pos = self.predict_ball_position()

                # 确保预测位置在画面内
                predicted_x = max(0, min(width - 1, predicted_pos[0]))
                predicted_y = max(0, min(height - 1, predicted_pos[1]))

                predicted_center = (int(predicted_x), int(predicted_y))
                self.ball_positions.append(predicted_center)
                self.last_valid_pos = predicted_center

                # 更新速度（应用重力和阻力）
                self.velocity[0] *= 0.98  # 水平方向阻力
                self.velocity[1] += self.acceleration[1]  # 应用重力

                # 绘制预测的球位置
                cv2.circle(overlay, predicted_center, 5, (255, 0, 0), 1)  # 蓝色表示预测位置

        # 更新球的检测状态
        self.ball_detected = ball_detected_this_frame or (len(self.ball_positions) > 0 and
                                                          current_time - self.last_detection_time < self.detection_timeout)

    def update_velocity_and_acceleration(self, current_pos):
        # 如果有历史位置，计算速度
        if len(self.ball_positions) > 0:
            prev_pos = self.ball_positions[-1]
            # 计算新速度
            new_velocity = [
                current_pos[0] - prev_pos[0],
                current_pos[1] - prev_pos[1]
            ]
            # 应用平滑滤波
            alpha = 0.7  # 平滑系数
            self.velocity[0] = alpha * self.velocity[0] + (1 - alpha) * new_velocity[0]
            self.velocity[1] = alpha * self.velocity[1] + (1 - alpha) * new_velocity[1]

    def predict_ball_position(self):
        # 基于物理模型预测球的位置
        if self.last_valid_pos is None:
            return (0, 0)

        # 应用速度和加速度进行预测
        predicted_x = self.last_valid_pos[0] + self.velocity[0]
        predicted_y = self.last_valid_pos[1] + self.velocity[1]

        return (predicted_x, predicted_y)

    def update_game_state(self):
        if not self.ball_detected or len(self.ball_positions) < 2:
            return

        # 获取最新的球位置
        current_pos = self.ball_positions[-1]
        previous_pos = self.ball_positions[-2]

        # 判断球当前所在半场
        current_side = self.get_ball_side(current_pos)
        previous_side = self.get_ball_side(previous_pos)

        # 检测球是否过网
        if previous_side is not None and current_side is not None and previous_side != current_side:
            self.rally_count += 1
            print(f"球过网! 回合数: {self.rally_count}")

        # TODO: 根据实际比赛规则实现更复杂的计分逻辑

    def get_ball_side(self, position):
        # 判断球在左半场还是右半场
        if self.net_line is None:
            return None

        net_x = self.net_line[0][0]
        return "left" if position[0] < net_x else "right"

    def draw_info(self, overlay, info_overlay):
        height, width = overlay.shape[:2]

        # 绘制计分板
        score_text = f"A: {self.score_a}  -  B: {self.score_b}"
        server_text = f"发球方: {self.current_server}"
        rally_text = f"回合数: {self.rally_count}"
        conf_text = f"置信度阈值: {self.conf_threshold:.2f}"

        # 绘制半透明背景
        cv2.rectangle(info_overlay, (10, 10), (300, 130), self.bg_color, -1)

        # 绘制文本 - 使用支持中文的函数
        info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, score_text, (20, 40),
                                                            self.font_color, 20)
        info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, server_text, (20, 70),
                                                            self.font_color, 20)
        info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, rally_text, (20, 100),
                                                            self.font_color, 20)

        # 绘制置信度阈值信息
        info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, conf_text, (20, height - 30),
                                                            (0, 255, 255), 15)

        # 如果在调试模式，显示更多信息
        if self.debug_mode:
            debug_texts = [
                f"球检测: {'是' if self.ball_detected else '否'}",
                f"轨迹点数: {len(self.ball_positions)}",
                f"速度: ({self.velocity[0]:.1f}, {self.velocity[1]:.1f})"
            ]

            y_pos = height - 60
            for text in debug_texts:
                info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, text, (20, y_pos),
                                                                    (255, 255, 0), 15)
                y_pos -= 20

        # 绘制球场边框（如果已定义）
        if self.court_defined:
            # 绘制球场多边形
            cv2.polylines(overlay, [self.court_polygon], True, (0, 255, 255), 2)
            # 绘制球网
            cv2.line(overlay, self.net_line[0], self.net_line[1], (255, 0, 0), 2)

        # 绘制球的轨迹
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                # 渐变色轨迹
                alpha = i / len(self.ball_positions)
                color = (int(255 * (1 - alpha)), int(255 * alpha), int(255 * alpha))
                cv2.line(overlay, self.ball_positions[i - 1], self.ball_positions[i], color, 2)

        # 绘制操作提示
        help_text = "操作: c=设置球场, +/-=调整置信度, r=重置, d=调试模式, q=退出"
        cv2.rectangle(info_overlay, (width - 600, height - 30), (width - 10, height - 10), self.bg_color, -1)
        info_overlay = cv2_add_chinese_text_to_info_overlay(info_overlay, help_text, (width - 590, height - 15),
                                                            (255, 255, 255), 15)


if __name__ == "__main__":
    # 使用用户提供的模型路径
    model_path = r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train86_pk_far_small_final43\weights\best.pt"

    # 创建计分器实例
    scorer = PickleBallScorer(
        model_path=model_path,
        source=r"Y:\deeplearning\ultralytics-8.3.163\TESTVIDEO\pkvideo3.mp4",  # 使用指定视频
        save_video=False  # 设置为True可保存视频
    )

    # 开始计分
    scorer.start_scoring()