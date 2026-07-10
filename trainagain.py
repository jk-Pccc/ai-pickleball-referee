from ultralytics import YOLO
import torch

# 检查CUDA是否可用
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# 加载预训练模型
model = YOLO('yolo11m.pt')

# 开始训练
results = model.train(
    data="datasets/Ai-pickleball-referee.v5-all-train.yolov11/data.yaml",
    epochs=20,
    patience=50,
    imgsz=640,
    batch=-1,
    optimizer='Adam',
    lr0=0.001,
    lrf=0.01,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.5,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    workers=8,
    amp=True,
    cache=True,
    # device=0  # 指定使用GPU 0
)

# 评估模型
metrics = model.val()
print(f"最终mAP50: {metrics.box.map50:.4f}")
print(f"最终mAP50-95: {metrics.box.map:.4f}")

# # 导出模型
# model.export(format='onnx')