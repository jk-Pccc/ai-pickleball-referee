from sympy import false

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    model.train(
        data=r"coco8.yaml",
        epochs=10,  #训练次数
        imgsz=640,  #指定输入图像的尺寸（单位：像素）。训练时，所有图片会被自动缩放为 640x640 像素，统一输入尺寸方便模型计算。
        batch=2,    #指定 “批次大小”。即每次训练时，模型会同时处理 2 张图片，计算梯度并更新一次参数（批次大小需根据显卡显存调整，显存小则设小，避免内存溢出）。
        cache=False,#控制是否缓存数据集到内存 / 磁盘。False表示不缓存（适合小数据集测试；若数据集大，设为True可加速后续训练，因为无需重复读取图片）
        workers=0,  #指定数据加载的线程数（用于并行读取图片）。Windows 系统建议设为 0（避免多进程冲突）；Linux/Mac 可根据 CPU 核心数设为 4、8 等，加速数据加载。
        # val=False 是否验证
    )
