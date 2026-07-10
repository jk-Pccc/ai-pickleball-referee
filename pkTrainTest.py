from pathlib import Path

import yaml

from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 定义数据集配置文件路径 - 使用相对路径
    data_yaml_path = r"datasets/Ai-pickleball.yolov11-NEW2/data.yaml"

    # 2. 检查并修复data.yaml文件中的路径配置
    try:
        # 读取现有的data.yaml文件
        with open(data_yaml_path, encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        # 确保path指向正确的数据集目录
        dataset_dir = Path("datasets/Ai-pickleball.yolov11-NEW2")
        data_config["path"] = str(dataset_dir)

        # 确保train、val路径正确
        if "train" not in data_config:
            data_config["train"] = "train/images"
        if "val" not in data_config:
            data_config["val"] = "val/images" if (dataset_dir / "val/images").exists() else "train/images"

        # 保存修复后的配置
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print("已修复data.yaml文件，确保路径正确")
    except Exception as e:
        print(f"警告: 无法修改data.yaml文件: {e}")

    # 3. 加载YOLO11模型
    # 使用相对路径加载模型
    model_path = r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train86_pk_far_small_final42\weights\best.pt"
    # if not os.path.exists(model_path):
    #     # 如果相对路径不存在，尝试使用绝对路径
    #     model_path = r"Y:/deeplearning/ultralytics-8.3.163/runs/detect/train85_pk_airborne4/weights/best.pt"

    print(f"加载模型: {model_path}")
    model = YOLO(model_path)  # 基于现有模型微调

    # 4. 针对空中球检测的锚框优化
    custom_anchors = [[6, 8, 10, 15, 16, 22], [28, 36, 45, 60, 70, 95], [110, 150, 180, 220, 250, 300]]

    # 5. 开始训练
    print(f"开始训练，使用数据集: {data_yaml_path}")
    try:
        model.train(
            # 核心数据配置
            data=data_yaml_path,
            # 训练参数（针对空中球优化）
            epochs=150,
            imgsz=720,
            batch=4,
            workers=0,
            val=True,
            # 优化器与学习率
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.003,
            cos_lr=True,
            # 数据增强
            augment=True,
            degrees=15.0,
            translate=0.2,
            scale=0.9,
            shear=2.0,
            perspective=0.005,
            hsv_h=0.04,
            hsv_s=0.9,
            hsv_v=0.6,
            fliplr=0.5,
            flipud=0.3,
            mosaic=0.8,
            mixup=0.15,
            copy_paste=0.2,
            # 训练设置
            name="train86_pk_far_small_final4",
            exist_ok=False,
            deterministic=True,
            cache=True,
            # 模型设置
            box=9.0,
            cls=1.5,
            dfl=2.0,
            close_mosaic=2,
            single_cls=True,  # 如果只检测匹克球一个类别
            # 其他参数
            patience=12,
            save=True,
            plots=True,
            verbose=True,
        )
        print("训练完成！")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 尝试使用不同的方法修复问题
        print("尝试使用备用方法...")
        # 创建临时data.yaml文件，确保路径正确
        temp_yaml_path = "temp_data.yaml"
        temp_config = {
            "path": str(dataset_dir),
            "train": "train/images",
            "val": "val/images" if (dataset_dir / "val/images").exists() else "train/images",
            "nc": 1,
            "names": ["Pickleball"],
        }
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(temp_config, f, default_flow_style=False)
        print(f"创建了临时配置文件: {temp_yaml_path}")
        # 重新训练
        model.train(
            data=temp_yaml_path,
            epochs=15,
            imgsz=720,
            batch=4,
            workers=0,
            val=True,
            # name="train86_pk_far_small_final4",
            exist_ok=False,
        )
