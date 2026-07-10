from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train78\weights\best.pt")
    model.train(
        data=r"Y:\deeplearning\ultralytics-8.3.163\ultralytics\cfg\datasets\diy_dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=2,
        cache=False,
        workers=8,
        val=True,  # 是否验证
        # optimizer="AdamW",

    )