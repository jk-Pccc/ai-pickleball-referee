from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"Y:\deeplearning\ultralytics-8.3.163\runs\detect\train11\weights\best.pt")
    model.tune(
        data=r"datasets\pickleball.v4i.yolov11\data.yaml",
        epochs=10,
        iterations=300,  # iterations参数在tune()方法中是有效的
        optimizer="AdamW",
        imgsz=640,
        batch=2
    )