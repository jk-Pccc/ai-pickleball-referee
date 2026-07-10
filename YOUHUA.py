from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs/detect/train82_pk_video4/weights/best.pt")
    model.train(
        data=r"datasets/Ai-pickleball-referee.v5-all-train.yolov11/data.yaml",
        epochs=10,
        imgsz=640,
        batch=2,
        cache=False,
        workers=0,
        # val=False
    )
