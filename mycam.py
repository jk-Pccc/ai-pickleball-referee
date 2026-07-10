import cv2

from ultralytics import YOLO

model = YOLO(r"runs/detect/train82_pk_video4/weights/best.pt")

results = model(
    source="TESTVIDEO/pkvideo3.mp4",
    stream=True,
    show=True,
    save=True,
)

for result in results:
    plotted = result.plot()
    cv2.imshow("YOLO Inference", plotted)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
