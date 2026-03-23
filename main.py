from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source=0,   # webcam
    show=True
)