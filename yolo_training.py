from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

model.train(
    data='data.yaml',
    epochs=2,
    imgsz=640,
    batch=8,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    name='actions_yolov11s_adamw'
)