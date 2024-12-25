from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='pet_waste_detector'
)