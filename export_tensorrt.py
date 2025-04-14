from ultralytics import YOLOv10
import os

for model_name in [
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
]: 
    if not os.path.exists(f"{model_name}.pt"):
        print(f"{model_name}.pt not found, skipping export.")
        continue
    if os.path.exists(f"{model_name}.engine"):
        print(f"{model_name}.engine already exists, skipping export.")
        continue
    model = YOLOv10(f"{model_name}.pt")  # Load a pretrained YOLOv10 model
    model.export(format="engine", half=True)  # creates 'yolo11n.engine'



