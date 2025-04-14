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

    model = YOLOv10(f"{model_name}.pt")  # Load a pretrained YOLOv10 model
    if model_name == 'yolov10m':
        print(f'{model = }')



