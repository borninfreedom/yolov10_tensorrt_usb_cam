from ultralytics import YOLOv10
import os
# "yolov10n",
# "yolov10s",
# "yolov10m",
# "yolov10b",
# "yolov10l",
# "yolov10x",
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

# model = YOLOv10("./yolov10m.pt")  # Load a pretrained YOLOv10 model
# # model = YOLOv10("runs/V10train/exp3_fp16/weights/best.engine")
# model.export(format="engine",half=True)  # creates 'yolo11n.engine'
#model.predict(source="data/test/images/cju0qkwl35piu0993l0dewei2.jpg", imgsz=640, conf=0.05,save=True)  #单张图片测试
# print(f'{model = }')
# print(f'{model.__dict__ = }')
# model.predict(source="test_imgs", imgsz=640, conf=0.05,save=True,save_dir="predictions")   #整个文件夹测试

