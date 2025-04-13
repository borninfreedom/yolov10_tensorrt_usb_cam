from ultralytics import YOLOv10
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r"D:\xianyu_project\250227Yolov10weibingdetect\yolov10-main\ultralytics\cfg\models\v10\yolov10n_DynamicConv.yaml"
#数据集配置文件
data_yaml_path = r'D:\xianyu_project\250227Yolov10weibingdetect\yolov10-main\data\data.yaml'
#预训练模型
pre_model_name = 'yolov10n.pt'
if __name__ == '__main__':
    # #加载预训练模型
    # model = YOLOv10(model_yaml_path).load(pre_model_name)
    # 不加载预训练模型
    model = YOLOv10(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=256,
                          epochs=10,
                          batch=4,
                          workers=0,
                          optimizer='SGD',  # using SGD
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/V10train',
                          name='exp',
                          )
