import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import gradio as gr
import cv2
# from gradio_webrtc import WebRTC
from fastrtc import WebRTC
from twilio.rest import Client
import os
import numpy as np
import time
import PIL.Image as Image
from collections import deque
from dotenv import load_dotenv
load_dotenv()  # 添加在代码开头

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

if account_sid and auth_token:
    print('USE twilio')
    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }
else:
    rtc_configuration = None

# FPS 计算类
class FPS:
    def __init__(self, avg_frames=30):
        self.timestamps = deque(maxlen=avg_frames)  # 用于存储时间戳
    
    def update(self):
        self.timestamps.append(time.time())  # 添加当前时间戳
        
    def get(self):
        if len(self.timestamps) < 2:
            return 0  # 如果时间戳不足，返回 0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])  # 计算平均 FPS

fps_counter = FPS()  # 创建 FPS 计数器实例


from ultralytics import YOLOv10

# model = YOLOv10("runs/V10train/exp3/weights/best.pt")
model = YOLOv10("runs/V10train/exp3_fp16/weights/best.engine")
print(f'{model = }')


def yolov10_inference_video(video_path):
    fps_counter = FPS()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 更新 FPS 计数器
        fps_counter.update()

        # 模型推理
        results = model.predict(source=frame, imgsz=640, conf=0.05)
        r = results[0]
        im_bgr = r.plot()  # Ultralytics 默认输出 BGR numpy

        # 计算 FPS
        fps = fps_counter.get()
        cv2.putText(im_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将帧返回给 Gradio
        yield im_bgr  # 使用 yield 逐帧返回推理结果

    cap.release()


def yolov10_inference(image):
    fps_counter.update()
    if 1:
        results = model.predict(source=image, imgsz=640, conf=0.05,save=False)   #整个文件夹测试
    
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])

        # 计算 FPS
        fps = fps_counter.get()
        print(f"FPS: {fps:.1f}")  # 在控制台打印 FPS

        return im

# def yolov10_inference_webrtc(frame: np.ndarray) -> np.ndarray:
#     # frame: BGR numpy
#     results = model.predict(source=frame, imgsz=640, conf=0.05)
#     r = results[0]
#     im_bgr = r.plot()      # Ultralytics 默认输出 BGR numpy
#     return im_bgr          # Gradio 会自动把 BGR 转成 RGB 展示



def app():
    with gr.Blocks():
        with gr.Tabs():  # 添加选项卡
            with gr.TabItem("WebRTC Stream"):  # WebRTC 输入选项卡
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(sources=["webcam"],type="pil", streaming=True)
                    image.stream(
                        fn=yolov10_inference, inputs=[image], outputs=[image],stream_every=0.1
                    )
                    #     image = WebRTC(label="Stream", rtc_configuration=rtc_configuration, height=640, width=640)
                    # image.stream(
                    #     fn=yolov10_inference, inputs=[image], outputs=[image], time_limit=500
                    # )
            with gr.TabItem("Image Upload"):  # 图片输入选项卡
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        output_image = gr.Image(label="Output Image",type="pil")
                    input_image.change(
                        fn=yolov10_inference, inputs=[input_image], outputs=[output_image]
                    )
            with gr.TabItem("Video Upload"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_output = gr.Video(label="Output Video")
                    video_input.change(
                        fn=lambda video_path: list(yolov10_inference_video(video_path)),inputs=[video_input], outputs=[video_output]
                    )



gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(server_name = '0.0.0.0')