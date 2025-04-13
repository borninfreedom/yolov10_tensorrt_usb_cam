import cv2
import os

def extract_frames(video_path, output_dir, interval=50):
    """
    视频抽帧核心函数
    :param video_path: 输入视频路径
    :param output_dir: 输出目录
    :param interval: 抽帧间隔（默认50帧）
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件，请检查路径有效性")

    # 获取视频元数据
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频总帧数：{total_frames} | 帧率：{fps:.1f} FPS")

    # 初始化计数器
    current_frame = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 抽帧逻辑
        if current_frame % interval == 0:
            # 生成标准化文件名（补零对齐）
            filename = f"frame_{saved_count:04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # 保存图像（设置JPG质量参数）
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            # 打印进度
            if saved_count % 10 == 0:
                progress = (current_frame / total_frames) * 100
                print(f"处理进度：{progress:.1f}% | 已保存 {saved_count} 帧")

        current_frame += 1

    # 释放资源
    cap.release()
    print(f"抽帧完成！共保存 {saved_count} 张图片")

if __name__ == "__main__":
    # 参数配置
    video_file = "test_video/test_video.mp4"        # 替换为实际视频路径
    output_folder = "test_img_from_video"  # 输出目录名称
    
    # 执行抽帧（间隔50帧）
    extract_frames(video_file, output_folder, interval=50)