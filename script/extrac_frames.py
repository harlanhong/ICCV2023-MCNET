import cv2
import os

# 创建一个文件夹来保存帧
frames_dir = "out-of-domain-newton"
os.makedirs(frames_dir, exist_ok=True)

# 打开视频文件
video = cv2.VideoCapture('out-of-domain-newton.mp4')

frame_num = 0
while True:
    # 读取一帧
    ret, frame = video.read()

    # 如果帧没有正确读取，那么我们已经到了视频的结尾
    if not ret:
        break

    # 写出帧到文件
    frame_filename = os.path.join(frames_dir, f"frame-{frame_num}.png")
    cv2.imwrite(frame_filename, frame)
    
    frame_num += 1

# 关闭视频文件
video.release()

print(f"Saved {frame_num} frames to directory {frames_dir}.")
