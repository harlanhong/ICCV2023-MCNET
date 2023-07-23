import cv2
import numpy as np
import glob
import os
video1s = sorted(glob.glob('/data/fhongac/workspace/src/ECCV2022/hdtf-rec/*'))
video2s = sorted(glob.glob('/ssddata/fhongac/origDataset/HDTF/frames_split/test/*'))
psnr_value = []
def load_frames(video_path):
    if os.path.isfile(video_path):
        video = cv2.VideoCapture(video_path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # Convert the frame to RGB (OpenCV uses BGR) and normalize to [0, 1]
            frames.append(frame)
        video.release()
    else:
        imgs = glob.glob(video_path+'/*')
        frames = [cv2.resize(cv2.imread(im),(512,512)) for im in imgs]
    return frames
for idx, (v1,v2) in enumerate(zip(video1s,video2s)):
    print(v1,v2)
    # 打开视频文件
    video1 = load_frames(v1)
    video2 = load_frames(v2)
    video1 = video1[:500]
    video2 = video2[:500]

    # 初始化PSNR累计值
    psnr_total = 0

    # 循环读取视频帧
    for frame1, frame2 in zip(video1,video2):

        # 将帧转换为灰度图像
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算帧的MSE（Mean Squared Error）
        mse = np.mean((frame1_gray - frame2_gray) ** 2)

        # 计算帧的最大像素值
        max_value = np.max(frame1_gray)

        # 计算帧的PSNR
        psnr = 20 * np.log10(max_value / np.sqrt(mse))

        # 累计PSNR
        psnr_total += psnr

    # 计算平均PSNR
    average_psnr = psnr_total / len(video1)
    psnr_value.append(average_psnr)
    # 释放视频对象

    # 打印平均PSNR
print("Average PSNR:", sum(psnr_value)/len(psnr_value))
