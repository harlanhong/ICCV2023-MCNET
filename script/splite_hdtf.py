import os
import shutil
import random
source_dir = "/ssddata/fhongac/origDataset/HDTF/hdtf_video_frames"  # 源目录
test_dir = "/ssddata/fhongac/origDataset/HDTF/frames_split/test"  # 目标目录
train_dir = "/ssddata/fhongac/origDataset/HDTF/frames_split/train"  # 目标目录

videos = os.listdir(source_dir)

selected_folders = random.sample(videos, 20) 

for folder in videos:
    src_path = os.path.join(source_dir, folder)  # 源文件夹路径
    if folder in selected_folders:
        shutil.copytree(src_path, os.path.join(test_dir,folder))  # 复制文件夹
    else:
        shutil.copytree(src_path, os.path.join(train_dir,folder))  # 复制文件夹
    print(f"Folder {src_path}.")
