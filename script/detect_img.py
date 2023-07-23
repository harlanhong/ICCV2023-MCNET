import os

def delete_folders_without_images(folder_path):
    # 获取当前文件夹下的所有子文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # 检查子文件夹中是否存在图片文件
        has_images = False
        for file_name in os.listdir(subfolder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                has_images = True
                break
        
        # 如果子文件夹中没有图片文件，则删除该子文件夹
        if not has_images:
            print(f"删除子文件夹：{subfolder}")
            os.rmdir(subfolder)  # 如果子文件夹非空，使用os.rmdir会引发异常，请注意备份重要数据

# 在当前文件夹下运行代码

delete_folders_without_images('/ssddata/fhongac/origDataset/HDTF/hdtf_video_frames')
