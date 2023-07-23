import torch
import lpips
import cv2
import glob
from tqdm import tqdm
# Initialize the LPIPS model (using the default network)
loss_fn = lpips.LPIPS(net='alex').cuda()
import os
import pdb
from pytorch_ssim import ssim
# Specify the paths of your videos
video1_paths = sorted(glob.glob('/data/fhongac/workspace/src/ECCV2022/hdtf-rec/*'))
video2_paths = sorted(glob.glob('/ssddata/fhongac/origDataset/HDTF/frames_split/test/*'))
# Function to load frames from video
def load_frames(video_path):
    if os.path.isfile(video_path):
        video = cv2.VideoCapture(video_path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # Convert the frame to RGB (OpenCV uses BGR) and normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        video.release()
    else:
        imgs = glob.glob(video_path+'/*')
        frames = [cv2.resize(cv2.cvtColor(cv2.imread(im),cv2.COLOR_BGR2RGB),(512,512))/ 255.0 for im in imgs]
    return frames

total_dis = []
for i,(v1,v2) in tqdm(enumerate(zip(video1_paths,video2_paths))):
    # Load the frames from your videos
    print(v1,v2)
    video1_frames = load_frames(v1)
    video2_frames = load_frames(v2)
    video1_frames= video1_frames[:500]
    video2_frames= video2_frames[:500]
    # Make sure the videos have the same number of frames
    assert len(video1_frames) == len(video2_frames)
   
    # Calculate LPIPS for each pair of frames and average
    distances = []
    for frame1, frame2 in zip(video1_frames, video2_frames):
        # Convert your frames to PyTorch tensors, add an extra batch dimension, and rearrange dimensions to [B, C, H, W]
        frame1_tensor = torch.from_numpy(frame1).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        frame2_tensor = torch.from_numpy(frame2).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        # Compute LPIPS
        with torch.no_grad():
            distance = ssim(frame1_tensor.float(), frame2_tensor.float())
        distances.append(distance.item())

    # Average LPIPS over all frames
    average_lpips = sum(distances) / len(distances)
    total_dis.append(average_lpips)
print('Average SSIM:', sum(total_dis)/len(total_dis))
