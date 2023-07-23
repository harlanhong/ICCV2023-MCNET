import cv2
from glob import glob
from tqdm import tqdm
import os
def read_video(filename):
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

if __name__ == '__main__':
    dataset = '/data/fhongac/origDataset/fashion'
    videos = sorted(glob(dataset+'/*/*.mp4'))
    print(videos)
    for vid in tqdm(videos):
        frame_path = vid.replace('fashion','fashion_frames')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        frames = read_video(vid)
        for idxf, fr in tqdm(enumerate(frames)):
            cv2.imwrite(frame_path+'/%s' % str(idxf).zfill(6)+'.jpg',fr)

