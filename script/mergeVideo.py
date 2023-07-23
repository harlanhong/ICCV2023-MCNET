import imageio
import numpy as np
from imageio import mimread,imsave
from skimage import io, img_as_float32
from skimage.color import gray2rgb
import cv2
from PIL import Image
path = 'nikita_2drivers_cmp.mp4'
reader = imageio.get_reader('{}'.format(path))
fps = reader.get_meta_data()['fps']
video = np.array(mimread('{}'.format(path),memtest=False))
video = np.array([gray2rgb(frame) for frame in video])
num, h,w,c=video.shape
ind_w = w // 5
rst1 = video[:,:,4*ind_w:5*ind_w]

# rst1 = np.resize(rst1,(num,256,256,c))

# 创建一个新的数组，存放调整大小后的视频帧
resized_video_data = np.empty((rst1.shape[0], 256, 256, rst1.shape[3]))

# 遍历每一帧，将其调整为 256x256
for i in range(rst1.shape[0]):
    frame = rst1[i]
    resized_frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_LINEAR)
    resized_video_data[i] = resized_frame


path = 'd-s2.mp4'
video = np.array(mimread('{}'.format(path),memtest=False))
video = np.array([gray2rgb(frame) for frame in video])

rst = np.concatenate((video,resized_video_data),2)


imageio.mimsave('d-s2-compare.mp4', rst, fps=fps)
