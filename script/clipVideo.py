import imageio
import numpy as np
from imageio import mimread,imsave
from skimage import io, img_as_float32
from skimage.color import gray2rgb
import cv2
path = 'nikita_2drivers_cmp.mp4'
reader = imageio.get_reader('{}'.format(path))
fps = reader.get_meta_data()['fps']
video = np.array(mimread('{}'.format(path),memtest=False))
video = np.array([gray2rgb(frame) for frame in video])
num, h,w,c=video.shape
ind_w = w // 5
driving = video[:,:,:ind_w]
source1 = video[:,:,ind_w:2*ind_w]
source2 = video[:,:,3*ind_w:4*ind_w]

cv2.imwrite('s1.jpg',source1[0])
cv2.imwrite('s2.jpg',source2[0])
# interval = num//9
# video=video[interval*2:,...]
imageio.mimsave('driving.mp4', driving, fps=fps)