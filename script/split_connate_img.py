import cv2
import os
import pdb
img = "out-of-domain-mona/frame-143.png"

img = cv2.imread(img)

h,w,_ = img.shape

w_ = w // 3
pdb.set_trace()
cv2.imwrite('source-mona.jpg', img[:,:w_])
cv2.imwrite('driving.jpg', img[:,w_:2*w_])
cv2.imwrite('result.jpg', img[:,2*w_:])
