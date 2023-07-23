from imageio import mimread, imread, mimsave
import numpy as np
import warnings
import os

def frames2array(file, is_video, image_shape=None, column=0):
    if is_video:
        if os.path.isdir(file):
            images = [imread(os.path.join(file, name))  for name in sorted(os.listdir(file))]
            video = np.array(images)
        elif file.endswith('.png') or file.endswith('.jpg'):
            ### Frames is stacked (e.g taichi ground truth)
            image = imread(file)
            if image.shape[2] == 4:
                image = image[..., :3]

            video = np.moveaxis(image, 1, 0)
#            print (image_shape)
            video = video.reshape((-1, ) + image_shape + (3, ))
            video = np.moveaxis(video, 1, 2)
        elif file.endswith('.gif') or file.endswith('.mp4'):
            video = np.array(mimread(file))
        else:
            warnings.warn("Unknown file extensions  %s" % file, Warning)
            return []
    else:
        ## Image is given, interpret it as video with one frame
        image = imread(file)
        if image.shape[2] == 4:
            image = image[..., :3]
        video = image[np.newaxis]

    if image_shape is None:
        return video
    else:
        ### Several images stacked together select one based on column number
        return video[:, :, (image_shape[1] * column):(image_shape[1] * (column + 1))]


def draw_video_with_kp(video, kp_array):
    from skimage.draw import circle
    video_array = np.copy(video)
    for i in range(len(video_array)):
        for kp_ind, kp in enumerate(kp_array[i][1:-4]):
            rr, cc = circle(kp[1], kp[0], 2, shape=video_array.shape[1:3])
            video_array[i][rr, cc] = (255, 0, 0)
    return video_array


if __name__ == "__main__":
    import pandas as pd
    file_names = ['00000137.jpg', '00004204.jpg']
    frames = [0, 0]
    videos = []
    for i, file_name in enumerate(file_names):
        video  = frames2array('../mtm/data/taichi/test/' + file_name, is_video=True, image_shape=(64, 64))
        df1 = pd.read_pickle('taichi_pose_gt_fine.pkl')

        kp_array = np.array(df1[df1['file_name'] == file_name]['value'])
        videos.append(draw_video_with_kp(video[frames[i]:(frames[i] + 1)], kp_array[frames[i]:(frames[i] + 1)]))
    mimsave('1.gif', np.concatenate(videos, axis=1))
   
 
