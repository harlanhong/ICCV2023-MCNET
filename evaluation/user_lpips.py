import torch
# from IPython import embed
import os
from lpips_pytorch import LPIPS, lpips
def compute_lpips(path1,path2):
    use_gpu = True         # Whether to use GPU
    spatial = True         # Return a spatial map of perceptual distance.
    # Linearly calibrated models (LPIPS)
    # define as a criterion module (recommended)
    criterion = LPIPS(
        net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
        version='0.1'  # Currently, v0.1 is supported
    )
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
    # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
    
    
    ## Example usage with dummy tensors
    im1_path_list = []
    im2_path_list = []
    imgs = os.listdir(path1)
    for im in imgs:
        path1 = os.path.join(path1, im)
        im1_path_list.append(path1)
        path2 = os.path.join(path2, im)
        im2_path_list.append(path2)

    dist_ = []
    for i in range(len(im1_path_list)):
        dummy_im0 = lpips.im2tensor(lpips.load_image(im1_path_list[i]))
        dummy_im1 = lpips.im2tensor(lpips.load_image(im2_path_list[i]))
        if(use_gpu):
            dummy_im0 = dummy_im0.cuda()
            dummy_im1 = dummy_im1.cuda()
        dist = criterion(dummy_im0, dummy_im1)
        # dist = loss_fn.forward(dummy_im0, dummy_im1)
        dist_.append(dist.mean().item())
    print('Avarage Distances: %.3f' % (sum(dist_)/len(im0_path_list)))
    return sum(dist_)/len(im0_path_list)

if __name__ == '__main__':
    compute_lpips('/data/fhongac/workspace/src/ECCV2022/log/vox-adv-256baseline_invertmb_ocsmooth0.1/vox_same_id/generate','/data/fhongac/workspace/src/ECCV2022/log/vox-adv-256baseline_invertmb_ocsmooth0.1/vox_same_id/gt')