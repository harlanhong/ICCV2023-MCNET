import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torch
from arcface.arcface_arch import ResNetArcFace
import collections
import torch.nn.functional as F
import pdb
import torchvision.transforms as T

def gray_resize_for_identity(out, size=128):
    out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
    out_gray = out_gray.unsqueeze(1)
    out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
    return out_gray
if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str)
    opt = parser.parse_args()

    source_fold = os.path.join(opt.fold,'source')
    gt_fold = os.path.join(opt.fold,'gt')
    generate_fold = os.path.join(opt.fold,'generate')

    # evaluate_CSIM_PRMSE_AUCON(source_fold,gt_fold,generate_fold):

    imgs = os.listdir(generate_fold)
    PRMSE = 0
    AUCON = 0
    counter = 1e-9
    CSIM = 0
    csim_counter = 1e-9
    ##########################################CSIM##############################################################
    from facenet_pytorch import MTCNN, InceptionResnetV1

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256, margin=0)
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
   
    network_identity = ResNetArcFace(block = 'IRBlock', layers = [2, 2, 2, 2], use_se= False)
    checkpoint = torch.load('arcface/arcface_resnet18.pth', map_location='cuda')
    ckp_net = collections.OrderedDict((k.replace('module.',''),v) for k,v in checkpoint.items())
    network_identity.load_state_dict(ckp_net)
    network_identity.eval()
    network_identity.cuda()
    num = len(imgs)
    valid_num = num//3*2
    #####################################################################################################################
    with torch.no_grad():
        for im in tqdm(imgs):
            try:
                source = Image.open(os.path.join(source_fold,im))
                generate = Image.open(os.path.join(generate_fold,im))
                # Get cropped and prewhitened image tensor
                img_cropped = mtcnn(source,save_path='src.jpg')
                # source_emb = resnet(img_cropped[0].unsqueeze(0))

                img_cropped = mtcnn(generate,save_path='gen.jpg')
                source = Image.open('src.jpg')
                source = T.ToTensor()(source).cuda().unsqueeze(0)
                img_cropped = mtcnn(generate,save_path='dst.jpg')
                generate = Image.open('dst.jpg')
                generate = T.ToTensor()(generate).cuda().unsqueeze(0)
                out_gray = gray_resize_for_identity(source)
                gt_gray = gray_resize_for_identity(generate)
                identity_gt = network_identity(gt_gray)
                identity_out = network_identity(out_gray)
                # generate_emb = resnet(img_cropped[0].unsqueeze(0))
                # sim = F.cosine_similarity(source_emb, generate_emb)
                sim = F.cosine_similarity(identity_gt, identity_out)
                CSIM+=sim.item()
                csim_counter+=1
            except Exception as e:
                print(e) 
    print(' CSIM: {}'.format(CSIM/csim_counter))

# CUDA_VISIBLE_DEVICES=0 python evaluate_CSIM.py --fold /ssddata/fhongac/src/scalabel_warp/log/vox-256-3down-ZoomAug-image-stn-one-branch-inpainterv3fiducialConditionv3/vox_cross_id
# CUDA_VISIBLE_DEVICES=1 python evaluate_CSIM_AUCON.py --fold /ssddata/fhongac/src/scalabel_warp/log/vox-256-3down-ZoomAug-image-stn-one-branch-inpainterv3fiducialConditionv3/celebv

# CUDA_VISIBLE_DEVICES=0 python evaluate_CSIM.py --fold /data/fhongac/workspace/src/DaGAN_Origin/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id