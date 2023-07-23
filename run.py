import shutil
import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset
# from frames_dataset_zoom import FramesDataset as FramesDataset_zoom
import pdb
import modules.generator as G
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
import modules.keypoint_detector as KPD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.tensorboard import SummaryWriter 
import train
import train_single_optim
from train_avd import train_avd
from reconstruction import reconstruction
from animate import animate
import random
from modules.avd_network import AVDNetwork
import numpy as np
from collections import OrderedDict
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
seed_torch()
def mb_load(ckp, model,flag):
    model.mbUnit.mb.data = ckp['generator']['module.mbUnit.mb']
    if flag == 'finetune':
        model.mbUnit.mb.requires_grad = True
    if flag == 'fixed':
        model.mbUnit.mb.requires_grad = False
    return model


if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate","train_avd"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--use_depth",action='store_true',help='depth mode')
    parser.add_argument("--rgbd",action='store_true',help='depth mode')

    # alter model
    parser.add_argument("--generator",required=True,help='depth mode')
    parser.add_argument("--kp_detector",default='KPDetector',type=str,help='depth mode')
    parser.add_argument("--GFM",default='GeneratorFullModel')
    
    parser.add_argument("--batchsize",type=int, default=-1,help='depth mode')
    parser.add_argument("--kp_num",type=int, default=-1,help='depth mode')
    parser.add_argument("--kp_distance",type=int, default=0,help='depth mode')
    parser.add_argument("--depth_constraint",type=int, default=0,help='depth mode')
    parser.add_argument("--app_mask",type=float, default=0, help='depth mode')
    parser.add_argument("--mb_consistent",type=float, default=0, help='depth mode')
    parser.add_argument("--feat_consistent",type=float, default=0, help='depth mode')
    parser.add_argument("--bi_feat_consistent",type=float, default=0, help='depth mode')
    parser.add_argument("--l1",type=float, default=0, help='depth mode')
    parser.add_argument("--image_ffl",type=float, default=0, help='depth mode')
    parser.add_argument("--feat_ffl",type=float, default=0, help='depth mode')
    parser.add_argument("--warp_loss",type=float, default=0, help='depth mode')
    parser.add_argument("--mb_ffl",type=float, default=0, help='depth mode')

    parser.add_argument("--occlusion_smooth",type=float, default=0, help='depth mode')
    parser.add_argument("--hierachy_constraint",type=float, default=0, help='depth mode')
    parser.add_argument("--equivariance_keypoint",type=float, default=0, help='depth mode')
    parser.add_argument("--kl_mb",type=float, default=0, help='depth mode')
    parser.add_argument("--kl_feat",type=float, default=0, help='depth mode')
    parser.add_argument("--vq_commit",type=float, default=0, help='depth mode')
    parser.add_argument("--attn_regular",type=float, default=0, help='depth mode')
    parser.add_argument("--qv_style_similar",type=float, default=0, help='depth mode')
    parser.add_argument("--feat_gap",type=float, default=0, help='depth mode')
    parser.add_argument("--sample_feat_consistent",type=float, default=0, help='depth mode')
    parser.add_argument("--reconstruction",type=float, default=0, help='depth mode')
    parser.add_argument("--kp_prior",type=float, default=0, help='depth mode')
    parser.add_argument("--identity",type=float, default=0, help='depth mode')
    parser.add_argument("--FDIT",type=float, default=0, help='depth mode')
    parser.add_argument("--mbunit",type=str, default='', help='depth mode')
    parser.add_argument("--clip_grad",type=float, default=10, help='depth mode')


    parser.add_argument("--mb_pretrained",action='store_true')
    parser.add_argument("--mb_finetune",action='store_true')
    parser.add_argument("--single_optim",action='store_true')
    parser.add_argument("--styleGAN",action='store_true')
    parser.add_argument("--sft_cross",action='store_true')
    parser.add_argument("--multi_scale",action='store_true')
    parser.add_argument("--mb_channel",type=int, default=512, help='depth mode')
    parser.add_argument("--mb_spatial",type=int, default=32, help='depth mode')



    parser.add_argument("--generator_gan",type=float, default=1, help='depth mode')
    parser.add_argument("--memsize",type=int, default=8, help='depth mode')
    parser.add_argument("--linear_grow_mb_weight",action='store_true')

    parser.add_argument("--depth_path",type=str,help='the path of depth weight')
    parser.add_argument("--no_jacobain",action='store_true')
    parser.add_argument("--no_disc_use_kp",action='store_true')

    parser.add_argument("--name",type=str)

    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    if opt.checkpoint is not None:
        if opt.mode != "train_avd":
            log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        else:
            log_dir = os.path.join(opt.log_dir, opt.name)
    else:
        log_dir = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        

    print("Training...")

    dist.init_process_group(backend='nccl', init_method='env://') 
    torch.cuda.set_device(opt.local_rank)
    device=torch.device("cuda",opt.local_rank)
    config['train_params']['loss_weights']['equivariance_keypoint'] = opt.equivariance_keypoint
    config['train_params']['loss_weights']['depth_constraint'] = opt.depth_constraint
    config['train_params']['loss_weights']['kp_distance'] = opt.kp_distance
    config['train_params']['loss_weights']['app_mask'] = opt.app_mask
    config['train_params']['loss_weights']['mb_consistent'] = opt.mb_consistent
    config['train_params']['loss_weights']['occlusion_smooth'] = opt.occlusion_smooth
    config['train_params']['loss_weights']['hierachy_constraint'] = opt.hierachy_constraint
    config['train_params']['loss_weights']['kl_mb'] = opt.kl_mb
    config['train_params']['loss_weights']['kl_feat'] = opt.kl_feat
    config['train_params']['loss_weights']['feat_consistent'] = opt.feat_consistent
    config['train_params']['loss_weights']['vq_commit'] = opt.vq_commit
    config['train_params']['loss_weights']['attn_regular'] = opt.attn_regular
    config['train_params']['loss_weights']['bi_feat_consistent'] = opt.bi_feat_consistent
    config['train_params']['loss_weights']['qv_style_similar'] = opt.qv_style_similar
    config['train_params']['loss_weights']['feat_gap'] = opt.feat_gap
    config['train_params']['loss_weights']['generator_gan'] = opt.generator_gan
    config['train_params']['loss_weights']['sample_feat_consistent'] = opt.sample_feat_consistent
    config['train_params']['loss_weights']['l1'] = opt.l1
    config['train_params']['loss_weights']['warp_loss'] = opt.warp_loss
    config['train_params']['loss_weights']['reconstruction'] = opt.reconstruction
    config['train_params']['loss_weights']['kp_prior'] = opt.kp_prior
    config['train_params']['loss_weights']['image_ffl'] = opt.image_ffl
    config['train_params']['loss_weights']['feat_ffl'] = opt.feat_ffl
    config['train_params']['loss_weights']['mb_ffl'] = opt.mb_ffl
    config['train_params']['loss_weights']['identity'] = opt.identity
    config['train_params']['loss_weights']['FDIT'] = opt.FDIT




    config['model_params']['generator_params']['memsize'] = opt.memsize
    if opt.no_jacobain:
        config['model_params']['common_params']['estimate_jacobian'] = False
        config['train_params']['loss_weights']['equivariance_jacobian'] = 0
    
    if opt.batchsize != -1:
        config['train_params']['batch_size'] = opt.batchsize
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    if opt.no_disc_use_kp:
        config['model_params']['discriminator_params']['use_kp'] = False

    
    # create generator
    generator = getattr(G, opt.generator)(**config['model_params']['generator_params'],**config['model_params']['common_params'],**{'mbunit':opt.mbunit,'mb_spatial':opt.mb_spatial,'mb_channel':opt.mb_channel})
    
    if opt.mb_pretrained:
        checkpoint_path = 'log/Unet_Reconstruction_no_adv/00000099-checkpoint.pth.tar'
        ckp = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        generator = mb_load(ckp,generator,flag='fixed')
    if opt.mb_finetune:
        checkpoint_path = 'log/Unet_Reconstruction_no_adv/00000099-checkpoint.pth.tar'
        ckp = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        generator = mb_load(ckp,generator,flag='finetune')
   
    generator.to(device)
    if opt.verbose:
        print(generator)
    generator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
    
    # create discriminator
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    discriminator.to(device)
    if opt.verbose:
        print(discriminator)
    discriminator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    
        
    kp_detector = getattr(KPD, opt.kp_detector)(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
    kp_detector.to(device)
    if opt.verbose:
        print(kp_detector) >> 'd.txt'
    kp_detector= torch.nn.SyncBatchNorm.convert_sync_batchnorm(kp_detector)
    #Create the record of network arch
    # with open(os.path.join(log_dir,'network.txt'),"w") as file:
    #     file.write(kp_detector)
    #     file.write(discriminator)
    #     file.write(generator)
    if opt.mode == "train_avd":
        avd_network = AVDNetwork(num_kp=config['model_params']['common_params']['num_kp'],
                             **config['model_params']['avd_network_params'])
        avd_network.to(device)

    if torch.cuda.device_count() == 1:
        kp_detector = DDP(kp_detector,device_ids=[opt.local_rank],broadcast_buffers=False)
        discriminator = DDP(discriminator,device_ids=[opt.local_rank],broadcast_buffers=False)
        generator = DDP(generator,device_ids=[opt.local_rank],broadcast_buffers=False)
        if opt.mode == "train_avd":
            avd_network = DDP(avd_network,device_ids=[opt.local_rank],broadcast_buffers=False)
            
    else:
        kp_detector = DDP(kp_detector,device_ids=[opt.local_rank])
        discriminator = DDP(discriminator,device_ids=[opt.local_rank])
        generator = DDP(generator,device_ids=[opt.local_rank])
        if opt.mode == "train_avd":
            avd_network = DDP(avd_network,device_ids=[opt.local_rank])
    dataset = FramesDataset(is_train=(opt.mode == 'train' or opt.mode == 'train_avd'), **config['dataset_params'])
    dataset.__getitem__(0)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if not os.path.exists(os.path.join(log_dir,'log')):
        os.makedirs(os.path.join(log_dir,'log'))
    writer = SummaryWriter(os.path.join(log_dir,'log'))
    # print(config)
    if opt.mode == 'train':
        if opt.single_optim:
            train_single_optim.train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.local_rank,device,opt,writer)
        else:
            train.train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.local_rank,device,opt,writer)
    elif opt.mode == 'train_avd':
        train_avd(config,  generator, discriminator, kp_detector, avd_network, opt.checkpoint, log_dir, dataset, opt.local_rank,device,opt,writer)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset,opt)
