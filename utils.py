import os
import random
import numpy as np
import csv
import cv2
import pdb
from collections import defaultdict
import sys
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import math
import imageio
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread,imsave
import pandas as pd
from evaluation.extract import cmp_akd, cmp_aed,cmp_aed_corss,extract_face_id,extract_arcface_id,cmp_CSIM_corss
from glob import glob
import random

def count_test_video(path):
    vis = {video[:video.find('#',8)] for video in
                                os.listdir(path)}
    print(vis)
    print(len(vis))

def create_same_id_test_set(path):
    vis = os.listdir(path)
    videos = np.random.choice(vis, replace=False, size=100)
    f = open('./data/vox_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = os.path.join(path,v[0],pair[0])
        dst = os.path.join(path,v[0],pair[1])
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def modify_same_id_voxceleb():
    f = open('./data/vox_evaluation_v2.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","best_frame"])
    pairs = pd.read_csv('data/vox_evaluation.csv')
    source = pairs['source'].tolist()
    driving = pairs['driving'].tolist()
    best_frame = pairs['driving'].tolist()

    source = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    anchor = np.array(best_frame).reshape(-1,1)

    content = np.concatenate((source,driving,anchor),1)
    csv_writer.writerows(content)
def create_cross_id_test_set(path):
    vis = os.listdir(path)
    ids2video = defaultdict(list)
    num = len('id10283')
    for vi in vis:
        ids2video[vi[:num]].append(vi)
    ids = list(ids2video.keys())
    videos = np.random.choice(vis, replace=False, size=100)
    f = open('./data/vox_cross_id_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        id = np.random.choice(ids, replace=False, size=1)
        vis = np.random.choice(ids2video[id[0]], replace=False, size=1)
        imgs = os.listdir(os.path.join(path,vis[0]))
        img = np.random.choice(imgs, replace=False, size=1)
        src = os.path.join(path,vis[0],img[0])

        other_id = list(set(ids).difference(set(id)))
        id = np.random.choice(other_id, replace=False, size=1)
        vis = np.random.choice(ids2video[id[0]], replace=False, size=1)
        imgs = os.listdir(os.path.join(path,vis[0]))
        img = np.random.choice(imgs, replace=False, size=1)
        dst = os.path.join(path,vis[0],img[0])

        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def concate_compared_results(resust_path,cp_path):
    imgs = os.listdir(resust_path)
    for im in tqdm(imgs):
        ours = cv2.imread(os.path.join(resust_path,im))
        fomm = cv2.imread(os.path.join(cp_path,im))
        rst = np.concatenate((ours,fomm),1).astype(np.uint8)
        cv2.imwrite(os.path.join('compare',im),rst)
def render(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    cvimg = cv2.resize(cv2.imread(path),(256,256))
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    depth_source = outputs[("disp", 0)][0]
    depth_source = depth_source.permute(1,2,0).detach().cpu().numpy()
    heatmap = depth_source/np.max(depth_source)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img1 = heatmap*0.6+cvimg
    cv2.imwrite('{}.jpg'.format(path),superimposed_img1)

def depth_gray(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    depth_source = outputs[("disp", 0)][0]
    depth_source = depth_source.permute(1,2,0).detach().cpu().numpy()*depth_source.permute(1,2,0).detach().cpu().numpy()
    heatmap = 1-depth_source/np.max(depth_source)
    heatmap = np.uint8(255 * heatmap)
    cv2.imwrite('heatmap.jpg',heatmap)

def depth_rgb(path):
    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    disp = outputs[("disp", 0)]
    # Saving colormapped depth image
    disp_resized = torch.nn.functional.interpolate(disp, (256, 256), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(colormapped_im)
    # plt.colorbar(mapper)
    plt.savefig(path+'.pdf')
    # plt.savefig(path+'.png')
    plt.clf()


def process_celeV(path):
    train_path = os.path.join(path,'train')
    test_path = os.path.join(path,'test')
    ids = os.listdir(path)
    f = open('./data/celeV_cross_id_evaluation.csv','w',encoding='utf-8')

    # sample 2000 image sets from each identity
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)
    # if not os.path.exists(test_path):
    #     os.makedirs(test_path)
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        src_id = np.random.choice(ids, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,src_id[0],'Image'))
        src_imgs = np.random.choice(imgs, replace=False, size=1)
        src = os.path.join(path,src_id[0],'Image',src_imgs[0])

        res_ids = list(set(ids).difference(set(src_id)))

        dst_id = np.random.choice(res_ids, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,dst_id[0],'Image'))
        dst_imgs = np.random.choice(imgs, replace=False, size=1)
        dst = os.path.join(path,dst_id[0],'Image',dst_imgs[0])
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def compare():
    x2face = '/data/fhongac/workspace/gitrepo/X2Face/UnwrapMosaic/FID/celebv'
    fomm = '/data/fhongac/workspace/gitrepo/first-order-model/FID/celebv'
    osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/FID/celebv'
    dagan = '/data/fhongac/workspace/src/parallel-fom-rgbd/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/celebv/concate'
    
    imgs = os.listdir(x2face)
    for i in tqdm(range(len(imgs))):
        im = imgs[i]
        img_x2face = os.path.join(x2face,im)
        img_x2face = cv2.imread(img_x2face)

        img_fomm = os.path.join(fomm,im)
        img_fomm = cv2.imread(img_fomm)

        img_osfv = os.path.join(osfv,im)
        img_osfv = cv2.imread(img_osfv)

        img_dagan = os.path.join(dagan,im)
        img_dagan = cv2.imread(img_dagan)


        img = np.vstack((img_x2face, img_fomm,img_osfv,img_dagan))
        cv2.imwrite('FID/multiMethod/{}.jpg'.format(i),img)
def aus(path):
    import cv2
    frame = cv2.imread(path)
    from feat import Detector
    detector = Detector()  
    # image_prediction = detector.detect_image(path)
    out1 = detector.detect_image('FID/source/0.jpg')
    out1.plot_aus(12, muscles={'all': "heatmap"}, gaze = None)
    plt.savefig('a.jpg')
    out2 = detector.detect_image('FID/source/1.jpg')
    p1 = out1.facepose().values
    p2 = out2.facepose().values
    
    # landmarks = detector.detect_landmarks(frame, face)  
    # score = detector.detect_aus(frame,landmarks[0])
def evaluate_CSIM_PRMSE_AUCON(source_fold,gt_fold,generate_fold):
    from feat import Detector
    # from feat.utils import read_pictures
    detector = Detector()
#     detector = Detector(
#     face_model="retinaface",
#     landmark_model="mobilefacenet",
#     au_model='svm',
#     emotion_model="resmasknet",
#     facepose_model="img2pose",
# )
    # x2face = '/data/fhongac/workspace/gitrepo/X2Face/UnwrapMosaic/FID'
    # fomm = '/data/fhongac/workspace/gitrepo/first-order-model/FID'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/FID'
    # dagan = '/data/fhongac/workspace/src/parallel-fom-rgbd/FID'
    # test = dagan
    # path = sys.argv[1]
    imgs = os.listdir(generate_fold)
    PRMSE = 0
    AUCON = 0
    counter = 1e-9
    CSIM = 0
    csim_counter = 1e-9
    ##########################################CSIM##############################################################
    # from facenet_pytorch import MTCNN, InceptionResnetV1

    # # If required, create a face detection pipeline using MTCNN:
    # mtcnn = MTCNN(image_size=256, margin=0).cuda()
    # # Create an inception resnet (in eval mode):
    # resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    #####################################################################################################################
    for im in tqdm(imgs):
        gt = os.path.join(gt_fold,im)
        gen = os.path.join(generate_fold,im)
        try:
            out_gt = detector.detect_image(gt)
            out_generat = detector.detect_image(gen)
            gt_aus = out_gt.aus.values
            generate_aus = out_generat.aus.values
            gt_pose = out_gt.facepose.values
            generate_pose = out_generat.facepose.values
            row,num = generate_aus.shape
            prmse=np.sqrt(np.power(gt_pose-generate_pose,2).sum()/3)
            if math.isnan(prmse):
                print(im)
                raise RuntimeError('NaN')
            PRMSE+=prmse
            generate_aus = generate_aus>0.5
            gt_aus = gt_aus>0.5
            rst = ~ (generate_aus^gt_aus)
            correct = rst.sum()
            AUCON+=(correct/num)
            counter+=1
        except Exception as e:
            print(e)
        # try:
        #     source = Image.open(os.path.join(source_fold,im))
        #     generate = Image.open(os.path.join(generate_fold,im))

        #     # Get cropped and prewhitened image tensor
        #     img_cropped = mtcnn(source,save_path='src.jpg')
        #     # img_cropped = T.ToTensor()(source).cuda()
        #     # Calculate embedding (unsqueeze to add batch dimension)
        #     source_emb = resnet(img_cropped.unsqueeze(0))

        #     # Get cropped and prewhitened image tensor
        #     img_cropped = mtcnn(generate,save_path='dst.jpg')
        #     # img_cropped = T.ToTensor()(generate).cuda()
        #     # Calculate embedding (unsqueeze to add batch dimension)
        #     generate_emb = resnet(img_cropped.unsqueeze(0))
        #     CSIM+=torch.cosine_similarity(source_emb,generate_emb).item()
        #     csim_counter+=1
        # except Exception as e:
            # print(e)
    print(' PRMSE: {}, AUCON : {}, CSIM: {}'.format(PRMSE/counter, AUCON/counter,CSIM/csim_counter))

def mergeimgs(paths, save_name):
    pth = paths[0]
    imgps = os.listdir(pth)
    if not os.path.exists('Compare/{}'.format(save_name)):
        os.makedirs('Compare/{}'.format(save_name))
    for i in tqdm(range(len(imgps))):
        imgp = imgps[i]
        cats = []
        for idx, pth in enumerate(paths):
            img = os.path.join(pth,imgp)
            img = cv2.imread(img)
            # if idx!=0:
            #     img = cv2.resize(img,(256,256))
            cats.append(img)
        img = np.hstack(cats)
        cv2.imwrite('Compare/{}/{}.jpg'.format(save_name,i),img)

def create_animate_pair():
    f = open('./data/vox_cross_id_animate.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source_frame","driving_video"])
    pairs = pd.read_csv('data/vox_cross_id_evaluation.csv')
    source = pairs['source'].tolist()
    driving = pairs['driving'].tolist()
    source_frames = []
    driving_videos = []

    for src, dst in zip(source,driving):
        video = os.path.dirname(dst).replace('vox1_frames','vox1')
        source_frames.append(src)
        driving_videos.append(video)
    source_frames = np.array(source_frames).reshape(-1,1)
    driving_videos = np.array(driving_videos).reshape(-1,1)
    content = np.concatenate((source_frames,driving_videos),1)
    csv_writer.writerows(content)
    f.close()

def merge_abla_imgs(paths):
    pth = paths[0]
    imgps = os.listdir(pth)
    for i in tqdm(range(len(imgps))):
        imgp = imgps[i]
        cats = []
        for pth in paths:
            img = os.path.join(pth,imgp)
            img = cv2.imread(img)
            cats.append(img)
        img = np.vstack(cats)
        cv2.imwrite('FID/abla/{}.jpg'.format(i),img)

def mergevideos():
    videos_path1 = 'animation'
    videos_path2 = '/data/fhongac/workspace/gitrepo/first-order-model/animation'
    videos = os.listdir(videos_path1)
    save_path = 'merge_animation'
    for vi in tqdm(videos):
        fomm = np.array(mimread('{}/{}'.format(videos_path2,vi),memtest=False))
        ours = np.array(mimread('{}/{}'.format(videos_path1,vi),memtest=False))
        reader = imageio.get_reader('{}/{}'.format(videos_path2,vi))
        fps = reader.get_meta_data()['fps']
        if len(fomm.shape) == 3:
            fomm = np.array([gray2rgb(frame) for frame in fomm])
        if fomm.shape[-1] == 4:
            fomm = fomm[..., :3]
        if len(ours.shape) == 3:
            ours = np.array([gray2rgb(frame) for frame in ours])
        if ours.shape[-1] == 4:
            ours = ours[..., :3]
        fomm = fomm[:,:,-256:,:]
        src_dst = ours[:,:,:512,:]
        ours = ours[:,:,-256:,:]
        merge = np.concatenate((src_dst,fomm,ours),2)
        imageio.mimsave('{}/{}'.format(save_path,vi), merge, fps=fps)

def extractFrames():
    videos_pairs = pd.read_csv('data/vox_cross_id_animate.csv')
    source = videos_pairs['source_frame'].tolist()
    driving = videos_pairs['driving_video'].tolist()
    frame_pairs = pd.read_csv('data/vox_cross_id_evaluation.csv')
    # source = videos_pairs['source_frame'].tolist()
    driving_frame = frame_pairs['driving'].tolist()
    concate = 'FID/video_cross_id'
    generate = 'FID/video_generate'
    videos = 'animation'
    for i, (src, dst,number) in tqdm(enumerate(zip(source,driving,driving_frame))):
        video = np.array(mimread('{}/{}.mp4'.format(videos,i),memtest=False))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        num = int(os.path.basename(number)[:7])
        video_array = img_as_float32(video)
        frame = (video_array[num]*255).astype(np.uint8)
        imsave('{}/{}.jpg'.format(concate,i),frame)
        imsave('{}/{}.jpg'.format(generate,i),frame[:,-256:,:])

class depth_network(nn.Module):
    def __init__(self):
        super(depth_network, self).__init__()
        self.depth_encoder = depth.ResnetEncoder(18, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
    def forward(self,x):
        outputs = self.depth_decoder(self.depth_encoder(x))
        return outputs

def viewNetworkStructure():
    network = depth_network().cuda()
    print(network)
    import hiddenlayer as h
    vis_graph = h.build_graph(network, torch.zeros([1,3,256,256]).cuda())   # 获取绘制图像的对象
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    vis_graph.save("network_graph/depth_network.png")   # 保存图像的路径
   
def drawKPline():
    kp10 = [2.292730636,0.870793269,0.719648837]
    kp15 = [2.335680558,0.872849592,0.7229482939818654]
    kp20 = [2.268743373,0.882716346, 0.67557838]
    kp25 = [3.395401378,0.827983638,0.662669217]
    data = np.array([kp10,kp15,kp20,kp25])
    x=[0,1,2,3]
    
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    l1=plt.plot(x,data[:,0],'r--',label='PRMSE')
    l2=plt.plot(x,data[:,1],'g--',label='AUCON')
    l3=plt.plot(x,data[:,2],'b--',label='CSIM')

    plt.plot(x,data[:,0],'ro-',x,data[:,1],'g+-',x,data[:,2],'b^-')
    plt.grid(linestyle=':')
    # ax.tick_params(bottom=False)
    plt.xticks(x,["kp=10","kp=15","kp=20","kp=25"])  #去掉横坐标值
    # plt.yticks([])  #去掉纵坐标值
    # plt.setp(ax.get_xticklabels(), visible=False)
    # plt.setp(ax.get_yticklabels(), visible=False)
    plt.legend()
    plt.savefig('network_graph/kp.pdf')

def all_depth(path):
    imgs = os.listdir(path+'/gt')

    depth_encoder = depth.ResnetEncoder(18, False).cuda()
    depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    loaded_dict_enc = torch.load('depth/models/weights_19/encoder.pth')
    loaded_dict_dec = torch.load('depth/models/weights_19/depth.pth')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_decoder.load_state_dict(loaded_dict_dec)
    depth_encoder.eval()
    depth_decoder.eval()
    for im in tqdm(imgs):
        driving = os.path.join(path,'gt',im)
        source = os.path.join(path,'generate',im)
    img = Image.open(path).convert('RGB').resize((256,256))
    tensor_img = T.ToTensor()(img).unsqueeze(0).cuda()
    outputs = depth_decoder(depth_encoder(tensor_img))
    disp = outputs[("disp", 0)]
    # Saving colormapped depth image
    disp_resized = torch.nn.functional.interpolate(disp, (256, 256), mode="bilinear", align_corners=False)
    disp_resized_np = disp_resized.squeeze().detach().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')

    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    plt.axis('off')
    plt.imshow(colormapped_im)

    # plt.colorbar(mapper)
    plt.savefig(path+'.pdf')
    # plt.savefig(path+'.png')

    plt.clf()


def changevideos():
    # videos_path1 = 'animation'
    # videos_path2 = '/data/fhongac/workspace/gitrepo/first-order-model/animation'
    # videos = os.listdir(videos_path1)
    # save_path = 'merge_animation'
    # for vi in tqdm(videos):
    #     fomm = np.array(mimread('{}/{}'.format(videos_path2,vi),memtest=False))
    #     ours = np.array(mimread('{}/{}'.format(videos_path1,vi),memtest=False))
    #     reader = imageio.get_reader('{}/{}'.format(videos_path2,vi))
    #     fps = reader.get_meta_data()['fps']
    #     if len(fomm.shape) == 3:
    #         fomm = np.array([gray2rgb(frame) for frame in fomm])
    #     if fomm.shape[-1] == 4:
    #         fomm = fomm[..., :3]
    #     if len(ours.shape) == 3:
    #         ours = np.array([gray2rgb(frame) for frame in ours])
    #     if ours.shape[-1] == 4:
    #         ours = ours[..., :3]
    #     fomm = fomm[:,:,-256:,:]
    #     src_dst = ours[:,:,:512,:]
    #     ours = ours[:,:,-256:,:]
    #     merge = np.concatenate((src_dst,fomm,ours),2)
    #     imageio.mimsave('{}/{}'.format(save_path,vi), merge, fps=fps)
    # 155
    # path = 'merge_animation/155.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/7PbDDjXgYzU/id10287#bP0bKbQQlzc#003638#003940_disp.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/155.mp4'
    # save = 'FID/animation/155.mp4'

    # path = 'merge_animation/523.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/7PbDDjXgYzU/id10287#4oOmqI1myzY#000381#000729_disp.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/523.mp4'
    # save = 'FID/animation/523.mp4'

    # path = 'merge_animation/705.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/705.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/705.mp4'
    # save = 'FID/animation/705.mp4'

    # path = 'merge_animation/2062.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/2062.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/2062.mp4'
    # save = 'FID/animation/2062.mp4'

    # path = 'merge_animation/1841.mp4'
    # disp = '/data/fhongac/workspace/src/depthEstimate/1841.mp4'
    # osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/1841.mp4'
    # save = 'FID/animation/1841.mp4'

    path = 'merge_animation/1758.mp4'
    disp = '/data/fhongac/workspace/src/depthEstimate/1758.mp4'
    osfv = '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/1758.mp4'
    save = 'FID/animation/1758.mp4'
    
    video = np.array(mimread('{}'.format(path),memtest=False))
    reader = imageio.get_reader('{}'.format(path))
    fps = reader.get_meta_data()['fps']
    video = np.array([gray2rgb(frame) for frame in video])

    disp = np.array(mimread('{}'.format(disp),memtest=False))
    disp = np.array([gray2rgb(frame) for frame in disp])
    
    osfv = np.array(mimread('{}'.format(osfv),memtest=False))
    osfv = np.array([gray2rgb(frame) for frame in osfv])

    bz,h,w,c = video.shape
    up_video = np.concatenate((video[:,:,:int(w/2),:],disp),2)
    down_video = np.concatenate((video[:,:,int(w/2):int(w/4)*3,:],osfv,video[:,:,int(w/4)*3:,:]),2)

    up_zeros = np.ones((bz,20,256*3,3))*255
    mid_zeros = np.ones((bz,40,256*3,3))*255
    down_zeros = np.ones((bz,40,256*3,3))*255
    video = np.concatenate((up_zeros,up_video, mid_zeros, down_video,down_zeros),1)
    imageio.mimsave('{}'.format(save), video, fps=fps)

    print('aa')

def mergevideo(paths,save_name):
    pth = paths[0]
    vps = os.listdir(pth)
    if not os.path.exists('Compare/{}'.format(save_name)):
        os.makedirs('Compare/{}'.format(save_name))
    for i in tqdm(range(len(vps))):
        imgp = vps[i]
        cats = []
        fps = None
        for pth in paths:
            vp = os.path.join(pth,imgp)
            video = np.array(mimread('{}'.format(vp),memtest=False))
            if not fps:
                reader = imageio.get_reader('{}'.format(vp))
                fps = reader.get_meta_data()['fps']
            video = np.array([gray2rgb(frame) for frame in video])
            cats.append(video)
        cats = np.concatenate(cats,1)
        imageio.mimsave('Compare/{}/{}.mp4'.format(save_name,i), cats, fps=fps*2)
        # cv2.imwrite('Compare/{}/{}.jpg'.format(save_name,i),img)

def vec_sten():
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns
    
    # We want to get TSNE embedding with 2 dimensions
    
    
    generated = 'log/Unet_Baseline/vox_cross_id/generate'
    # generated = 'log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_cross_id/generate'
    df = extract_arcface_id(False,generated, (256,256), 0)
    id_maps, ids = id_collect()
    feats = np.array(df['value'].values)
    # pdb.set_trace()
    X = np.stack(feats,0)

    n_components = 2
    tsne = TSNE(n_components,init='pca', random_state=501)
    tsne_result = tsne.fit_transform(X)
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': ids})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=20)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig('baseline_cross.jpg')
def id_collect():
    idx = len('/data/fhongac/origDataset/vox1_frames/test/')
    # pairs = pd.read_csv('data/vox_evaluation_v2.csv')
    pairs = pd.read_csv('data/vox_cross_id_evaluation_best_frame.csv')
    sources = pairs['source'].values
    n = len(sources)
    maps = {}
    ids = []
    for i in range(n):
        addr = sources[i]
        iden = addr[idx:idx+7]
        ids.append(iden)
        if iden in maps:
            maps[iden].append(i)
        else:
            maps[iden] = []
            maps[iden].append(i)
    return maps ,ids
def Video_construction(ids, ours, others):
    for id_ in ids:
        mcnet = os.path.join(ours,str(id_)+'.mp4')
        reader = imageio.get_reader('{}'.format(mcnet))
        fps = reader.get_meta_data()['fps']
        video = np.array(mimread('{}'.format(mcnet),memtest=False))
        video = np.array([gray2rgb(frame) for frame in video])
        num, h,w,c=video.shape
        idx = w//3*2
        src_and_dst = video[:,:,:idx,:]
        mcnet = video[:,:,idx:,:]
        result = []
        result.append(src_and_dst)
        for method in others:
            mp = os.path.join(method,str(id_)+'.mp4')
            video = np.array(mimread('{}'.format(mp),memtest=False))
            video = np.array([gray2rgb(frame) for frame in video])
            mds = video[:,:,idx:,:]
            result.append(mds)
        result.append(mcnet)
        cats = np.concatenate(result,2)
        imageio.mimsave('Compare/select_video/{}.mp4'.format(id_), cats, fps=fps*2)
        print('Compare/select_video/{}.mp4'.format(id_))
        # cv2.imwrite('Compare/{}/{}.jpg'.format(save_name,i),img)
def clipVideo(path):
    reader = imageio.get_reader('{}'.format(path))
    fps = reader.get_meta_data()['fps']
    video = np.array(mimread('{}'.format(path),memtest=False))
    video = np.array([gray2rgb(frame) for frame in video])
    num, h,w,c=video.shape
    # num, h,w,c=video.shape
    bound = np.ones((num,224//3,w,c))*255
    video = np.concatenate((bound,video,bound),1)
    # interval = num//9
    # video=video[interval*2:,...]
    imageio.mimsave(path, video, fps=fps)
#串连所有的memory bank
def mergeMetaMb(path,name, rows_num=16, columns_num=32, start=0):
    rows_num = rows_num
    columns_num = columns_num
    whole_img = []
    column_barria = np.zeros((32,2,3))
    for i in range(rows_num):
        row = []
        for j in range(columns_num):
            num = i*columns_num+j + start
            pth = os.path.join(path,'{}.jpg'.format(num))
            img = cv2.imread(pth)
            row.append(img)
            row.append(column_barria)
            
        row = np.concatenate(row[:-1],1)
        h,w,c= row.shape
        row_barria = np.zeros((2,w,3))       
        whole_img.append(row)
        whole_img.append(row_barria)
    whole_img = np.concatenate(whole_img[:-1],0)
    cv2.imwrite(name,whole_img)

def mergeMetaMb_sub(path,name):
    mbs = [3,9,33,13,23,459,108,149,196,171,418,369]
    rows_num = 2
    columns_num = 6
    whole_img = []
    column_barria = np.zeros((32,2,3))
    for i in range(rows_num):
        row = []
        for j in range(columns_num):
            num = i*columns_num+j
            idx = mbs[num]
            pth = os.path.join(path,'{}.jpg'.format(idx))
            img = cv2.imread(pth)
            row.append(img)
            row.append(column_barria)
            
        row = np.concatenate(row[:-1],1)
        h,w,c= row.shape
        row_barria = np.zeros((2,w,3))       
        whole_img.append(row)
        whole_img.append(row_barria)
    whole_img = np.concatenate(whole_img[:-1],0)
    cv2.imwrite(name,whole_img)

# 画柱状图
def AKD():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [0.1236,0.1097, 0.1087,0.1065]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((0.1,0.13))
    # plt.yticks([])
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    # plt.ylabel('Number of Patents')
    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('AKD.pdf')
    plt.clf()

def AED():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [1.303,1.237, 1.227,1.203]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((1.115,1.35))
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    # plt.ylabel('Number of Patents')
    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('AED.pdf')
    plt.clf()
def l1_bar():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [0.0356,0.0336, 0.0333,0.0331]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((0.0330,0.0358))
    # plt.ylabel('Number of Patents')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('l1.pdf')
    plt.clf()
def LPIPS_bar():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [0.182,0.175,0.175,0.174]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((0.173,0.183))
    # plt.ylabel('Number of Patents')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])

    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('LPIPS.pdf')
    plt.clf()
def PSNR_bar():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [31.701,31.896,31.932,31.942]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((31.600,31.943))
    # plt.ylabel('Number of Patents')
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('PSNR.pdf')
    plt.clf()   
def SSIM_bar():
    # -*- coding: utf-8 -*-
    import matplotlib.pyplot as plt
    
    name_list = ['Baseline','MCNet w/o F','MCNet w/o kp','MCNet']
    num_list = [81.1,82.3,82.3,82.5]
    plt.bar(name_list, num_list, color=['aquamarine','darkseagreen', 'dodgerblue','crimson'], width=0.7)
    plt.ylim((81.0,82.6))
    ax = plt.gca()
    ax.axes.yaxis.set_ticklabels([])
    # plt.ylabel('Number of Patents')
    plt.xticks([])
    # plt.legend()
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.savefig('SSIM.pdf')
    plt.clf()
def Bars():
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="white", context="talk")
    rs = np.random.RandomState(8)

    # Set up the matplotlib figure
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,3), sharex=True)

    # Generate some sequential data
    x = np.array(list("ABCDEFGHIJ"))
    y1 = np.arange(1, 11)
    sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel("Sequential")

    # Center the data to make it diverging
    y2 = y1 - 5.5
    sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("Diverging")

    # Randomly reorder the data to make it qualitative
    y3 = rs.choice(y1, len(y1), replace=False)
    sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
    ax3.axhline(0, color="k", clip_on=False)
    ax3.set_ylabel("Qualitative")

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
def sum_meta(path):
    rows_num = 16
    columns_num = 32
    total=[]
    for i in range(rows_num):
        for j in range(columns_num):
            num = i*columns_num+j
            pth = os.path.join(path,'{}.jpg'.format(num))
            img = cv2.imread(pth)
            total.append(img)
    pdb.set_trace()
    total = np.stack(total).mean(0).astype(np.uint8)
    # total = (total/(columns_num*rows_num)).astype(np.uint8)
    print(total)
    cv2.imwrite('mb_sum.jpg',total)

def cropVideo(path):
    vps = os.listdir(path)
    for vp in vps:
        if '.mp4' in vp:
            reader = imageio.get_reader('{}'.format(os.path.join(path,vp)))
            fps = reader.get_meta_data()['fps']
            video = np.array(mimread('{}'.format(os.path.join(path,vp)),memtest=False))
            video = np.array([gray2rgb(frame) for frame in video])
            num, h,w,c=video.shape
            # num, h,w,c=video.shape
            video = video[:,:,-w//3:,:]
            imageio.mimsave('single_demo/{}'.format(vp), video, fps=fps)
            print(vp)

def create_same_id_train_set(path):
    vis = os.listdir(path)
    videos = np.random.choice(vis, replace=False, size=100)
    f = open('./data/vox_train_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)
        imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = os.path.join(path,v[0],pair[0])
        dst = os.path.join(path,v[0],pair[1])
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_vox2_same_id_train_set(path):
    path = '/data/fhongac/origDataset/Voxceleb2/vox2_train_frames/mp4/'
    videos = sorted(glob(path+"/*/*/*"))
    videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/vox2_train_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)[0]
        imgs = sorted(glob(v+"/*.jpg"))
        # imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_HDTF_same_id_train_set():
    path = '/ssddata/fhongac/origDataset/HDTF/frames_split/test'
    videos = sorted(glob(path+"/*"))
    # videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/HDTF_test_evaluation_new.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in tqdm(range(2083)):
        v = np.random.choice(videos, replace=False, size=1)[0]
        print(v)
        imgs = sorted(glob(v+"/*.jpg"))
        if len(imgs)<2:
            continue
        # imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_HDTF_HHFQ_test_set(path):
    ffhq = '/ssddata/fhongac/gitrepo/GRAM/datasets/ffhq'
    ffhq = sorted(glob(ffhq+"/*.png"))

    kids = '/ssddata/fhongac/origDataset/kids_face'
    kids = sorted(glob(kids+"/*.png"))

    # videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/HDTF_HHFQ_test_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in tqdm(range(2083)):
        adult = np.random.choice(ffhq, replace=False, size=1)[0]
        kid = np.random.choice(kids, replace=False, size=1)[0]
        pair = [adult,kid]
        if random.random() < 0.5:
            pair = pair[::-1]
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_fashion_set(path):
    
    
    # videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/HDTF_HHFQ_test_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in tqdm(range(2083)):
        adult = np.random.choice(ffhq, replace=False, size=1)[0]
        kid = np.random.choice(kids, replace=False, size=1)[0]
        pair = [adult,kid]
        if random.random() < 0.5:
            pair = pair[::-1]
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def create_taichi_test_set(path):
    path = '/ssddata/fhongac/origDataset/taichi/taichi/test'
    videos = sorted(glob(path+"/*"))
    videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/taichi_test_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)[0]
        imgs = sorted(glob(v+"/*.png"))
        # imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()
def create_ted_test_set():
    path = '/ssddata/fhongac/origDataset/ted/ted/TED384-v2/test'
    videos = sorted(glob(path+"/*"))
    videos = np.random.choice(videos, replace=False, size=100)
    f = open('./data/ted_test_evaluation_temp.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        v = np.random.choice(videos, replace=False, size=1)[0]
        imgs = sorted(glob(v+"/*.png"))
        pdb.set_trace()
        # imgs = os.listdir(os.path.join(path,v[0]))
        pair = np.random.choice(imgs, replace=False, size=2)
        src = pair[0]
        dst = pair[1]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

def stastic_face_scale():
    path = '/data/fhongac/origDataset/vox1_frames/train'
    imgs = sorted(glob(path+'/*/*.png'))
    from facenet_pytorch import MTCNN, InceptionResnetV1

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256, margin=0)
    # Get cropped and prewhitened image tensor
    log_file = open('statistic_train.txt', 'a')
    save_dict = {}
    for img in tqdm(imgs):
        # print(img)
        try:
            cv_img = Image.open(img)
            size = cv_img.size
            total_pixels = size[0]*size[1]
            img_cropped,batch_boxes = mtcnn(cv_img, save_path='src.jpg')
            box = batch_boxes[0]
            box = [
            int(max(box[0] , 0)),
            int(max(box[1] , 0)),
            int(min(box[2], size[0])),
            int(min(box[3] , size[1])),
            ]
            crop_size = (box[2]-box[0])*(box[3]-box[1])
            loss_string = img+' {}/{}'.format(str(crop_size),str(total_pixels))+" {},{},{},{}".format(box[0],box[1],box[2],box[3])
            print(loss_string, file=log_file)
            log_file.flush()
            # print(img)
            key = (crop_size*100)//total_pixels
            if not key in save_dict:
                save_dict[key] = []
            save_dict[key].append(img)
        except Exception as e:
            print(e)
    np.save('statstic_train.npy',save_dict)

def create_different_scale():
    dict_load=np.load('statstic.npy', allow_pickle=True).tolist()

    f1 = open('./data/scale_test_30_30_evaluation.csv','w',encoding='utf-8')
    scale30 = dict_load[1]+dict_load[2]
    scale60 = dict_load[3]+dict_load[4]+dict_load[5]
    scale80 = dict_load[6]+dict_load[7]
    #scale_test_30_60_evaluation
    source = []
    driving = []
    csv_writer = csv.writer(f1)
    csv_writer.writerow(["source","driving","frame"])
    
    for i in range(2083):
        # imgs = os.listdir(os.path.join(path,v[0]))
        src = np.random.choice(scale30, replace=False, size=1)[0]
        dst = np.random.choice(scale30, replace=False, size=1)[0]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f1.close()

    #scale_test_30_80_evaluation
    f = open('./data/scale_test_60_60_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        # imgs = os.listdir(os.path.join(path,v[0]))
        src = np.random.choice(scale60, replace=False, size=1)[0]
        dst = np.random.choice(scale60, replace=False, size=1)[0]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()

    #scale_test_80_80_evaluation
    f = open('./data/scale_test_80_80_evaluation.csv','w',encoding='utf-8')
    source = []
    driving = []
    csv_writer = csv.writer(f)
    csv_writer.writerow(["source","driving","frame"])
    for i in range(2083):
        # imgs = os.listdir(os.path.join(path,v[0]))
        src = np.random.choice(scale80, replace=False, size=1)[0]
        dst = np.random.choice(scale80, replace=False, size=1)[0]
        source.append(src)
        driving.append(dst)
    sources = np.array(source).reshape(-1,1)
    driving = np.array(driving).reshape(-1,1)
    content = np.concatenate((sources,driving),1)
    csv_writer.writerows(content)
    f.close()
 
    print('aa')

def select_images():
    save_dict = {}
    save_dict['30_30'] = [38, 117, 261]
    save_dict['30_60'] = [0, 12,104] 
    save_dict['30_80'] = [270,276, 281, 501]
    save_dict['60_30'] = [4, 65, 105]
    save_dict['60_60'] = [16, 65, 295] 
    save_dict['60_80'] = [99, 461,784 ,1408]
    save_dict['80_30'] = [94, 643, 811]
    save_dict['80_60'] = [6, 120,358,388]
    save_dict['80_80'] = [5,33,115,231]
    save_dir = 'Compare/rebuttal_pic'
    import shutil
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for key in save_dict:
        imgs = save_dict[key]
        for img in imgs:
            baseline = 'baselin_{}/vox_same_id/generate/{}.jpg'.format(key,img)
            woISCM = 'woISCM_{}/vox_same_id/generate/{}.jpg'.format(key,img)
            ours = '{}/vox_same_id/generate/{}.jpg'.format(key,img)
            src = '{}/vox_same_id/source/{}.jpg'.format(key,img)
            driving = '{}/vox_same_id/gt/{}.jpg'.format(key,img)
            folder = os.path.join(save_dir,key,str(img))
            if not os.path.exists(folder):
                os.makedirs(folder)
            shutil.copy(baseline,folder+'/baseline.jpg')
            shutil.copy(woISCM,folder+'/woISCM.jpg')
            shutil.copy(ours,folder+'/ours.jpg')
            shutil.copy(src,folder+'/src.jpg')
            shutil.copy(driving,folder+'/driving.jpg')

def combine():
    v1 = '/data/fhongac/workspace/src/ECCV2022/Goodwin/1.mp4'
    v2 = '/data/fhongac/workspace/src/ECCV2022/Goodwin/mcnet_1.mp4'
    # reader = imageio.get_reader(v1)
    # fps = reader.get_meta_data()['fps']
    video = np.array(imageio.mimread(v1,memtest=False))
    video = np.array([gray2rgb(frame) for frame in video])
    num, h,w,c=video.shape

    v2 = np.array(imageio.mimread(v2,memtest=False))
    v2 = np.array([gray2rgb(frame) for frame in v2])
    num, h,w,c=v2.shape
    rst = v2[:, :,-w//3:,:]

    video = np.concatenate((video,rst),2)
    imageio.mimsave('Goodwin/merge_1.mp4', video, fps=24)

if __name__ == '__main__':
    # create_ted_test_set()
    # create_HDTF_same_id_train_set()
    # create_different_scale()
    # combine()
    # mergeMetaMb_sub('ted384_mb','ted_mb.jpg')
    # mergeMetaMb('ted384_mb','ted_mb.jpg')
    # mergeMetaMb('worec_mb','aa_worec_mb_grid_sub.jpg',rows_num=1, columns_num=8, start=32)
    mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware_hd_split/concate',
            '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/log/hdtf_face-vid2vid/htdf_same_id/generate'],'htdf_512_same_id')  

    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/MCNet_ted_384/ted_same_id/concate',
    #         '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/checkpoint/ted_same_id/generate'],'ted_same_id')  

    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/concate',
    #         '/data/fhongac/workspace/src/ECCV2022/log/Unet_Baseline/vox_same_id/generate',
    #         '/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV19_kp15_run2/vox_same_id/generate'],'same_ablation')  

    # create_ted_test_set()
    # stastic_face_scale()
    exit()
    # ours = ['30_30','30_60','30_80','60_30','60_60','60_80','80_30','80_60','80_80']
    # woISCM = ['woISCM_30_30','woISCM_30_60','woISCM_30_80','woISCM_60_30','woISCM_60_60','woISCM_60_80','woISCM_80_30','woISCM_80_60','woISCM_80_80']
    # baseline = ['baselin_30_30','baselin_30_60','baselin_30_80','baselin_60_30','baselin_60_60','baselin_60_80','baselin_80_30','baselin_80_60','baselin_80_80']
    # for bl, wo, full in zip(baseline,woISCM,ours):
    #     mergeimgs([bl+'/vox_same_id/concate',
    #             wo+'/vox_same_id/generate',
    #             full+'/generate',],'compare_'+full)  

    # cropVideo('demo')
    # evaluate_CSIM_PRMSE_AUCON('log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/source', 'log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/gt','log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/generate')
    # evaluate_CSIM_PRMSE_AUCON('log/Unet_Baseline/vox_same_id/source', 'log/Unet_Baseline/vox_same_id/gt','log/Unet_Baseline/vox_same_id/generate')
    exit(0)
    # create_ted_test_set('/data/fhongac/origDataset/vox1_frames/train')
    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/concate',
    #         'log/Unet_Baseline/vox_same_id/generate'],'Child-HHFQ-compare')  
    # exit(0)
    # CSIM = cmp_CSIM_corss('/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/log/InpaintingNetwork/vox_same_id')
    # CSIM = cmp_CSIM_corss('/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/vox_same_id')
    # print(CSIM)
    # CSIM = cmp_CSIM_corss('/data/fhongac/workspace/gitrepo/articulated-animation/checkpoints/vox_same_id')
    # CSIM = cmp_CSIM_corss('/data/fhongac/workspace/gitrepo/articulated-animation/checkpoints/vox_same_id')

    
    # create_test_set('/data/fhongac/origDataset/vox1_frames/test')
    # concate_compared_results('FID/generate','/data/fhongac/workspace/gitrepo/first-order-model/FID/generate')
    # concate_compared_results('FID/concate','/data/fhongac/workspace/gitrepo/first-order-model/FID/concate')
    # create_cross_id_test_set('/data/fhongac/origDataset/vox1_frames/test')
    # render('ppt_figure/0000021.jpg')
    # depth_rgb('ppt_figure/293.jpg')
    # process_celeV('/data/fhongac/origDataset/CelebV')
    # compare()
    # evaluate_CSIM_PRMSE_AUCON('log/Unet_PartitionMemoryUnitV3_no_adv/vox_same_id/source','log/Unet_PartitionMemoryUnitV3_no_adv/vox_same_id/gt','log/Unet_PartitionMemoryUnitV3_no_adv/vox_same_id/generate')
    # evaluate_PRMSE_AUCON()  # CUDA_VISIBLE_DEVICES=7 python utils.py 
    # viewNetworkStructure()
    # aus('11.png')
    # mergeimgs(['/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/vox_cross_id/concate',
    #         '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_cross_id/concate',
    #         '/data/fhongac/workspace/src/parallel-fom-rgbd/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id/concate'])    
    # create_animate_pair()
    # merge_abla_imgs(['/data/fhongac/workspace/gitrepo/first-order-model/FID/cross_id','log/vox-adv-256baseline/vox_cross_id/concate', 'log/vox-adv-256rgbd_kp_num15/vox_cross_id/concate','log/vox-adv-256rgbd_kp_num15_rgbd_attnv2_wo_D/vox_cross_id/concate','log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id/concate'])  
    # extractFrames()
    # mergevideos()
    # drawKPline()
    # changevideos()
    # concate_compared_results('log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/concate','//data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/log/InpaintingNetwork/vox_same_id/generate')
    # modify_same_id_voxceleb()

    #ICLR 2023
    #compare same id
    mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_same_id/concate',
            '/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/vox_same_id/generate',
            # '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_same_id/generate',
            '/data/fhongac/workspace/gitrepo/articulated-animation/checkpoints/vox_same_id/generate',
            '/data/fhongac/workspace/src/DaGAN_Origin/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_same_id/generate',
            '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/log/InpaintingNetwork/vox_same_id/generate'],'HDTF_compare_same_id')  
     #compare cross id
    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_cross_id/concate',
    #         '/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/vox_cross_id/generate',
    #         # '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_same_id/generate',
    #         '/data/fhongac/workspace/gitrepo/articulated-animation/checkpoints/vox_cross_id/generate',
    #         '/data/fhongac/workspace/src/DaGAN_Origin/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/vox_cross_id/generate',
    #         '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/log/InpaintingNetwork/vox_cross_id/generate'],'compare_cross_id')  

     #compare celebv
    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/celebv/concate',
    #         '/data/fhongac/workspace/gitrepo/first-order-model/checkpoint/celebv/generate',
    #         # '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_same_id/generate',
    #         '/data/fhongac/workspace/gitrepo/articulated-animation/checkpoints/celebv/generate',
    #         '/data/fhongac/workspace/src/DaGAN_Origin/log/vox-adv-256rgbd_kp_num15_rgbd_attnv2/celebv/generate',
    #         '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/log/InpaintingNetwork/celebv/generate'],'celebv')  
    #compare cross id video
    # mergevideo(['/data/fhongac/workspace/src/ECCV2022/animation',
    #         '/data/fhongac/workspace/gitrepo/first-order-model/animation',
    #         # '/data/fhongac/workspace/gitrepo/One-Shot_Free-View_Neural_Talking_Head_Synthesis/checkpoint/vox_same_id/generate',
    #         '/data/fhongac/workspace/gitrepo/articulated-animation/animation',
    #         '/data/fhongac/workspace/src/DaGAN_Origin/animation',
    #         '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/animation'],'compare_cross_id_video')  
    
    #compare ablation
    # mergeimgs(['/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_cross_id/concate',
    #         '/data/fhongac/workspace/src/ECCV2022/log/Unet_Baseline/vox_cross_id/generate',
    #         '/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV19_kp15_run2/vox_cross_id/generate'],'ablation')  

    # Video_construction([210,773,778,889,1237, 856,1247],'/data/fhongac/workspace/src/ECCV2022/animation', [
    #         '/data/fhongac/workspace/gitrepo/first-order-model/animation',
    #         '/data/fhongac/workspace/gitrepo/articulated-animation/animation',
    #         '/data/fhongac/workspace/src/DaGAN_Origin/animation',
    #         '/data/fhongac/workspace/gitrepo/Thin-Plate-Spline-Motion-Model/animation'])
    #clip video
    # clipVideo('/data/fhongac/workspace/src/ECCV2022/Compare/select_video/1237.mp4')
    #mb grid
    mergeMetaMb('/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_cross_id/meta_mb','all_mb_grid.jpg',rows_num=32, columns_num=16, start=0)
    # sum_meta('/data/fhongac/workspace/src/ECCV2022/log/ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware/vox_cross_id/meta_mb')

    # SSIM_bar()
    # l1_bar()
    # PSNR_bar()
    # LPIPS_bar()
    # AED()
    # AKD()
