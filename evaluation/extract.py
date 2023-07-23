import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.transform import resize
from .util import frames2array
from imageio import mimsave
import torch.nn.functional as F
import pdb
from collections import OrderedDict
def extract_vgg(in_folder, is_video, image_shape, column):
    from torchvision.models import vgg
    from torchvision import transforms
    from torch import nn
    import torch

    class VggConv(nn.Module):
            def __init__(self):
                super(VggConv, self).__init__()
                self.original_model = vgg.vgg16(pretrained=True)
            def forward(self, x):
                x = self.original_model.features(x)
                return x

    net = VggConv().cuda()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
                     transforms.ToTensor(),
                     normalize])
    for file in tqdm(sorted(os.listdir(in_folder))):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            with torch.no_grad():
                frame = frame.astype('float32') / 255.0
                frame = transform(frame)
                frame = frame.unsqueeze(0).cuda()
                feat = net(frame).data.cpu().numpy()
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(feat)

    return pd.DataFrame(out_df)



def extract_face_pose(in_folder, is_video, image_shape, column):
    import face_alignment

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            kp = fa.get_landmarks(frame)
            if kp is not None:
               kp = kp[0]
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(kp)

    return pd.DataFrame(out_df)

def gray_resize_for_identity(out, size=128):
    out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
    out_gray = out_gray.unsqueeze(1)
    out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
    return out_gray
def extract_arcface_id(is_video, in_folder, image_shape, column):
    from .OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable
    import torch
    from imageio import mimsave
    from stylegan2 import ResNetArcFace
    net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False).cuda().eval()
    checkpoint = torch.load('checkpoints/arcface_resnet18.pth', map_location=torch.device('cuda'))
    ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint.items())
    net.load_state_dict(ckp_generator)

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                frame_gray = gray_resize_for_identity(frame)
                id_vec = net(frame_gray)[0].data.cpu().numpy()
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)



def extract_face_id(is_video, in_folder, image_shape, column):
    from .OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable
    import torch
    from imageio import mimsave
    from stylegan2 import ResNetArcFace
    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
    # net = ResNetArcFace(block='IRBlock', layers=(2, 2, 2, 2), use_se=False).cuda().eval()


    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                id_vec = net(frame)[0].data.cpu().numpy()
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)


def extract_body_pose(in_folder, is_video, image_shape, column):
    from pose_estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
    import torch
    from pose_estimation.network.rtpose_vgg import get_model
    from pose_estimation.network.post import decode_pose

    weight_name = 'pose_estimation/network/weight/pose_model.pth'

    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    model = torch.nn.DataParallel(model).cuda()
    model.float()
    model.eval()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]# B,G,R order
            multiplier = get_multiplier(frame)

            with torch.no_grad():
                orig_paf, orig_heat = get_outputs(multiplier, frame, model, 'rtpose')

                # Get results of flipped image
                swapped_img = frame[:, ::-1, :]
                flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, model, 'rtpose')

                # compute averaged heatmap and paf
                paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)

            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            _, _, joint_list, _ = decode_pose(frame, param, heatmap, paf)
            
            joint_list = np.array(joint_list)
            tmp = -np.ones((18, 2))
            if len(joint_list) != 0:
                tmp[joint_list[:, -1].astype(int)] = joint_list[:, :2]
            joint_list = tmp

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(joint_list)


    return pd.DataFrame(out_df)


def extract_body_id(in_folder, is_video, image_shape, column):
    from reid_baseline.model import ft_net
    from torch import nn
    from torchvision import transforms
    import torch

    net = ft_net(751)
    net.load_state_dict(torch.load('reid_baseline/reid_model.pth'))
    net.model.fc = nn.Sequential()
    net.classifier = nn.Sequential()
    net.cuda()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [255., 255., 255.]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder)):
        video = frames2array(os.path.join(in_folder, file), is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = data_transforms(frame).cuda()
            with torch.no_grad():
                id_vec = net(frame.unsqueeze(0))
                id_vec = id_vec.data.cpu().numpy()

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    return pd.DataFrame(out_df)

def cmp_akd(path,size=[256,256]):
    gt = os.path.join(path,'gt')
    generated = os.path.join(path,'generate')

    df1 = extract_face_pose(gt, False, size, 0)
    df2 = extract_face_pose(generated, False, size, 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    #df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
    df2 = df2.sort_values(by=['file_name', 'frame_number'])

    scores = []
    for i in range(df1.shape[0]):
        try:
            file_name1 = df1['file_name'].iloc[i].split('.')[0]
            file_name2 = df2['file_name'].iloc[i].split('.')[0]
            assert file_name1 == file_name2
            assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
            if df2['value'].iloc[i] is not None: 
                scores.append(np.mean(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float)))
        except Exception as e:
            print(e)
    return np.mean(scores)

def cmp_aed(path,size=[256,256]):
    gt = os.path.join(path,'gt')
    generated = os.path.join(path,'generate')
    df1 = extract_face_id(False, gt, size, 0)
    df2 = extract_face_id(False,generated, size, 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    #df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
    df2 = df2.sort_values(by=['file_name', 'frame_number'])
    assert df1.shape == df2.shape
    scores = []
    for i in range(df1.shape[0]):
        file_name1 = df1['file_name'].iloc[i].split('.')[0]
        file_name2 = df2['file_name'].iloc[i].split('.')[0]
        assert file_name1 == file_name2
        assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
        scores.append(np.sum(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float) ** 2))
    return np.mean(scores)
def cmp_aed_corss(path):
    gt = os.path.join(path,'source')
    generated = os.path.join(path,'generate')
    df1 = extract_face_id(False, gt, (256,256), 0)
    df2 = extract_face_id(False,generated, (256,256), 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    #df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
    df2 = df2.sort_values(by=['file_name', 'frame_number'])
    assert df1.shape == df2.shape
    scores = []
    for i in range(df1.shape[0]):
        file_name1 = df1['file_name'].iloc[i].split('.')[0]
        file_name2 = df2['file_name'].iloc[i].split('.')[0]
        assert file_name1 == file_name2
        assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
        scores.append(np.sum(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float) ** 2))
    return np.mean(scores)

def cmp_CSIM_corss(path):
    gt = os.path.join(path,'source')
    generated = os.path.join(path,'generate')
    df1 = extract_arcface_id(False, gt, (256,256), 0)
    df2 = extract_arcface_id(False,generated, (256,256), 0)

    df1 = df1.sort_values(by=['file_name', 'frame_number'])
    #df2 = df1.sort_values(by=['file_name', 'frame_number'], ascending=False)
    df2 = df2.sort_values(by=['file_name', 'frame_number'])
    assert df1.shape == df2.shape
    scores = []
    for i in range(df1.shape[0]):
        file_name1 = df1['file_name'].iloc[i].split('.')[0]
        file_name2 = df2['file_name'].iloc[i].split('.')[0]
        assert file_name1 == file_name2
        assert df1['frame_number'].iloc[i] == df2['frame_number'].iloc[i]
        # scores.append(np.sum(np.abs(df1['value'].iloc[i] - df2['value'].iloc[i]).astype(float) ** 2))
        scores.append((np.matmul(df1['value'].iloc[i],df2['value'].iloc[i]))/(np.linalg.norm(df1['value'].iloc[i])*np.linalg.norm(df2['value'].iloc[i])))
    return np.mean(scores)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--in_folder", default="test", help="Folder with images")
    parser.add_argument("--out_file", default="test.pkl", help="Extracted values")
    parser.add_argument("--is_video", dest='is_video', action='store_true', help="If this is a video.")
    parser.add_argument("--column", default=0, type=int, help="Some generation tools stack multiple images together,"
                                                              " the index of the comlumn with right images")
    parser.add_argument("--image_shape", default=(64, 64), type=lambda x: tuple([int(a) for a in x.split(',')]),
                        help="Image shape")

    parser.add_argument("--type", default='body_id', choices=['face_id', 'face_pose', 'body_id', 'body_pose', 'vgg'],
                        help="Type of info to extract")

    args = parser.parse_args()

    func = locals()["extract_" + args.type]
    out_file = args.out_file
    del args.type, args.out_file

    df = func(**vars(args))

    df.to_pickle(out_file)


