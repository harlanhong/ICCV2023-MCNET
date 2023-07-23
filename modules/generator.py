from curses import keyname
from curses.ascii import BS
from pickle import TRUE
import torch
from torch import nn
import torch.nn.functional as F
from modules.util import *
from modules.dense_motion import *
import pdb
from stylegan2.stylegan2_arch import ModulatedConv2d,StyleConv
from modules.dynamic_conv import Dynamic_conv2d,Multi_Dynamic_conv2d,Dynamic_conv2d_conditional
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class SelfAttention(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,source,feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64
                driving : input feature maps( B X C X W X H) 256,64,64
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = source.size()
        proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
        proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out

        return out,attention     

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class MemoryUnit(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,spatial_dim, channel_dim):
        super(MemoryUnit,self).__init__()
        self.spatial_dim = spatial_dim
        self.channel_dim = channel_dim
        self.mb = torch.nn.Parameter(torch.randn(1,channel_dim,spatial_dim,spatial_dim))
        # self.mb_conv = nn.Conv2d(in_channels = int(channel_dim/4), out_channels = channel_dim , kernel_size= 3,padding=1)
        
        nn.init.kaiming_uniform_(self.mb, mode='fan_in', nonlinearity='relu')

        self.query_conv = nn.Conv2d(in_channels = channel_dim , out_channels = channel_dim , kernel_size= 1,padding=0)

        self.conv_kv =  Dynamic_conv2d(in_planes=channel_dim, out_planes=channel_dim*2, kernel_size=3, ratio=0.25, padding=1,groups=1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.to_embedding= Rearrange('b c w h -> b (w h) c')
        self.to_map= Rearrange('b (w h) c -> b c w h', w=spatial_dim)
        self.softmax  = nn.Softmax(dim=1) #
        # self.norm = BatchNorm2d(self.channel_dim, affine=True)
    def forward(self,feature,output_dict):
        """
            inputs :
                feature : input feature maps( B X C X W X H)
                res_dict : dictionary
            returns :
                out : self attention value
        """
        expand_mb=self.mb.expand_as(feature).contiguous()
        key,value = self.conv_kv(feature, expand_mb).chunk(2, dim = 1) #(bs,c,w,h)
        output_dict['visual_feat_{}'.format(self.spatial_dim)] = feature
        query = self.query_conv(feature)
        output_dict['visual_value_{}'.format(self.spatial_dim)] = value
        output_dict['visual_key_{}'.format(self.spatial_dim)] = key
        output_dict['visual_query_{}'.format(self.spatial_dim)] = query
        output_dict['visual_memory_{}'.format(self.spatial_dim)] = self.mb

        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/4,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v)
        residual =  self.to_map(residual)
        # residual = self.norm(residual)
        output_dict['visual_residual_{}'.format(self.spatial_dim)] = residual
        return  residual

class NormMemoryUnit(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,spatial_dim, channel_dim):
        super(NormMemoryUnit,self).__init__()
        self.spatial_dim = spatial_dim
        self.channel_dim = channel_dim
        self.mb = torch.nn.Parameter(torch.randn(1,channel_dim,spatial_dim,spatial_dim))
        # self.mb_conv = nn.Conv2d(in_channels = int(channel_dim/4), out_channels = channel_dim , kernel_size= 3,padding=1)
        
        nn.init.kaiming_uniform_(self.mb, mode='fan_in', nonlinearity='relu')

        self.query_conv = nn.Conv2d(in_channels = channel_dim , out_channels = channel_dim , kernel_size= 1,padding=0)

        self.conv_kv =  Dynamic_conv2d(in_planes=channel_dim, out_planes=channel_dim*2, kernel_size=3, ratio=0.25, padding=1,groups=1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.to_embedding= Rearrange('b c w h -> b (w h) c')
        self.to_map= Rearrange('b (w h) c -> b c w h', w=spatial_dim)
        self.softmax  = nn.Softmax(dim=1) #
        # self.norm = BatchNorm2d(self.channel_dim, affine=True)
        self.conv = nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=3,
                              padding=1, groups=1)
        self.norm = nn.InstanceNorm2d(channel_dim, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
    def forward(self,feature,output_dict):
        """
            inputs :
                feature : input feature maps( B X C X W X H)
                res_dict : dictionary
            returns :
                out : self attention value
        """
        expand_mb=self.mb.expand_as(feature).contiguous()
        key,value = self.conv_kv(feature, expand_mb).chunk(2, dim = 1) #(bs,c,w,h)
        output_dict['visual_feat_{}'.format(self.spatial_dim)] = feature
        query = self.query_conv(feature)
        output_dict['visual_value_{}'.format(self.spatial_dim)] = value
        output_dict['visual_key_{}'.format(self.spatial_dim)] = key
        output_dict['visual_query_{}'.format(self.spatial_dim)] = query
        output_dict['visual_memory_{}'.format(self.spatial_dim)] = self.mb

        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/4,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v)
        residual =  self.to_map(residual)
        # residual = self.norm(residual)
        residual = self.conv(residual)
        residual = self.norm(residual)
        residual = F.relu(residual)
        output_dict['visual_residual_{}'.format(self.spatial_dim)] = residual
        return  residual


#channel cross split + pct diff + residual, 19(relu after qkv)+ feature aware mb，(在39的基础上去掉res block) + keypoint
class ExpendMemoryUnit(nn.Module):
    """ depth-aware attention Layer"""
    def __init__(self,spatial_dim, channel_dim, half=True):
        super(ExpendMemoryUnit,self).__init__()
        self.scale=1 if not half else 2
        self.spatial_dim = spatial_dim//self.scale
        self.channel_dim = channel_dim
        self.channel_32 = 512//self.scale
        self.channel_64 = 256//self.scale
        self.channel_128 = 128//self.scale
        self.mb = torch.nn.Parameter(torch.randn(1,channel_dim,spatial_dim,spatial_dim))
        # self.mb_conv = nn.Conv2d(in_channels = int(channel_dim/4), out_channels = channel_dim , kernel_size= 3,padding=1)
        nn.init.kaiming_uniform_(self.mb, mode='fan_in', nonlinearity='relu')

        self.feat_forward_proj_32 = nn.Conv2d(in_channels = self.channel_32, out_channels = channel_dim , kernel_size= 1,padding=0)
        self.feat_forward_proj_64 = nn.Conv2d(in_channels = self.channel_64, out_channels = channel_dim , kernel_size= 1,padding=0)
        self.feat_forward_proj_128 = nn.Conv2d(in_channels = self.channel_128, out_channels = channel_dim , kernel_size= 1,padding=0)
        
        self.query_conv = nn.Conv2d(in_channels = channel_dim , out_channels = channel_dim , kernel_size= 1,padding=0)

        self.conv_kv =  Dynamic_conv2d(in_planes=channel_dim, out_planes=channel_dim*2, kernel_size=3, ratio=0.25, padding=1,groups=1)
        
        self.diff_conv = nn.Conv2d(in_channels = channel_dim , out_channels = channel_dim , kernel_size= 1,padding=0)

        self.mb_window = spatial_dim//4
        # self.feat_window = conditional_spatial//4
        self.to_embedding= Rearrange('b c w h -> b (w h) c')
        self.to_map= Rearrange('b (w h) c -> b c w h', w=spatial_dim)
        self.conv_32 = nn.Conv2d(in_channels=channel_dim, out_channels=self.channel_32, kernel_size=3,
                              padding=1, groups=1)
        self.norm_32 = nn.InstanceNorm2d(self.channel_32, affine=True)
        self.conv_64 = nn.Conv2d(in_channels=channel_dim, out_channels=self.channel_64, kernel_size=3,
                              padding=1, groups=1)
        self.norm_64 = nn.InstanceNorm2d(self.channel_64, affine=True)
        self.conv_128 = nn.Conv2d(in_channels=channel_dim, out_channels=self.channel_128, kernel_size=3,
                              padding=1, groups=1)
        self.norm_128 = nn.InstanceNorm2d(self.channel_128, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        self.style_route = nn.Sequential(
            nn.Linear(channel_dim+30, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.styleconv = StyleConv(
                    channel_dim,
                    channel_dim,
                    kernel_size=3,
                    num_style_feat=512,
                    demodulate=True,
                    resample_kernel=(1, 3, 3, 1),
                )
    def forward(self,feature,output_dict,**args):
        """
            inputs :
                feature : input feature maps( B X C X W X H)
                res_dict : dictionary
            returns :
                out : self attention value
        """
        if self.scale==2:
            out_same, out_cs = feature[:,::2,...],feature[:,1::2,...]
            # out_same, out_cs = torch.split(feature, int(feature.size(1) // 2), dim=1)
        bs,c,w,h = out_cs.shape
        mb_bs,mb_c,mb_w,mb_h = self.mb.shape

        feat = eval('self.feat_forward_proj_{}'.format(w))(out_cs)
        scale_code = self.avg_pool(feat).view(bs,-1)
        keypoints = args['keypoints'].view(bs,-1)
        scale_code = torch.cat((scale_code,keypoints),-1)
        style_code = self.style_route(scale_code)
        expand_mb=self.mb.expand(bs,mb_c,mb_w,mb_h).contiguous()
        expand_mb = self.styleconv(expand_mb,style_code)
        # expand_mb=aware_mb.expand(bs,mb_c,mb_w,mb_h).contiguous()
        key,value = F.relu(self.conv_kv(feat, expand_mb)).chunk(2, dim = 1) #(bs,c,w,h)
        query = F.relu(self.query_conv(feat))
        
        if 'feat_list' in output_dict:
            output_dict['feat_list'].append(feat)
        else:
            output_dict['feat_list'] = [feat]
        if 'value_list' in output_dict:
            output_dict['value_list'].append(value)
        else:
            output_dict['value_list']=[value]
        output_dict['visual_before_{}'.format(w)] = feature
        output_dict['visual_feat_{}'.format(w)] = feat
        output_dict['visual_value_{}'.format(w)] = value
        output_dict['visual_key_{}'.format(w)] = key
        output_dict['visual_query_{}'.format(w)] = query
        output_dict['visual_memory_{}'.format(w)] = expand_mb

        # value_windows = self.window_partition(value.permute(0,2,3,1).contiguous(), self.mb_window)  # nW*B, window_size, window_size, C
        # query_windows = self.window_partition(query.permute(0,2,3,1).contiguous(), w//4)  # nW*B, window_size, window_size, C
        # key_windows = self.window_partition(key.permute(0,2,3,1).contiguous(),self.mb_window)  # nW*B, window_size, window_size, C

        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/4,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v)
        residual = rearrange(residual, 'b (x y) c -> b c x y', x=w,y=h)
        # residual = self.window_reverse(residual, w//4, w, w).permute(0,3,1,2).contiguous()
        output_dict['visual_residual_{}'.format(w)] = residual
        diff = F.relu(self.diff_conv(feat-residual))
        out = diff+feat
        out = eval('self.conv_{}'.format(w))(out)
        out = eval('self.norm_{}'.format(w))(out)
        out = F.relu(out)
        out = torch.stack((out_same, out), dim=2)
        out = out.view(bs,c*2,h,w)
        output_dict['visual_after_{}'.format(w)] = out
        return  out

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows


    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def return_latent(self):
        latent_code = F.adaptive_avg_pool2d(self.mb,(4,4))
        return latent_code

class Unet_Generator_keypoint_aware(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False, **kwargs):
        super(Unet_Generator_keypoint_aware, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        self.num_down_blocks = num_down_blocks
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = [max_features, max_features, max_features//2]
        out_features = [max_features//2, max_features//4, max_features//8]
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        # for i in range(num_bottleneck_blocks):
        #     self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        # Memory Bank
        self.mbUnit = eval(kwargs['mbunit'])(kwargs['mb_spatial'],kwargs['mb_channel'])

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation,align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, kp_driving, kp_source, **kwargs):
        # Encoding (downsampling) part
        out = self.first(source_image)
        encoder_map = [out]

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)
        output_dict = {}

        # feat_driving = self.first(kwargs['driving_image'])
        # for i in range(len(self.down_blocks)):
        #     feat_driving = self.down_blocks[i](feat_driving)
        # output_dict['feat_driving']=feat_driving
        
        # Transforming feature representation according to deformation and occlusion
        dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                    kp_source=kp_source)
        output_dict['mask'] = dense_motion['mask']
        output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

        if 'occlusion_map' in dense_motion:
            occlusion_map = dense_motion['occlusion_map']
            output_dict['occlusion_map'] = occlusion_map
        else:
            occlusion_map = None
        deformation = dense_motion['deformation']
        output_dict['deformation'] = deformation
        out_ij = self.deform_input(out.detach(), deformation)
        out = self.deform_input(out, deformation)

        out_ij = self.occlude_input(out_ij, occlusion_map.detach())
        out = self.occlude_input(out, occlusion_map)
        output_dict["visual_encode_ij"] = out_ij
        output_dict["Fwarp"] = out

        warped_encoder_maps = []
        
        # output_dict["recon_Fwarp_feat"] = out
        # Decoding part
        # out = self.mb_smooth(out)
        warped_encoder_maps.append(out)
        out = self.mbUnit(out,output_dict,keypoints = kp_source['value'])
        output_dict['visual_reconstruction'] = out
        for i in range(self.num_down_blocks):
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            out = self.up_blocks[i](out)
            
            encode_i = encoder_map[-(i+2)]
            encode_ij = self.deform_input(encode_i.detach(), deformation)
            encode_i = self.deform_input(encode_i, deformation)
            
            occlusion_ind = 0
            # if self.multi_mask:
            # occlusion_ind = i+1
            encode_ij = self.occlude_input(encode_ij, occlusion_map.detach())
            encode_i = self.occlude_input(encode_i, occlusion_map)
            output_dict["visual_encode_{}".format(i)] = encode_ij
            warped_encoder_maps.append(encode_i)
            if(i==self.num_down_blocks-1):
                break
            encode_i = self.mbUnit(encode_i,output_dict,keypoints = kp_source['value'])
            out = torch.cat([out, encode_i], 1)

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source
        output_dict["warped_encoder_maps"] = warped_encoder_maps

        occlusion_last = occlusion_map
        occlusion_last = F.interpolate(occlusion_last, size=out.shape[2:], mode='bilinear',align_corners=True)

        out = out * (1 - occlusion_last) + encode_i
        out = self.final(out)
        out = torch.sigmoid(out)
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out

        return output_dict
    def get_encode(self, driver_image, occlusion_map):
        out = self.first(driver_image)
        encoder_map = []
        encoder_map.append(out)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out.detach())
            # out_mask = self.occlude_input(out.detach(), occlusion_map.detach())
            encoder_map.append(out.detach())

        return encoder_map
