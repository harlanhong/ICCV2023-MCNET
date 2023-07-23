from ast import Gt
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
import pdb
import os
from collections import OrderedDict
import modules.losses as losses
import losses
def checkNaN(ret):
    for k in ret:
        if 'visual' in k:
            try:
                if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                    print(f"! [Numerical Error] {k} contains nan or inf.")
            except Exception as e:
                print(e)

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result
        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian
    def hessian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian
    

def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class VallinaGeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params,opt):
        super(VallinaGeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.module.scales
        self.pyramid = ImagePyramide(self.scales, generator.module.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.opt = opt
        self.loss_weights = train_params['loss_weights']
        self.entropyloss = losses.EntropyLoss()
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        if opt.rgbd:
            self.depth_encoder = depth.ResnetEncoder(18, False).cuda()
            self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
            loaded_dict_enc = torch.load(os.path.join(opt.depth_path,'encoder.pth'),map_location='cpu')
            loaded_dict_dec = torch.load(os.path.join(opt.depth_path,'depth.pth'),map_location='cpu')
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
            self.depth_encoder.load_state_dict(filtered_dict_enc)
            self.depth_decoder.load_state_dict(loaded_dict_dec)
            self.set_requires_grad(self.depth_encoder, False) 
            self.set_requires_grad(self.depth_decoder, False) 
            self.depth_decoder.eval()
            self.depth_encoder.eval()
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self, x, weight=None):
        depth_source = None
        depth_driving = None
        
        kp_source = self.kp_extractor(x['source'],isSource=True)
        kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source, driving_depth = depth_driving,driving_image=x['driving'])
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        loss_values = {}
        
        pyramide_real = self.pyramid(x['source'])
        pyramide_generated = self.pyramid(generated['prediction'])
        if self.loss_weights['attn_regular']!=0:
            loss_values["attn_regular"]=self.loss_weights['attn_regular']*self.entropyloss(generated["attn"])
        if self.loss_weights['vq_commit'] != 0:
            loss_values["vq_commit"]=self.loss_weights['vq_commit']*generated["commit_loss"].sum()
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:

            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))

            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total
        if self.loss_weights['feat_consistent']:
            value_total = 0
            if 'visual_value_32' in generated:
                driving_feat = generated['visual_feat_32']
                recon_driving_feat = generated['visual_value_32']
                value_total += self.loss_weights['feat_consistent']*torch.abs(driving_feat.detach()-recon_driving_feat).mean()
            if 'visual_value_64' in generated:
                driving_feat = generated['visual_feat_64']
                recon_driving_feat = generated['visual_value_64']
                value_total += self.loss_weights['feat_consistent']*torch.abs(driving_feat.detach()-recon_driving_feat).mean()
            if 'visual_value_128' in generated:
                driving_feat = generated['visual_feat_128']
                recon_driving_feat = generated['visual_value_128']
                value_total += self.loss_weights['feat_consistent']*torch.abs(driving_feat.detach()-recon_driving_feat).mean()
            if 'visual_value_256' in generated:
                driving_feat = generated['visual_feat_256']
                recon_driving_feat = generated['visual_value_256']
                value_total += self.loss_weights['feat_consistent']*torch.abs(driving_feat.detach()-recon_driving_feat).mean()
            if 'visual_shift_value_64' in generated:
                driving_feat = generated['visual_shift_feat_64']
                recon_driving_feat = generated['visual_shift_value_64']
                value_total += self.loss_weights['feat_consistent']*torch.abs(driving_feat.detach()-recon_driving_feat).mean()
        
            loss_values['feat_consistent'] = value_total
        if self.loss_weights['bi_feat_consistent']:
            driving_feat = generated['visual_feat']
            recon_driving_feat = generated['visual_feat_reconstruction']
            value_total = self.loss_weights['feat_consistent']*(torch.abs(driving_feat.detach()-recon_driving_feat).mean()+torch.abs(recon_driving_feat.detach()-driving_feat).mean())/2
            loss_values['bi_feat_consistent'] = value_total
        return loss_values, generated


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params,opt):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.module.scales
        self.pyramid = ImagePyramide(self.scales, generator.module.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.opt = opt
        self.loss_weights = train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            self.vgg.eval()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        self.gauss_kernel = losses.get_gaussian_kernel(21).cuda()
    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def forward(self, x, weight=1,epoch=0):
        depth_source = None
        depth_driving = None
        
        kp_source = self.kp_extractor(x['source'],isSource=True)
        kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source, driving_depth = depth_driving,driving_image=x['driving'])
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        #Check NaN
        checkNaN(generated)
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:

            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))

            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)
            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['kp_distance']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            source_dist_loss = (-torch.sign((torch.sqrt((sk*sk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            driving_dist_loss = (-torch.sign((torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            # driving_dist_loss = (torch.sign(1-(torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()))+1).mean()
            value_total = self.loss_weights['kp_distance']*(source_dist_loss+driving_dist_loss)
            loss_values['kp_distance'] = value_total


        if self.loss_weights['feat_consistent']:
            value_total = 0
            def l1(target, source):
                if target.shape[2] != source.shape[2] or target.shape[3] != source.shape[3]:
                    target = F.interpolate(target, size=source.shape[2:], mode='bilinear',align_corners=True)
                return torch.abs(target.detach()-source).mean()

            feat_list = generated['feat_list']
            value_list = generated['value_list']
            for feat, value in zip(feat_list,value_list):
                value_total+=self.loss_weights['feat_consistent']*l1(feat,value)
     
            loss_values['feat_consistent'] = value_total
        
 
   
        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.module.scales
        self.pyramid = ImagePyramide(self.scales, generator.module.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values


