from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid_2d
from torchvision import models
import numpy as np
from torch.autograd import grad
from torchvision import transforms
import modules.arcfacenet as arcfacenet

from modules.spynet import *
import os
import argparse

from modules.wm3dr.lib.decode import decode
from modules.wm3dr.lib.model import *
from modules.wm3dr.lib.pt_renderer import PtRender
from modules.wm3dr.lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  #construct_meshes,
)

from modules.flow_util import *


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
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
    Create image pyramide for computing pyramide perceptual loss.
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
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
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



def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--arch', default='resnet_50', type=str)
    parser.add_argument('--load_model', default='./modules/wm3dr/model/final.pth', type=str)
    parser.add_argument('--BFM', default='./BFM/mSEmTFK68etc.chj', type=str)    

    return parser.parse_args()


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):        
        super(GeneratorFullModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.scale_factor = train_params['scale_factor']        
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        self.num_channels=3
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']


        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(self.num_channels, self.scale_factor).cuda()        
            self.down1 = AntiAliasInterpolation2d(self.num_channels, 120/256).cuda()  
        
        
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
                
        if sum(self.loss_weights['face_features']) != 0:
            self.arcfacenet = arcfacenet.iresnet50()
            arcfacenet_state_dict = torch.load(train_params['arcfacenet'])
            self.arcfacenet.load_state_dict(arcfacenet_state_dict)
            if torch.cuda.is_available():
                self.arcfacenet = self.arcfacenet.cuda()
                self.arcfacenet.eval()
            print('Loading arcfacenet')

        
        self.num_kp=train_params['num_kp'] 


    def source_mesh_reconstruct(self, sourceimg):

        opt = opts()

        render = PtRender(opt).cuda().eval()
        opt.heads = {'hm': 1, 'params': 257}
        model = create_model(opt.arch, opt.heads)
        facemodel = loadmodel(model, opt.load_model)
        facemodel.cuda().eval()

        output, topk_scores, topk_inds, topk_ys, topk_xs = decode(sourceimg, facemodel)
      
        params = _tranpose_and_gather_feat(output['params'], topk_inds)

        id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))

        render.set_RotTransLight(angles, translation, topk_inds.view(-1)) 

        # reconstruct shape
        canoShape_ = render.Shape_formation(id_coeff, ex_coeff)     #################
        rotShape = render.RotTrans(canoShape_)

        projection_shape, landmark = render.get_Landmarks(rotShape)
        
        return {'mesh': projection_shape, 'lmk':landmark, 'id_coeff': id_coeff}             
        
        
    def driving_parameter_extraction(self, inputimg):
       
        opt = opts()

        render = PtRender(opt).cuda().eval()
        opt.heads = {'hm': 1, 'params': 257}
        model = create_model(opt.arch, opt.heads)
        facemodel = loadmodel(model, opt.load_model)
        facemodel.cuda().eval()

        output, topk_scores, topk_inds, topk_ys, topk_xs = decode(inputimg, facemodel)
      
        params = _tranpose_and_gather_feat(output['params'], topk_inds)

        id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))
 
        return {'ex_coeff':ex_coeff, 'angles':angles, 'translation':translation, 'topk_inds': topk_inds}        
        
        
    def driving_mesh_reconstruct(self, id_coeff, ex_coeff, angles, translation, topk_inds):
       
        opt = opts()
        render = PtRender(opt).cuda().eval()    
        
        render.set_RotTransLight(angles, translation, topk_inds.view(-1)) 

        # reconstruct shape
        canoShape_ = render.Shape_formation(id_coeff, ex_coeff)     #################
        rotShape = render.RotTrans(canoShape_)

        projection_shape, landmark = render.get_Landmarks(rotShape)

        return {'mesh':projection_shape, 'lmk':landmark}  


    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2) 
        gram = features.bmm(features_t) / (c * h * w)   
        return gram



    def forward(self, x):

        rendered_source=x['rendered_source'] 
        rendered_driving=x['rendered_driving'] 

        eyemask_source=x['eyemask_source'] 
        eyemask_driving=x['eyemask_driving'] 
        
        source_reconstruction=self.source_mesh_reconstruct(x['source'])
        
        
        kp_driving =self.driving_parameter_extraction(x['driving'])   
        
        id_coeff=source_reconstruction['id_coeff']
        driving_ex_coeff, driving_angles = kp_driving['ex_coeff'], kp_driving['angles']
        driving_translation, driving_topk_inds = kp_driving['translation'], kp_driving['topk_inds']
        
        driving_reconstruction=self.driving_mesh_reconstruct(id_coeff,driving_ex_coeff, driving_angles,
                                                             driving_translation, driving_topk_inds)
        
        
        
        source_reconstruction.update({ 'rendered_source':rendered_source, 'eyemask_source':eyemask_source}) 
        driving_reconstruction.update({'rendered_driving':rendered_driving,'eyemask_driving':eyemask_driving}) 
        
        generated = self.generator(x['source'], kp_source=source_reconstruction, kp_driving=driving_reconstruction)
        generated.update({'kp_source': source_reconstruction, 'kp_driving': driving_reconstruction})        
        

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        sparse_pyramide_generated = self.pyramid(generated['sparse_deformed'])            

        driving_image_downsample = self.down(x['driving'])         
        source_image_downsample = self.down(x['source']) 
        
        # #################       
        optical_flow_generated=generated['dense_flow_foreground'].to(device)  
        optical_flow_generated_initial=generated['sparseflow'].to(device) 

        optical_flow_real= spynet_estimate(x['source'], x['driving']).to(device)        
        optical_flow_real = convert_flow_to_deformation(optical_flow_real)  
                
      
        ### Perceptual Loss---Initial
        if sum(self.loss_weights['perceptual_initial']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(sparse_pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value

            loss_values['initial_vggper'] = value_total   
                     
           
        
        ### GAN Loss---Initial
        if self.loss_weights['generator_gan_initial'] != 0:
            discriminator_maps_sparse_generated = self.discriminator(sparse_pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_sparse_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_sparse_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan_initial'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_sparse_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching_initial'] = value_total        
        
 ##############################################################################################################################################       
           #### optical flow loss 
        if self.loss_weights['optical_flow'] != 0:
            value = torch.abs(optical_flow_real.to(device).detach()-optical_flow_generated.to(device).detach()).mean()
            value = value.requires_grad_()
            value_total = self.loss_weights['optical_flow'] * value            
            loss_values['flow'] = value_total   
            
        ### Perceptual Loss---Final            
        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['vgg'] = value_total


        ### face Indentity Loss---Final            
        if sum(self.loss_weights['face_features']) != 0:
            value_total = 0
            for scale in self.scales:           
                pyramide_generated_face = (pyramide_generated['prediction_' + str(scale)] - 0.5) / 0.5
                pyramide_real_face = (pyramide_real['prediction_' + str(scale)] - 0.5) / 0.5
                x_facefeature = self.arcfacenet(pyramide_generated_face)
                y_facefeature = self.arcfacenet(pyramide_real_face)

                for i, weight in enumerate(self.loss_weights['face_features']):
                    value = torch.abs(x_facefeature[i] - y_facefeature[i].detach()).mean()
                    value_total += self.loss_weights['face_features'][i] * value
            loss_values['indentity'] = value_total
            
            
        ### Texture Loss---Final            
        if sum(self.loss_weights['texture']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['texture']):
                    x_gram=self.gram_matrix(x_vgg[i])
                    y_gram=self.gram_matrix(y_vgg[i])
                    value = torch.mean((x_gram - y_gram)**2)
                    value_total += (self.loss_weights['texture'][i] * value) 
            loss_values['texture'] = value_total   
            
        ### Pixel Loss---Final      
        if self.loss_weights['pixelrec'] != 0:
            pixelwise_loss = nn.L1Loss(reduction='mean')  #平均绝对误差,L1-损失
            
            value = pixelwise_loss(x['driving'], generated['prediction']).mean()
            value_total = self.loss_weights['pixelrec'] * value      

            loss_values['pixel'] = value_total                  
            
            
        ### GAN Loss---Final            
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

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


                       
        return loss_values, generated


    
    
    
    
class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        sparse_pyramide_generated = self.pyramid(generated['sparse_deformed'].detach())
        
        discriminator_maps_real = self.discriminator(pyramide_real)
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_sparse_generated = self.discriminator(sparse_pyramide_generated)
        

        loss_values = {}


        if self.loss_weights['discriminator_gan'] != 0:    
            value_total = 0
            for scale in self.scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['discriminator_gan'] * value
            loss_values['disc_gan'] = value_total

        return loss_values
