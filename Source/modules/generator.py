import torch
from torch import nn
import torch.nn.functional as F
from modules.util import *
from modules.utils2 import *
from modules.dense_motion import DenseMotionNetwork
from modules.flow_util import *

    
class GeneEncoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(GeneEncoder, self).__init__()        
        
        down_blocks = []
        for i in range(num_blocks):
            
            down_blocks.append(DownBlock2d(min(max_features, block_expansion * (2 ** (i + 1))),
                                           min(max_features, block_expansion * (2 ** (i + 2))),
                                           kernel_size=3, padding=1))           
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            out=down_block(outs[-1])
            outs.append(out)      
            
        return outs

    
class SFTLayer1(nn.Module):
    def __init__(self, in_features=256):
        super(SFTLayer1, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_features, in_features, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_features, int(in_features/2), 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_features, in_features, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_features, int(in_features/2), 1)
 
    def forward(self, out_spatial, out_sft, out_same):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(out_spatial), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(out_spatial), 0.1, inplace=True))
        out_sft = out_sft * (scale + 1) + shift
        out = torch.cat([out_same, out_sft], dim=1)
        return out    
    
class SFTLayer(nn.Module):
    def __init__(self, in_features):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_features, in_features, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_features, int(in_features/2), 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_features, in_features, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_features, int(in_features/2), 1)
 
    def forward(self, out_spatial, out_sft, out_same):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(out_spatial), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(out_spatial), 0.1, inplace=True))
        out_sft = out_sft * (scale + 1) + shift
        out = torch.cat([out_same, out_sft], dim=1)
        return out

    

class GeneDecoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(GeneDecoder, self).__init__()

        up_blocks = []
        sft_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 2)))
            out_filters = min(max_features, block_expansion * (2 ** (i + 1)))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
            sft_blocks.append(SFTLayer(out_filters))
            
        self.up_blocks = nn.ModuleList(up_blocks)
        self.sft_blocks = nn.ModuleList(sft_blocks) 
        
        self.out_filters = (block_expansion + in_features)*2
        
        self.down_1_8 = AntiAliasInterpolation2d(1, 0.125).cuda()  
        self.down_2_8 = AntiAliasInterpolation2d(2, 0.125).cuda()           
        self.SFTLayer1=SFTLayer1()
        
        
    def forward(self, x, flow, occlusion):

        flow_blocks = []  
        occlusion_blocks = []
        for i in range(3): 
            self.downflow=AntiAliasInterpolation2d(2, 1/2**(i)).cuda()
            self.occlusion=AntiAliasInterpolation2d(1, 1/2**(i)).cuda()            
            denseflow=self.downflow(flow.permute(0,3,1,2)).permute(0,2,3,1)
            denseocclusion=self.occlusion(occlusion)
            flow_blocks.append(denseflow)     
            occlusion_blocks.append(denseocclusion)
        
        
        out = x.pop()
        
        denseflow_8=self.down_2_8(flow.permute(0,3,1,2)).permute(0,2,3,1)
        denseocclusion_8=self.down_1_8(occlusion)      
        
        out_deform_1 =  F.grid_sample(out, denseflow_8) * denseocclusion_8    
        out_same, out_sft = torch.split(out_deform_1, int(out_deform_1.size(1) // 2), dim=1)
        out=self.SFTLayer1(out, out_sft, out_same)

        for i, up_block in enumerate(self.up_blocks):
            out = up_block(out)
            skip = x.pop()
            flow=flow_blocks.pop()
            occlusion=occlusion_blocks.pop()
            out_deform =  F.grid_sample(skip, flow) * occlusion        
            
            out_same, out_sft = torch.split(out_deform, int(out_deform.size(1) // 2), dim=1)
            out_deform =  self.sft_blocks[i](out, out_sft, out_same)
    
            out = torch.cat([out, out_deform], dim=1)
            
        return out    


class GeneHourglass2d(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion=32, in_features=32, num_blocks=3, max_features=256):
        super(GeneHourglass2d, self).__init__()
        self.geneencoder = GeneEncoder(block_expansion, in_features, num_blocks, max_features)
        self.genedecoder = GeneDecoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.genedecoder.out_filters
        self.final = nn.Conv2d(self.out_filters, 3, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, x, flow, occlusion):
        out=self.geneencoder(x)
        out=self.genedecoder(out, flow, occlusion)
        out = self.final(out)
        out = F.sigmoid(out)
        return out
    


class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, block_expansion, dense_motion_params=None):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(**dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))


        self.GeneHourglass2d=GeneHourglass2d()
        
        self.image_channel=image_channel


    def forward(self, source_image, kp_source, kp_driving):

        
        ################################ generation network  ################################ 
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        output_dict['sourcelmk'] = kp_source['lmk']       
        output_dict['drivinglmk'] = kp_driving['lmk']       

  
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image, kp_driving=kp_driving, kp_source=kp_source)
            
            output_dict['meshflow'] = dense_motion['meshflow']
            output_dict['sparseflow'] = dense_motion['sparseflow']  
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']    

            output_dict['dense_flow_foreground'] = dense_motion['dense_flow_foreground'] 
            output_dict['matting_mask'] = dense_motion['matting_mask'] 
                        
            output_dict['sourcelmk'] = dense_motion['sourcelmk']      
            output_dict['drivinglmk'] = dense_motion['drivinglmk']       
            
            output_dict['eye_mask_source_tensor'] = dense_motion['eye_mask_source_tensor'] 
            output_dict['eye_mask_driving_tensor'] =  dense_motion['eye_mask_driving_tensor']    

            output_dict['eyemask_reconstruct_source_tensor'] = dense_motion['eyemask_reconstruct_source_tensor'] 
            output_dict['eyemask_reconstruct_driving_tensor'] = dense_motion['eyemask_reconstruct_driving_tensor']      
                    
            
        # # Encoding (downsampling) part
        out=self.first(source_image)
        out=self.GeneHourglass2d(out, dense_motion['dense_flow_foreground'], dense_motion['matting_mask'])
        output_dict['prediction'] = out     
        return output_dict    