from torch import nn
import torch.nn.functional as F
import torch
from modules.util import *
from modules.utils2 import *
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from scipy.interpolate import griddata
from modules.flowwarp import *
from modules.flow_util import *

import cv2
from PIL import Image
from skimage.draw import circle, ellipse
from skimage.draw import line
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread

import cv2, random
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import imghdr

import time

        
class SPADE_Flow(nn.Module):
    def __init__(self):
        super(SPADE_Flow, self).__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.first = SameBlock2d(12, 32, kernel_size=(7, 7), padding=(3, 3))
        
        meshmotiondown_blocks = []
        num_down_blocks=3
        max_features=512
        block_expansion=32
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            meshmotiondown_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motiondown_blocks = nn.ModuleList(meshmotiondown_blocks)        
        
        self.seg = DownBlock2d(ic+2, ic, kernel_size=(3, 3), padding=(1, 1))
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)        
        self.up = nn.Upsample(scale_factor=2)

        self.down_2_2 = AntiAliasInterpolation2d(2, 0.5).cuda()  
        self.down_3_4 = AntiAliasInterpolation2d(3, 0.25).cuda()  
        self.down_2_8 = AntiAliasInterpolation2d(2, 0.125).cuda()  
        
        self.meshflow = nn.Conv2d(64, 2, kernel_size=(7, 7), padding=(3, 3))          
        self.down_sample_flow = AntiAliasInterpolation2d(2, 0.25).cuda()  
        
        self.meshocclusion = nn.Conv2d(64, 1, kernel_size=(7, 7), padding=(3, 3))            
        self.down_sample_occlusion = AntiAliasInterpolation2d(1, 0.25).cuda()          
        self.sigmoid = nn.Sigmoid()       
                
        
    def forward(self, source_image,source_eye_mask, deformed_image,driving_eye_mask, sparseflow):
        
        
        input=torch.cat([source_image,source_eye_mask, deformed_image,driving_eye_mask], dim=1)    
        
        out=self.first(input)
        #print(out.shape)
        
        for i in range(3):
            out = self.motiondown_blocks[i](out)         
        
        sparseflow=self.down_2_8(sparseflow)
        
        seg = torch.cat([out,sparseflow], dim=1) 
        seg = self.seg(seg)
                
        x = self.fc(out)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.up(x)   
        x = self.up_0(x, seg)   
        x = self.up(x)    
        x = self.up_1(x, seg)    
        x = self.up(x)     
        
        flow=convert_flow_to_deformation(self.meshflow(x))
        occlusion=self.sigmoid(self.meshocclusion(x))
        
        return occlusion, flow



class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, num_channels, num_down_blocks, num_bottleneck_blocks, max_features):
        super(DenseMotionNetwork, self).__init__()
 
        self.flowprediction=SPADE_Flow()

            
        self.num_down_blocks=num_down_blocks
        self.num_channels = num_channels
          

        self.down_sample_image = AntiAliasInterpolation2d(3, 0.25)    
        self.down_sample_flow = AntiAliasInterpolation2d(2, 0.25) 
        

     
        # deform model        
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        
        motiondown_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            motiondown_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motiondown_blocks = nn.ModuleList(motiondown_blocks)

        motionup_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            motionup_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.motionup_blocks = nn.ModuleList(motionup_blocks)    

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))        
        
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
 


    
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)  #########    
    
    def create_deformed_source_image(self, source_image, sparse_motion):
        
        out = self.first(source_image)  
        
        for i in range(self.num_down_blocks):
            out = self.motiondown_blocks[i](out) 
        
        #out=warp(out, sparse_motion)
        out=self.deform_input(out, sparse_motion)
        
        out = self.bottleneck(out)
        
        for i in range(self.num_down_blocks):
            out = self.motionup_blocks[i](out)

        out = self.final(out)  
        return out

    def make_coordinate_grid_2d(self, image_size):
        h, w = image_size
        x = np.arange(w)
        y = np.arange(h)
        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)
        xx = x.reshape(1, -1).repeat(h, axis=0)
        yy = y.reshape(-1, 1).repeat(w, axis=1)
        meshed = np.stack([xx, yy], 2)
        return meshed

    def make_coordinate_grid_torch(self, spatial_size, type):
        h, w = spatial_size
        x = torch.arange(w).type(type)
        y = torch.arange(h).type(type)
        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)
        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)
        meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
        return meshed    
    
      
    
    def construct_Fapp(self, reference_projected_mesh_points, drive_projected_mesh_points, image_size):
        '''
        compute Fapp from projected mesh points
        reference_projected_mesh_points: the projected mesh points of reference image
        drive_projected_mesh_points: the driving projected mesh points
        '''
        ## resize to -1 ~ 1
        reference_projected_mesh_points = (reference_projected_mesh_points / (image_size - 1) * 2) - 1
 
        ### compute the max heigh of face
        face_max_h = np.max(drive_projected_mesh_points[:, 1]).astype(np.int)
        ## resize to -1 ~ 1
        drive_projected_mesh_points = (drive_projected_mesh_points / (image_size - 1) * 2) - 1
      
        drive_projected_mesh_points_yx = drive_projected_mesh_points[:, [1, 0]]

        ## compute sparse dense flow
        sparse_dense_flow = reference_projected_mesh_points - drive_projected_mesh_points

        ## compute average head motion
        mean_dense_flow = np.mean(sparse_dense_flow, axis=0) #np.mean(sparse_dense_flow, axis=0)
        ## compute the dense flow in head-related region
        grid_nums = complex(str(image_size) + "j")
        grid_y, grid_x = np.mgrid[-1:1:grid_nums, -1:1:grid_nums]
        dense_foreground_flow_x = griddata(drive_projected_mesh_points_yx, sparse_dense_flow[:, 0], (grid_y, grid_x), method='nearest')
        dense_foreground_flow_y = griddata(drive_projected_mesh_points_yx, sparse_dense_flow[:, 1], (grid_y, grid_x), method='nearest')
        
        ## compute dense flow in torso related region
        dense_foreground_flow_x[face_max_h:, :] = mean_dense_flow[0]
        dense_foreground_flow_y[face_max_h:, :] = mean_dense_flow[1]
        Fapp = np.stack([dense_foreground_flow_x, dense_foreground_flow_y], 2)

        return Fapp
    

    
    def eye_mask_get(self, lmk_driving, lmk_source, bs):

        eye_mask_driving=[]
        for batch in range(bs):
            background_img_driving = np.zeros((256, 256,3), dtype=np.uint8)   
            ##left eye region
            for n in range(36,42):
                x= lmk_driving[batch][n][0].data.cpu().numpy()
                y = lmk_driving[batch][n][1].data.cpu().numpy()
                next_point = n+1
                if n==41:
                    next_point = 36 

                x2 = lmk_driving[batch][next_point][0].data.cpu().numpy()
                y2 = lmk_driving[batch][next_point][1].data.cpu().numpy()
                cv2.ocl.setUseOpenCL(False)  
                cv2.setNumThreads(0)  
                cv2.line(background_img_driving,(int(x),int(y)),(int(x2),int(y2)),(255,255,255),6)  

            ###right eye region
            for nn in range(42,48):
                x_right= lmk_driving[batch][nn][0].data.cpu().numpy()
                y_right = lmk_driving[batch][nn][1].data.cpu().numpy()
                next_point_right = nn+1
                if nn==47:
                    next_point_right = 42 

                x2_right = lmk_driving[batch][next_point_right][0].data.cpu().numpy()
                y2_right = lmk_driving[batch][next_point_right][1].data.cpu().numpy()
                cv2.ocl.setUseOpenCL(False)   
                cv2.setNumThreads(0)  
                cv2.line(background_img_driving,(int(x_right),int(y_right)),(int(x2_right),int(y2_right)),(255,255,255),6) 
            
            background_img_driving=background_img_driving/255
            eye_mask_driving.append(background_img_driving)

            
        eye_mask_driving_tensor=torch.from_numpy(np.array(eye_mask_driving)).cuda()

        eye_mask_source=[]
        for batch in range(bs):
            
            background_img_source = np.zeros((256, 256,3), dtype=np.uint8)   
            
            ##left eye region
            for n in range(36,42):
                x= lmk_source[batch][n][0].data.cpu().numpy()
                y = lmk_source[batch][n][1].data.cpu().numpy()
                next_point = n+1
                if n==41:
                    next_point = 36 

                x2 = lmk_source[batch][next_point][0].data.cpu().numpy()
                y2 = lmk_source[batch][next_point][1].data.cpu().numpy()
                cv2.ocl.setUseOpenCL(False)   
                cv2.setNumThreads(0)  
                cv2.line(background_img_source,(int(x),int(y)),(int(x2),int(y2)),(255,255,255),6)  

            ###right eye region
            for nn in range(42,48):
                x_right= lmk_source[batch][nn][0].data.cpu().numpy()
                y_right = lmk_source[batch][nn][1].data.cpu().numpy()
                next_point_right = nn+1
                if nn==47:
                    next_point_right = 42 

                x2_right = lmk_source[batch][next_point_right][0].data.cpu().numpy()
                y2_right = lmk_source[batch][next_point_right][1].data.cpu().numpy()
                cv2.ocl.setUseOpenCL(False)  
                cv2.setNumThreads(0)  
                cv2.line(background_img_source,(int(x_right),int(y_right)),(int(x2_right),int(y2_right)),(255,255,255),6) 

            background_img_source=background_img_source/255
            eye_mask_source.append(background_img_source)
            time.sleep(0.003)
            
        eye_mask_source_tensor=torch.from_numpy(np.array(eye_mask_source)).cuda()
        return eye_mask_driving_tensor, eye_mask_source_tensor
    
    
    
    
    
    def rendered_reconstruct(self, eye_mask_source_tensor, lmk_source, source_au45, bs ):

        eyemask_reconstruct_source=[]           
        for batch in range(bs):
            eye_mask_source_numpy=eye_mask_source_tensor[batch].data.cpu().numpy()            
            eye_mask_source_numpy = (255 * eye_mask_source_numpy).astype(np.uint8)            

#           ##left eye region
            source_36_eye_left_x= lmk_source[batch][36][0].data.cpu().numpy()
            source_36_eye_left_y = lmk_source[batch][36][1].data.cpu().numpy()        
            source_37_eye_left_x= lmk_source[batch][37][0].data.cpu().numpy() ###
            source_37_eye_left_y = lmk_source[batch][37][1].data.cpu().numpy() ###               
            source_38_eye_left_x= lmk_source[batch][38][0].data.cpu().numpy() ###  
            source_38_eye_left_y = lmk_source[batch][38][1].data.cpu().numpy()  ###                
            source_39_eye_left_x= lmk_source[batch][39][0].data.cpu().numpy()
            source_39_eye_left_y = lmk_source[batch][39][1].data.cpu().numpy() 
            source_40_eye_left_x= lmk_source[batch][40][0].data.cpu().numpy() ###  
            source_40_eye_left_y = lmk_source[batch][40][1].data.cpu().numpy() ###   
            source_41_eye_left_x= lmk_source[batch][41][0].data.cpu().numpy() ##
            source_41_eye_left_y = lmk_source[batch][41][1].data.cpu().numpy() ###

            ##right eye region
            source_42_eye_right_x= lmk_source[batch][42][0].data.cpu().numpy()
            source_42_eye_right_y = lmk_source[batch][42][1].data.cpu().numpy()        
            source_43_eye_right_x= lmk_source[batch][43][0].data.cpu().numpy() ###
            source_43_eye_right_y = lmk_source[batch][43][1].data.cpu().numpy() ###               
            source_44_eye_right_x= lmk_source[batch][44][0].data.cpu().numpy() ###  
            source_44_eye_right_y = lmk_source[batch][44][1].data.cpu().numpy()  ###                
            source_45_eye_right_x= lmk_source[batch][45][0].data.cpu().numpy()
            source_45_eye_right_y = lmk_source[batch][45][1].data.cpu().numpy() 
            source_46_eye_right_x= lmk_source[batch][46][0].data.cpu().numpy() ###  
            source_46_eye_right_y = lmk_source[batch][46][1].data.cpu().numpy() ###   
            source_47_eye_right_x= lmk_source[batch][47][0].data.cpu().numpy() ##
            source_47_eye_right_y = lmk_source[batch][47][1].data.cpu().numpy() ###

            ##3 retarget the eye region
            source_au45_bs=source_au45[batch][0].data.cpu().numpy()       
            indensity=source_au45_bs/5

            if source_au45_bs>= 0.2:
                source_37_eye_left_y_new = source_37_eye_left_y + (source_41_eye_left_y-source_37_eye_left_y)*indensity
                source_38_eye_left_y_new = source_38_eye_left_y + (source_40_eye_left_y-source_38_eye_left_y)*indensity
                source_43_eye_right_y_new = source_43_eye_right_y + (source_47_eye_right_y-source_43_eye_right_y)*indensity
                source_44_eye_right_y_new = source_44_eye_right_y + (source_46_eye_right_y-source_44_eye_right_y)*indensity                
                source_gaze_circle_left_x=(source_37_eye_left_x+source_38_eye_left_x)/2
                source_gaze_circle_left_y= (source_37_eye_left_y_new+source_41_eye_left_y)/2            
                source_gaze_circle_right_x= (source_44_eye_right_x+source_43_eye_right_x)/2
                source_gaze_circle_right_y= (source_47_eye_right_y+source_43_eye_right_y_new)/2
                source_gaze_radius= (source_47_eye_right_y-source_43_eye_right_y_new)/2
                
                ### rendered eye mask retarget   
                cv2.ocl.setUseOpenCL(False)   
                cv2.setNumThreads(0)  
                cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(255,0,0),5)  
                cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(255,0,0),5)   
                
                cv2.line(eye_mask_source_numpy,(int(source_36_eye_left_x),int(source_36_eye_left_y)),(int(source_37_eye_left_x),int(source_37_eye_left_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_37_eye_left_x),int(source_37_eye_left_y)),(int(source_38_eye_left_x),int(source_38_eye_left_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_38_eye_left_x),int(source_38_eye_left_y)),(int(source_39_eye_left_x),int(source_39_eye_left_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_39_eye_left_x),int(source_39_eye_left_y)),(int(source_38_eye_left_x),int(source_38_eye_left_y_new)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_38_eye_left_x),int(source_38_eye_left_y_new)),(int(source_37_eye_left_x),int(source_37_eye_left_y_new)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_37_eye_left_x),int(source_37_eye_left_y_new)),(int(source_36_eye_left_x),int(source_36_eye_left_y)),(0,0,0),5)
                                
                cv2.line(eye_mask_source_numpy,(int(source_42_eye_right_x),int(source_42_eye_right_y)),(int(source_43_eye_right_x),int(source_43_eye_right_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_43_eye_right_x),int(source_43_eye_right_y)),(int(source_44_eye_right_x),int(source_44_eye_right_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_44_eye_right_x),int(source_44_eye_right_y)),(int(source_45_eye_right_x),int(source_45_eye_right_y)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_45_eye_right_x),int(source_45_eye_right_y)),(int(source_44_eye_right_x),int(source_44_eye_right_y_new)),(0,0,0),5)
                cv2.line(eye_mask_source_numpy,(int(source_44_eye_right_x),int(source_44_eye_right_y_new)),(int(source_43_eye_right_x),int(source_43_eye_right_y_new)),(0,0,0),5) 
                cv2.line(eye_mask_source_numpy,(int(source_43_eye_right_x),int(source_43_eye_right_y_new)),(int(source_42_eye_right_x),int(source_42_eye_right_y)),(0,0,0),5) 
            
            else:
                          
                source_gaze_circle_left_x=(source_37_eye_left_x+source_38_eye_left_x)/2
                source_gaze_circle_left_y= (source_37_eye_left_y+source_41_eye_left_y)/2
                source_gaze_circle_right_x= (source_44_eye_right_x+source_43_eye_right_x)/2
                source_gaze_circle_right_y= (source_47_eye_right_y+source_43_eye_right_y)/2
                source_gaze_radius= (source_47_eye_right_y-source_43_eye_right_y)/2     

                ### rendered eye mask retarget 
                cv2.ocl.setUseOpenCL(False)  
                cv2.setNumThreads(0)  
                cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(255,0,0),5)  
                cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(255,0,0),5)                            
                
            
            eye_mask_source_numpy=eye_mask_source_numpy/255
            eyemask_reconstruct_source.append(eye_mask_source_numpy)
            time.sleep(0.003)
            

        eyemask_reconstruct_source_tensor=torch.FloatTensor(np.array(eyemask_reconstruct_source)).cuda()      
        
        return eyemask_reconstruct_source_tensor       
    
    
    
    
    def forward(self, source_image, kp_driving, kp_source):
        
        bs, c , h, w = source_image.shape    
        
        mesh_driving=kp_driving['mesh']
        mesh_source=kp_source['mesh']
        
        lmk_driving=kp_driving['lmk']
        lmk_source=kp_source['lmk']        

        rendered_source=kp_source['rendered_source'] 
        rendered_driving=kp_driving['rendered_driving'] 

        eyemask_source=kp_source['eyemask_source'] 
        eyemask_driving=kp_driving['eyemask_driving']  
        
        out_dict = dict()
    
        sparseflow=[]
        for singlebs in range(bs):
            mesh_driving_single=mesh_driving[singlebs,:,:].squeeze(dim=0).cpu().detach().numpy()
            mesh_source_single=mesh_source[singlebs,:,:].squeeze(dim=0).cpu().detach().numpy()
            sparseflow_single=self.construct_Fapp(mesh_source_single, mesh_driving_single, h)            
            
            sparseflow.append(sparseflow_single)
            time.sleep(0.003)
        
        out_dict['meshflow'] = torch.tensor(sparseflow).cuda().to(torch.float32)  
        
         

        ###############
        grid_mesh = self.make_coordinate_grid_2d((h,w))
        sparseflownew = grid_mesh + sparseflow
        
        sparsedeformation=self.down_sample_flow(torch.tensor(sparseflownew).cuda().to(torch.float32).permute(0,3,1,2)).permute(0,2,3,1)
        
        out_dict['sparseflow'] = sparsedeformation   
        
        deformed_source =  self.create_deformed_source_image(source_image, sparsedeformation) 

        out_dict['sparse_deformed'] = deformed_source        

        occlusion, deformation =self.flowprediction(source_image,eyemask_source, deformed_source,eyemask_driving, torch.tensor(sparseflow).cuda().to(torch.float32).permute(0,3,1,2))

        out_dict['dense_flow_foreground'] = deformation 

        out_dict['matting_mask'] = occlusion 
        
        out_dict['sourcelmk'] = lmk_source 
        out_dict['drivinglmk'] = lmk_driving    
        
        out_dict['eye_mask_source_tensor'] = eyemask_source.permute(0,2,3,1)
        out_dict['eye_mask_driving_tensor'] = eyemask_driving.permute(0,2,3,1) 
        
        out_dict['eyemask_reconstruct_source_tensor'] = rendered_source.permute(0,2,3,1) 
        out_dict['eyemask_reconstruct_driving_tensor'] = rendered_driving.permute(0,2,3,1) 
        
        return out_dict