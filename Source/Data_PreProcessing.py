import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte
import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch

import time
import random
import pandas as pd
import collections
import itertools
from scipy.spatial import ConvexHull
from animate import normalize_kp
import scipy.io as io
import json
import cv2
import math
import torch.nn.functional as F
from arithmetic.value_decoder import *
import argparse
from modules.wm3dr.lib.decode import decode
from modules.wm3dr.lib.model import * 
from modules.wm3dr.lib.pt_renderer import PtRender
from modules.wm3dr.lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  construct_meshes,
)


from os.path import exists, join, basename, splitext
import shutil
import torchvision.transforms as transforms


def source_mesh_reconstruct(sourceimg):

    B=sourceimg.shape[0]
    opt = opts()
    opt.batch_size = B

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
    projection_2dmesh, landmark = render.get_Landmarks(rotShape)

    ############ # Pytorch3D render
    Albedo = render.Texture_formation(tex_coeff)  #torch.Size([bs, 35709, 3])
    Texture, lighting = render.Illumination_new(Albedo, canoShape_, gamma)
    Texture = torch.clamp(Texture, 0, 1)
    rotShape = rotShape.view(B, 1, -1, 3)
    Texture = Texture.view(B, 1, -1, 3)

    meshes = construct_meshes(rotShape, Texture, render.BFM.tri.view(1, -1))      

    rendered, masks, depth = render(meshes) # RGB
    masks = masks.unsqueeze(-1) 

    return {'mesh': projection_2dmesh, 'lmk':landmark, 'depth':depth, 'mask':masks, 'rendered':rendered, 'id_coeff': id_coeff, 'tex_coeff':tex_coeff, 'gamma': gamma}             
        
        
def driving_mesh_reconstruct(id_coeff, tex_coeff, gamma, ex_coeff, angles, translation, topk_inds):
    B=id_coeff.shape[0]       
    opt = opts()
    opt.batch_size = B
    render = PtRender(opt).cuda().eval()    

    render.set_RotTransLight(angles, translation, topk_inds.view(-1)) 

    coeff=torch.cat([angles,gamma,translation],dim=1)

    # reconstruct shape
    canoShape_ = render.Shape_formation(id_coeff, ex_coeff)     #################
    rotShape = render.RotTrans(canoShape_)
    projection_2dmesh, landmark = render.get_Landmarks(rotShape)

    ############ # Pytorch3D render
    Albedo = render.Texture_formation(tex_coeff)  #torch.Size([bs, 35709, 3])
    Texture, lighting = render.Illumination_new(Albedo, canoShape_, gamma)
    Texture = torch.clamp(Texture, 0, 1)
    rotShape = rotShape.view(B, 1, -1, 3)
    Texture = Texture.view(B, 1, -1, 3)
    meshes = construct_meshes(rotShape, Texture, render.BFM.tri.view(1, -1))  

    rendered, masks, depth = render(meshes) # RGB
    masks = masks.unsqueeze(-1) 
   
    return {'mesh':projection_2dmesh, 'lmk':landmark, 'depth':depth, 'mask':masks, 'rendered':rendered}            
    

def face_parameter_extraction(inputimg):
    
    B=inputimg.shape[0]
    opt = opts()
    opt.batch_size = B

    render = PtRender(opt).cuda().eval()
    opt.heads = {'hm': 1, 'params': 257}
    model = create_model(opt.arch, opt.heads)
    facemodel = loadmodel(model, opt.load_model)
    facemodel.cuda().eval()

    output, topk_scores, topk_inds, topk_ys, topk_xs = decode(inputimg, facemodel)
    params = _tranpose_and_gather_feat(output['params'], topk_inds)
    
    id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))

    return {'ex_coeff':ex_coeff, 'angles':angles, 'translation':translation, 'topk_inds': topk_inds}          
    
    

def eye_mask_get(lmk_driving, bs):

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
            cv2.line(background_img_driving,(int(x_right),int(y_right)),(int(x2_right),int(y2_right)),(255,255,255),6) 

        background_img_driving=background_img_driving/255
        eye_mask_driving.append(background_img_driving)

    eye_mask_driving_tensor=torch.from_numpy(np.array(eye_mask_driving)).cuda()

    return eye_mask_driving_tensor


    
def rendered_reconstruct(rendered_source_mask, eye_mask_source_tensor, lmk_source, source_au45, bs ):

    rendered_reconstruct_source=[]   
    eyemask_reconstruct_source=[]           
    for batch in range(bs):
        rendered_source_mask_numpy=rendered_source_mask[batch].data.cpu().numpy()            
        rendered_source_mask_numpy = (255 * rendered_source_mask_numpy).astype(np.uint8)

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
        #indensity=(5-source_au45_bs)/5
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

            ### rendered face retarget
            cv2.circle(rendered_source_mask_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(0,0,255),5)  
            cv2.circle(rendered_source_mask_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(0,0,255),5)      

            cv2.line(rendered_source_mask_numpy,(int(source_36_eye_left_x),int(source_36_eye_left_y)),(int(source_37_eye_left_x),int(source_37_eye_left_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_37_eye_left_x),int(source_37_eye_left_y)),(int(source_38_eye_left_x),int(source_38_eye_left_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_38_eye_left_x),int(source_38_eye_left_y)),(int(source_39_eye_left_x),int(source_39_eye_left_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_39_eye_left_x),int(source_39_eye_left_y)),(int(source_38_eye_left_x),int(source_38_eye_left_y_new)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_38_eye_left_x),int(source_38_eye_left_y_new)),(int(source_37_eye_left_x),int(source_37_eye_left_y_new)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_37_eye_left_x),int(source_37_eye_left_y_new)),(int(source_36_eye_left_x),int(source_36_eye_left_y)),(125,133,156),5)

            cv2.line(rendered_source_mask_numpy,(int(source_42_eye_right_x),int(source_42_eye_right_y)),(int(source_43_eye_right_x),int(source_43_eye_right_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_43_eye_right_x),int(source_43_eye_right_y)),(int(source_44_eye_right_x),int(source_44_eye_right_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_44_eye_right_x),int(source_44_eye_right_y)),(int(source_45_eye_right_x),int(source_45_eye_right_y)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_45_eye_right_x),int(source_45_eye_right_y)),(int(source_44_eye_right_x),int(source_44_eye_right_y_new)),(125,133,156),5)
            cv2.line(rendered_source_mask_numpy,(int(source_44_eye_right_x),int(source_44_eye_right_y_new)),(int(source_43_eye_right_x),int(source_43_eye_right_y_new)),(125,133,156),5) 
            cv2.line(rendered_source_mask_numpy,(int(source_43_eye_right_x),int(source_43_eye_right_y_new)),(int(source_42_eye_right_x),int(source_42_eye_right_y)),(125,133,156),5) 


            ### rendered eye mask retarget   
            cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(0,0,255),5)  
            cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(0,0,255),5)   

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
            #source_gaze_radius_left=(source_41_eye_left_y-source_37_eye_left_y)/2
            source_gaze_circle_right_x= (source_44_eye_right_x+source_43_eye_right_x)/2
            source_gaze_circle_right_y= (source_47_eye_right_y+source_43_eye_right_y)/2
            source_gaze_radius= (source_47_eye_right_y-source_43_eye_right_y)/2     

            ### rendered face retarget
            cv2.circle(rendered_source_mask_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(0,0,255),5)  
            cv2.circle(rendered_source_mask_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(0,0,255),5)                  

            ### rendered eye mask retarget     
            cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_right_x),round(source_gaze_circle_right_y)),round(source_gaze_radius),(0,0,255),5)  
            cv2.circle(eye_mask_source_numpy,(round(source_gaze_circle_left_x),round(source_gaze_circle_left_y)),round(source_gaze_radius),(0,0,255),5)                            

        rendered_source_mask_numpy=rendered_source_mask_numpy/255
        rendered_reconstruct_source.append(rendered_source_mask_numpy)

        eye_mask_source_numpy=eye_mask_source_numpy/255
        eyemask_reconstruct_source.append(eye_mask_source_numpy)


    rendered_reconstruct_source_tensor=torch.FloatTensor(np.array(rendered_reconstruct_source)).cuda()
    eyemask_reconstruct_source_tensor=torch.FloatTensor(np.array(eyemask_reconstruct_source)).cuda()      

    return rendered_reconstruct_source_tensor,eyemask_reconstruct_source_tensor       
    



def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--arch', default='resnet_50', type=str)
    parser.add_argument('--load_model', default='./modules/wm3dr/model/final.pth', type=str)
    parser.add_argument('--BFM', default='./BFM/mSEmTFK68etc.chj', type=str)

    return parser.parse_args()

def delete_checkpoints(path):
    for root, dirs, files in os.walk(path):
        for dir_ in dirs:
            if dir_ == '.ipynb_checkpoints':
                shutil.rmtree(os.path.join(root, dir_))    
                
def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img                
                

if __name__ == "__main__":
    
    width=256
    height=256
    



    InputDir='./train/'  ### The training dataset of VoxCeleb or Others
    Output_EyeMaskDir='./train_eyemask_reconstruct/' ## The path to store the precessed output
    Output_RenderDir='./train_rendered_reconstruct/' ## The path to store the precessed output
    Output_ProcessOutCSVDir='./train_OutCSV/' ## The path to store the precessed output

    delete_checkpoints(InputDir)

    folderlist=os.listdir(InputDir)[:]
    print(len(folderlist))

    for folder in folderlist:

        os.makedirs(os.path.join(Output_EyeMaskDir, folder), exist_ok=True)
        os.makedirs(os.path.join(Output_RenderDir, folder), exist_ok=True)
        os.makedirs(os.path.join(Output_ProcessOutCSVDir, folder), exist_ok=True)

        for file in os.listdir(os.path.join(InputDir, folder)):
            name= os.path.splitext(file)[0]
            
            transform = transforms.ToTensor()
            original_img = load_img(os.path.join(InputDir, folder,file))
            img_rec = transform(original_img.astype(np.float32))/255
            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis])
                reference = reference.cuda()    # require GPU
                
                ##Parameter extraction
                reference_reconstruction = source_mesh_reconstruct(reference)    ####################################
                reference_id_coeff=reference_reconstruction['id_coeff']
                reference_tex_coeff=reference_reconstruction['tex_coeff']     
                reference_gamma=reference_reconstruction['gamma']    

                
                os.system("./opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction -au_static -aus -f %s -out_dir %s >/dev/null 2>&1"%(os.path.join(InputDir, folder,file), os.path.join(Output_ProcessOutCSVDir, folder))) 

                csvname=name+".csv"

                aus_info = pd.read_csv(os.path.join(Output_ProcessOutCSVDir, folder, csvname))
                au45_intensity = np.array(aus_info['AU45_r'], dtype='float32')  #np.array(aus_info['AU45_r'][0])   
                au45value=torch.FloatTensor([au45_intensity]).cuda()  ######################################################## 

                #### reconstruct rendered face with eye mask
                source_lmk=reference_reconstruction['lmk']
                source_rendered=reference_reconstruction['rendered']
                source_eye_mask_tensor=eye_mask_get( source_lmk ,source_lmk.shape[0])
                source_rendered_reconstruct,source_eyemask_reconstruct=rendered_reconstruct(source_rendered, source_eye_mask_tensor, source_lmk, au45value, source_lmk.shape[0] ) 
                source_rendered_reconstruct_np=(255 * np.transpose(source_rendered_reconstruct.data.cpu().numpy(), [0, 1, 2, 3])[0]).astype(np.uint8)                
                cv2.imwrite(os.path.join(Output_RenderDir, folder, name+'.png'),source_rendered_reconstruct_np)
                
                #### reconstruct eye mask 
                source_eyemask_reconstruct_np=(255 * np.transpose(source_eyemask_reconstruct.data.cpu().numpy(), [0, 1, 2, 3])[0]).astype(np.uint8)
                cv2.imwrite(os.path.join(Output_EyeMaskDir, folder, name+'.png'),source_eyemask_reconstruct_np)

