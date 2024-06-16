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
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
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


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'],strict=False) 
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    return generator

def make_prediction(reference_frame, source_reconstruction, driving_reconstruction, generator):
   
    out = generator(reference_frame, kp_source=source_reconstruction, kp_driving=driving_reconstruction)
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction



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


if __name__ == "__main__":
    #parser = ArgumentParser()
    frames=250 #150
    width=256
    height=256
    
    Qstep=8
    exp_dim=6
    epoch=130
    modeldir = 'train'  
    
    config_path='./checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint_path='./checkpoint/'+modeldir+'/'+str(epoch-1).zfill(8)+'-checkpoint.pth.tar' 
    experiment_dir = './experiment/'+str(epoch)+'epoch/'+str(exp_dim)+'dim/'+str(Qstep)+'step/'+modeldir+'/'
    os.makedirs(experiment_dir+'/dec/',exist_ok=True)     # the real decoded video

    generator = load_checkpoints(config_path, checkpoint_path, cpu=False)  

    
    seqlist= ['001']
    qplist=["47"] 
    
    
    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):         
 
            driving_kp = experiment_dir+'/kp/'+seq+'_QP'+str(QP)+'/'
            
            dir_dec= experiment_dir+'/dec/'  
            os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
            decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'
            
            decode_rendered_seq=dir_dec+seq+'_QP'+str(QP)+'_rendered.rgb'     
            decode_orirendered_seq=dir_dec+seq+'_QP'+str(QP)+'_orirendered.rgb'            
            decode_eyemask_seq=dir_dec+seq+'_QP'+str(QP)+'_eyemask.rgb'            

            
            dir_enc = experiment_dir+'enc/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm       
            
            savetxt=experiment_dir+'/resultBit/'
            os.makedirs(savetxt,exist_ok=True)   
            
            imgsavepth=experiment_dir+'/oriimg/'+seq+'_QP'+str(QP)+'/'
            processed_output=experiment_dir+'/blinkau/'+seq+'_QP'+str(QP)+'/'     
            
            f_dec=open(decode_seq,'w') 
            f_dec_rendered=open(decode_rendered_seq,'w')     
            f_dec_orirendered=open(decode_orirendered_seq,'w')             
            f_dec_eyemask=open(decode_eyemask_seq,'w')             


            ref_rgb_list=[]                 #  the rgb format of refernce frame list
            ref_norm_list=[]                #  the normalized rgb [0-1] format of reference frame list
            ref_kp_list=[]                  #  the key-point [0-1] list of each reference frame 
            seq_kp_integer=[]               #  the quantilized compact feature list of the whole sequence
            
            start=time.time() 
            generate_time = 0
            
            sum_bits = 0
            
            for frame_idx in range(0, frames):    
                
                frame_idx_str = str(frame_idx).zfill(4)   
                                
                if frame_idx in [0]:      # I-frame          
                    
                    os.system("./vtm/decode.sh "+dir_enc+'frame'+frame_idx_str)

                    bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits

                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                    img_rec.tofile(f_dec) 

                    ref_rgb_list.append(img_rec)

                    img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU

                        reference_reconstruction = source_mesh_reconstruct(reference)    ####################################
                        reference_id_coeff=reference_reconstruction['id_coeff']
                        reference_tex_coeff=reference_reconstruction['tex_coeff']     
                        reference_gamma=reference_reconstruction['gamma']     


                        img_np=(255 * np.transpose(reference.data.cpu().numpy(), [0, 2, 3, 1])[0]).astype(np.uint8)

                        imgname=frame_idx_str+'_dec.png'
                        cv2.imwrite(os.path.join(imgsavepth, imgname) , img_np[..., ::-1])


                        os.system("./opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction -au_static -aus -f %s -out_dir %s >/dev/null 2>&1"%(os.path.join(imgsavepth, imgname), processed_output)) 

                        csvname=frame_idx_str+"_dec.csv"

                        aus_info = pd.read_csv(os.path.join(processed_output, csvname))
                        au45_intensity = np.array(aus_info['AU45_r'], dtype='float32')  #np.array(aus_info['AU45_r'][0])   
                        au45value=torch.FloatTensor([au45_intensity]).cuda()  ######################################################## 

                        #### reconstruct rendered face
                        source_lmk=reference_reconstruction['lmk']
                        source_rendered=reference_reconstruction['rendered']
                        source_eye_mask_tensor=eye_mask_get( source_lmk ,source_lmk.shape[0])
                        source_rendered_reconstruct,source_eyemask_reconstruct=rendered_reconstruct(source_rendered, source_eye_mask_tensor, source_lmk, au45value, source_lmk.shape[0] ) 
                        source_rendered_reconstruct_np=(255 * np.transpose(source_rendered_reconstruct.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                        source_rendered_reconstruct_np.tofile(f_dec_rendered) 
                        
                        source_rendered_np=(255 * np.transpose(source_rendered.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                        source_rendered_np.tofile(f_dec_orirendered)    
            
                        source_eyemask_reconstruct_np=(255 * np.transpose(source_eyemask_reconstruct.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                        source_eyemask_reconstruct_np.tofile(f_dec_eyemask)    

                        
                        reference_reconstruction.update({ 'rendered_source':source_rendered_reconstruct.permute(0,3,1,2), 'eyemask_source':source_eyemask_reconstruct.permute(0,3,1,2)})  

                        
                        #########3    
                        au45value=torch.round(au45value*100)              
                        au45value=au45value.int()
                        au45_list=au45value.tolist()
                        au45_str=str(au45_list)
                        au45_str="".join(au45_str.split())                           

                        #####################################
                        kp_cur = face_parameter_extraction(reference)  
                        
                        exp=kp_cur['ex_coeff'][:,:exp_dim]
                        exp_list=exp.tolist()
                        exp_str=str(exp_list)
                        exp_str="".join(exp_str.split())

                        angles=kp_cur['angles']
                        angles_list=angles.tolist()
                        angles_str=str(angles_list)
                        angles_str="".join(angles_str.split())         

                        translation=kp_cur['translation']
                        translation_list=translation.tolist()
                        translation_str=str(translation_list)
                        translation_str="".join(translation_str.split())     

                        inds=kp_cur['topk_inds']
                        inds_list=inds.tolist()
                        inds_str=str(inds_list)
                        inds_str="".join(inds_str.split())                 

                        exp_frame= eval('[%s]'%repr(exp_list).replace('[', '').replace(']', ''))   
                        angles_frame= eval('[%s]'%repr(angles_list).replace('[', '').replace(']', ''))  
                        translation_frame= eval('[%s]'%repr(translation_list).replace('[', '').replace(']', ''))
                        inds_frame= eval('[%s]'%repr(inds_list).replace('[', '').replace(']', ''))            
                        au45_frame= eval('[%s]'%repr(au45_list).replace('[', '').replace(']', ''))            

                        kp_integer=exp_frame+angles_frame+translation_frame+inds_frame+au45_frame  ## 64+3+3+1+1  
                        kp_integer=str(kp_integer)      
                        seq_kp_integer.append(kp_integer)              

   
                else:                                      ### check whether refresh reference                                  
                    ### entropy-decoding
                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_idx_str+'.bin'            
                    kp_dec = final_decoder_expgolomb(bin_save)
                    kp_difference = data_convert_inverse_expgolomb(kp_dec)
      
                    
                    
                    kp_difference=[i/ 50 for i in kp_difference[:2]] + [i/ Qstep for i in kp_difference[2:exp_dim]] + [i /50 for i in kp_difference[exp_dim:exp_dim+3]] +[i /Qstep for i in kp_difference[exp_dim+3:exp_dim+5]]  +[ i /50 for i in kp_difference[exp_dim+5:exp_dim+6]]  + [i for i in kp_difference[exp_dim+6:exp_dim+7]] + [i for i in kp_difference[exp_dim+7:exp_dim+8]]                               
                    
                    
                    kp_previous=json.loads(str(seq_kp_integer[frame_idx-1]))
                
                    kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', ''))  
                    
                    
                    if exp_dim==64:
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)
                        
                    elif  exp_dim==48:   
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_48(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)
                                            
                    elif  exp_dim==32:   
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_32(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)
                    
                    elif  exp_dim==16:   
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_16(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)
                        
                    elif  exp_dim==8:   #else:
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_8(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)     
                        
                        
                    elif  exp_dim==6:   #else:
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_6(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)                             

                    elif  exp_dim==4:   #else:
                        kp_integer,exp_value,angle_value,tranlation_value,inds_value,au45_value=listformat_3dmm_4(kp_previous, kp_difference) 
                        seq_kp_integer.append(kp_integer)         
                                                   
                    kp_current_exp=torch.Tensor(exp_value).to('cuda:0')          
                    kp_current_angle=torch.Tensor(angle_value).to('cuda:0')          
                    kp_current_translation=torch.Tensor(tranlation_value).to('cuda:0')  
                    kp_current_inds=torch.Tensor(inds_value).to('cuda:0')  
                    kp_current_au45=torch.Tensor(au45_value).to('cuda:0')  
                    kp_current_au45=kp_current_au45/100

                    
                    driving_reconstruction=driving_mesh_reconstruct(reference_id_coeff, reference_tex_coeff, reference_gamma,
                                                                    kp_current_exp, kp_current_angle, kp_current_translation, kp_current_inds)
                    
                    ### driving rendered reconstruct
                    driving_lmk=driving_reconstruction['lmk']
                    driving_rendered=driving_reconstruction['rendered']
                    driving_eye_mask_tensor=eye_mask_get( driving_lmk ,driving_lmk.shape[0])
                    driving_rendered_reconstruct,driving_eyemask_reconstruct=rendered_reconstruct(driving_rendered, driving_eye_mask_tensor, 
                                                                                                  driving_lmk, kp_current_au45, driving_lmk.shape[0] )    


                    driving_rendered_reconstruct_np=(255 * np.transpose(driving_rendered_reconstruct.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                    driving_rendered_reconstruct_np.tofile(f_dec_rendered)    
                    
                    
                    driving_rendered_np=(255 * np.transpose(driving_rendered.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                    driving_rendered_np.tofile(f_dec_orirendered)                         
                                            
                    driving_eyemask_reconstruct_np=(255 * np.transpose(driving_eyemask_reconstruct.data.cpu().numpy(), [0, 3, 1, 2])[0]).astype(np.uint8)
                    driving_eyemask_reconstruct_np.tofile(f_dec_eyemask)    
                                           
                    
                    driving_reconstruction.update({'rendered_driving':driving_rendered_reconstruct.permute(0,3,1,2),'eyemask_driving':driving_eyemask_reconstruct.permute(0,3,1,2)})         
                    
                    
                    generate_start = time.time()
                    prediction = make_prediction(reference, reference_reconstruction, driving_reconstruction, generator) #######################
                    generate_end = time.time()
                    generate_time += generate_end - generate_start
                    pre=(prediction*255).astype(np.uint8)  
                    
                    pre.tofile(f_dec)                              

                    #frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_save)*8
                    sum_bits += bits
                               
            f_dec.close()     
            end=time.time()
            
            totalResult[seqIdx][qpIdx]=sum_bits           
            print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" 
                  %(end-start,generate_time,sum_bits))
    
    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)
    
    print(totalResult)
    np.set_printoptions(suppress=True, precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength
    
    np.savetxt(experiment_dir+'/resultBit.txt', totalResult, fmt = '%.5f')            
                
