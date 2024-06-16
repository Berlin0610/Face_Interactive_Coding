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
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

import argparse
from modules.wm3dr.lib.decode import decode
from modules.wm3dr.lib.model import * #create_model, load_model
from modules.wm3dr.lib.pt_renderer import PtRender
from modules.wm3dr.lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  #construct_meshes,
)

import matplotlib.pyplot as plt

def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--input_res', default=256, type=int)
    parser.add_argument('--arch', default='resnet_50', type=str)
    parser.add_argument('--load_model', default='./modules/wm3dr/model/final.pth', type=str)
    parser.add_argument('--BFM', default='./BFM/mSEmTFK68etc.chj', type=str)

    return parser.parse_args()

# def load_checkpoints(config_path, checkpoint_path, cpu=False):

#     with open(config_path) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
#                                         **config['model_params']['common_params'])
#     if not cpu:
#         generator.cuda()

#     if cpu:
#         checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     else:
#         checkpoint = torch.load(checkpoint_path)
 
#     generator.load_state_dict(checkpoint['generator'],strict=False) 
    
#     if not cpu:
#         generator = DataParallelWithCallback(generator)

#     generator.eval()
    
#     return generator


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)
    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)
    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)
    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat

#Ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz



def extract_params(x):

    opt = opts()


    render = PtRender(opt).cuda().eval()
    opt.heads = {'hm': 1, 'params': 257}
    model = create_model(opt.arch, opt.heads)
    facemodel = loadmodel(model, opt.load_model)
    facemodel.cuda().eval()

    output, topk_scores, topk_inds, topk_ys, topk_xs = decode(x, facemodel)

    params = _tranpose_and_gather_feat(output['params'], topk_inds)

    id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))


    return {'ex_coeff':ex_coeff, 'angles':angles, 'translation':translation, 'topk_inds': topk_inds}   


def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB


if __name__ == "__main__":
   
    parser = ArgumentParser()

    frames=250 
    width=256
    height=256
    
    Qstep=8
    exp_dim=6
    epoch=130
    modeldir = 'train'  
    
    dirname='./experiment/'
    
    config_path='./checkpoint/'+modeldir+'/vox-256.yaml'
    checkpoint_path='./checkpoint/'+modeldir+'/'+str(epoch-1).zfill(8) +'-checkpoint.pth.tar' 
    #generator = load_checkpoints(config_path, checkpoint_path, cpu=False)
 
    seqlist= ['001']
    qplist=["47"] 
    
    

    opt = opts()
    render = PtRender(opt).cuda().eval()
    opt.heads = {'hm': 1, 'params': 257}
    model = create_model(opt.arch, opt.heads)
    facemodel = loadmodel(model, opt.load_model)
    facemodel.cuda().eval()


    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):     
        
            start=time.time()
            
            original_seq='./Testrgb/'+str(seq)+'_'+str(width)+'x'+str(width)+'.rgb'  #'_1_8bit.rgb'        
            
            f_org=open(original_seq,'rb')
            
            listR,listG,listB=RawReader_planar(original_seq,width, height,frames)


            dir_enc =dirname+str(epoch)+'epoch/'+str(exp_dim)+'dim/'+str(Qstep)+'step/'+modeldir+'/enc/'+'/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm      

            kp_path=dirname+str(epoch)+'epoch/'+str(exp_dim)+'dim/'+str(Qstep)+'step/'+modeldir+'/kp/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(kp_path,exist_ok=True)     # the frames to be compressed by vtm                 

            imgsavepth=dirname+str(epoch)+'epoch/'+str(exp_dim)+'dim/'+str(Qstep)+'step/'+modeldir+'/oriimg/'+seq+'_QP'+str(QP)+'/'
            blinkausavepth=dirname+str(epoch)+'epoch/'+str(exp_dim)+'dim/'+str(Qstep)+'step/'+modeldir+'/blinkau/'+seq+'_QP'+str(QP)+'/'     


            img_savepth =imgsavepth#+"/"+str(seq)+'/'
            if not os.path.exists(img_savepth):
                os.makedirs(img_savepth)


            processed_output = blinkausavepth#+"/"+str(seq)+'/'
            if not os.path.exists(processed_output):
                os.makedirs(processed_output)

            kp_value_seq = []

            start=time.time() 
            
            sum_bits = 0            

            for frame_idx in range(0,frames):
                frame_idx_str = str(frame_idx).zfill(4)   
                img = np.fromfile(f_org,np.uint8,3*height*width).reshape((3,height,width))  #3xHxW RGB
                
                
                if frame_idx in [0]:      # I-frame                        
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                    img.tofile(f_temp)
                    f_temp.close()
                                                           
                    os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   

                    bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits
                    
                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                
                    img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU                
                

                        img_np=(255 * np.transpose(reference.data.cpu().numpy(), [0, 2, 3, 1])[0]).astype(np.uint8)

                        imgname=frame_idx_str+'.png'
                        cv2.imwrite(os.path.join(img_savepth, imgname) , img_np[..., ::-1])
    
        
                        os.system("./opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction -au_static -aus -f %s -out_dir %s >/dev/null 2>&1"%(os.path.join(img_savepth, imgname), processed_output))         

                        csvname=frame_idx_str+".csv"

                        aus_info = pd.read_csv(os.path.join(processed_output, csvname))
                        au45_intensity = np.array(aus_info['AU45_r'], dtype='float32')  #np.array(aus_info['AU45_r'][0])   
                        au45value=torch.FloatTensor([au45_intensity]).cuda()   

                        #########3    
                        au45value=torch.round(au45value*100)              
                        au45value=au45value.int()
                        au45_list=au45value.tolist()
                        au45_str=str(au45_list)
                        au45_str="".join(au45_str.split())   


                        ########3DMM parameters
                        output, topk_scores, topk_inds, topk_ys, topk_xs = decode(reference, facemodel)

                        params = _tranpose_and_gather_feat(output['params'], topk_inds)

                        id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))

                        exp=ex_coeff[:,:exp_dim]
                        exp_list=exp.tolist()
                        exp_str=str(exp_list)
                        exp_str="".join(exp_str.split())            


                        angles=angles
                        angles_list=angles.tolist()
                        angles_str=str(angles_list)
                        angles_str="".join(angles_str.split())         

                        translation=translation
                        translation_list=translation.tolist()
                        translation_str=str(translation_list)
                        translation_str="".join(translation_str.split())     

                        inds=topk_inds
                        inds_list=inds.tolist()
                        inds_str=str(inds_list)
                        inds_str="".join(inds_str.split())       


                        with open(kp_path+'/frame'+frame_idx_str+'.txt','w')as f:
                            f.write(exp_str)
                            f.write('\n'+angles_str)  
                            f.write('\n'+translation_str)            
                            f.write('\n'+inds_str)                 
                            f.write('\n'+au45_str)     


                        exp_frame= eval('[%s]'%repr(exp_list).replace('[', '').replace(']', ''))   
                        angles_frame= eval('[%s]'%repr(angles_list).replace('[', '').replace(']', ''))  
                        translation_frame= eval('[%s]'%repr(translation_list).replace('[', '').replace(']', ''))
                        inds_frame= eval('[%s]'%repr(inds_list).replace('[', '').replace(']', ''))      
                        au45_frame= eval('[%s]'%repr(au45_list).replace('[', '').replace(']', ''))            

                        kp_integer=exp_frame+angles_frame+translation_frame+inds_frame+au45_frame  ## 64+3+3+1+1
                        kp_value_seq.append(kp_integer)    




                else:
                    
                    interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                    #print(source_image)        
                    interframe = resize(interframe, (width, height))[..., :3]

                    with torch.no_grad(): 
                        interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

                        interframe = interframe.cuda()    # require GPU                
                

                         ### blink au45 value

                        img_np=(255 * np.transpose(interframe.data.cpu().numpy(), [0, 2, 3, 1])[0]).astype(np.uint8)

                        imgname=frame_idx_str+'.png'
                        cv2.imwrite(os.path.join(img_savepth, imgname) , img_np[..., ::-1])
  

                        os.system("./opencv-4.1.0/build/OpenFace/build/bin/FeatureExtraction -au_static -aus -f %s -out_dir %s >/dev/null 2>&1"%(os.path.join(img_savepth, imgname), processed_output)) 

                        csvname=frame_idx_str+".csv"

                        aus_info = pd.read_csv(os.path.join(processed_output, csvname))
                        au45_intensity = np.array(aus_info['AU45_r'], dtype='float32')  #np.array(aus_info['AU45_r'][0])   
                        au45value=torch.FloatTensor([au45_intensity]).cuda()   

                        #########3    
                        au45value=torch.round(au45value*100)              
                        au45value=au45value.int()
                        au45_list=au45value.tolist()
                        au45_str=str(au45_list)
                        au45_str="".join(au45_str.split())   


                        ########3DMM parameters
                        output, topk_scores, topk_inds, topk_ys, topk_xs = decode(interframe, facemodel)

                        params = _tranpose_and_gather_feat(output['params'], topk_inds)

                        id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = render.Split_coeff(params.view(-1, params.size(2)))

                        exp=ex_coeff[:,:exp_dim]
                        exp_list=exp.tolist()
                        exp_str=str(exp_list)
                        exp_str="".join(exp_str.split())            


                        angles=angles
                        angles_list=angles.tolist()
                        angles_str=str(angles_list)
                        angles_str="".join(angles_str.split())         

                        translation=translation
                        translation_list=translation.tolist()
                        translation_str=str(translation_list)
                        translation_str="".join(translation_str.split())     

                        inds=topk_inds
                        inds_list=inds.tolist()
                        inds_str=str(inds_list)
                        inds_str="".join(inds_str.split())       

                        with open(kp_path+'/frame'+frame_idx_str+'.txt','w')as f:
                            f.write(exp_str) ### 
                            f.write('\n'+angles_str)  ###3
                            f.write('\n'+translation_str)  ##3          
                            f.write('\n'+inds_str) ### 1                
                            f.write('\n'+au45_str)  ### 1   


                        exp_frame= eval('[%s]'%repr(exp_list).replace('[', '').replace(']', ''))   
                        angles_frame= eval('[%s]'%repr(angles_list).replace('[', '').replace(']', ''))  
                        translation_frame= eval('[%s]'%repr(translation_list).replace('[', '').replace(']', ''))
                        inds_frame= eval('[%s]'%repr(inds_list).replace('[', '').replace(']', ''))      
                        au45_frame= eval('[%s]'%repr(au45_list).replace('[', '').replace(']', ''))            


                        kp_integer=exp_frame+angles_frame+translation_frame+inds_frame+au45_frame  ## 64+3+3+1+1
                        kp_value_seq.append(kp_integer)   

                        
            rec_sem=[]
            for frame in range(1,frames):
                frame_idx = str(frame).zfill(4)
                if frame==1:
                    rec_sem.append(kp_value_seq[0])
                    
                    kp_difference=(np.array(kp_value_seq[frame])-np.array(kp_value_seq[frame-1])).tolist()
                    
                    kp_difference=[i * 50 for i in kp_difference[:2]] + [i * Qstep for i in kp_difference[2:exp_dim]] +  [i * 50 for i in kp_difference[exp_dim:exp_dim+3]]  + [i * Qstep for i in kp_difference[exp_dim+3:exp_dim+5]] + [i * 50 for i in kp_difference[exp_dim+5:exp_dim+6]] +[i for i in kp_difference[exp_dim+6:exp_dim+7]] + [i for i in kp_difference[exp_dim+7:exp_dim+8]]                      

                    kp_difference= list(map(round, kp_difference[:]))

                    frame_idx = str(frame).zfill(4)
                    bin_file=kp_path+'/frame'+str(frame_idx)+'.bin'

                    final_encoder_expgolomb(kp_difference,bin_file)     

                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits          
                    
                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

                    res_difference_dec=[i/ 50 for i in res_difference_dec[:2]] + [i/ Qstep for i in res_difference_dec[2:exp_dim]] + [i /50 for i in res_difference_dec[exp_dim:exp_dim+3]] +[i /Qstep for i in res_difference_dec[exp_dim+3:exp_dim+5]]  +[ i /50 for i in res_difference_dec[exp_dim+5:exp_dim+6]]  + [i for i in res_difference_dec[exp_dim+6:exp_dim+7]] + [i for i in res_difference_dec[exp_dim+7:exp_dim+8]]  

                    rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()

                    rec_sem.append(rec_semantics)
                    
                else:
                    
                    
                    kp_difference=(np.array(kp_value_seq[frame])-np.array(rec_sem[frame-1])).tolist()
                    
                    kp_difference=[i * 50 for i in kp_difference[:2]] + [i * Qstep for i in kp_difference[2:exp_dim]] +  [i * 50 for i in kp_difference[exp_dim:exp_dim+3]]  + [i * Qstep for i in kp_difference[exp_dim+3:exp_dim+5]] + [i * 50 for i in kp_difference[exp_dim+5:exp_dim+6]] +[i for i in kp_difference[exp_dim+6:exp_dim+7]] + [i for i in kp_difference[exp_dim+7:exp_dim+8]]                      


                    kp_difference= list(map(round, kp_difference[:]))

                    frame_idx = str(frame).zfill(4)
                    bin_file=kp_path+'/frame'+str(frame_idx)+'.bin'
                    
                
                    final_encoder_expgolomb(kp_difference,bin_file)     

                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits          
                    
                    res_dec = final_decoder_expgolomb(bin_file)
                    res_difference_dec = data_convert_inverse_expgolomb(res_dec)                  
                
                    res_difference_dec=[i/ 50 for i in res_difference_dec[:2]] + [i/ Qstep for i in res_difference_dec[2:exp_dim]] + [i /50 for i in res_difference_dec[exp_dim:exp_dim+3]] +[i /Qstep for i in res_difference_dec[exp_dim+3:exp_dim+5]]  +[ i /50 for i in res_difference_dec[exp_dim+5:exp_dim+6]]  + [i for i in res_difference_dec[exp_dim+6:exp_dim+7]] + [i for i in res_difference_dec[exp_dim+7:exp_dim+8]]  

                    
                    rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()
                                        
                    rec_sem.append(rec_semantics)

            end=time.time()
            print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))