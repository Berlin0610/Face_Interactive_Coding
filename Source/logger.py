import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle, ellipse
from skimage.draw import circle_perimeter

import matplotlib.pyplot as plt
import collections
from flowvisual import *

class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, he_estimator=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_he_estimator=None):
        checkpoint = torch.load(checkpoint_path)
        
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    
    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2]) #circle_perimeter ellipse   circle
            image[rr, cc] = (255,0,255)
        return image    
    

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []
        
        # Source image with keypoints
        source = source.data.cpu()
        #kp_source = out['kp_source']['value'][:, :, :].data.cpu().numpy()     # 3d -> 2d
        kp_source = out['sourcelmk'][:, :, :].data.cpu().numpy()     # 3d -> 2d        
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))
        
        # Driving image with keypoints
        #kp_driving = out['kp_driving']['value'][:, :, :].data.cpu().numpy()    # 3d -> 2d
        kp_driving = out['drivinglmk'][:, :, :].data.cpu().numpy()    # 3d -> 2d        
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        
    
        ### eye_mask_source_tensor 
        if 'eye_mask_source_tensor' in out:
            eye_mask_source_tensor = out['eye_mask_source_tensor'].data.cpu().numpy() 
            images.append(eye_mask_source_tensor)
        
        ### eyemask_reconstruct_source_tensor 
        if 'eyemask_reconstruct_source_tensor' in out:
            eyemask_reconstruct_source_tensor = out['eyemask_reconstruct_source_tensor'].data.cpu().numpy() 
            images.append(eyemask_reconstruct_source_tensor)
             
        ### eye_mask_driving_tensor 
        if 'eye_mask_driving_tensor' in out:
            eye_mask_driving_tensor = out['eye_mask_driving_tensor'].data.cpu().numpy() 
            images.append(eye_mask_driving_tensor)

        ### eyemask_reconstruct_driving_tensor 
        if 'eyemask_reconstruct_driving_tensor' in out:
            eyemask_reconstruct_driving_tensor = out['eyemask_reconstruct_driving_tensor'].data.cpu().numpy() 
            images.append(eyemask_reconstruct_driving_tensor)
            
            
        #Sparse motion
        if 'meshflow' in out:
            meshflow = out['meshflow'].data.cpu().numpy()

            bs, h, w, c = meshflow.shape
            flow=[]
            for batch in range(0,bs):
                sf =flow_to_image(meshflow[batch:batch+1,:,:,:].reshape(h, w, c))
                flow.append(sf)

            mesh_flow= np.array(flow)
            mesh_flow = np.transpose(mesh_flow, [0, 3, 1, 2])
            mesh_flow = torch.from_numpy(mesh_flow).type(source.type())  ###.type(dtype=torch.float64)
            mesh_flow = F.interpolate(mesh_flow, size=source.shape[1:3]).numpy()
            mesh_flow = np.transpose(mesh_flow, [0, 2, 3, 1])   
            images.append(mesh_flow)               
            

        #Sparse motion
        if 'sparseflow' in out:
            sparseflow = out['sparseflow'].data.cpu().numpy()

            bs, h, w, c = sparseflow.shape
            flow=[]
            for batch in range(0,bs):
                sf =flow_to_image(sparseflow[batch:batch+1,:,:,:].reshape(h, w, c))
                flow.append(sf)

            sparse_flow= np.array(flow)
            sparse_flow = np.transpose(sparse_flow, [0, 3, 1, 2])
            sparse_flow = torch.from_numpy(sparse_flow).type(source.type())  ###.type(dtype=torch.float64)
            sparse_flow = F.interpolate(sparse_flow, size=source.shape[1:3]).numpy()
            sparse_flow = np.transpose(sparse_flow, [0, 2, 3, 1])   
            images.append(sparse_flow)           
            
        ### sparse motion deformed image
        if 'sparse_deformed' in out:        
            sparse_deformed = out['sparse_deformed'].data.cpu().repeat(1, 1, 1, 1)
            sparse_deformed = F.interpolate(sparse_deformed, size=source.shape[1:3]).numpy()
            sparse_deformed = np.transpose(sparse_deformed, [0, 2, 3, 1])
            images.append(sparse_deformed)            
        
        
        # Result
        if 'prediction' in out: 
            prediction = out['prediction'].data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1])
            images.append(prediction)
  
            
        ## foreground_mask
        if 'foreground_mask' in out:
            occlusion_map = out['foreground_mask'].permute(0, 3, 1, 2)
            occlusion_map=occlusion_map.data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)            
                        
        #Dense motion
        if 'dense_flow_foreground' in out:
            denseflow = out['dense_flow_foreground'].data.cpu().numpy()

            bs, h, w, c = denseflow.shape
            flow=[]
            for batch in range(0,bs):
                df =flow_to_image(denseflow[batch:batch+1,:,:,:].reshape(h, w, c))
                flow.append(df)

            dense_flow= np.array(flow)
            dense_flow = np.transpose(dense_flow, [0, 3, 1, 2])
            dense_flow = torch.from_numpy(dense_flow).type(source.type()) 
            dense_flow = F.interpolate(dense_flow, size=source.shape[1:3]).numpy()
            dense_flow = np.transpose(dense_flow, [0, 2, 3, 1])
            images.append(dense_flow)               
        
        
        ## Occlusion map
        if 'matting_mask' in out:
            occlusion_map = out['matting_mask'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        #Dense motion
        if 'dense_flow_foreground_vis' in out:
            denseflow_final = out['dense_flow_foreground_vis'].data.cpu().numpy()

            bs, h, w, c = denseflow_final.shape
            flowfinal=[]
            for batch in range(0,bs):
                dffinal =flow_to_image(denseflow_final[batch:batch+1,:,:,:].reshape(h, w, c))
                flowfinal.append(dffinal)

            dense_flow_final= np.array(flowfinal)
            dense_flow_final = np.transpose(dense_flow_final, [0, 3, 1, 2])
            dense_flow_final = torch.from_numpy(dense_flow_final).type(source.type()) 
            dense_flow_final = F.interpolate(dense_flow_final, size=source.shape[1:3]).numpy()
            dense_flow_final = np.transpose(dense_flow_final, [0, 2, 3, 1])
            images.append(dense_flow_final)      
            
        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
