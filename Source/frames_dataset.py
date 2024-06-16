import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

from os.path import exists, join, basename, splitext
import pandas as pd
import torch
import re
from os.path import exists, join, basename, splitext
import pandas as pd
import numpy as np


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


# class FramesDataset(Dataset):
#     """
#     Dataset of videos, each video can be represented as:
#       - an image of concatenated frames
#       - '.mp4' or '.gif'
#       - folder with all frames
#     """

#     def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
#                  random_seed=0, pairs_list=None, augmentation_params=None):
#         self.root_dir = root_dir
#         self.videos = os.listdir(root_dir)
#         self.frame_shape = tuple(frame_shape)
#         self.pairs_list = pairs_list
#         self.id_sampling = id_sampling
#         if os.path.exists(os.path.join(root_dir, 'train_rendered_reconstruct')):
#             assert os.path.exists(os.path.join(root_dir, 'test_rendered_reconstruct'))
#             print("Use predefined train-test split.")
#             if id_sampling:
#                 train_videos = {os.path.basename(video).split('#')[0] for video in
#                                 os.listdir(os.path.join(root_dir, 'train_rendered_reconstruct'))}
#                 train_videos = list(train_videos)
#             else:
#                 train_videos = os.listdir(os.path.join(root_dir, 'train_rendered_reconstruct'))
#             test_videos = os.listdir(os.path.join(root_dir, 'test_rendered_reconstruct'))
            
#             self.root_dir = os.path.join(self.root_dir, 'train_rendered_reconstruct' if is_train else 'test_rendered_reconstruct')
#         else:
#             print("Use random train-test split.")
#             train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

#         if is_train:
#             self.videos = train_videos
#         else:
#             self.videos = test_videos

#         self.is_train = is_train

#         if self.is_train:
#             self.transform = AllAugmentationTransform(**augmentation_params)
#         else:
#             self.transform = None

#     def __len__(self):
#         return len(self.videos)

#     def __getitem__(self, idx):
#         if self.is_train and self.id_sampling:
#             name = self.videos[idx]
#             path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
#         else:
#             name = self.videos[idx]
#             path = os.path.join(self.root_dir, name)

#         video_name = os.path.basename(path)
        
#         out = {}
        
#         if self.is_train and os.path.isdir(path):
#             frames = os.listdir(path)
#             num_frames = len(frames)
#             frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            
#             video_array=[]
#             source_idx=str(frame_idx[0]).zfill(7) 
#             driving_idx=str(frame_idx[1]).zfill(7) 

#             source_img_path= source_idx + ".png"
#             driving_img_path= driving_idx + ".png"
            
#             video_array.append(img_as_float32(io.imread(os.path.join(path, source_img_path))))
#             video_array.append(img_as_float32(io.imread(os.path.join(path, driving_img_path))))
            

#             path_rendered_reconstruct='/mnt/bolin/Data/VoxCeleb/train/'+ path.split(os.path.sep)[6]
#             print(path_rendered_reconstruct)
            
#             video_array.append(img_as_float32(io.imread(os.path.join(path_rendered_reconstruct, source_img_path))))
#             video_array.append(img_as_float32(io.imread(os.path.join(path_rendered_reconstruct, driving_img_path))))

            
#             path_eyemask_reconstruct='/mnt/bolin/Data/VoxCeleb/train_eyemask_reconstruct/'+ path.split(os.path.sep)[6]  
#             print(path_eyemask_reconstruct)
            
#             video_array.append(img_as_float32(io.imread(os.path.join(path_eyemask_reconstruct, source_img_path))))
#             video_array.append(img_as_float32(io.imread(os.path.join(path_eyemask_reconstruct, driving_img_path))))

            
            
#         source = np.array(video_array[2], dtype='float32')
#         driving = np.array(video_array[3], dtype='float32')

#         rendered_source = np.array(video_array[0], dtype='float32')
#         rendered_driving = np.array(video_array[1], dtype='float32')

#         eyemask_source = np.array(video_array[4], dtype='float32')
#         eyemask_driving = np.array(video_array[5], dtype='float32')

#         out['driving'] = driving.transpose((2, 0, 1))
#         out['source'] = source.transpose((2, 0, 1))

#         out['rendered_driving'] = rendered_driving.transpose((2, 0, 1))
#         out['rendered_source'] = rendered_source.transpose((2, 0, 1))            

#         out['eyemask_driving'] = eyemask_driving.transpose((2, 0, 1))
#         out['eyemask_source'] = eyemask_source.transpose((2, 0, 1))      



#         out['name'] = video_name

#         return out
    
    
    
    
    
    
class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        
        if os.path.exists(os.path.join(root_dir, 'train')):
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            self.root_dir = os.path.join(self.root_dir, 'train' )

        if is_train:
            self.videos = train_videos
 
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))            
            
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            
            video_array = [img_as_float32(io.imread(os.path.join(path, str(frames[idx], encoding="utf-8")))) for idx in frame_idx]
            #video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            
            source_idx=str(frame_idx[0]).zfill(7) 
            driving_idx=str(frame_idx[1]).zfill(7)             
            
            video_array=[]
            
            # print(path)  
            ### original training data
            source_img_path= source_idx + ".png"
            driving_img_path= driving_idx + ".png"
            video_array.append(img_as_float32(io.imread(os.path.join(path, source_img_path))))
            video_array.append(img_as_float32(io.imread(os.path.join(path, driving_img_path))))
            

            ### original visual mesh of the corresponding data with eye mask      
            strinfo_visual = re.compile('/train/')
            path_render = strinfo_visual.sub('/train_rendered_reconstruct/',path)
            video_array.append(img_as_float32(io.imread(os.path.join(path_render, source_img_path))))
            video_array.append(img_as_float32(io.imread(os.path.join(path_render, driving_img_path))))
            
            ### original eye mask    
            strinfo_visual = re.compile('/train_rendered_reconstruct/')
            path_eyemask= strinfo_visual.sub('/train_eyemask_reconstruct/',path_render)
            video_array.append(img_as_float32(io.imread(os.path.join(path_eyemask, source_img_path))))
            video_array.append(img_as_float32(io.imread(os.path.join(path_eyemask, driving_img_path))))            
            


        out = {}
        if self.is_train:

            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            rendered_source = np.array(video_array[2], dtype='float32')
            rendered_driving = np.array(video_array[3], dtype='float32')

            eyemask_source = np.array(video_array[4], dtype='float32')
            eyemask_driving = np.array(video_array[5], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

            out['rendered_driving'] = rendered_driving.transpose((2, 0, 1))
            out['rendered_source'] = rendered_source.transpose((2, 0, 1))            

            out['eyemask_driving'] = eyemask_driving.transpose((2, 0, 1))
            out['eyemask_source'] = eyemask_source.transpose((2, 0, 1))      
            
            
        out['name'] = video_name

        return out
    
    

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
