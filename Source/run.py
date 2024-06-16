import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator

import torch

from train import train



if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="original", choices=["original", "spade"])
    parser.add_argument("--log_dir", default="./checkpoint/", help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--gpu_num", default=4, help="GPU number") 
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    opt.device_ids = list(range(opt.gpu_num))
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, 'train')
        

    if opt.gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif opt.gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if torch.cuda.is_available():
        print('cuda is available')
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)



    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, opt.checkpoint, log_dir, dataset, opt.device_ids)