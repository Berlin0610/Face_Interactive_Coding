#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms

##########################################################
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")


#torch.set_grad_enabled(True) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
arguments_strOne = './images/one.png'
arguments_strTwo = './images/two.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
	if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
	if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored


##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).to(device)#.cuda()

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)


##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super().__init__()

			def forward(self, tenInput):
				tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
				tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
				tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229

				return torch.cat([ tenRed, tenGreen, tenBlue ], 1)

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super().__init__()

				self.netBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)

			def forward(self, tenInput):
				return self.netBasic(tenInput)
		self.netPreprocess = Preprocess()

		self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch', file_name='spynet-' + arguments_strModel).items() })
	# end

	def forward(self, tenOne, tenTwo):
		tenFlow = []

		tenOne = [ self.netPreprocess(tenOne) ]
		tenTwo = [ self.netPreprocess(tenTwo) ]

		for intLevel in range(5):
			if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
				tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
				tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
		tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])
        
		for intLevel in range(len(tenOne)):
			tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
		return tenFlow

    
############
netNetwork = None

def spynet_estimate(tenOne, tenTwo):
    global netNetwork
    if netNetwork is None:
        netNetwork = Network().eval().to(device)
        
    # if torch.cuda.is_available():
    #     netNetwork = torch.nn.DataParallel(netNetwork,device_ids = [0,1,2,3]).to(device)#.cuda()#.to(device_ids[0])###########
        #netNetwork.to(device)
    tenOne=tenOne.to(device)
    tenTwo=tenTwo.to(device)
    #print(tenOne)
    #print(tenTwo)
    
    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenOne, tenTwo), size=(tenOne.shape[3], tenOne.shape[2]), mode='bilinear', align_corners=False)
    return tenFlow

