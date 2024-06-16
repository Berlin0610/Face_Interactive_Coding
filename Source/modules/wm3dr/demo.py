from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from modules.wm3dr.lib.decode import decode
from modules.wm3dr.lib.model import create_model, load_model
from modules.wm3dr.lib.pt_renderer import PtRender
from modules.wm3dr.lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  construct_meshes,
)


def opts():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=20, type=int)
  parser.add_argument('--input_res', default=256, type=int)
  parser.add_argument('--arch', default='resnet_50', type=str)
  parser.add_argument('--load_model', default='.././modules/wm3dr/model/final.pth', type=str)
  parser.add_argument('--BFM', default='.././modules/wm3dr/BFM/mSEmTFK68etc.chj', type=str)

  return parser.parse_args()

def wm3dr(image):
  

  opt = opts()
    
  #print('Creating model...')
  # opt.input_res = 64
  render = PtRender(opt).cuda().eval()
  opt.heads = {'hm': 1, 'params': 257}
  model = create_model(opt.arch, opt.heads)

  if opt.load_model != '':
    model = load_model(model, opt.load_model)
  model.cuda().eval()

  # pre_img, meta = preprocess(image.copy(), opt.input_res)
  # # print(pre_img)
  # # print(pre_img.shape)
  # output, topk_scores, topk_inds, topk_ys, topk_xs = decode(pre_img, model)
  output, topk_scores, topk_inds, topk_ys, topk_xs = decode(image, model)

  params = _tranpose_and_gather_feat(output['params'], topk_inds)
  B, C, _ = params.size()
  # B, C, _ = params.size()
  # if C == 0:
  #   print('no face!')


  # 3DMM formation
  # split coefficients
    # torch.Size([1, 80])
    # torch.Size([1, 64])
    # torch.Size([1, 80])
    # torch.Size([1, 33])
  id_coeff, ex_coeff, tex_coeff, coeff = render.Split_coeff(params.view(-1, params.size(2)))
  # print(id_coeff.shape)
  # print(ex_coeff.shape)  
  # print(tex_coeff.shape)
  # print(coeff.shape)  

  render.set_RotTransLight(coeff, topk_inds.view(-1))  #### 把angle和translation信息传进去

  # reconstruct shape
  canoShape_ = render.Shape_formation(id_coeff, ex_coeff)     #################
  rotShape = render.RotTrans(canoShape_)

  landmark = render.get_Landmarks(rotShape)
  landmark_gt=landmark.detach() 

#   Albedo = render.Texture_formation(tex_coeff)

#   Texture, lighting = render.Illumination(Albedo, canoShape_)
#   Texture = torch.clamp(Texture, 0, 1)

#   rotShape = rotShape.view(B, C, -1, 3)
#   #print(rotShape.shape)  
#   Texture = Texture.view(B, C, -1, 3)
#   # Pytorch3D render
#   meshes = construct_meshes(rotShape, Texture, render.BFM.tri.view(1, -1))        


#   rendered, gpu_masks, depth = render(meshes) # RGB
#   rendered = rendered.squeeze(0).detach().cpu().numpy()
#   print(rendered.shape)
  # gpu_masks = gpu_masks.squeeze(0).unsqueeze(-1).cpu().numpy()


  # # resize to original image
  # image = image.astype(np.float32) / 255.
  # rendered = cv2.resize(rendered, (max(h, w), max(h, w)))[:h, :w]

  return rotShape,landmark_gt
        
        
        

    
    


#       rendered, gpu_masks, depth = render(meshes) # RGB
#       rendered = rendered.squeeze(0).detach().cpu().numpy()
#       gpu_masks = gpu_masks.squeeze(0).unsqueeze(-1).cpu().numpy()


#       # resize to original image
#       image = image.astype(np.float32) / 255.
#       rendered = cv2.resize(rendered, (max(h, w), max(h, w)))[:h, :w]
#       gpu_masks = cv2.resize(gpu_masks, (max(h, w), max(h, w)), interpolation=cv2.INTER_NEAREST)[:h, :w, np.newaxis]
#       image_fuse = image * (1 - gpu_masks) + (0.9 * rendered[..., ::-1] + 0.1 * image) * gpu_masks
#       # image_fuse = image * (1 - gpu_masks) + rendered[..., ::-1] * gpu_masks

#       cv2.imwrite(outfile, (image_fuse * 255).astype(np.uint8))
#       # plt.imshow(image_fuse[..., ::-1])
#       # plt.show()


# if __name__ == '__main__':
#   opt = opts()
#   main(opt)

