# Interactive Face Video Coding: A Generative Compression Framework

This repository contains the source code for the paper [Interactive Face Video Coding: A Generative Compression Framework](https://arxiv.org/abs/2302.09919) by Bolin Chen, Zhao Wang, Binzhe Li, Shurun Wang, Shiqi Wang, Yan Ye.


### Installation

The version ```python3.8+``` and ```Pytorch3D``` are needed. To install the dependencies, please run:
```
pip install -r requirements.txt
```

For the OpenFace library, you have two solutions as follows:
```
1) Install it based on the opencv-4.1.0 cccording to the tutorial https://github.com/TadasBaltrusaitis/OpenFace. 
2) Directly download the compiled library via this link: [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/Ei3vdYy4whtCmI3lSBpH8ucBxsoEqk3mjqwdfeqR3uuEvg?e=uwNbol), and put it in the " opencv-4.1.0" folder.
```

In addition, please activate the VVC codec run
```
sudo chmod -R 777 vtm
```

The pretrained IFVC model can be downloaded  and unziped from the following link: [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/Em1LMSDv4xtJvDJVORmPusMBMY8_g0XpJ8-LnPOKtjj86w?e=pwzbpv), and put it in the the "./checkpoint" folder.

The pretrained "arcface_backbone_r50.pth" can be downloaded from following link: [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/ElYn9dQbNZ5Au7k1p5Q6CaoBtrLQDKsZ4yw2k66deKBJQQ?e=9Y5lHQ), and put it in the the "./config" folder.

The pretrained "WM3DR" models can be downloaded from following link [OneDrive](https://portland-my.sharepoint.com/:f:/g/personal/bolinchen3-c_my_cityu_edu_hk/EpdQGTuLhjRDnAptwnVW9XUBYltICZxtNQMfY8pFcHok_g?e=KnAdDc) or [WM3DR](https://github.com/kalyo-zjl/WM3DR), and put them in the  "./modules/wm3dr/BFM/mSEmTFK68etc.chj" and  "./modules/wm3dr/model/final.pth".


### Training

To train a model on [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

In addition, you should pre-process the VoxCeleb data using the "Data_PreProcessing.py", such that the training dataset includes 3Dmesh-realted data.

When finishing the downloading and pre-processing the dataset, you can train the model,
```
python run.py
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder. To check the loss values during training see ```log.txt```. You can also check training data reconstructions in the ```train-vis``` subfolder. You can change the training settings in corresponding ```./config/vox-256.yaml``` file.

### Inference

To encode a sequence, please put the provided testing sequence in ```./Testrgb/``` file and run
```
python IFVC_Encoder.py
```
After obtaining the bistream, please run
```
python IFVC_Decoder.py
```
For the testing sequence, it should be in the format of ```RGB:444``` at the resolution of ```256*256```.


### Additional notes

#### Reference

The training code refers to the FOMM: https://github.com/AliaksandrSiarohin/first-order-model.

The arithmetic-coding refers to https://github.com/nayuki/Reference-arithmetic-coding.

The 3DMM model refers to the WM3DR https://github.com/kalyo-zjl/WM3DR.

#### Citation:

```
@article{chen2023interactive,
  title={Interactive face video coding: A generative compression framework},
  author={Chen, Bolin and Wang, Zhao and Li, Binzhe and Wang, Shurun and Wang, Shiqi and Ye, Yan},
  journal={arXiv preprint arXiv:2302.09919},
  year={2023}
}
```
