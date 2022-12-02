# Face Interactive Coding: A Generative Compression Framework

## Bolin Chen&dagger;, Zhao Wang&sect;, Binzhe Li&dagger;, Shurun Wang&dagger;, Shiqi Wang&dagger; and Yan Ye&sect;

### &dagger; City University of Hong Kong and &sect; Alibaba Group

#### The first two authors (Bolin Chen and Zhao Wang) contributed equally to this work

## Abstract

In this paper, we propose a face interactive coding framework, which efficiently projects the talking face frames into low-dimensional and highly-disentangled facial semantics. Therefore, the video conferencing/chat system can be developed towards low-bandwidth communication and immersive interactivity. Technically, assisted by 3D morphable model (3DMM), the face videos exhibiting strong statistical regularities can be decoupled into facial semantic parameters, where these parameters are further compressed and transmitted to reconstruct 3D facial meshes. Moreover, joint with the additionally predicted eye-blink intensities, these reconstructed 3D facial meshes are evolved into explicit motion field characterization and facial attention guidance. As such, the talking face videos are enabled to be accurately reconstructed or controllably synthesized via the deep generative model at ultra-low bit rates. Experimental results have demonstrated the performance superiority and application prospects of our proposed scheme in video conferencing and live entertainment compared with the state-of-the-art video coding standard Versatile Video Coding (VVC) and the latest generative compression schemes.

## Quality Comparisons (Similar Bitrate)

### For better quality comparisons, please download the videos (mp4) from the "video" file.

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/195520658-a6523e1b-dcdb-4be7-ad17-b7873eb9d26c.mp4)](https://user-images.githubusercontent.com/80899378/195520658-a6523e1b-dcdb-4be7-ad17-b7873eb9d26c.mp4)


## Face Interactive Coding with Facial Semantics

### The key-reference frame is compressed by VVC codec to provide the texture reference and the subsequent inter frames are compactly represented with highly-disentangled facial semantics. By decoding these semantics and modifying the corresponding value of them at the decoder side, different interactive manners can be achieved in terms of eye motion, mouth motion, headpose and face region size. For better quality comparisons, please download the videos (mp4) from the "video" file.


## Virtual Character Animation with Facial Semantics

### Virtual characters stored at the decoder side are selected by end users and facial semantics are extracted and transmitted from the talking face sequence captured at the encoder side. For better quality comparisons, please download the videos (mp4) from the "video" file.

