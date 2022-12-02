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

### The key-reference frame is compressed by VVC codec to provide the texture reference and the subsequent inter frames are compactly represented with highly-disentangled facial semantics. By decoding these semantics and modifying the corresponding value of them at the decoder side, different interactive manners can be achieved in terms of eye motion, mouth motion, headpose and face region size. 


### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205346979-65cd41d8-a575-485f-8e2c-fecfb4ff296e.mp4)](https://user-images.githubusercontent.com/80899378/205346979-65cd41d8-a575-485f-8e2c-fecfb4ff296e.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347004-c993ccad-11ca-4129-81ab-67cd1f4d0f62.mp4)](https://user-images.githubusercontent.com/80899378/205347004-c993ccad-11ca-4129-81ab-67cd1f4d0f62.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347025-140bf020-2aa4-456c-a4ac-8537c213a5fc.mp4)](https://user-images.githubusercontent.com/80899378/205347025-140bf020-2aa4-456c-a4ac-8537c213a5fc.mp4)


### Example 4

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347052-1f1431de-91cf-4045-ab2c-e84ae9d37c8c.mp4)](https://user-images.githubusercontent.com/80899378/205347052-1f1431de-91cf-4045-ab2c-e84ae9d37c8c.mp4)


### Example 5

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347065-5c785684-80a8-4189-8e1f-c3fdbea55449.mp4)](https://user-images.githubusercontent.com/80899378/205347065-5c785684-80a8-4189-8e1f-c3fdbea55449.mp4)


### Example 6

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347073-55002555-33a6-4893-8c53-37974782366c.mp4)](https://user-images.githubusercontent.com/80899378/205347073-55002555-33a6-4893-8c53-37974782366c.mp4)


### Example 7

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347094-3af5febb-c8ab-4877-a3c8-9a390a9ece3b.mp4)](https://user-images.githubusercontent.com/80899378/205347094-3af5febb-c8ab-4877-a3c8-9a390a9ece3b.mp4)


### Example 8

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205347116-259c4593-e258-46ed-a63c-0c2ef15f2065.mp4)](https://user-images.githubusercontent.com/80899378/205347116-259c4593-e258-46ed-a63c-0c2ef15f2065.mp4)





## Virtual Character Animation with Facial Semantics

### Virtual characters stored at the decoder side are selected by end users and facial semantics are extracted and transmitted from the talking face sequence captured at the encoder side. 

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205348093-f9b38340-d96b-4aa9-aa10-f86fd7d58bba.mp4)](https://user-images.githubusercontent.com/80899378/205348093-f9b38340-d96b-4aa9-aa10-f86fd7d58bba.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205348113-79e77836-ecdf-4926-ba93-c8a0de2a6147.mp4)](https://user-images.githubusercontent.com/80899378/205348113-79e77836-ecdf-4926-ba93-c8a0de2a6147.mp4)


### Example 

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205348134-842a3e6a-14f7-4c76-ba91-cb764a84e1f6.mp4)](https://user-images.githubusercontent.com/80899378/205348134-842a3e6a-14f7-4c76-ba91-cb764a84e1f6.mp4)


