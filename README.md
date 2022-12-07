# Face Interactive Coding: A Generative Compression Framework


[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205655623-706a9715-ef8c-4fc2-9925-9110e9bbb128.mp4)](https://user-images.githubusercontent.com/80899378/205655623-706a9715-ef8c-4fc2-9925-9110e9bbb128.mp4)

## Abstract

In this paper, we propose a face interactive coding framework, which efficiently projects the talking face frames into low-dimensional and highly-disentangled facial semantics. Therefore, the video conferencing/chat system can be developed towards low-bandwidth communication and immersive interactivity. Technically, assisted by 3D morphable model (3DMM), the face videos exhibiting strong statistical regularities can be decoupled into facial semantic parameters, where these parameters are further compressed and transmitted to reconstruct 3D facial meshes. Moreover, joint with the additionally predicted eye-blink intensities, these reconstructed 3D facial meshes are evolved into explicit motion field characterization and facial attention guidance. As such, the talking face videos are enabled to be accurately reconstructed or controllably synthesized via the deep generative model at ultra-low bit rates. Experimental results have demonstrated the performance superiority and application prospects of our proposed scheme in video conferencing and live entertainment compared with the state-of-the-art video coding standard Versatile Video Coding (VVC) and the latest generative compression schemes. 



## Quality Comparisons (Similar Bitrate)

### To verify the performance, we compare our proposed face interactive coding scheme with the latest hybrid video coding standard VVC and five generative compression schemes, including 2D algorithms (FOMM, FOMM2.0 and CFTE) and 3D algorithms (Face_vid2vid and Face2FaceRHO). For better quality comparisons, please download the videos (mp4).

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445375-38109973-9d82-490a-8ec9-5a5e40094f72.mp4)](https://user-images.githubusercontent.com/80899378/205445375-38109973-9d82-490a-8ec9-5a5e40094f72.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445384-fc48de76-3f0d-49de-839b-f143d96d360c.mp4)](https://user-images.githubusercontent.com/80899378/205445384-fc48de76-3f0d-49de-839b-f143d96d360c.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445378-c1111319-8215-4ddb-bb74-5fcfd861d163.mp4)](https://user-images.githubusercontent.com/80899378/205445378-c1111319-8215-4ddb-bb74-5fcfd861d163.mp4)


### Example 4

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445390-c8701815-08e8-4a5d-b2ad-15a52281c212.mp4)](https://user-images.githubusercontent.com/80899378/205445390-c8701815-08e8-4a5d-b2ad-15a52281c212.mp4)


### Example 5

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445393-4dcfc0bc-2009-41ae-9674-3255f8c30879.mp4)](https://user-images.githubusercontent.com/80899378/205445393-4dcfc0bc-2009-41ae-9674-3255f8c30879.mp4)


### Example 6

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445403-7b8fba7b-5828-49f6-89be-893b2fd09678.mp4)](https://user-images.githubusercontent.com/80899378/205445403-7b8fba7b-5828-49f6-89be-893b2fd09678.mp4)


### Example 7

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445412-5d0b5bbc-0e1a-48f6-b01b-0795637c38aa.mp4)](https://user-images.githubusercontent.com/80899378/205445412-5d0b5bbc-0e1a-48f6-b01b-0795637c38aa.mp4)


### Example 8

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445413-4ea15cd6-05c7-4d1c-912a-09f97e4dd0ff.mp4)](https://user-images.githubusercontent.com/80899378/205445413-4ea15cd6-05c7-4d1c-912a-09f97e4dd0ff.mp4)


### Example 9

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205445414-483f3272-04ac-4ff2-bef2-58868feaedf8.mp4
)](https://user-images.githubusercontent.com/80899378/205445414-483f3272-04ac-4ff2-bef2-58868feaedf8.mp4
)


## Face Interactive Coding with Facial Semantics

### By decoding these facial semantics from the coding bitstream and modifying the corresponding value of them at the decoder side, different interactive manners can be achieved in terms of eye motion, mouth motion, headpose and face region size. For better quality comparisons, please download the videos (mp4).


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


## Face Interactive Coding with Facial Semantics

### To better show the superior facial interactive results, we provide visual examples about different interactivity degrees of eye motion, mouth motion, head posture and head translation. For better quality comparisons, please download the videos (mp4).

### Example--Eye Motion Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205454266-d46b9c8b-2150-45d5-bfd3-2936633a879d.mp4)](https://user-images.githubusercontent.com/80899378/205454266-d46b9c8b-2150-45d5-bfd3-2936633a879d.mp4)


### Example--Mouth Motion Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205454271-a45b3e5c-229e-41ff-a095-b6ff575e17b7.mp4)](https://user-images.githubusercontent.com/80899378/205454271-a45b3e5c-229e-41ff-a095-b6ff575e17b7.mp4)


### Example--Head Pose Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205454275-090a315a-5261-4830-97a1-e82ade31c04d.mp4)](https://user-images.githubusercontent.com/80899378/205454275-090a315a-5261-4830-97a1-e82ade31c04d.mp4)


### Example--Head Translation Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205454277-c7f1f0bf-76e5-436d-b725-10afb969d3a6.mp4)](https://user-images.githubusercontent.com/80899378/205454277-c7f1f0bf-76e5-436d-b725-10afb969d3a6.mp4)



## Virtual Character Animation with Facial Semantics

### To better protect user privacy in talking face communication, we provide a virtual character animation manner by regarding the virtual character as the key-reference frame and animating it with compact facial semantics at the decoder side. For better quality comparisons, please download the videos (mp4).

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205426549-ea260d28-a1ad-4983-b461-f3634b2475f9.mp4)](https://user-images.githubusercontent.com/80899378/205426549-ea260d28-a1ad-4983-b461-f3634b2475f9.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205426551-fc6ec78c-d43c-4cd3-8a71-3dc6f4bd7168.mp4)](https://user-images.githubusercontent.com/80899378/205426551-fc6ec78c-d43c-4cd3-8a71-3dc6f4bd7168.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/205426552-4b7d35c2-a4a6-4aff-b35a-f629026e4ba5.mp4)](https://user-images.githubusercontent.com/80899378/205426552-4b7d35c2-a4a6-4aff-b35a-f629026e4ba5.mp4)








