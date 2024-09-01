# Interactive Face Video Coding: A Generative Compression Framework


[![IMAGE ALT TEXT](https://github.com/user-attachments/assets/2bb7e8b3-a5f5-43d0-8572-7dd242cc58e7)](https://github.com/user-attachments/assets/2bb7e8b3-a5f5-43d0-8572-7dd242cc58e7)

## Abstract

In this paper, we propose a novel framework for Interactive Face Video Coding (IFVC), which allows humans to interact with the intrinsic visual representations instead of the signals. The proposed solution enjoys several distinct advantages, including ultra-compact representation, low delay interaction, and vivid expression and headpose animation. In particular, we propose the Internal Dimension Increase (IDI) based representation, greatly enhancing the fidelity and flexibility in rendering the appearance while maintaining reasonable representation cost. By leveraging strong statistical regularities, the visual signals can be effectively projected into controllable semantics in the three dimensional space (e.g., mouth motion, eye blinking, head rotation and head translation), which are compressed and transmitted. The editable bitstream, which naturally supports the interactivity at the semantic level, can synthesize the face frames via the strong inference ability of the deep generative model. Experimental results have demonstrated the performance superiority and application prospects of our proposed IFVC scheme. In particular, the proposed scheme not only outperforms the state-of-the-art video coding standard Versatile Video Coding (VVC) and the latest generative compression schemes in terms of rate-distortion performance for face videos, but also enables the interactive coding without introducing additional manipulation processes. Furthermore, the proposed framework is expected to shed lights on the future design of the digital human communication in the metaverse.


## Quality Comparisons (Similar Bitrate)

### To verify the performance, we compare our proposed IFVC scheme with the latest hybrid video coding standard VVC and five generative compression schemes, including FOMM, FOMM2.0, Face2FaceRHO, Face_vid2vid and CFTE. For better quality comparisons, please download the videos (mp4).


### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206108-a1c08768-dbd5-4be4-80a7-a9dc634001d5.mp4)](https://user-images.githubusercontent.com/80899378/217206108-a1c08768-dbd5-4be4-80a7-a9dc634001d5.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206276-3bb51d55-0420-4b9a-b96c-7d1a8f25c393.mp4)](https://user-images.githubusercontent.com/80899378/217206276-3bb51d55-0420-4b9a-b96c-7d1a8f25c393.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206360-5c51739f-3d82-46fb-a034-df657c55359a.mp4)](https://user-images.githubusercontent.com/80899378/217206360-5c51739f-3d82-46fb-a034-df657c55359a.mp4)


### Example 4

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206395-696a0b68-4b0b-4924-a970-ebc753cacf64.mp4)](https://user-images.githubusercontent.com/80899378/217206395-696a0b68-4b0b-4924-a970-ebc753cacf64.mp4)


### Example 5

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206448-8a380660-f501-4003-b07c-51aeac99de58.mp4)](https://user-images.githubusercontent.com/80899378/217206448-8a380660-f501-4003-b07c-51aeac99de58.mp4)


### Example 6

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206481-98cc3a4a-7cf5-4343-a2ec-4fe4a27f01a3.mp4)](https://user-images.githubusercontent.com/80899378/217206481-98cc3a4a-7cf5-4343-a2ec-4fe4a27f01a3.mp4)


### Example 7

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217293301-64005a07-9320-472b-a471-d28369c5a690.mp4)](https://user-images.githubusercontent.com/80899378/217293301-64005a07-9320-472b-a471-d28369c5a690.mp4)


### Example 8

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206635-ac939183-10e9-400b-9ea6-1045689e024b.mp4)](https://user-images.githubusercontent.com/80899378/217206635-ac939183-10e9-400b-9ea6-1045689e024b.mp4)


### Example 9

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217206681-56764e38-cb47-4a43-a234-5b04bbb2f494.mp4)](https://user-images.githubusercontent.com/80899378/217206681-56764e38-cb47-4a43-a234-5b04bbb2f494.mp4)


## Interactive Face Video Coding with Facial Semantics

### By modifying the corresponding facial semantics at the decoder side, different interactive manners can be achieved in terms of eye blinking, mouth motion, head rotation and head translation. For better quality comparisons, please download the videos (mp4).


### Example 1

[![IMAGE ALT TEXT](https://github.com/user-attachments/assets/c6780295-1dc4-4476-9f7a-cbd951e43eda)](https://github.com/user-attachments/assets/c6780295-1dc4-4476-9f7a-cbd951e43eda)


### Example 2

[![IMAGE ALT TEXT](https://github.com/user-attachments/assets/01cbf71c-a81a-4074-a53f-fd2f662dd4e0)](https://github.com/user-attachments/assets/01cbf71c-a81a-4074-a53f-fd2f662dd4e0)


## Interactive Face Video Coding with Facial Semantics

### To better show the superior interactive results, we provide visual examples about different interactivity degrees of eye motion, mouth motion, head rotation and head translation. For better quality comparisons, please download the videos (mp4).


### Example--Eye Motion Interactivity


[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217287951-fd9010cb-2f78-4a9c-85d4-ba1ca3563ebd.mp4)](https://user-images.githubusercontent.com/80899378/217287951-fd9010cb-2f78-4a9c-85d4-ba1ca3563ebd.mp4)


### Example--Mouth Motion Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217287963-16b2debc-a75d-4b71-b421-974e7e159aa7.mp4)](https://user-images.githubusercontent.com/80899378/217287963-16b2debc-a75d-4b71-b421-974e7e159aa7.mp4)


### Example--Head Rotation Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217287980-2a783ffd-246f-4fc0-8327-4bfabf7af98f.mp4)](https://user-images.githubusercontent.com/80899378/217287980-2a783ffd-246f-4fc0-8327-4bfabf7af98f.mp4)


### Example--Head Translation Interactivity

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/217288250-62ee55f3-0a36-4c3a-ab80-5b0fa1f32c77.mp4)](https://user-images.githubusercontent.com/80899378/217288250-62ee55f3-0a36-4c3a-ab80-5b0fa1f32c77.mp4)


### Example--Head Location Interactivity

[![IMAGE ALT TEXT](https://github.com/user-attachments/assets/b869ae49-08de-429a-8e94-c27e91664976)](https://github.com/user-attachments/assets/b869ae49-08de-429a-8e94-c27e91664976)




## Virtual Character Animation with Facial Semantics

### To better protect user privacy in face video communication, we provide a virtual character animation manner by treating the virtual character as the key-reference frame and animating it with compact facial semantics at the decoder side. For better quality comparisons, please download the videos (mp4).

### Example 1

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/212704149-21cac074-3597-4455-8dda-3c95dd80d8e0.mp4)](https://user-images.githubusercontent.com/80899378/212704149-21cac074-3597-4455-8dda-3c95dd80d8e0.mp4)


### Example 2

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/212704451-a28401ec-8ed5-417b-805c-c3cf4c757f6f.mp4)](https://user-images.githubusercontent.com/80899378/212704451-a28401ec-8ed5-417b-805c-c3cf4c757f6f.mp4)


### Example 3

[![IMAGE ALT TEXT](https://user-images.githubusercontent.com/80899378/212704167-4b6ebccd-7d86-441d-9564-2d67b004c4cd.mp4)](https://user-images.githubusercontent.com/80899378/212704167-4b6ebccd-7d86-441d-9564-2d67b004c4cd.mp4)


