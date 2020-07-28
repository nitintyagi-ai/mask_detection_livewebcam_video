# mask_detection_livewebcam_video
Detection of mask on images and live web cam video

--> Install requirement.txt for dependencies (Go to requirement.txt drive and run pip install - r requirement.txt)  
--> Download pretrained weights folder which is attached here (Its has caffee file to detect faces in video and image).  
--> Download dataset from mentioned link (https://github.com/prajnasb/observations/tree/master/experiements/data). Thanks to Prajna Bhandary for this dataset.  
--> create a folder named dataset. Create 2 more folder inside dataset 1. mask 2. without_mask and placed your images in these folder accordingly.  
--> Run train_mask.py file by using command python train_mask.py --dataset  provide\path of\ pretrained weights\folder  
--> You will recive 1 model weights file in pretrained weights folder named as Mask_Detection.h5  
--> Run mask_detection_image.py file by using command python mask_detection_image.py --image path/of /your/image/file  
--> To detect the mask in video, provide your machine camera permission.  

# Run mask_detection_video.py file to start live detection of mask on your face
# images and video
# Video Output
![alt text](https://github.com/nitintyagi-ai/mask_detection_livewebcam_video/blob/master/output/output4.gif?raw=true)
# Image Output
![alt text](https://github.com/nitintyagi-ai/mask_detection_livewebcam_video/blob/master/output/output2.PNG?raw=true)  
![alt text](https://github.com/nitintyagi-ai/mask_detection_livewebcam_video/blob/master/output/output3.PNG?raw=true)  
