import os
import torch
import numpy as np
import cv2
import time
import glob
from thop import profile
from queue import Queue
from PIL import Image
from EfficientNet import SSD352EFFB0, MultiBoxLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('paltform:',device)
n_classes = 5

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('=>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def fun_load_od_model(checkpoint_path):
    model_od = SSD352EFFB0(width_coeff=1,depth_coeff=1.1,n_classes=n_classes)
    model_od_dict = model_od.state_dict()
    model_refernce = torch.load(checkpoint_path, map_location=device)
    model_refernce = model_refernce['model'].state_dict()
    pretrained_dict = {k: v for k, v in model_refernce.items() if k in model_od_dict}
    model_od_dict.update(pretrained_dict)
    model_od.load_state_dict(model_od_dict)
    model_od = model_od.to(device)
    model_od.eval()
    print('load model (Object detection) : successful')
    return model_od



# Load Object Detection model
checkpoint_path = './BEST_checkpoint.pth.tar'
Tensor = torch.cuda.FloatTensor
createFolder('./demo')

count = 0
for img_path in glob.glob('../AIdea/ivslab_train/JPEGImages/All/1_40_00.mp4/*.jpg'):
    img_name = img_path.split('/')
    img_name = img_name[-1]
    print(img_name)
    ### test 
    count += 1 
    if count == 10 :
        break
        
    frame_np_img = cv2.imread(img_path)
    frame_pil_img = Image.fromarray(frame_np_img)
    frame_pil_img = frame_pil_img.to(device)
    
    predicted_locs, predicted_scores = model(frame_pil_img)
    bboxes, labels, scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
                                                             max_overlap=0.5, top_k=200)

    print('bboxes:',bboxes)
    
    for bbox in bboxes :
        print('box:',bbox)
        
        points = bbox.get('points')
        x1,y1,x2,y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
        cls_ = bbox.get('label')
        
        frame_np_img = cv2.putText(frame_np_img, cls_, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame_np_img = cv2.rectangle(frame_np_img, (x1,y1), (x2,y2), (0, 0, 255), 3)
    
        cv2.imwrite('./demo/demo_'+img_name,frame_np_img)

    
    
        