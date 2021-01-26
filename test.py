import os
import torch
import numpy as np
import cv2
import time
import glob
from thop import profile
from queue import Queue
from PIL import Image
from src.model_inference_ObjectDetect_Elanet import detect
from models.multi_tasks import ELANetV3_modified_sigapore

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('paltform:',device)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('=>>>>  Scuessfully Make folder :' + directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def fun_load_od_model(checkpoint_path):
    model_od = ELANetV3_modified_sigapore.SSD352(n_classes=5)
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
checkpoint_path = './models/weights/BEST_checkpoint.pth.tar'
Tensor = torch.cuda.FloatTensor
createFolder('./demo')
model_od = fun_load_od_model(checkpoint_path)

count = 0
for img_path in glob.glob('./ivslab_train/JPEGImages/All/1_40_00.mp4/*.jpg'):
    img_name = img_path.split('/')
    img_name = img_name[-1]
    print(img_name)
    ### test 
    count += 1 
    if count == 10 :
        break
        
    frame_np_img = cv2.imread(img_path)
    frame_pil_img = Image.fromarray(frame_np_img)
    
    t_start_decision = time.time()
    #_, bboxes = detect(model_od, frame_pil_img, min_score=0.50, max_overlap=0.5, top_k=50,device=device)
    # min_score=0.01, max_overlap=0.45,top_k=200
    _, bboxes = detect(model_od, frame_pil_img, min_score=0.25, max_overlap=0.45, top_k=50,device=device)
    t_end_decision = time.time()
    print('bboxes:',bboxes)
    
    for bbox in bboxes :
        print('box:',bbox)
        
        points = bbox.get('points')
        x1,y1,x2,y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
        cls_ = bbox.get('label')
        
        frame_np_img = cv2.putText(frame_np_img, cls_, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame_np_img = cv2.rectangle(frame_np_img, (x1,y1), (x2,y2), (0, 0, 255), 3)
    
        cv2.imwrite('./demo/demo_'+img_name,frame_np_img)

    
    
        