import os
import torch
import numpy as np
import cv2
import csv
import time
import glob
from thop import profile
from queue import Queue
from PIL import Image
from src.model_inference_ObjectDetect_Elanet import detect
from models.multi_tasks import ELANetV3_modified_sigapore

def convert_xyxy_to_xywh(x1,y1,x2,y2):
    w = x2 - x1
    h = y2 - y1
    return x1,y1,w,h
    
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('paltform:',device)

# Load Object Detection model
checkpoint_path = './models/weights/BEST_checkpoint.pth.tar'
Tensor = torch.cuda.FloatTensor
model_od = fun_load_od_model(checkpoint_path)


label_dict = {'vehicle':1, 'pedestrian':2, 'scooter':3, 'bicycle':4} 

# detect on write to csv file
columns = ["image_filename", "label_id", "x", "y", "w", "h","confidence"]
with open('./ivslab_test_public/submission.csv', 'w', newline='') as file:
    csvfile = csv.writer(file)
    img_list = sorted( glob.glob('./ivslab_test_public/JPEGImages/All/*.jpg'))
    csvfile.writerow(columns)
 
    for img_path in img_list:
        img_name = img_path.split('/')
        img_name = img_name[-1]
        print(img_name)
        frame_np_img = cv2.imread(img_path)
        frame_pil_img = Image.fromarray(frame_np_img)

        #_, bboxes = detect(model_od, frame_pil_img, min_score=0.50, max_overlap=0.5, top_k=50,device=device)
        # min_score=0.01, max_overlap=0.45,top_k=200
        _, bboxes = detect(model_od, frame_pil_img, min_score=0.5, max_overlap=0.5, top_k=50,device=device)

        for bbox in bboxes :
            #print('box:',bbox)

            points = bbox.get('points')
            x1,y1,x2,y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
            cls_ = bbox.get('label')

            frame_np_img = cv2.putText(frame_np_img, cls_, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            frame_np_img = cv2.rectangle(frame_np_img, (x1,y1), (x2,y2), (0, 0, 255), 3)
            cv2.imwrite('./ivslab_test_public/test/test_'+img_name,frame_np_img)
            
            x,y,w,h = convert_xyxy_to_xywh(x1,y1,x2,y2)
            cls_label = label_dict[cls_]

            csvfile.writerow([img_name,cls_label,x,y,w,h,0])
            
