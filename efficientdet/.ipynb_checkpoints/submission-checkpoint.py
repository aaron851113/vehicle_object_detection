import time
import torch
from torch.backends import cudnn
from matplotlib import colors
from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import csv
import glob
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box



"""
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
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd352.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((352, 352))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


label_dict = {'vehicle':1, 'pedestrian':2, 'scooter':3, 'bicycle':4} 

# detect on write to csv file
columns = ["image_filename", "label_id", "x", "y", "w", "h","confidence"]
with open('../../AIdea/ivslab_test_public/submission.csv', 'w', newline='') as file:
    csvfile = csv.writer(file)
    img_list = sorted( glob.glob('../../AIdea/ivslab_test_public/JPEGImages/All/*.jpg'))
    csvfile.writerow(columns)
 
    for img_path in img_list:
        img_name = img_path.split('/')
        img_name = img_name[-1]
        print(img_name)
        
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        
        det_boxes, det_labels, det_scores = detect(original_image, img_name, min_score=0.3, max_overlap=0.5, top_k=50, save_img=True)
        
        if det_labels == ['background']:
            continue
        
        for i in range(len(det_labels)):
            x1,y1,x2,y2 = int(det_boxes[i][0]), int(det_boxes[i][1]), int(det_boxes[i][2]), int(det_boxes[i][3])
            label = det_labels[i]
            score = round(det_scores[0][i].item(),2)
            
            cls_label = label_dict[label]
            x,y,w,h = convert_xyxy_to_xywh(x1,y1,x2,y2)
            
            csvfile.writerow([img_name,cls_label,x,y,w,h,score])
            
