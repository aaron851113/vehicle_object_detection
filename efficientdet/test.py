# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
import cv2
import numpy as np
import json
import os
import glob
from torch.backends import cudnn
from matplotlib import colors
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def display(preds, imgs, imwrite=True):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            print(score,x1, y1, x2, y2,score)
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            print('Create Test dir.')
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

compound_coef = 1
input_size = 640
#img_path = '../../AIdea/ivslab_train/JPEGImages/All/1_40_00.mp4/1_40_00.mp4_00007.jpg'
img_list = sorted(glob.glob('../../AIdea/ivslab_train/JPEGImages/All/1_40_00.mp4/*.jpg'))

color_list = standard_to_bgr(STANDARD_COLORS)

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.005
iou_threshold = 0.005

cudnn.benchmark = True

obj_list = []
with open(os.path.join('../data', 'label_map.json'),'r') as f:
    obj_json = json.load(f)
    for obj in obj_json:
        if "background" in obj:
            continue
        obj_list.append(obj)
    print('obj_list:',obj_list)

i=0

for img_path in img_list:
    if i>5:
        break
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)

    x = x.to(torch.float32).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'efficientdet-d{compound_coef}_15.pth', map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    model = model.cuda()
    
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        print(out)



    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imwrite=True)
    i+=1