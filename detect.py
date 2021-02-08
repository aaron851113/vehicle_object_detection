from torchvision import transforms
from utils import *
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def detect(original_image, img_name, img_path, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """
    # Transform
    image = normalize(to_tensor(resize(original_image)))
    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [Elan_od_singapore_rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image
    
    # Annotate
    annotate_image = original_image
    #frame_np_img = cv2.imread(img_path)
    #frame_pil_img = Image.fromarray(frame_np_img)
    frame_pil_img = cv2.cvtColor(np.array(annotate_image), cv2.COLOR_RGB2BGR)

    for i in range(len(det_labels)):
        x1,y1,x2,y2 = int(det_boxes[i][0]), int(det_boxes[i][1]), int(det_boxes[i][2]), int(det_boxes[i][3])
        label = det_labels[i]
        score = round(det_scores[0][i].item(),2)
        score = str(score)
        
        frame_pil_img = cv2.putText(frame_pil_img, label+'-'+score, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        frame_pil_img = cv2.rectangle(frame_pil_img, (x1,y1), (x2,y2), (0, 0, 255), 3)
    
    cv2.imwrite('./demo/demo_'+img_name,frame_pil_img)


if __name__ == '__main__':
    count = 0
    for img_path in glob.glob('../AIdea/ivslab_train/JPEGImages/All/1_40_00.mp4/*.jpg'):
        img_name = img_path.split('/')
        img_name = img_name[-1]
        print(img_name)
        ### test 
        count += 1 
        if count == 5 :
            break
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        detect(original_image, img_name, img_path, min_score=0.2, max_overlap=0.5, top_k=200)
