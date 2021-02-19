import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, dim, keep_difficult=False):
    #def __init__(self, data_folder, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.dim = dim

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        if self.split == 'TRAIN':
            with open(os.path.join(data_folder, 'TRAIN_images.json'), 'r') as j:
                self.images = json.load(j)
            with open(os.path.join(data_folder, 'TRAIN_objects.json'), 'r') as j:
                self.objects = json.load(j)
        else :
            with open(os.path.join(data_folder, 'VALID_images.json'), 'r') as j:
                self.images = json.load(j)
            with open(os.path.join(data_folder, 'VALID_objects.json'), 'r') as j:
                self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        
        label_list = []
        with open(os.path.join(self.data_folder, 'label_map.json'),'r') as f:
            label_list = json.load(f)

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = []
        labels = []
        for object in objects:
            boxes.append(object['point'])
            label_str = object['label']
            label_int = label_list[label_str]
            labels.append(label_int)
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.LongTensor(labels)  # (n_objects)
        #difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, dim=self.dim, split=self.split)

        return image, boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        #difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            #difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    
    
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PascalVOCDataset("./data",split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               collate_fn=train_dataset.collate_fn,num_workers=1,
                                               pin_memory=True)
    for i, (images, boxes, labels) in enumerate(train_loader):
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        print(images,boxes,labels)