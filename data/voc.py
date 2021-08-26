import torch
import torch.utils.data as data

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import random
import cv2

from cfg import ssd_cfg
from cfg import voc_classes
from cfg import voc_mean
from cfg import voc_root
from .augmentations import Augmentation

class VOCTransform():
    def __init__(self, classes, ignore_difficult=True):
        """
        Inputs:
            classes (list): all classnames of dataset
            ignore_difficult (bool): ignore difficult instances or not
        """
        self.class_to_idx = dict(
            zip(classes, range(len(classes)))
        )
        self.ignore_difficult = ignore_difficult
        
    def __call__(self, target, width=1, height=1):
        """
        Get information of boxes (coordinates and label) in annotation and
        do some transformation.
        
        Inputs:
            target (ET.Element): annotation of input image
            width (int): width of input image
            height (int): height of input image
        Outputs:
            out (list[list[float]]): a list containing lists of
                coordinates and label
        """
        out = []
        for obj in target.iter('object'): # for every <object> element in .xml
            if int(obj.find('difficult').text) == 1: # if the instance is difficult
                if self.ignore_difficult: # if we want ignore the difficult ones
                    continue
                    
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt /= width if i % 2 == 0 else height
                bndbox.append(cur_pt)
            label_idx = self.class_to_idx[name]
            bndbox.append(label_idx)
            out += [bndbox] # bndbox = [xmin, ymin, xmax, ymax, label_idx]
        return out # out = [[xmin, ymin, xmax, ymax, label_idx], [...], [...], ...]
        
class VOCDataset(data.Dataset):
    def __init__(self,
        root=voc_root,
        sets=[('2012', 'trainval')],
        ann_transform=VOCTransform(voc_classes),
        augment=Augmentation(size=ssd_cfg['min_dim'], mean=voc_mean, complicated=True)):
        """
        Inputs:
            root (str): root directory eg. 'xxx/data/VOCdevkit'
            sets (list[(str, str)]): datasets to use
            ann_transform (callable): read annotation and do some transformation
            augment (callable): augment data
        """
        self.root = root
        self.sets = sets
        self.ann_transform = ann_transform
        self.augment = augment
        
        self._img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._ann_path = os.path.join('%s', 'Annotations', '%s.xml')
        
        self.ids = []
        for (year, style) in sets:
            p = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(p, 'ImageSets', 'Main', style + '.txt')):
                # p (str): root directory to 'ImageSet/Main/xxx.txt'
                # line (str): names of images in 'p/JPEGImages'
                self.ids.append((p, line.strip()))
                
    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt
        
    def __len__(self):
        return len(self.ids) # number of images to use
        
    def pull_item(self, index):
        """
        Inputs:
            index (int): index of data in dataset
        Outputs:
            img (tensor, [3, h, w]): image in dataset, RGB
            target (ndarray, [n, 5]): target[:, :4] contains coordinates of 
                boxes, target[:, 4] contains labels
        """
        # read image
        img_id = self.ids[index]
        img = cv2.imread(self._img_path % img_id) # HWC, BGR
        height, width, channels = img.shape
        
        # read annotation
        ann = ET.parse(self._ann_path % img_id).getroot()
        target = np.array(self.ann_transform(ann, width, height))
        
        # augment data
        img, boxes, labels = self.augment(img, target[:, :4], target[:, 4])
        
        img = img[:, :, (2, 1, 0)] # to RGB
        img = torch.from_numpy(img).permute(2, 0, 1) # to CHW
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target
        
    def pull_image(self, index):
        """
        Inputs:
            index (int): index of data in dataset
        Outputs:
            img_id[1] (str): name of image in 'JPEGImages'
            img (ndarray, [h, w, 3]): image in dataset, BGR
        """
        img_id = self.ids[index]
        img = cv2.imread(self._img_path % img_id)
        return img_id[1], img
        
    def pull_annotation(self, index, width=1, height=1):
        """
        Inputs:
            index (int): index of data in dataset
            width (int): width of input image
            height (int): height of input image
        Outputs:
            img_id[1] (str): name of image in 'JPEGImages'
            target (ndarray, [n, 5]): target[:, :4] contains coordinates of 
                boxes, target[:, 4] contains labels
        """
        img_id = self.ids[index]
        ann = ET.parse(self._ann_path % img_id).getroot()
        target = np.array(self.ann_transform(ann, width, height))
        return img_id[1], target
        
def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b
    
def draw_annotation(img, boxes, labels, colors=None, classes=voc_classes):
    """
    Inputs:
        img (ndarray, [h, w, 3]): image of BGR
        boxes (ndarray, [n, 4]): coordinates of boxes
        labels (ndarray, [n,]): labels
        colors (list[(b, g, r)]): colors of boxes and labels
        classes (list[str]): names of all classes
    Outputs:
        img (ndarray, [h, w, 3]): image of BGR
    """
    dim = len(boxes.shape)
    if dim != 2:
        raise ValueError('len(boxes.shape) must be 2.')
    
    white = (255, 255, 255)
    face = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.4
    thick = 1
    
    num = boxes.shape[0]
    if colors is None:
        colors = []
        for i in range(num):
            colors.append((create_random_color()))
    
    for i in range(num):
        c = colors[i]
        x1, y1, x2, y2 = boxes[i][:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thickness=1)
        
        text = classes[int(labels[i])]
        tw, th = cv2.getTextSize(text, face, scale, thick)[0]
        cv2.rectangle(img, (x1, y1), (x1 + tw, y1 + th + 3), c, -1)
        cv2.putText(img, text, (x1, y1 + th), face, scale, white, thick, cv2.LINE_AA)
    
    return img

def draw_prediction(img, box, classname, confidence, color=None):
    """
    Inputs:
        img (ndarray, [h, w, 3]): image of BGR
        box (ndarray, [4]): coordinates of box
        classname (str): name of class
        confidence (float): confidence of prediction
        color (tuple(b, g, r)): color of box
    Outputs:
        img (ndarray, [h, w, 3]): image of BGR
    """
    dim = len(box.shape)
    if dim != 1:
        raise ValueError('len(boxes.shape) must be 1.')
    
    white = (255, 255, 255)
    face = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.4
    thick = 1
    
    if color is None:
        color = (create_random_color())
    
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
    
    text = '%s: %s' % (classname, str(confidence))
    tw, th = cv2.getTextSize(text, face, scale, thick)[0]
    cv2.rectangle(img, (x1, y1), (x1 + tw, y1 + th + 3), color, -1)
    cv2.putText(img, text, (x1, y1 + th), face, scale, white, thick, cv2.LINE_AA)
    
    return img
