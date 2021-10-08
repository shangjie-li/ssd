import torch
import numpy as np
import cv2
from numpy import random

def intersect(boxes, box):
    max_xy = np.minimum(boxes[:, 2:], box[2:])
    min_xy = np.maximum(boxes[:, :2], box[:2])
    dxdy = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return dxdy[:, 0] * dxdy[:, 1]

def jaccard_numpy(boxes, box):
    inter = intersect(boxes, box)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    union = area_a + area_b - inter
    return inter / union

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class ToFloats():
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels

class ToAbsoluteCoords():
    def __call__(self, img, boxes=None, labels=None):
        h, w, c = img.shape
        boxes[:, 0] *= w
        boxes[:, 1] *= h
        boxes[:, 2] *= w
        boxes[:, 3] *= h
        return img, boxes, labels

class ToPercentCoords():
    def __call__(self, img, boxes=None, labels=None):
        h, w, c = img.shape
        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h
        return img, boxes, labels

class RandomContrast():
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2): # randomly create a number between [0, 2)
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, boxes, labels

class ConvertColor():
    def __init__(self, current='BGR', transform='HSV'):
        self.current = current
        self.transform = transform
        
    def __call__(self, img, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, boxes, labels

class RandomSaturation():
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0
        
    def __call__(self, img, boxes=None, labels=None): # for HSV image
        if random.randint(2): # randomly create a number between [0, 2)
            img[:, :, 1] *= random.uniform(self.lower, self.upper)
        return img, boxes, labels

class RandomHue():
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        
    def __call__(self, img, boxes=None, labels=None): # for HSV image
        if random.randint(2): # randomly create a number between [0, 2)
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img, boxes, labels

class RandomBrightness():
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2): # randomly create a number between [0, 2)
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels

class RandomLightingNoise():
    def __init__(self):
        self.perms = (
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0),
        )
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2): # randomly create a number between [0, 2)
            swap = self.perms[random.randint(len(self.perms))]
            img = img[:, :, swap]
        return img, boxes, labels

class PhotometricDistort():
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current='BGR', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(),
        ]
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()
        
    def __call__(self, img, boxes, labels):
        img = img.copy()
        img, boxes, labels = self.rb(img, boxes, labels)
        if random.randint(2):
            t = Compose(self.pd[:-1])
        else:
            t = Compose(self.pd[1:])
        img, boxes, labels = t(img, boxes, labels)
        return self.rln(img, boxes, labels)

class Expand():
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels
        
        height, width, depth = img.shape
        img, boxes, labels = img.copy(), boxes.copy(), labels.copy()
        
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        
        h, w = int(height * ratio), int(width * ratio)
        expand_img = np.zeros((h, w, depth), dtype=img.dtype)
        expand_img[:, :, :] = self.mean
        expand_img[int(top):int(top + height), int(left):int(left + width)] = img
        
        expand_boxes = boxes
        expand_boxes[:, :2] += (int(left), int(top))
        expand_boxes[:, 2:] += (int(left), int(top))

        return expand_img, expand_boxes, labels

class RandomSampleCrop():
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # min_iou and max_iou threshold
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # no iou threshold
            (None, None),
        )

    def __call__(self, img, boxes, labels):
        height, width, _ = img.shape
        img, boxes, labels = img.copy(), boxes.copy(), labels.copy()
        
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, boxes, labels
            
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            
            for _ in range(50):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                # crop the image
                current_img = img[rect[1]:rect[3], rect[0]:rect[2], :]
                
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                if not mask.any():
                    continue
                
                # adjust the boxes
                current_boxes = boxes[mask, :]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                
                # crop the labels
                current_labels = labels[mask]

                return current_img, current_boxes, current_labels

class RandomMirror():
    def __call__(self, img, boxes, labels):
        h, w, _ = img.shape
        if random.randint(2): # randomly create a number between [0, 2)
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = w - boxes[:, 2::-2] # (x1, x2) = w - (x2, x1)
        return img, boxes, labels

class Resize():
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size)) # assume coords are percent form 
        return img, boxes, labels

class SubtractMeans():
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        
    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels

class Augmentation():
    def __init__(self, size, mean):
        """
        Inputs:
            size (int): size of image to resize, h = w = size
            mean (list[uint8, uint8, uint8]): mean value of image
        """
        self.mean = mean
        self.size = size
        self.augment = [
            ToFloats(),
            ToAbsoluteCoords(), # percent coords to absolute coords
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(), # absolute coords to percent coords
            Resize(self.size), # Scale to size * size
            SubtractMeans(self.mean),
        ]
        
    def __call__(self, img, boxes, labels):
        t = Compose(self.augment)
        return t(img, boxes, labels)
