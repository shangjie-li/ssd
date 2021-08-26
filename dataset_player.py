import sys
import numpy as np
import matplotlib.pyplot as plt # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from cfg import ssd_cfg
from cfg import voc_classes
from cfg import voc_mean
from cfg import voc_root
from data.voc import VOCDataset, VOCTransform
from data.voc import draw_annotation
from data.augmentations import Augmentation

if __name__ == '__main__':
    dataset = VOCDataset(
        root=voc_root,
        sets=[('2012', 'trainval')],
        ann_transform=VOCTransform(voc_classes),
        augment=Augmentation(size=ssd_cfg['min_dim'], mean=voc_mean, complicated=True)
    )
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i, len(dataset)))
        print('Original Data:')
        img_id, img = dataset.pull_image(i)
        print(' image_id:', img_id, '\n size:', img.shape)
        ann_id, ann = dataset.pull_annotation(i)
        print(' annotation_id:', ann_id, '\n boxes:\n', ann[:, :4])
        
        img = draw_annotation(img, ann[:, :4], ann[:, 4])
        cv2.imshow('Original Data', img)
        
        im, gt = dataset[i]
        print('Augmented Data:')
        print(' size:', im.shape, '\n boxes:\n', gt[:, :4])
        print()
        
        imga = im.permute(1, 2, 0).numpy() # to HWC
        imga = imga[:, :, (2, 1, 0)] # to BGR
        imga = imga.copy().astype(np.uint8) # to Ints, copy() is needed for some reason
        
        h, w, _ = imga.shape
        boxes, labels = gt[:, :4], gt[:, 4]
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        
        imga = draw_annotation(imga, boxes, labels)
        cv2.imshow('Augmented Data', imga)
        
        # press 'Esc' to shut down, and every key else to continue
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
