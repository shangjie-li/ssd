import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import os
import sys
import time
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2
    
from cfg import ssd_cfg
from cfg import voc_classes
from cfg import voc_mean
from cfg import voc_root
from data.voc import VOCDataset
from data.voc import VOCTransform
from data.voc import draw_annotation
from data.voc import draw_prediction
from data.augmentations import Augmentation
from layers.ssd import build_ssd

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')
    
parser = argparse.ArgumentParser(description='SSD Evaluation')
parser.add_argument('--dataset_root', default=voc_root, type=str,
    help='dataset root directory path')
parser.add_argument('--dataset_year', default='2007', type=str,
    help='year of dataset')
parser.add_argument('--dataset_style', default='test', type=str,
    help='style of dataset')
parser.add_argument('--trained_model', default='ssd300_mAP_77.43_v2.pth', type=str,
    help='file name of trained model')
parser.add_argument('--weights_folder', default='weights', type=str,
    help='directory for weights')
parser.add_argument('--output_folder', default='results', type=str,
    help='directory for evaluation results')
parser.add_argument('--cuda', default=True, type=str2bool,
    help='use CUDA to train model')
parser.add_argument('--conf_thresh', default=0.6, type=float,
    help='confidence threshold of objects')
parser.add_argument('--top_k', default=20, type=int,
    help='number of objects to consider per class')
parser.add_argument('--nms_thresh', default=0.5, type=float,
    help='overlap threshold for nms')
parser.add_argument('--display', default=False, type=str2bool,
    help='display detection result')
    
args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
            "using CUDA. \nRun with --cuda for optimal evaluation speed.")

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

def transform_image_for_ssd(img, size=ssd_cfg['min_dim'], mean=voc_mean):
    # convert to float
    img = img.astype(np.float32)
    # resize
    img = cv2.resize(img, (size, size))
    # subtract mean
    img -= mean
    
    img = img[:, :, (2, 1, 0)] # to RGB
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) # to CHW
    
def evaluate():
    net = build_ssd(phase='test', cfg=ssd_cfg,
        conf_thresh=args.conf_thresh, top_k=args.top_k, nms_thresh=args.nms_thresh)
    
    weights = os.path.join(args.weights_folder, args.trained_model)
    print('Start evaluation, loading {}...'.format(weights))
    net.load_weights(weights)
    
    net.eval()
    if args.cuda:
        cudnn.benchmark = True
        net = net.cuda()
        
    dataset = VOCDataset(
        root=args.dataset_root,
        sets=[(args.dataset_year, args.dataset_style)],
        ann_transform=VOCTransform(voc_classes)
    )
    
    current_time = time.asctime(time.localtime(time.time()))
    eval_result = os.path.join(args.output_folder, current_time + '.txt')
    
    time_cost = 0
    num = len(dataset)
    for i in range(num):
        if args.display:
            _, img_a = dataset.pull_image(i)
            _, img_p = dataset.pull_image(i)
            
        ann_id, ann = dataset.pull_annotation(i)
        with open(eval_result, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: ' + ann_id + '\n')
            for k in range(ann.shape[0]):
                box = ann[k, :4]
                classname = voc_classes[int(ann[k, 4])]
                f.write('label: ' + ' || '.join(str(b) for b in box) +
                    ' ' + classname + '\n')
                
        if args.display:
            img_a = draw_annotation(img_a, ann[:, :4], ann[:, 4])
        
        img_id, img = dataset.pull_image(i)
        h, w, _ = img.shape
        x = transform_image_for_ssd(img)
        if args.cuda:
            x = x.cuda()
        
        t0 = time.time()
        out = net(x)
        time_cost += time.time() - t0
        fps = (i + 1) / time_cost
        
        scale = torch.Tensor([w, h, w, h])
        num_preds = 0
        out = out.squeeze(0)
        num_classes = out.size(0)
        
        for cl in range(1, num_classes): # ignore background
            classname = voc_classes[cl - 1]
            num_obj = out[cl, :, 4].nonzero().numel()
            if num_obj > 0 and num_preds == 0:
                with open(eval_result, mode='a') as f:
                    f.write('PREDICTIONS: \n')
                    
            for j in range(num_obj):
                score = round(out[cl, j, 4].item(), 3)
                box = (out[cl, j, :4].clamp(min=0.0, max=1.0) * scale).cpu().numpy()
                num_preds += 1
                
                with open(eval_result, mode='a') as f:
                    f.write(
                        str(num_preds) +
                        ' label: ' + ' || '.join(str(round(b, 1)) for b in box) +
                        ' ' + classname + ' ' + str(score) + '\n'
                    )
                
                if args.display:
                    img_p = draw_prediction(img_p, box, classname, score)
                
        print('Evaluating image {:d}/{:d}...  Speed: {:.2f}fps'.format(
            i + 1, num, fps))
            
        if args.display:
            cv2.imshow('Annotation', img_a)
            cv2.imshow('Prediction', img_p)
            
            # press 'Esc' to shut down, and every key else to continue
            key = cv2.waitKey(0)
            if key == 27:
                break
            else:
                continue
    
if __name__ == '__main__':
    evaluate()
    
