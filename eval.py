import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

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
from utils.utils import create_plot_color
from utils.utils import smooth_data

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')
    
parser = argparse.ArgumentParser(description='SSD Evaluation')
parser.add_argument('--dataset_root', default=voc_root, type=str,
    help='dataset root directory path')
parser.add_argument('--dataset_year', default='2007', type=str,
    help='year of dataset')
parser.add_argument('--dataset_style', default='test', type=str,
    help='style of dataset')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str,
    help='file name of trained model')
parser.add_argument('--output_folder', default='results', type=str,
    help='directory for saving prediction results')
parser.add_argument('--cuda', default=True, type=str2bool,
    help='use CUDA to train model')
parser.add_argument('--conf_thresh', default=0.1, type=float,
    help='confidence threshold of objects')
parser.add_argument('--top_k', default=20, type=int,
    help='number of objects to consider per class')
parser.add_argument('--nms_thresh', default=0.5, type=float,
    help='overlap threshold for nms')
parser.add_argument('--display', default=False, type=str2bool,
    help='display prediction results')
    
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

def get_voc_result_file_name(classname=None):
    # e.g. data/VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    file_dir = os.path.join(voc_root, 'VOC' + args.dataset_year, 'results')
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if classname:
        return os.path.join(file_dir, 'det_' + args.dataset_style + '_%s.txt' % (classname))
    else:
        return file_dir

def wirte_pred_result_for_evaluation(all_boxes, dataset):
    for cls_idx, classname in enumerate(voc_classes):
        print('Writing {:s} VOC result file...'.format(classname))
        
        file_path = get_voc_result_file_name(classname)
        with open(file_path, 'wt') as f:
            for img_idx, img_name in enumerate(dataset.ids):
                infos = all_boxes[cls_idx + 1][img_idx]
                
                if infos == []:
                    continue
                for k in range(infos.shape[0]):
                    f.write(
                        # id, score, xmin, ymin, xmax, ymax
                        '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                            img_name[1], infos[k, 4],
                            infos[k, 0], infos[k, 1], infos[k, 2], infos[k, 3],
                        )
                    )
    
def parse_voc_ground_truth(file_name):
    tree = ET.parse(file_name)
    objects = []
    for obj in tree.findall('object'):
        x = {}
        x['name'] = obj.find('name').text
        x['pose'] = obj.find('pose').text
        x['truncated'] = int(obj.find('truncated').text)
        x['difficult'] = int(obj.find('difficult').text)
        x['bbox'] = [
            int(obj.find('bndbox').find('xmin').text) - 1,
            int(obj.find('bndbox').find('ymin').text) - 1,
            int(obj.find('bndbox').find('xmax').text) - 1,
            int(obj.find('bndbox').find('ymax').text) - 1,
        ]
        objects.append(x)
    return objects

def compute_iou(boxes_gt, box):
    area_a = (boxes_gt[:, 2] - boxes_gt[:, 0]) * (boxes_gt[:, 3] - boxes_gt[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    
    xmin = np.maximum(boxes_gt[:, 0], box[0])
    ymin = np.maximum(boxes_gt[:, 1], box[1])
    xmax = np.minimum(boxes_gt[:, 2], box[2])
    ymax = np.minimum(boxes_gt[:, 3], box[3])
    w = np.maximum(xmax - xmin, 0)
    h = np.maximum(ymax - ymin, 0)
    
    inter = w * h
    union = area_a + area_b - inter
    return inter / union

def compute_ap(rec, prec):
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap

def do_voc_evaluation(classname, iou_thresh=0.5):
    # read ground truth
    image_set_file = os.path.join(voc_root, 'VOC' + args.dataset_year,
        'ImageSets', 'Main', args.dataset_style + '.txt')
    with open(image_set_file, 'r') as f:
        lines = f.readlines()
    image_names = [line.strip() for line in lines]
    anno_path = os.path.join(voc_root, 'VOC' + args.dataset_year,
        'Annotations', '%s.xml')
    
    all_gts, all_gts_for_one_class, num_pos = {}, {}, 0
    for i, image_name in enumerate(image_names):
        all_gts[image_name] = parse_voc_ground_truth(anno_path % image_name)
    for image_name in image_names:
        gts = [obj for obj in all_gts[image_name] if obj['name'] == classname]
        boxes = np.array([x['bbox'] for x in gts])
        is_difficult = np.array([x['difficult'] for x in gts]).astype(np.bool)
        is_detected = [False] * len(gts)
        num_pos += sum(~is_difficult)
        all_gts_for_one_class[image_name] = {
            'boxes': boxes,
            'is_difficult': is_difficult,
            'is_detected': is_detected,
        }
    
    # read prediction value and compute AP
    pred_file = get_voc_result_file_name(classname)
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    dets = [line.strip().split(' ') for line in lines]
    num = len(dets)
    if num > 0:
        image_ids = [x[0] for x in dets]
        boxes = np.array([[float(z) for z in x[2:]] for x in dets])
        scores = np.array([float(x[1]) for x in dets])
        
        indices = np.argsort(-scores)
        image_ids = [image_ids[idx] for idx in indices]
        boxes = boxes[indices, :]
        
        tp, fp = np.zeros(num), np.zeros(num)
        for i in range(num):
            box = boxes[i, :].astype(np.float32)
            gts = all_gts_for_one_class[image_ids[i]]
            boxes_gt = gts['boxes'].astype(np.float32)
            iou_max = -np.inf
            if boxes_gt.size > 0:
                overlaps = compute_iou(boxes_gt, box)
                iou_max = np.max(overlaps)
                idxm = np.argmax(overlaps)
            if iou_max > iou_thresh:
                if not gts['is_difficult'][idxm]:
                    if not gts['is_detected'][idxm]:
                        tp[i] = 1
                        gts['is_detected'][idxm] = 1
                    else:
                        fp[i] = 1
            else:
                fp[i] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / num_pos
        prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps) # avoid divide by zero
        return rec, prec, compute_ap(rec, prec)
    else:
        return None, None, None

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
    dataset = VOCDataset(
        root=args.dataset_root,
        sets=[(args.dataset_year, args.dataset_style)],
        ann_transform=VOCTransform(voc_classes)
    )
    p = os.path.join(args.dataset_root, 'VOC' + args.dataset_year,
        'ImageSets', 'Main', args.dataset_style + '.txt')
    current_time = time.asctime(time.localtime(time.time()))
    pred_result_file = os.path.join(args.output_folder, current_time + '.txt')
    print('Loading dataset from {}...'.format(p))
    print('Saving prediction results to {}...'.format(pred_result_file))
    
    net = build_ssd(phase='test', cfg=ssd_cfg,
        conf_thresh=args.conf_thresh, top_k=args.top_k, nms_thresh=args.nms_thresh)
    weights = args.trained_model
    print('Starting evaluating, loading model {}...'.format(weights))
    net.load_weights(weights)
    
    net.eval()
    if args.cuda:
        cudnn.benchmark = True
        net = net.cuda()
        net(torch.randn(1, 3, 300, 300).cuda()) # initialize the net
    else:
        net(torch.randn(1, 3, 300, 300)) # initialize the net
        
    # the element of all_boxes[class][image] is a [N, 5] ndarray
    # in [[x1, y1, x2, y2, score], ...]
    num = len(dataset)
    all_boxes = [[[] for _ in range(num)] for _ in range(ssd_cfg['num_classes'])]
    interrupted = False
    time_cost = 0
    
    for i in range(num):
        ann_id, ann = dataset.pull_annotation(i)
        if args.display:
            _, img_a = dataset.pull_image(i)
            _, img_p = dataset.pull_image(i)
            
        with open(pred_result_file, mode='a') as f:
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
        if num_classes != ssd_cfg['num_classes']:
            raise ValueError("out.size(0) must be equal to ssd_cfg['num_classes']")
        
        for cls in range(1, num_classes): # ignore background
            classname = voc_classes[cls - 1]
            num_objs = out[cls, :, 4].nonzero().numel()
            if num_objs > 0 and num_preds == 0:
                with open(pred_result_file, mode='a') as f:
                    f.write('PREDICTIONS: \n')
                    
            scores = out[cls, :num_objs, 4].cpu().numpy()
            boxes = out[cls, :num_objs, :4].clamp(min=0.0, max=1.0) * scale
            boxes = boxes.cpu().numpy()
            
            for j in range(num_objs):
                score, box = scores[j], boxes[j]
                num_preds += 1
                with open(pred_result_file, mode='a') as f:
                    f.write(
                        str(num_preds) +
                        ' label: ' + ' || '.join(str(round(b, 1)) for b in box) +
                        ' ' + classname + ' ' + str(round(score, 3)) + '\n'
                    )
                if args.display:
                    img_p = draw_prediction(img_p, box, classname, round(score, 3))
            all_boxes[cls][i] = np.hstack((boxes, scores[:, None]))
            
        print('Evaluating image {:d}/{:d}...  Speed: {:.2f}fps.'.format(
            i + 1, num, fps))
        if args.display:
            cv2.imshow('Annotation', img_a)
            cv2.imshow('Prediction', img_p)
            print('Press [Esc] to shut down or every key else to continue.')
            key = cv2.waitKey(0)
            if key == 27:
                interrupted = True
                break
            else:
                continue
        
    if not interrupted:
        fig, ax = plt.subplots(figsize=(8, 8))
        print('\nWriting VOC results to {}...'.format(get_voc_result_file_name()))
        wirte_pred_result_for_evaluation(all_boxes, dataset)
        
        aps = []
        print('\nAverage Precision:')
        for cls_idx, classname in enumerate(voc_classes):
            rec, prec, ap = do_voc_evaluation(classname)
            if ap:
                print('{:s}: {:.3f}'.format(classname, ap))
                aps.append(ap)
                rec, prec = smooth_data(rec, prec)
                ax.scatter(rec, prec, s=10, color=create_plot_color(), label=classname)
            else:
                print('{:s}: None'.format(classname))
                aps.append(0)
        print('\nMean Average Precision: {:.3f}'.format(np.mean(aps)))
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.legend(bbox_to_anchor=(0, 0), loc='lower left', ncol=4)
        plt.show()

if __name__ == '__main__':
    evaluate()
