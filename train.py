import torch
import torch.nn as nn
import torch.optim as optim
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

from cfg import voc_mean
from cfg import voc_root
from cfg import voc_classes
from cfg import ssd_cfg
from data.augmentations import Augmentation
from data.voc import VOCDataset
from data.voc import VOCTransform
from utils.utils import collate
from utils.utils import adjust_lr
from layers.ssd import build_ssd
from layers.multi_box_loss import MultiBoxLoss

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')
    
parser = argparse.ArgumentParser(description='SSD Training')
parser.add_argument('--dataset_root', default=voc_root, type=str,
    help='dataset root directory path')
parser.add_argument('--dataset_year', default='2012', type=str,
    help='year of dataset')
parser.add_argument('--dataset_style', default='trainval', type=str,
    help='style of dataset')
parser.add_argument('--resume', default=None, type=str,
    help='file path of model to resume training')
parser.add_argument('--pretrained_model', default='weights/vgg16_reducedfc.pth', type=str,
    help='file path of pretrained model')
parser.add_argument('--batch_size', default=16, type=int,
    help='batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
    help='number of workers used in data loading')
parser.add_argument('--cuda', default=True, type=str2bool,
    help='use CUDA to train model')
parser.add_argument('--lr', default=1e-3, type=float,
    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
    help='momentum value for optimize')
parser.add_argument('--weight_decay', default=5e-4, type=float,
    help='weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
    help='gamma update for SGD')
parser.add_argument('--saving_folder', default='weights', type=str,
    help='directory for saving weights')
parser.add_argument('--saving_interval', default=10000, type=int,
    help='interval of saving model')

args = parser.parse_args()

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
            "using CUDA. \nRun with --cuda for optimal training speed.")
            
if not os.path.exists(args.saving_folder):
    os.mkdir(args.saving_folder)
    
def train():
    dataset = VOCDataset(
        root=args.dataset_root,
        sets=[(args.dataset_year, args.dataset_style)],
        ann_transform=VOCTransform(voc_classes),
        augment=Augmentation(size=ssd_cfg['min_dim'], mean=voc_mean, complicated=True)
    )
    data_loader = data.DataLoader(dataset, args.batch_size,
        num_workers=args.num_workers, shuffle=True, collate_fn=collate,
        pin_memory=True)
    p = os.path.join(args.dataset_root, 'VOC' + args.dataset_year,
        'ImageSets', 'Main', args.dataset_style + '.txt')
    print('Loading dataset from {}...'.format(p))
    
    ssd_net = build_ssd(phase='train', cfg=ssd_cfg)
    if args.resume:
        resume_weights = args.resume
        print('Resuming training, loading {}...'.format(resume_weights))
        ssd_net.load_weights(resume_weights)
    else:
        vgg_weights = args.pretrained_model
        print('Starting training, loading pretrained model {}...'.format(vgg_weights))
        ssd_net.load_vgg_weights(vgg_weights)
        ssd_net.init_extra_weights()
        
    # DataParallel wraps the underlying module, but when saving and loading
    # we don't want that.
    net = ssd_net
    net.train()
    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net = net.cuda()
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg=ssd_cfg, iou_thresh=0.5, neg_pos_ratio=3,
        cuda=args.cuda)
    
    step_index = 0
    epoch, iter_num_per_epoch = 0, len(dataset) // args.batch_size
    batch_iterator = iter(data_loader)
    print('\nUsing the specified args:\n', args, '\n')
    
    for it in range(1, ssd_cfg['max_iter'] + 1):
        if it % iter_num_per_epoch == 0:
            epoch += 1
            
        if it in ssd_cfg['lr_steps']:
            step_index += 1
            adjust_lr(optimizer, args.gamma, step_index, args.lr)
            
        try:
            imgs, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            imgs, targets = next(batch_iterator)
            
        if args.cuda:
            imgs = imgs.cuda()
            targets = [ann.cuda() for ann in targets]
            
        t0 = time.time()
        out = net(imgs) # inference
        
        optimizer.zero_grad()
        l_loss, c_loss = criterion(out, targets)
        
        loss = l_loss + c_loss
        loss.backward()
        
        optimizer.step()
        t1 = time.time()
        
        ll = l_loss.item()
        cl = c_loss.item()
        
        if it % 10 == 0:
            print('Epoch: %s || Iter: %s || Loss: %.4f || Timer: %.4f sec.' % (
                repr(epoch), repr(it), ll + cl, t1 - t0))
                
        if it % args.saving_interval == 0 or it == ssd_cfg['max_iter']:
            print('Saving state, iter:', it)
            torch.save(ssd_net.state_dict(), args.saving_folder + '/ssd300_' +
                repr(epoch) + '_' + repr(it) + '.pth')

if __name__ == '__main__':
    train()
