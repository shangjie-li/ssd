import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import os
import sys

from cfg import ssd_cfg
from .prior_boxes import PriorBoxes
from .utils import decode, nms

class Detect():
    def __init__(self, cfg, conf_thresh=0.01, top_k=200, nms_thresh=0.5):
        """
        Inputs:
            cfg (dict): basic parameters of ssd
            conf_thresh (float): confidence threshold of objects
            top_k (int): number of objects to consider per class
            nms_thresh (float): overlap threshold for nms
        """
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if self.nms_thresh <= 0:
            raise ValueError('nms_thresh must be non negative.')
        self.variance = self.cfg['variance']
        
    def forward(self, loc_data, conf_data, priors):
        """
        Inputs:
            loc_data (tensor, [batch_size, num_priors, 4]): 
                coords of prediction boxes (percent, g^ccwh form)
            conf_data (tensor, [batch_size, num_priors, num_classes]):
                confidence of each class (0 for background) for each prior box
            priors (tensor, [num_priors, 4]):
                coords of prior boxes (percent, ccwh form)
        Outputs:
            output (tensor, [batch_size, num_classes, top_k, 5]):
                output[:, :, :, :4]: coords of prediction boxes (percent, xyxy form)
                output[:, :, :, 4]: confidences
        """
        
        # Warning: Legacy autograd function with non-static forward method
        # is deprecated and will be removed in PyTorch 1.3.
        
        batch_size, num_priors = loc_data.size(0), loc_data.size(1)
        output = torch.zeros(batch_size, self.num_classes, self.top_k, 5)
        
        softmax = nn.Softmax(dim=-1)
        conf_preds = softmax(conf_data) # [batch_size, num_priors, num_classes]
        
        # change dimension order of num_priors and num_classes
        conf_preds = conf_preds.permute(0, 2, 1) # [batch_size, num_classes, num_priors]
        
        for i in range(batch_size):
            # decoded_boxes (tensor, [num_priors, 4])
            decoded_boxes = decode(loc_data[i], priors, self.variance)
            
            for cl in range(1, self.num_classes): # ignore background
                
                # Note: 
                # a. Mask of tensor (torch.uint8) should have the same size
                #    of the tensor, the size of a[mask] <= size of mask.
                # b. Indices of tensor (torch.long) needn't have the same size
                #    of the tensor, the size of a[indices] = size of indices.
                
                mask = conf_preds[i][cl].gt(self.conf_thresh) # [num_priors]
                boxes = decoded_boxes[mask] # [n, 4]
                scores = conf_preds[i][cl][mask] # [n]
                
                # indices (tensor, [count])
                indices = nms(boxes, scores, self.top_k, self.nms_thresh)
                count = indices.size(0)
                output[i, cl, :count] = torch.cat(
                    [
                    boxes[indices], # [count, 4]
                    scores[indices].unsqueeze(1), # [count, 1]
                    ], dim=1
                )
        return output

class L2Norm(nn.Module):
    def __init__(self, n_channels, gamma):
        super(L2Norm, self).__init__()
        
        self.n_channels = n_channels
        self.gamma = gamma
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)
        
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        return self.weight[None, :, None, None] * x

def init_param(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

class SSD(nn.Module):
    def __init__(self, phase, cfg, vgg, extra_layers, head,
        conf_thresh=0.01, top_k=200, nms_thresh=0.5):
        """
        Inputs:
            phase (str): 'test' or 'train'
            cfg (dict): basic parameters of ssd
            vgg (list[nn.layer]): VGG layers from conv1_1 to conv7
            extra_layers (list[nn.layer]): extra layers in SSD from conv8_1 to conv11_2
            head (tuple(loc_layers, conf_layers)): loc_layers and conf_layers
            conf_thresh (float): confidence threshold of objects
            top_k (int): number of objects to consider per class
            nms_thresh (float): overlap threshold for nms
        """
        super(SSD, self).__init__()
        
        self.phase = phase
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        
        pb = PriorBoxes(self.cfg)
        self.priors = pb.forward() # self.priors (tensor, [num_priors, 4])
        
        self.vgg = nn.ModuleList(vgg)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extra_layers)
        
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        
        if phase == 'test':
            self.detect = Detect(cfg, self.conf_thresh, self.top_k, self.nms_thresh)
        
    def load_weights(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(path,
                map_location=lambda storage, loc: storage))
            print('\tDone.')
        else:
            raise ValueError('Only .pkl and .pth are supported.')
            
    def load_vgg_weights(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.pkl' or '.pth':
            print('Loading vgg weights into state dict...')
            self.vgg.load_state_dict(torch.load(path))
            print('\tDone.')
        else:
            raise ValueError('Only .pkl and .pth are supported.')
            
    def init_extra_weights(self):
        self.extras.apply(init_param)
        self.loc.apply(init_param)
        self.conf.apply(init_param)
        
    def forward(self, x):
        """
        Inputs:
            x (tensor, [batch_size, 3, h, w]): input image(s)
        Outputs:
            depending on phase:
            'test':
                output of detection (tensor, [batch_size, num_classes, top_k, 5]):
                    output[:, :, :, :4]: coords of prediction boxes (percent, xyxy form)
                    output[:, :, :, 4]: confidences
            'train':
                predictions and prior boxes (list[loc_data, conf_data, priors])
        """
        sources, loc, conf = [], [], []
        
        # get features with first 23 layers (conv4_3) of vgg then add to sources
        for k in range(23):
            x = self.vgg[k](x)
        sources.append(self.L2Norm(x))
        
        # get features with all layers of vgg then add to sources
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        
        # get features with extra layers then add to sources
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
                
        # sources inludes conv4_3, conv7, conv8_2, conv9_2, conv10_2 and conv11_2
        # make self.loc and self.conf act on layers in sources
        # change the order of dimension to [batch_size, h, w, c]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # c = num_box * 4, num_box = 4 or 6
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            
            # c = num_box * num_classes (includeing background), num_box = 4 or 6
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)
        
        if self.phase == 'test':
            output = self.detect.forward(
                # (tensor, [batch_size, num_priors, 4])
                loc.view(loc.size(0), -1, 4).detach(),
                # (tensor, [batch_size, num_priors, num_classes (including background)])
                conf.view(conf.size(0), -1, self.num_classes).detach(),
                # (tensor, [num_priors, 4])
                self.priors.type_as(x)
            )
        else:
            output = (
                # (tensor, [batch_size, num_priors, 4])
                loc.view(loc.size(0), -1, 4),
                # (tensor, [batch_size, num_priors, num_classes (including background)])
                conf.view(conf.size(0), -1, self.num_classes),
                # (tensor, [num_priors, 4])
                self.priors.type_as(x)
            )
        return output

# 13 conv layers of VGG16, 'M' for maxpooling, 'C' for maxpooling and
# ceil_mode=True
base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512]

def build_vgg(base, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    # By now, we have (by default):
    # layers[0 - 3]: conv1_1 & relu, conv1_2 & relu
    # layers[5 - 8]: conv2_1 & relu, conv2_2 & relu
    # layers[10 - 15]: conv3_1 & relu, conv3_2 & relu, conv3_3 & relu
    # layers[17 - 22]: conv4_1 & relu, conv4_2 & relu, conv4_3 & relu
    # layers[24 - 29]: conv5_1 & relu, conv5_2 & relu, conv5_3 & relu
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    
    return layers

# 8 extra conv layers
extras = [256, 512, 128, 256, 128, 256, 128, 256]

def build_extra_layers(extras, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(extras):
        if k == 1 or k == 3:
            layers += [nn.Conv2d(in_channels, extras[k],
                kernel_size=(1, 3)[flag], stride=2, padding=1)]
        else:
            layers += [nn.Conv2d(in_channels, extras[k],
                kernel_size=(1, 3)[flag])]
        flag = not flag
        in_channels = extras[k]
    return layers

# number of boxes per point in feature map
mbox = [4, 6, 6, 6, 4, 4]

def build_ssd(phase='test', cfg=ssd_cfg, conf_thresh=0.01, top_k=200, nms_thresh=0.5):
    if phase != 'test' and phase != 'train':
        raise ValueError('Only test and train are supported.')
    
    vgg = build_vgg(base, 3)
    extra_layers = build_extra_layers(extras, 1024)
    
    num_classes = cfg['num_classes'] # num_classes includes background
    loc_layers, conf_layers = [], []
    vgg_layer_idx = [21, -2] # conv4_3 and conv7
    for k, v in enumerate(vgg_layer_idx): # k = 0, 1
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, mbox[k] * 4,
                kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(vgg[v].out_channels, mbox[k] * num_classes,
                kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], start=2): # k = 2, 3, 4, 5
        loc_layers += [
            nn.Conv2d(v.out_channels, mbox[k] * 4,
                kernel_size=3, padding=1)]
        conf_layers += [
            nn.Conv2d(v.out_channels, mbox[k] * num_classes,
                kernel_size=3, padding=1)]
    head_layers = (loc_layers, conf_layers)
    return SSD(phase, cfg, vgg, extra_layers, head_layers,
        conf_thresh, top_k, nms_thresh)
