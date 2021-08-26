from __future__ import division

import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBoxes():
    """
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBoxes, self).__init__()
        
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        
    def forward(self):
        boxes = []
        # for every feature map
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2): # i in [0, f - 1), j in [0, f - 1)
                # compute center coordinates of box
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f
                
                # compute size of box (
                #  (s_k, s_k), 
                #  (sqrt(s_k * s_(k + 1)), sqrt(s_k * s_(k + 1))),
                #  (s_k * 1.4, s_k / 1.4), (s_k / 1.4, s_k * 1.4),
                #  (s_k * 1.7, s_k / 1.7), (s_k / 1.7, s_k * 1.7),
                # ), in fact, it's size ratio of box and input image (percent form)
                # the order of prior boxes shouldn't be changed
                
                s_k = self.min_sizes[k] / self.image_size
                boxes += [cx, cy, s_k, s_k]
                
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                boxes += [cx, cy, s_k_prime, s_k_prime]
                
                for ar in self.aspect_ratios[k]:
                    boxes += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    boxes += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
                    
        output = torch.Tensor(boxes).view(-1, 4) # output (tensor, [n, 4]): percent coords
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
        
