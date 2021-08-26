import torch
import torch.nn as nn
import torch.nn.functional as F

from cfg import ssd_cfg
from .utils import encode, match

def compute_loc_loss(loc_data, loc_targets, conf_targets):
    """
    Compute loc loss for only positive targets, using smooth l1 loss function.
    
    Inputs:
        loc_data (tensor, [batch_size, num_priors, 4]): 
            coords of prediction boxes (percent, g^ccwh form)
        loc_targets (tensor, [batch_size, num_priors, 4]):
            ground truth coords of prior boxes (percent, g^ccwh form)
        conf_targets (tensor, [batch_size, num_priors]):
            ground truth labels of prior boxes (value between [0, num_classes),
            0 for background)
    """
    pos_mask = conf_targets > 0 # [batch_size, num_priors]
    num_positive = pos_mask.sum()
    
    mask = pos_mask.unsqueeze(-1).expand_as(loc_data) # [batch_size, num_priors, 4]
    loc_data = loc_data[mask].view(-1, 4)
    loc_targets = loc_targets[mask].view(-1, 4)
    loss = F.smooth_l1_loss(loc_data, loc_targets, reduction='sum')
    return loss / num_positive

def get_cross_entropy_loss(conf_data, conf_targets):
    """
    Get cross entropy loss of every prior box in predictions.
        loss of cross entropy: loss = -Sum{li * log(pi)}
            li: 1 for positive target, 0 for negative target
            pi: probability of each class (output of softmax function)
        softmax function: pi = exp(xi - x_max) / Sum{exp(xi - xmax)}
            xi: confidence of each class (prediction of network)
        thus, loss(i) = log(sum(exp(xi - x_max))) + xmax - xi
        
    Inputs:
        conf_data (tensor, [batch_size, num_priors, num_classes]):
            confidence of each class (0 for background) for each prior box
        conf_targets (tensor, [batch_size, num_priors]):
            ground truth labels of prior boxes (value between [0, num_classes),
            0 for background)
    Outputs:
        cross entropy loss (tensor, [batch_size, num_priors])
    """
    batch_size, num_priors, num_classes = conf_data.shape
    x = conf_data.view(-1, num_classes) # [n, num_classes], n = batch_size * num_priors
    x_max = x.max()
    
    lse = torch.log(torch.sum((torch.exp(x - x_max)), dim=1, keepdim=True)) # [n, 1]
    ce_loss = lse + x_max - x.gather(dim=1, index=conf_targets.view(-1, 1)) # [n, 1]
    return ce_loss.view(batch_size, num_priors)

def compute_conf_loss(conf_data, conf_targets, neg_pos_ratio=3):
    """
    Compute conf loss for positive and negative targets using cross entropy
    loss function.
    
    Inputs:
        conf_data (tensor, [batch_size, num_priors, num_classes]):
            confidence of each class (0 for background) for each prior box
        conf_targets (tensor, [batch_size, num_priors]):
            ground truth labels of prior boxes (value between [0, num_classes),
            0 for background)
        neg_pos_ratio (int): ratio of negative and positive prior boxes
            counted into loss
    """
    pos_mask = conf_targets > 0 # [batch_size, num_priors]
    num_positive = pos_mask.sum()
    
    ce_loss = get_cross_entropy_loss(conf_data, conf_targets)
    ce_loss[pos_mask] = 0
    _, idx = ce_loss.sort(dim=1, descending=True)
    _, idx_rank = idx.sort(dim=1)
    neg_mask = idx_rank < pos_mask.sum(dim=1, keepdim=True) * neg_pos_ratio
    
    mask = pos_mask + neg_mask # [batch_size, num_priors]
    conf_data = conf_data[mask.unsqueeze(-1).expand_as(conf_data)].view(-1, conf_data.size(-1))
    conf_targets = conf_targets[mask].view(-1)
    loss = F.cross_entropy(conf_data, conf_targets, reduction='sum')
    return loss / num_positive

class MultiBoxLoss(nn.Module):
    def __init__(self, cfg=ssd_cfg, iou_thresh=0.5, neg_pos_ratio=3, cuda=True):
        super(MultiBoxLoss, self).__init__()
        
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.iou_thresh = iou_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.cuda = cuda
        self.variance = self.cfg['variance']
        
    def forward(self, predictions, targets):
        """
        Inputs:
            predictions (list[loc_data, conf_data, priors]):
                loc_data (tensor, [batch_size, num_priors, 4]): 
                    coords of prediction boxes (percent, g^ccwh form)
                conf_data (tensor, [batch_size, num_priors, num_classes]):
                    confidence of each class (0 for background) for each prior box
                priors (tensor, [num_priors, 4]):
                    coords of prior boxes (percent, ccwh form)
            targets (tensor, [batch_size, n, 5]):
                targets[:, :, :4]: coords of target boxes (percent, xyxy form)
                targets[:, :, 4]: target labels of target boxes (no background)
        """
        loc_data, conf_data, priors = predictions
        batch_size, num_priors = loc_data.size(0), loc_data.size(1)
        
        # loc_targets: ground truth coords of prior boxes
        loc_targets = torch.Tensor(batch_size, num_priors, 4)
        
        # conf_targets: ground truth labels of prior boxes
        conf_targets = torch.LongTensor(batch_size, num_priors)
        
        for idx in range(batch_size):
            truths, labels = targets[idx][:, :-1], targets[idx][:, -1]
            
            # match prior boxes and target boxes in a single image and change
            # value in loc_targets and conf_targets
            loc_t, conf_t = match(self.iou_thresh, truths, labels, priors,
                self.variance)
            loc_targets[idx] = loc_t
            conf_targets[idx] = conf_t
            
        if self.cuda:
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()
            
        loc_loss = compute_loc_loss(loc_data, loc_targets, conf_targets)
        conf_loss = compute_conf_loss(conf_data, conf_targets, self.neg_pos_ratio) 
        return loc_loss, conf_loss
