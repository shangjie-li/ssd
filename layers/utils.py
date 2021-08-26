import torch

def convert_ccwh_to_xyxy(boxes):
    return torch.cat(
        [
        boxes[:, :2] - boxes[:, 2:]/2, # xmin: c - w/2, ymin: c - h/2
        boxes[:, :2] + boxes[:, 2:]/2, # xmax: c + w/2, ymax: c + h/2
        ], dim=1
    )

def intersect(boxes_a, boxes_b):
    a = boxes_a.size(0)
    b = boxes_b.size(0)
    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(1).expand(a, b ,2),
        boxes_b[:, 2:].unsqueeze(0).expand(a, b, 2),
    )
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(1).expand(a, b, 2),
        boxes_b[:, :2].unsqueeze(0).expand(a, b, 2),
    )
    dxdy = torch.clamp((max_xy - min_xy), min=0)
    return dxdy[:, :, 0] * dxdy[:, :, 1] # [a, b]
    
def jaccard_tensor(boxes_a, boxes_b):
    a = boxes_a.size(0)
    b = boxes_b.size(0)
    inter = intersect(boxes_a, boxes_b)
    
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_a = area_a.unsqueeze(1).expand(a, b)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    area_b = area_b.unsqueeze(0).expand(a, b)
    union = area_a + area_b - inter
    
    # iou = inter / (area(a) + area(b) - inter)
    return inter / union # [a, b]

def encode(truths, priors, variances):
    """
    Encode the ground truth coords for each prior box.
    
    Inputs:
        truths (tensor, [num_priors, 4]): ground truth coords of prior boxes
            (percent, xyxy form)
        priors (tensor, [num_priors, 4]): coords of prior boxes (percent, ccwh form)
        variances (list[float]): variances of prior boxes (to adjust distribution)
    Outputs:
        boxes (tensor, [num_priors, 4]): ground truth coords of prior boxes
            (percent, g^ccwh form)
    """
    # g_cx = (cx' - cx) / w, g_cy = (cy' - cy) / h
    g_cxcy = (truths[:, :2] + truths[:, 2:]) / 2 - priors[:, :2]
    g_cxcy = g_cxcy / priors[:, 2:]
    g_cxcy = g_cxcy / variances[0]
    
    # g_w = log(w' / w), g_h = log(h' / h)
    g_wh = (truths[:, 2:] - truths[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh)
    g_wh = g_wh / variances[1]
    
    boxes = torch.cat([g_cxcy, g_wh], dim=1)
    return boxes

def match(thresh, truths, labels, priors, variances):
    """
    Match prior boxes and target boxes in a single image, and do some
    encode work.
    
    Inputs:
        truths (tensor, [a, 4]): coords of target boxes (percent, xyxy form)
        labels (tensor, [a]): target labels of target boxes (no background)
        priors (tensor, [b, 4]): coords of prior boxes (percent, ccwh form)
        variances (list[float]): variances of prior boxes
    Outputs:
        loc (tensor, [b, 4]): ground truth coords of prior boxes
        conf (tensor, [b]): ground truth labels of prior boxes (0 for background)
    """
    iou = jaccard_tensor(truths, convert_ccwh_to_xyxy(priors)) # [a, b]
    
    # best_p_for_truths_iou (tensor, [a])
    # best_p_for_truths_idx (tensor, [a]): value between [0, b)
    best_p_for_truths_iou, best_p_for_truths_idx = iou.max(dim=1)
    
    # best_t_for_priors_iou (tensor, [b])
    # best_t_for_priors_idx (tensor, [b]): value between [0, a)
    best_t_for_priors_iou, best_t_for_priors_idx = iou.max(dim=0)
    
    for i in range(best_p_for_truths_idx.size(0)):
        k = best_p_for_truths_idx[i]
        best_t_for_priors_iou[k] = 1.0
        best_t_for_priors_idx[k] = i
        
    loc = encode(truths[best_t_for_priors_idx], priors, variances)
    conf = labels[best_t_for_priors_idx] + 1
    conf[best_t_for_priors_iou < thresh] = 0 # 0 for background
    return loc, conf

def decode(preds, priors, variances):
    """
    Decode the prediction coords for each prior box.
    
    Inputs:
        preds (tensor, [num_priors, 4]): coords of prediction boxes
            (percent, g^ccwh form)
        priors (tensor, [num_priors, 4]): coords of prior boxes (percent, ccwh form)
        variances (list[float]): variances of prior boxes
    Outputs:
        boxes (tensor, [num_priors, 4]): decoded box predictions (percent, xyxy form)
    """
    cxcy = preds[:, :2] * variances[0]
    cxcy = cxcy * priors[:, 2:] + priors[:, :2]
    
    wh = preds[:, 2:] * variances[1]
    wh = torch.exp(wh) * priors[:, 2:]
    
    boxes = torch.cat([cxcy, wh], dim=1)
    boxes = convert_ccwh_to_xyxy(boxes)
    return boxes
    
def nms(boxes, scores, top_k=200, iou_thresh=0.5):
    """
    Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    
    Inputs:
        boxes (tensor, [n, 4]): coords of prediction boxes (percent, xyxy form)
        scores (tensor, [n]): confidences
        iou_thresh (float): iou threshold for suppressing
        top_k (int): maximum number of predictions to consider
    Outputs:
        indices of boxes to keep (tensor, [m])
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    top_k = min(top_k, boxes.size(0))
    _, idx = scores.sort(dim=0, descending=True)
    idx = idx[:top_k] # [top_k]
    
    indices = []
    while idx.numel() > 0:
        i = idx[0].item()
        indices.append(i) # keep the box of largest confidence
        if idx.size(0) == 1:
            break
        
        idx = idx[1:] # remove the largest confidence element
        
        xx1 = x1[idx].clamp(min=x1[i].item())
        yy1 = y1[idx].clamp(min=y1[i].item())
        xx2 = x2[idx].clamp(max=x2[i].item())
        yy2 = y2[idx].clamp(max=y2[i].item())
        
        w = (xx2 - xx1).clamp(min=0.0)
        h = (yy2 - yy1).clamp(min=0.0)
        
        inter = w * h
        union = area[i] + area[idx] - inter
        iou = inter / union
        idx = idx[iou.le(iou_thresh)] # keep elements with iou <= iou_thresh
        
    return torch.Tensor(indices).type(torch.long)
