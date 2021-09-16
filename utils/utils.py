import math
import torch
import random
import numpy as np

def collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def adjust_lr(optimizer, gamma, step, lr):
    lr = lr * (gamma ** (step))
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_letter(x):
    if x < 0 or x > 15:
        raise ValueError('x must be between 0 and 15')
    x = int(x)
    info = ['a', 'b', 'c', 'd', 'e', 'f']
    if x < 10:
        return x
    else:
        return info[x - 10]

def convert_decimal_to_hex(x):
    if x < 0 or x > 255:
        raise ValueError('x must be between 0 and 255!')
    x = int(x)
    y = str(get_letter(x // 16)) + str(get_letter(x % 16))
    return y

def create_plot_color():
    r = convert_decimal_to_hex(random.randint(0, 255))
    g = convert_decimal_to_hex(random.randint(0, 255))
    b = convert_decimal_to_hex(random.randint(0, 255))
    color = '#' + r + g + b
    return color

def smooth_data(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    if xs.size != ys.size:
        raise ValueError('xs.size must be equal to ys.size.')
    
    xs_out, ys_out = [], []
    for t in np.arange(0., 1.0, 0.01):
        if np.sum(xs >= t) == 0:
            break
        else:
            xs_out.append(t)
            ys_out.append(np.max(ys[xs >= t]))
    return xs_out, ys_out
