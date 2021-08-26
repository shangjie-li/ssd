import torch

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

