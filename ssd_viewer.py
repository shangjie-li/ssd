import torch

from cfg import ssd_cfg
from layers.ssd import build_ssd

if __name__ == '__main__':
    model = build_ssd(phase='test', cfg=ssd_cfg)
    print('Structure of SSD:\n', model)
    print('\nSize of prior boxes:\n', model.priors.shape)
    
    batch_size = 10
    data = torch.randn(batch_size, 3, 300, 300)
    print('\nOutput of SSD:\n', model(data).shape)
