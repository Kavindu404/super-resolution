# resize_transforms.py

import torch
import torch.nn.functional as F

class ResizeTensor(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, tensor):
        tensor = F.interpolate(tensor.unsqueeze(0), size=self.new_size, mode='bilinear', align_corners=False)
        return tensor.squeeze(0)
