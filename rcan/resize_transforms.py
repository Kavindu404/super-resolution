import torch
import torch.nn.functional as F

class ResizeTensor(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, tensor):
        if tensor.dim() == 3:
            if tensor.shape[-2:] != self.new_size:
                tensor = F.interpolate(tensor.unsqueeze(0), size=self.new_size, mode='bilinear', align_corners=False)
                tensor = tensor.squeeze(0)
        elif tensor.dim() > 3:
            batch_size = tensor.shape[0]
            resized_tensors = []
            for i in range(batch_size):
                t = tensor[i]
                if t.shape[-2:] != self.new_size:
                    resized_tensor = F.interpolate(t.unsqueeze(0), size=self.new_size, mode='bilinear', align_corners=False)
                    resized_tensor = resized_tensor.squeeze(0)
                    resized_tensors.append(resized_tensor)
                else:
                    resized_tensors.append(t)
            tensor = torch.stack(resized_tensors)
        return tensor
