import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import glob
import os   
from resize_transforms import ResizeTensor

def get_dataset(pth):
    data_tensors = []
    files = glob.glob(os.path.join(pth, '*2d.nc'))
    for file in files:
        data = xr.open_dataset(file)
        if 'ssh' in data.data_vars:
            data_tensors.append(torch.stack((torch.tensor(data.ssh.values).squeeze(), torch.tensor(data.u_barotropic_velocity.values).squeeze(), torch.tensor(data.v_barotropic_velocity.values).squeeze()), dim=0))
        
    return torch.stack([t for t in data_tensors], dim=0)

