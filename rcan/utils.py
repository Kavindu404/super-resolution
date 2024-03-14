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
    for file in files[:10]:
        data = xr.open_dataset(file)
        if 'ssh' in data.data_vars:
            data.ssh.values[torch.isnan(torch.tensor(data.ssh.values))]=0
            data.u_barotropic_velocity.values[torch.isnan(torch.tensor(data.u_barotropic_velocity.values))]=0
            data.v_barotropic_velocity.values[torch.isnan(torch.tensor(data.v_barotropic_velocity.values))]=0
            data_tensors.append(torch.stack((torch.tensor(data.ssh.values).squeeze(), torch.tensor(data.u_barotropic_velocity.values).squeeze(), torch.tensor(data.v_barotropic_velocity.values).squeeze()), dim=0))
        
    return torch.stack([t for t in data_tensors], dim=0)

