# %% We are going to read examples from the two resolutions 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import torch
from utils import get_dataset
from resize_transforms import ResizeTensor
from os.path import join

years = [2010]

lr_path = "/unity/f1/BOEM_GOMb0.04/data/"
hr_path = "/unity/g2/BOEM_GOMb0.01/data/"

# %%
# for year in years:

year = 2010
# cur_hr_path = join(hr_path, str(year), f"010_archv.{year}_001_02_2d.nc") # This has the SSH
# cur_lr_path = join(lr_path, str(year), f"020_archv.{year}_001_02_2d.nc")
cur_hr_path = join(hr_path, str(year))
cur_lr_path = join(lr_path, str(year))

train_dset = get_dataset(cur_lr_path)
# test_dset = get_dataset(cur_hr_path)



# lr_ds = xr.open_dataset(cur_lr_path)


train_samples = ResizeTensor((250, 250))(train_dset)

a=0

# hr_ds = xr.open_dataset(cur_hr_path)

# %% Print the variables in the dataset
# print(lr_ds.data_vars)

# # %% You can start with SSH from the '2d' files
# # %% Suggested variables to start with will be water_temp, salinity, and u and v velocities

# # %% Plotting for temperature
# lr_temp = lr_ds.water_temp
# hr_temp = hr_ds.water_temp

# # %% Print the dimensions of the data
# print(f"Low resolution data: {lr_temp.shape} and High resolution data: {hr_temp.shape}")


# # %% Plot the data
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# # Flip the axis to have the same orientation
# axs[0].imshow(lr_temp[0,38,:,:], cmap='jet', origin='lower')
# axs[1].imshow(hr_temp[0,38,:,:], cmap='jet', origin='lower')
# plt.show()

# # %% Plotting for salinity
# lr_sal = lr_ds.salinity
# hr_sal = hr_ds.salinity

# # %% Print the dimensions of the data
# print(f"Low resolution data: {lr_sal.shape} and High resolution data: {hr_sal.shape}")