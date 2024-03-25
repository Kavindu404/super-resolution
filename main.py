import torch
import os
import json
import argparse
from rcan.utils import get_data
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "--v", help="model_meta_version")
    parser.add_argument("--train", "--x", nargs= '+', help="Training years for the model", default=["2021", "2022"])
    parser.add_argument("--val", "--y", help="Validation years for the model", default="2023")
    # parser.add_argument("--retrain", "--rt", help="Retraining the model", default=False)
    
    args = parser.parse_args()
    
    meta_version = args.version
    train_years = args.train
    val_year = args.val
    # rt = args.retrain

    lr_path = "/unity/f1/BOEM_GOMb0.04/data/"
    hr_path = "/unity/g2/BOEM_GOMb0.01/data/"

    train_pths = [os.join(lr_path, str(year)) for year in train_years]
    val_pths = [os.join(hr_path, str(val_year))]

    train_data = get_data(train_pths)
    val_data = get_data(val_pths)
    
    data_root = '../../../../data/'
    model_dp = 'models/'
    data_dir = model_dp + "datasets"
    
    meta_fp = 'model_meta.json'
    with open(meta_fp) as json_file:
        meta = json.load(json_file)
    
    meta = meta[meta_version]

    device = "cuda"
    gpus = meta["parameters"]["n_gpus"]
    num_epochs = meta["parameters"]["num_epochs"]
    input_dim = meta["parameters"]["input_dim"]
    output_dim = meta["parameters"]["output_dim"]
    scaling_factor = meta["parameters"]["scale"]
    

    model_dir = model_dp + str(meta_version)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logs_dir = model_dir + "/logs"

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    ckpt_dir = model_dir + "/ckpt"
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    writer = SummaryWriter(log_dir=logs_dir)

    




