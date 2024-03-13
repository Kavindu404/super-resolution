import torch
import os
import json
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "--v", help="model_meta_version")
    parser.add_argument("--data", "--x", help="Input data for the model", default="X_gmv5_s250")
    parser.add_argument("--label", "--y", help="Labels for the model", default="y_gmv5_s250")
    parser.add_argument("--retrain", "--rt", help="Retraining the model", default=False)
    
    args = parser.parse_args()
    
    meta_version = args.version
    data_file = args.data
    label_file = args.label
    rt = args.retrain
    
    data_root = '../../../../data/'
    model_dp = data_root + 'models/base/'
    data_dir = model_dp + "datasets"
    
    meta_fp = model_dp + 'model_meta.json'
    with open(meta_fp) as json_file:
        meta = json.load(json_file)
    
    meta = meta[meta_version]

    device = "cuda"
    num_epochs = meta["parameters"]["num_epochs"]
    input_dim = meta["parameters"]["input_dim"]
    hidden_dim = meta["parameters"]["hidden_dim"]
    num_layers = meta["parameters"]["num_layers"]
    output_dim = meta["parameters"]["output_dim"]
    dropout = meta["parameters"]["dropout"]
    lr = meta["parameters"]["lr"]
    batch_size = meta["parameters"]["train_batch_size"]
    val_batch_size = meta["parameters"]["val_batch_size"]
    weight_decay = meta["parameters"]["weight_decay"]
    num_heads = meta["parameters"]["num_heads"]
    version = meta["parameters"]["version"]

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




