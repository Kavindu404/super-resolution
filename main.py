import torch
import os
import json
import argparse
from rcan.utils import get_data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from rcan.model import UNetSR
from rcan.resize_transforms import ResizeTensor
from rcan.dataset import SRDataset
from rcan.loss import nll_loss
from rcan.metric import accuracy


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

    train_data_pths = [os.join(lr_path, str(year)) for year in train_years]
    train_lbl_pths = [os.join(hr_path, str(year)) for year in train_years]
    val_data_pths = [os.join(lr_path, str(val_year))]
    val_lbl_pths = [os.join(hr_path, str(val_year))]

    train_data = ResizeTensor((256, 256))(get_data(train_data_pths))
    val_data = ResizeTensor((256, 256))(get_data(val_data_pths))

    train_lbl = ResizeTensor((1024, 1024))(get_data(train_lbl_pths))
    val_lbl = ResizeTensor((1024, 1024))(get_data(val_lbl_pths))
    
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
    batch_size = meta["parameters"]["batch_size"]
    

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

    model = UNetSR(in_channels=input_dim, out_channels=output_dim, num_filters=64, upscale_factor=scaling_factor)

    if len(gpus) > 1:
        print(f"Using {len(gpus)} GPUs for training.")
        model = DataParallel(model)
    
    train_ds = SRDataset(train_data, train_lbl)
    val_ds = SRDataset(val_data, val_lbl)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    criterion = nll_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
         
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            _, y_pred = torch.max(outputs, 1)
        train_loss /= len(train_ds)
        train_acc = accuracy(y_pred, y)

        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        y_true = []
        y_preds = []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                
                loss = criterion(outputs, y)
                test_loss += loss.item() * X.size(0)
                _, y_prd = torch.max(outputs, 1)
            test_loss /= len(val_dl)
            acc = accuracy(y_preds, y_true)

            if acc > best_acc:
                best_acc = acc
                pres = round(best_acc, 4)
                if isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch.save(model, ckpt_dir+"srnet_"+str(pres)+".pt")

            scheduler.step(acc)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', test_loss, epoch)
            writer.add_scalar('Acc/val', acc, epoch)

            print(f"Epoch {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {test_loss:.4f}, train accuracy: {train_acc}, val accuracy:{acc:.4f}")
    




