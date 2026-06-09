import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms_v2
import pydicom as dicom
import pandas as pd
import numpy as np
import utils
import cbis
import unet
import skimage
import scipy
import matplotlib.pyplot as plt
import argparse
import os

# picking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")

# path to the root folder of the data
parser = argparse.ArgumentParser(description="")
parser.add_argument("data_root_filepath", help="Path to the project data root directory.")
args = parser.parse_args()

data_root_filepath = args.data_root_filepath

learning_rate = 5e-3
batch_size = 20
epochs = 10

# defyning transforms to augment data
train_transforms = test_transforms = transforms_v2.Compose(
    [
        transforms_v2.RandomHorizontalFlip()
    ]
)

# loading data
train_data = cbis.CBIS_Dataset(data_root_filepath=data_root_filepath, train=True, transform=train_transforms)
test_data = cbis.CBIS_Dataset(data_root_filepath=data_root_filepath, train=False, transform=test_transforms)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# defyning loss function
loss_fn = unet.dice_loss

# creating model
model = unet.UNet(n_class=2)
model.to(device)

# defyning optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# training and evaluating the model
train_losses = []
test_losses = []
for t in range(epochs):
    print(f"\nEpoch {t+1}\n------------")

    # training
    train_loss = unet.train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
    train_losses.append(train_loss)
    
    # testing
    test_loss = unet.test_loop(test_dataloader, model, loss_fn, device)
    test_losses.append(test_loss)

    # logging
    print(f"\nAvg. train loss={train_loss:.6f}\nAvg. test loss={test_loss:.6f}\n", flush=True)

#### LOGGING ####
if not os.path.exists(f"{data_root_filepath}/logs"):
    os.makedirs(f"{data_root_filepath}/logs")

# saving up the loss history
history = pd.DataFrame({
    "epoch": range(1, epochs+1),
    "train_loss": train_losses,
    "test_loss": test_losses
})
history.to_csv(f"{data_root_filepath}/logs/loss_history.csv", index=False)