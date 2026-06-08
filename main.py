import torch
from torch.utils.data import DataLoader, Dataset
import pydicom as dicom
import pandas as pd
import numpy as np
import utils
import unet
import skimage
import scipy
import matplotlib.pyplot as plt
import argparse

# picking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")

# path to the root folder of the data
parser = argparse.ArgumentParser(description="")
parser.add_argument("data_root_filepath", help="Path to the project data root directory.")
args = parser.parse_args()

data_root_filepath = args.data_root_filepath
#ROOT_FILEPATH = "/run/media/gianluca/EXTERNAL_US/CBIS-DDSM"

learning_rate = 1e-3
batch_size = 2
epochs = 3

# loading data
train_data = utils.CBIS_Dataset(data_root_filepath=data_root_filepath, train=True)
test_data = utils.CBIS_Dataset(data_root_filepath=data_root_filepath, train=False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# defyning loss function
loss_fn = unet.dice_loss

model = unet.UNet(n_class=2)
model.to(device)

# defyning optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

# training loop
for t in range(epochs):
    print(f"Epoch {t+1}\n------------")
    unet.train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
    unet.test_loop(test_dataloader, model, loss_fn, device)