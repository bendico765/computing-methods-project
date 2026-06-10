import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms_v2
import pydicom as dicom
import pandas as pd
import numpy as np
import utils
import metrics
import cbis
import unet
import skimage
import scipy
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# picking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")

# path to the root folder of the data
parser = argparse.ArgumentParser(description="")
parser.add_argument("data_root_filepath", help="Path to the project data root directory.")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=20)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--random-state", type=int, default=None)
args = parser.parse_args()

data_root_filepath = args.data_root_filepath

# save configuration for the current run
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(f"{data_root_filepath}/runs"):
    os.makedirs(f"{data_root_filepath}/runs")

if not os.path.exists(f"{data_root_filepath}/runs/{run_name}"):
    os.makedirs(f"{data_root_filepath}/runs/{run_name}")

# Hyperparameters
learning_rate = args.lr
batch_size = args.batch_size
epochs = args.epochs

random_state = args.random_state

print(f"\nConfiguration------------")
print(f"Learning rate:{learning_rate}")
print(f"Batch size:{batch_size}")
print(f"Epochs:{epochs}")

# defyning transforms to augment data
train_transforms = test_transforms = transforms_v2.Compose(
    [
        transforms_v2.RandomHorizontalFlip()
    ]
)

### LOADING DATA
df = pd.read_csv(f"{data_root_filepath}/lesions.csv")

# keeping only MLO and masses
df = df[(df["image view"] == "MLO") & (df["kind"] == "Mass")]
df_train_val, df_test = train_test_split(df, test_size=0.1,random_state=random_state)
df_train, df_val = train_test_split(df_train_val, test_size=0.22, random_state=random_state)

train_data = cbis.CBIS_Dataset(data_root_filepath, df_train)
validation_data = cbis.CBIS_Dataset(data_root_filepath, df_val)
test_data = cbis.CBIS_Dataset(data_root_filepath, df_test)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print(f"\nData samples (70-20-10 split)------------")
print(f"Training:{len(train_data)}")
print(f"Validation:{len(validation_data)}")
print(f"Test:{len(test_data)}")

# defyning loss function
loss_fn = metrics.dice_loss

# creating model
model = unet.UNet(n_class=2)
model.to(device)

# defyning optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)

### TRAINING AND EVALUATING THE MODEL
train_losses = []
test_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n------------")

    # training
    train_loss = unet.train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
    train_losses.append(train_loss)
    
    # testing on validation
    test_loss = unet.test_loop(validation_dataloader, model, loss_fn, device)
    test_losses.append(test_loss)

    # each few epoch save some predicted samples
    if epoch % 4 == 0:
        utils.save_prediction(
            model, 
            validation_dataloader,
            epoch, 
            device,
            f"{data_root_filepath}/runs/{run_name}/prediction_samples"
        )

    # logging
    print(f"\nAvg. train loss={train_loss:.6f}\nAvg. test loss={test_loss:.6f}\n", flush=True)

    # saving model checkpoints
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/checkpoints"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/checkpoints")

    torch.save({
            "epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_loss": train_loss,
            "test_loss": test_loss
        },
            f"{data_root_filepath}/runs/{run_name}/checkpoints/checkpoint_{epoch}.pth"
    )

#### LOGGING ####
if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/logs"):
    os.makedirs(f"{data_root_filepath}/runs/{run_name}/logs")

# saving up the loss history
history = pd.DataFrame({
    "epoch": range(1, epochs+1),
    "train_loss": train_losses,
    "test_loss": test_losses
})
history.to_csv(f"{data_root_filepath}/runs/{run_name}/logs/loss_history.csv", index=False)