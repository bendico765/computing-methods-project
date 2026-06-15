import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms_v2
import pandas as pd
import optuna
import cbis
import engine
import argparse
import os
import utils
from datetime import datetime

import metrics
import unet
from sklearn.model_selection import train_test_split

import visualization

# picking device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:{device}")

# path to the root folder of the data
parser = argparse.ArgumentParser(description="")
parser.add_argument("data_root_filepath", help="Path to the project data root directory.")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for hyperparameter optimization")
parser.add_argument("--batch-size", type=int, default=20, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for hyperparameter optimization and to re-train the final model")
parser.add_argument("--random-state", type=int, default=None, help="Random state used for loading up data")
parser.add_argument(
    "--enable-optimization",
    action="store_true",
    help="Whether to optimize the model, or use the command-line provided arguments. If this flag is not specified, just use the command line arguments to train a model (on train and validation sets) and evaluate on the test set.")
parser.add_argument("--test", action="store_true", help="Whether or not to retrain the best model on the test and validation set, and run it on the test")
args = parser.parse_args()

# Command-line parameters
data_root_filepath = args.data_root_filepath
learning_rate = args.lr
n_trials = args.n_trials
batch_size = args.batch_size
epochs = args.epochs
random_state = args.random_state
enable_optimization = args.enable_optimization

# save configuration for the current run
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(f"{data_root_filepath}/runs"):
    os.makedirs(f"{data_root_filepath}/runs")

if not os.path.exists(f"{data_root_filepath}/runs/{run_name}"):
    os.makedirs(f"{data_root_filepath}/runs/{run_name}")


# defining transforms to augment data
train_transforms = validation_transforms = transforms_v2.Compose(
    [
        transforms_v2.RandomHorizontalFlip(),
        transforms_v2.RandomVerticalFlip(),
        transforms_v2.RandomRotation(degrees=(-270, 270))
    ]
)

### LOADING DATA
df = pd.read_csv(f"{data_root_filepath}/lesions.csv")

# keeping only masses
df = df[df["kind"] == "Mass"]

# dividing data in test and train data
df_train_val, df_test = train_test_split(df, test_size=0.1,random_state=random_state)

### HYPERPARAMETER OPTIMIZATION
if enable_optimization:
    # divide data in training and validation set
    df_train, df_val = train_test_split(df_train_val, test_size=0.22, random_state=random_state)

    train_data = cbis.CBIS_Dataset(data_root_filepath, df_train, train_transforms)
    validation_data = cbis.CBIS_Dataset(data_root_filepath, df_val, validation_transforms)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    print(f"\nData samples (70-20-10 split)------------", flush=True)
    print(f"Training:{len(train_data)}", flush=True)
    print(f"Validation:{len(validation_data)}", flush=True)
    print(f"Test:{len(df_test)}", flush=True)

    print("\nOptimizing hyperparameters--------", flush=True)

    # create directory to save the trials for the search
    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/trials"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/trials")

    # use optuna for hyperparameters optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(
        engine.Objective(
            f"{data_root_filepath}/runs/{run_name}/trials",
            train_dataloader,
            validation_dataloader,
            batch_size,
            epochs,
            device
        ),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    print(f"Best hyperparameters: {study.best_params}", flush=True)
    print(f"User attrs: {study.best_trial.user_attrs}", flush=True)

    learning_rate = study.best_params["lr"]
    epochs = best_trial.user_attrs["epochs"]

    del train_data
    del validation_data
    del df_train
    del df_val

### RETRAIN THE BEST MODEL ON THE WHOLE DATASET AND TEST IT
if args.test:
    trainval_data = cbis.CBIS_Dataset(data_root_filepath, df_train_val, transform=train_transforms)
    test_data = cbis.CBIS_Dataset(data_root_filepath, df_test)

    trainval_dataloader = DataLoader(dataset=trainval_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    print("\nTRAINING THE FINAL MODEL AND EVALUATING ON THE TEST SET", flush=True)
    print(f"\nData samples (90-10 split)------------", flush=True)
    print(f"Training:{len(df_train_val)}", flush=True)
    print(f"Test:{len(df_test)}", flush=True)

    # creating model
    model = unet.UNet(n_class=2)
    model.to(device)

    # defining optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    # define loss
    loss_fn = metrics.DiceLoss()

    if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/final_retrain"):
        os.makedirs(f"{data_root_filepath}/runs/{run_name}/final_retrain")

    ### TRAIN THE MODEL
    train_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}\n------------", flush=True)

        # training
        train_loss = engine.train_loop(trainval_dataloader, model, loss_fn, optimizer, batch_size, device)
        train_losses.append(train_loss)

        # each few epoch save some predicted samples
        if epoch % 4 == 0:
            utils.save_prediction(
                model,
                trainval_dataloader,
                epoch,
                device,
                f"{data_root_filepath}/runs/{run_name}/prediction_samples"
            )

        # logging
        print(f"\nAvg. train loss={train_loss:.6f}\n", flush=True)

        # saving model checkpoints
        if not os.path.exists(f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints"):
            os.makedirs(f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_loss": train_loss
        },
            f"{data_root_filepath}/runs/{run_name}/final_model/checkpoints/checkpoint_{epoch}.pth"
        )

    # Make inference on the test set
    print("\nINFERENCE ON THE TEST SET", flush=True)
    test_loss = engine.test_loop(
        test_dataloader,
        model,
        loss_fn,
        device
    )
    print(f"Test loss={test_loss:.6f}", flush=True)