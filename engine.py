import torch
from torch.utils.data import DataLoader
import unet
import metrics
import os
import pandas as pd
import utils

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device="cpu"):
    model.train() # set the model to training mode
    
    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device) # (B, 1, W, H)
        y = y.to(device) # (B, 2, W, H) 
        
        # compute prediction and loss
        logits = model(X)
        pred_probs = torch.softmax(logits, dim=1)
        loss = loss_fn(y.float(), pred_probs)
        
        # backpropagation
        loss.backward() # backpropagate the prediction loss
        optimizer.step() # adjust the parameters by the gradients collected in the backward pass
        optimizer.zero_grad() # reset the gradients of model parameters

        loss, current = loss.item(), batch * batch_size + len(X)
        total_loss += loss
        
        print(f"loss: {loss:>6f}  [{current:>5d}/{size:>5d}]")

    return total_loss/len(dataloader)

def test_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval() # set model to evaluation mode
    
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            logits = model(X)
            pred_probs = torch.softmax(logits, dim=1)
            test_loss += loss_fn(y.float(), pred_probs).item()
    
    return test_loss / len(dataloader) # return average on the batches

class Objective:
    def __init__(self, 
                 trial_folder_filepath: str,
                 train_dataloader: DataLoader, 
                 validation_dataloader: DataLoader, 
                 batch_size: int, 
                 epochs: int, 
                 device: str):
        self.trial_folder_filepath = trial_folder_filepath
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
    def __call__(self, trial):
        # create folder to save trial information
        if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}"):
            os.makedirs(f"{self.trial_folder_filepath}/{trial.number}")

        # setting hyperparameters range of values
        learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        
        # creating model
        model = unet.UNet(n_class=2)
        model.to(self.device)
        
        # defyning optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate
        )
    
        # defyning loss function
        loss_fn = metrics.DiceLoss()
    
        # initializing early stopping
        early_stopper = utils.EarlyStopping(patience=5, min_delta=0.001)
        
        ### Training and evaluating the model
        train_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}\n------------")
            
            # training
            train_loss = train_loop(
                self.train_dataloader, 
                model, 
                loss_fn, 
                optimizer, 
                self.batch_size, 
                self.device
            )
            train_losses.append(train_loss)
            
            # validation
            val_loss = test_loop(self.validation_dataloader, model, loss_fn, self.device)
            val_losses.append(val_loss)
            
            print(f"\nAvg. train loss={train_loss:.6f}\nAvg. val loss={val_loss:.6f}\n", flush=True)
    
            # checking early stopping
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered")
                self.epochs = epoch + 1
                break

        ### LOGGING ###
        if not os.path.exists(f"{self.trial_folder_filepath}/{trial.number}/logs"):
            os.makedirs(f"{self.trial_folder_filepath}/{trial.number}/logs")

        # saving up the loss history
        history = pd.DataFrame({
            "epoch": range(1, self.epochs+1),
            "train_loss": train_losses,
            "val_loss": val_losses
        })
        history.to_csv(f"{self.trial_folder_filepath}/{trial.number}/logs/loss_history.csv", index=False)
        
        return val_loss
