import torch
import os
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience: int =5, min_delta: float=0):
        """
        :param patience: How many epochs to wait after last time validation loss improved.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state_dict = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Check if the validation loss improved significantly
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            # Save the best model state
            self.best_model_state_dict = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_prediction(
        model,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        epoch: int,
        device: torch.device,
        output_dir: str,
        n_samples: int =3
):
    """

    """
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        X, y = next(iter(dataloader))

        X = X.to(device)
        logits = model(X)
        pred_probs = torch.softmax(logits, dim=1)

        X = X.to("cpu")
        y = y.to("cpu")
        pred_probs = pred_probs.to("cpu")

        # number of samples to show minimum between n_samples and batch_size
        n = min(n_samples, X.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        for i in range(n): # iterate samples
            # compute loss
            loss = loss_fn(y[i].float().unsqueeze(0), pred_probs[i].unsqueeze(0)).item()

            # input image
            axes[i,0].imshow(X[i,0], cmap="gray")
            axes[i,0].set_title("Input")
            axes[i,0].axis("off")

            # ground truth
            axes[i,1].imshow(y[i,1], cmap="gray")
            axes[i,1].set_title("Ground Truth")
            axes[i,1].axis("off")

            # prediction
            axes[i,2].imshow(pred_probs[i,1], cmap="gray")
            axes[i,2].set_title(f"Prediction (Loss: {loss:.4f})")
            axes[i,2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/epoch_{epoch:03d}.png")
        plt.close()
