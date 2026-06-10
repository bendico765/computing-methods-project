import numpy as np
import scipy
import pandas as pd
import torch
import skimage
import os
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            checkpoint_path (str): Path to save the best model weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
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

def save_prediction(model, dataloader, epoch, device, output_dir, n_samples=3):
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
            axes[i,2].set_title("Prediction")
            axes[i,2].axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/epoch_{epoch:03d}.png")
        plt.close()

def crop_to_mask(image, mask, padding=20):
    # Find nonzero coordinates
    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding
    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)

    y_max = min(y_max + padding, image.shape[0])
    x_max = min(x_max + padding, image.shape[1])

    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask  = mask[y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask

def preprocess_image(image: np.array, mask: np.array, padding: int = 20) -> np.array:
    IMAGE_TARGET_SHAPE = (512, 512)
    
    # application of median filtering
    median_filtered_image = scipy.ndimage.median_filter(image, size=3)

    # application of clahe
    clahe_image = skimage.exposure.equalize_adapthist(median_filtered_image)
    
    # application of unsharp
    preproc_image = skimage.filters.unsharp_mask(clahe_image)

    # crop around the roi
    preproc_image, _ = crop_to_mask(preproc_image, mask, padding)

    # downscale the image
    preproc_image = skimage.transform.resize(preproc_image, IMAGE_TARGET_SHAPE) 

    return preproc_image

def preprocess_mask(image: np.array, mask: np.array, padding: int = 20) -> np.array:
    MASK_TARGET_SHAPE = (512, 512)

    # crop around the roi
    _, preproc_mask = crop_to_mask(image, mask, padding)
   
    # downscale the image
    preproc_mask = skimage.transform.resize(preproc_mask, MASK_TARGET_SHAPE)

    return preproc_mask

def preprocess(image: np.array, mask: np.array, padding: int = 20) -> tuple:
    return preprocess_image(image), preprocess_mask(preproc_mask)