import numpy as np
import scipy
import pandas as pd
import torch
import skimage
import os
import matplotlib.pyplot as plt

def save_prediction(model, dataloader,epoch, device, output_dir, n_samples=3):
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        images, masks = next(iter(dataloader))

        images = images.to(device)
        outputs = model(images)

        images = images.to("cpu")
        outputs = outputs.to("cpu")

        # number of samples to show minimum between n_samples and batch_size
        n = min(n_samples, images.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
        for i in range(n):
            # input image
            axes[i,0].imshow(images[i,0], cmap="gray")
            axes[i,0].set_title("Input")
            axes[i,0].axis("off")

            # ground truth
            axes[i,1].imshow(masks[i,1], cmap="gray")
            axes[i,1].set_title("Ground Truth")
            axes[i,1].axis("off")

            # prediction
            axes[i,2].imshow(outputs[i,1], cmap="gray")
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