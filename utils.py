import numpy as np
import scipy
import pandas as pd
import torch
import skimage

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