from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class CBIS_Dataset(Dataset):
    def __init__(self, data_root_filepath: str, df: pd.DataFrame, transform=None):
        self.transform = transform

        self.image_paths = np.array([
            f"{data_root_filepath}/{filepath}"
            for filepath in df["preprocessed fullimage tensor filepath"]
        ])

        self.mask_paths = np.array([
            f"{data_root_filepath}/{filepath}"
            for filepath in df["preprocessed mask tensor filepath"]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.load(
            self.image_paths[idx],
            weights_only=True
        )

        mask = torch.load(
            self.mask_paths[idx],
            weights_only=True
        )

        if self.transform:
            image = self.transform(image)

        return image, mask