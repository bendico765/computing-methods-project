from torch.utils.data import Dataset
import torch
import pandas as pd

class CBIS_Dataset(Dataset):
    def __init__(self, data_root_filepath, train, transform=None):
        self.transform = transform
        lesions_df = pd.read_csv(f"{data_root_filepath}/lesions.csv")
        
        # keeping only MLO and masses
        lesions_df = lesions_df[
            (lesions_df["image view"] == "MLO") & 
            (lesions_df["kind"] == "Mass")
        ]
        
        # based on parameter, keep only training or test instances
        if train:
            lesions_df = lesions_df[lesions_df["training or test"] == "training"]
        else:
            lesions_df = lesions_df[lesions_df["training or test"] == "Test"]

        lesions_df = lesions_df.head(10)
        
        self.images = []
        self.masks = []
        
        # load and preprocess the lesions
        for index, row in lesions_df.iterrows():
            image_tensor = torch.load(
                f"{data_root_filepath}/" + row["preprocessed fullimage tensor filepath"],
                weights_only=True
            ).float()
            mask_tensor = torch.load(
                f"{data_root_filepath}/" + row["preprocessed mask tensor filepath"],
                weights_only=True
            ).float()

            self.images.append(image_tensor)
            self.masks.append(mask_tensor)
            
        self.n_elements = len(self.images)
    
    def __len__(self):
        return self.n_elements

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]

        if self.transform:
            image = self.transform(image)

        return image, mask