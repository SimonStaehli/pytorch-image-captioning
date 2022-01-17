from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import pandas as pd
import os

class CustomDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, annot_transform=None, image_transform=None):
        self.annotation_file = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.annot_transform = annot_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
                
        img_path = os.path.join(self.img_dir, self.annotation_file.iloc[idx, 0])
        image = read_image(img_path)
        label = self.annotation_file.iloc[idx, 1]
        if self.image_transform:
            image = self.image_transform(image)
        if self.annot_transform:
            label = self.annot_transform([label])
            
        return image, label