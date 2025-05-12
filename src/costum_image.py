from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, data_frame, transform=None, target_transform=None, batch_size=32):
        self.data_frame = data_frame
        self.transform = transform
        self.batch_size = batch_size
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_frame['filepaths'])

    def __getitem__(self, idx):
        img_path = self.data_frame['filepaths'].iloc[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.data_frame['labels'].iloc[idx]
        age = self.data_frame['age'].iloc[idx]
        age_group = self.data_frame['age_group'].iloc[idx]
        gender = self.data_frame['sex'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.tensor(label, dtype=torch.long), age, age_group, gender