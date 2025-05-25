from sklearn.model_selection import train_test_split
from torchvision import transforms as T
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def stratified_split(df):
    lesion_to_label = df.groupby("lesion_id")["labels"].first()
    lesion_ids = lesion_to_label.index.to_numpy()
    lesion_labels = lesion_to_label.values

    train_ids, test_ids = train_test_split(
        lesion_ids,
        test_size=0.2,
        random_state=42,
        stratify=lesion_labels,
        shuffle=True
    )

    train_labels = lesion_to_label[train_ids].values
    train_ids_final, val_ids = train_test_split(
        train_ids,
        test_size=0.25,
        random_state=42,
        stratify=train_labels,
        shuffle=True
    )

    train_df = df[df["lesion_id"].isin(train_ids_final)].reset_index(drop=True)
    valid_df = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["lesion_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, valid_df, test_df

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Resize((256, 256)),
    T.ToTensor()
])

eval_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])
