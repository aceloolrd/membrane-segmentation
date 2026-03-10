import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_dataframe(image_dir: str, mask_dir: str) -> pd.DataFrame:
    images = sorted([os.path.join(image_dir, f).replace("\\", "/") for f in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir, f).replace("\\", "/") for f in os.listdir(mask_dir)])
    return pd.DataFrame({"image": images, "mask": masks})


class GranulometryDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]["image"]
        mask_path = self.dataframe.iloc[idx]["mask"]

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0)
        return image, mask


class GranulometryDataModule:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_height: int = 512,
        img_width: int = 512,
        batch_size: int = 2,
        num_workers: int = 0,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.dataframe = dataframe
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.random_state = random_state

        self.train_transform = A.Compose([
            A.Resize(img_height, img_width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(img_height, img_width),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

        self.train_dataset = None
        self.val_dataset = None

    def setup(self):
        train_df, val_df = train_test_split(
            self.dataframe, test_size=self.test_size, random_state=self.random_state
        )
        self.train_dataset = GranulometryDataset(train_df, transform=self.train_transform)
        self.val_dataset = GranulometryDataset(val_df, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
