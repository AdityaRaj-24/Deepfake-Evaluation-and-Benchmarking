"""
dataset.py

PyTorch Dataset for the processed FaceForensics++ dataset.
"""

from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeDataset(Dataset):
    """
    Binary Deepfake Dataset.

    Labels
    ------
    0 : Real
    1 : Fake
    """

    def __init__(
        self,
        dataset_root,
        metadata_file,
        split="train",
        image_size=256,
        transform=None,
    ):
        self.dataset_root = Path(dataset_root)

        self.metadata = pd.read_csv(metadata_file)
        self.metadata = (
            self.metadata[self.metadata["split"] == split]
            .reset_index(drop=True)
        )

        if transform is None:

            if split == "train":

                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

            else:

                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

        else:
            self.transform = transform

        print(
            f"[{split.upper()}] Loaded {len(self.metadata)} images."
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        image_path = self.dataset_root / row["filename"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32,
        )

        return image, label


if __name__ == "__main__":

    train_dataset = DeepfakeDataset(
        dataset_root="../dataset",
        metadata_file="../metadata.csv",
        split="train",
    )

    print(f"Training images: {len(train_dataset)}")

    image, label = train_dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label)