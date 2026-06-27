"""
dataset.py
==========
PyTorch Dataset for the processed FaceForensics++ deepfake benchmark.

Labels
------
    0 : Real
    1 : Fake

Model-aware normalisation
-------------------------
Different architectures have different normalisation expectations.
Meso4 was originally trained without ImageNet normalisation.
Passing it ImageNet-normalised inputs harms its shallow feature extractor.

Use the get_normalization(model_name) helper to retrieve the correct
mean/std pair for each architecture, and pass the result as the
`normalization` argument when constructing the dataset.

Quick usage:

    from data import DeepfakeDataset, get_normalization

    norm = get_normalization("meso4")          # (mean, std) or None
    ds = DeepfakeDataset(..., normalization=norm)

For Xception, PatchResNet, and MultipleAttention the default ImageNet
statistics are used.  For Meso4, no normalisation beyond ToTensor() is
applied (pixel values remain in [0, 1]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Normalisation registry
# ---------------------------------------------------------------------------

# ImageNet statistics — appropriate for ImageNet-pretrained backbones
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Meso4 — the original paper applies no channel normalisation; pixel values
# remain in [0, 1] after ToTensor().  Forcing ImageNet stats on a shallow
# randomly-initialised CNN suppresses learning in the first convolutional
# layer because the input distribution is very different from what the
# random weights expect.
_MESO4_MEAN = None
_MESO4_STD  = None


def get_normalization(model_name: str) -> Optional[tuple[list[float], list[float]]]:
    """
    Return the (mean, std) normalisation pair for a given architecture,
    or None if no channel normalisation should be applied.

    Parameters
    ----------
    model_name : str
        One of: "meso4", "xception", "patch_resnet", "multiple_attention".

    Returns
    -------
    (mean, std) | None
    """
    model_name = model_name.lower()
    if model_name == "meso4":
        return None           # no ImageNet normalisation for Meso4
    if model_name in ("xception", "patch_resnet", "multiple_attention"):
        return _IMAGENET_MEAN, _IMAGENET_STD
    raise ValueError(
        f"Unknown model_name: {model_name!r}. "
        "Choose from: meso4, xception, patch_resnet, multiple_attention"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    """
    Binary deepfake image dataset.

    Reads image paths and labels from a metadata CSV file. The CSV must
    contain at least these columns:

        filename  — path relative to ``dataset_root``
        label     — 0 (real) or 1 (fake)
        split     — "train", "val", or "test"

    Parameters
    ----------
    dataset_root : str | Path
        Root directory of the image dataset.
    metadata_file : str | Path
        Path to the metadata CSV file.
    split : str
        Which split to load: "train", "val", or "test".
    image_size : int
        Images are resized to (image_size, image_size).
    transform : callable, optional
        Custom torchvision transform that overrides the default pipeline.
        If provided, ``normalization`` is ignored.
    normalization : (mean, std) | None, optional
        Channel-wise normalisation applied after ToTensor().  Set to None
        to skip normalisation (required for Meso4).  Defaults to ImageNet
        statistics when not specified.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        metadata_file: str | Path,
        split: str = "train",
        image_size: int = 256,
        transform=None,
        normalization: Optional[tuple[list[float], list[float]]] = (
            _IMAGENET_MEAN, _IMAGENET_STD
        ),
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split

        # ------------------------------------------------- load metadata
        metadata_path = Path(metadata_file)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        all_metadata = pd.read_csv(metadata_path)

        required_cols = {"filename", "label", "split"}
        missing = required_cols - set(all_metadata.columns)
        if missing:
            raise ValueError(
                f"Metadata CSV is missing required columns: {missing}"
            )

        self.metadata = (
            all_metadata[all_metadata["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.metadata) == 0:
            raise RuntimeError(
                f"No samples found for split={split!r} in {metadata_path}. "
                "Check the 'split' column values in the CSV."
            )

        # Validate labels
        unique_labels = set(self.metadata["label"].unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Unexpected label values in split={split!r}: {unique_labels}. "
                "Expected values in {0, 1}."
            )

        # ------------------------------------------------- build transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform(
                split=split,
                image_size=image_size,
                normalization=normalization,
            )

        n_real = (self.metadata["label"] == 0).sum()
        n_fake = (self.metadata["label"] == 1).sum()
        print(
            f"[{split.upper():5s}] {len(self.metadata):6d} images  "
            f"(real={n_real}, fake={n_fake})"
        )

    # ------------------------------------------------------------------
    # Transform builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform(
        split: str,
        image_size: int,
        normalization: Optional[tuple[list[float], list[float]]],
    ):
        """Build the default transform pipeline for a given split."""
        ops = [transforms.Resize((image_size, image_size))]

        if split == "train":
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                ),
            ]

        ops.append(transforms.ToTensor())

        if normalization is not None:
            mean, std = normalization
            ops.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(ops)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]

        image_path = self.dataset_root / row["filename"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}  "
                f"(dataset_root={self.dataset_root}, filename={row['filename']})"
            )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def class_counts(self) -> dict[str, int]:
        """Return a dict with 'real' and 'fake' sample counts."""
        return {
            "real": int((self.metadata["label"] == 0).sum()),
            "fake": int((self.metadata["label"] == 1).sum()),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Adjust paths for standalone execution
    ds = DeepfakeDataset(
        dataset_root="../dataset",
        metadata_file="../metadata.csv",
        split="train",
        normalization=get_normalization("meso4"),
    )

    print(f"\nDataset size : {len(ds)}")
    image, label = ds[0]
    print(f"Image shape  : {image.shape}")
    print(f"Image range  : [{image.min():.3f}, {image.max():.3f}]")
    print(f"Label        : {label.item()}")
