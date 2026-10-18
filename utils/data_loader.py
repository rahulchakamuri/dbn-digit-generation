"""
MNIST Dataset Loading and Preprocessing Pipeline.

Provides utilities for:
1. Downloading and caching the official MNIST dataset.
2. Normalizing pixel values to [0.0, 1.0].
3. Flattening 28x28 images into 784-dimensional Bernoulli visible vectors.
4. Flexible subsetting for Quick Demo Mode vs Full Training Mode.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FlattenTransform:
    """Transform that normalizes and flattens a 28x28 image to a 784-dim tensor."""
    def __init__(self, binarize: bool = False, threshold: float = 0.5):
        self.binarize = binarize
        self.threshold = threshold

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor is shape (1, 28, 28) with values in [0.0, 1.0]
        flattened = tensor.view(-1)
        if self.binarize:
            return (flattened > self.threshold).float()
        return flattened


def get_transforms(binarize: bool = False, threshold: float = 0.5) -> transforms.Compose:
    """Return the torchvision transform pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),  # Scales PIL image to [0.0, 1.0]
        FlattenTransform(binarize=binarize, threshold=threshold)
    ])


def load_mnist(
    data_dir: Path,
    batch_size: int = 64,
    subset_size: Optional[int] = None,
    binarize: bool = False,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load MNIST train and test sets, apply flattening/normalization, and return DataLoaders.

    Args:
        data_dir: Directory to cache the MNIST files
        batch_size: Mini-batch size for DataLoader
        subset_size: Optional integer to subset training data (for fast demo training)
        binarize: Whether to strictly threshold pixel values to {0, 1}
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducible subsetting

    Returns:
        (train_loader, test_loader, metadata_dict)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transforms(binarize=binarize)

    try:
        train_dataset = datasets.MNIST(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download or load MNIST dataset from {data_dir}. "
            f"Please verify your internet connection or disk permissions. Details: {e}"
        ) from e

    # Subsetting for demo mode if requested
    total_train = len(train_dataset)
    if subset_size is not None and 0 < subset_size < total_train:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator)[:subset_size].tolist()
        active_train_dataset = Subset(train_dataset, indices)
        used_train_samples = subset_size
    else:
        active_train_dataset = train_dataset
        used_train_samples = total_train

    train_loader = DataLoader(
        active_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 is safe on Windows
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    metadata = {
        "dataset_name": "MNIST Handwritten Digits",
        "total_train_available": total_train,
        "active_train_samples": used_train_samples,
        "test_samples": len(test_dataset),
        "input_shape": (28, 28),
        "flattened_dim": 784,
        "pixel_range": "[0.0, 1.0]",
        "binarized": binarize,
        "num_classes": 10
    }

    return train_loader, test_loader, metadata


def get_sample_real_images(
    data_dir: Path,
    num_samples: int = 16,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fetch a fixed batch of real MNIST images for visual comparison.

    Returns:
        images: Float tensor of shape (num_samples, 28, 28)
        labels: Long tensor of shape (num_samples,)
    """
    # Load without flattening
    raw_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    torch.manual_seed(seed)
    indices = torch.randperm(len(raw_dataset))[:num_samples]

    images_list = []
    labels_list = []
    for idx in indices:
        img, lbl = raw_dataset[idx.item()]
        images_list.append(img.squeeze(0))  # shape (28, 28)
        labels_list.append(lbl)

    return torch.stack(images_list), torch.tensor(labels_list)


def get_dataset_info(data_dir: Path) -> Dict[str, Any]:
    """Return descriptive dictionary about MNIST for dashboard explorer."""
    return {
        "name": "MNIST (Modified National Institute of Standards and Technology)",
        "creator": "Yann LeCun, Corinna Cortes, Christopher J.C. Burges",
        "classes": [str(i) for i in range(10)],
        "train_count": 60000,
        "test_count": 10000,
        "dimensions": "28 x 28 pixels (grayscale)",
        "features": 784,
        "value_range": "Normalized floats [0.0, 1.0] (0 = background/black, 1 = foreground stroke/white)"
    }
