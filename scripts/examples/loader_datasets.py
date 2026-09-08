"""Two independent Dataset patterns. Run with a class-directory image root."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from imgread import Loader


def supported_extension(path):
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def numpy_to_tensor(array):
    """Convert Loader's HWC uint8 NumPy output to contiguous CHW float32."""
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255)


def image_folder(root):
    # ImageFolder passes an individual filename to its loader callback.
    return ImageFolder(root, loader=Loader(), is_valid_file=supported_extension,
                       transform=numpy_to_tensor)


class IndexedDataset(Dataset):
    def __init__(self, samples):
        self.samples = tuple((str(path), label) for path, label in samples)
        self.labels = tuple(label for _, label in self.samples)
        self.loader = Loader(path for path, _ in self.samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return numpy_to_tensor(self.loader[index]), self.labels[index]


def indexed_dataset(root):
    folder = image_folder(root)
    return IndexedDataset(folder.samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("path", "index"), default="path")
    parser.add_argument("--workers", type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    dataset = image_folder(args.root) if args.mode == "path" else indexed_dataset(args.root)
    # Batch size one also accepts images of different dimensions.
    batches = DataLoader(dataset, batch_size=1, num_workers=args.workers,
                         prefetch_factor=2 if args.workers else None,
                         persistent_workers=bool(args.workers),
                         multiprocessing_context="spawn" if args.workers else None,
                         pin_memory=False)
    for images, labels in batches:
        print(tuple(images.shape), labels.tolist())


if __name__ == "__main__":
    main()
