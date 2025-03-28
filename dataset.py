import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, root="./", dataset_type="MNIST", train=True, image_size=256):
        if dataset_type == "MNIST":
            normalize_mean_std = (0.5,), (0.5,)
            dataset_class = torchvision.datasets.MNIST
            raw_folder = os.path.join(root, "MNIST", "raw")
        elif dataset_type == "CIFAR-10":
            normalize_mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            dataset_class = torchvision.datasets.CIFAR10
            raw_folder = os.path.join(root, "CIFAR10", "raw")
        else:
            raise ValueError("Unsupported dataset type. Choose 'MNIST' or 'CIFAR10'.")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize_mean_std),
        ])

        if not os.path.exists(raw_folder):
            download = True
        else:
            download = False

        self.dataset = dataset_class(
            root=root,
            train=train,
            download=download,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label