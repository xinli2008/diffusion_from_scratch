import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import os

class  MNISTDataset(Dataset):
    def __init__(self, root = "./", train = True, image_size = 64):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Check if the dataset already exists in the specified path
        raw_folder = os.path.join(root, "MNIST", "raw")
        if not (os.path.exists(raw_folder) and os.listdir(raw_folder)):
            download = True
        else:
            download = False
            
        self.dataset = torchvision.datasets.MNIST(
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