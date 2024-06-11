import torch

image_size = 48
timestep = 1000
device = "cuda:0" if torch.cuda.is_available() else "cpu"